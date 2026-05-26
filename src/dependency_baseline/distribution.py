"""Distribution/prototype regressors for observed single-cell bags."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import logging
from pathlib import Path
import time
from collections.abc import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from dependency_baseline.artifacts import (
    ArtifactStore,
    CvPaths,
    checkpoint_path,
    create_run_context,
    job_key,
    manifest_base,
    set_seed,
    write_cv_config,
    write_json,
)
from dependency_baseline.config import BaselineConfig, SelectionConfig
from dependency_baseline.datasets import (
    filter_names,
    merge_selection,
    repeated_stratified_splitter,
    split_manifest,
    stratification_bins,
)
from dependency_baseline.evaluation import (
    _ensemble_metrics,
    _ensemble_predictions,
    _train_gene_lookup,
)
from dependency_baseline.metrics import ranking_metrics, regression_metrics
from dependency_baseline.models import (
    _inner_validation_split,
    _resolve_torch_device,
    _set_torch_seed,
)
from dependency_baseline.single_cell import (
    _attach_run_log,
    _cv_paths,
    _detach_run_log,
    _feature_token,
    _utc_now,
    load_cell_bag_data,
)


LOGGER = logging.getLogger(__name__)


CellBags = Sequence[np.ndarray]


@dataclass(frozen=True)
class DistributionBagData:
    bags: tuple[np.ndarray, ...]
    y: np.ndarray
    n_cells: np.ndarray
    genes: np.ndarray
    metadata: pd.DataFrame
    feature_set: str
    input_dim: int
    control_bag: np.ndarray
    source_bags: tuple[np.ndarray, ...] = ()
    source_control_bags: tuple[np.ndarray, ...] = ()
    source_genes: np.ndarray | None = None
    source_n_cells: np.ndarray | None = None


class FrozenGMMDistributionRegressor:
    """Frozen diagonal-GMM occupancy features followed by a sklearn regressor."""

    def __init__(
        self,
        gmm: GaussianMixture,
        estimator: object,
        view: str,
        model_name: str,
    ) -> None:
        self.gmm = gmm
        self.estimator = estimator
        self.view = view
        self.model_name = model_name
        self.control_feature_ = _occupancy_feature(gmm, None)

    def fit(
        self,
        bags: CellBags,
        y: np.ndarray,
        control_bag: np.ndarray,
    ) -> "FrozenGMMDistributionRegressor":
        self.control_feature_ = _occupancy_feature(self.gmm, control_bag)
        x_train = _features_for_bags(
            self.gmm,
            bags,
            self.view,
            self.control_feature_,
        )
        self.estimator.fit(x_train, y)
        self.n_features_in_ = int(x_train.shape[1])
        return self

    def predict(self, bags: CellBags) -> np.ndarray:
        x = _features_for_bags(self.gmm, bags, self.view, self.control_feature_)
        return np.asarray(self.estimator.predict(x), dtype=np.float64)

    def predict_source_weighted(self, data: DistributionBagData) -> np.ndarray:
        if self.view != "deltap" or not data.source_bags:
            return self.predict(data.bags)
        x = _source_weighted_features(
            self.gmm,
            data,
            self.view,
            self.control_feature_,
        )
        return np.asarray(self.estimator.predict(x), dtype=np.float64)


class CloudPredDistributionRegressor:
    """Trainable diagonal-Gaussian prototype occupancy regressor."""

    def __init__(
        self,
        *,
        input_dim: int,
        n_components: int,
        view: str,
        initial_means: np.ndarray,
        initial_vars: np.ndarray,
        hidden_units: tuple[int, ...] = (32,),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        max_epochs: int = 300,
        patience: int = 30,
        validation_fraction: float = 0.15,
        validation_bins: int = 10,
        batch_size: int = 32,
        max_cells_per_bag: int = 512,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.input_dim = input_dim
        self.n_components = n_components
        self.view = view
        self.initial_means = initial_means.astype(np.float32)
        self.initial_vars = initial_vars.astype(np.float32)
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.validation_bins = validation_bins
        self.batch_size = batch_size
        self.max_cells_per_bag = max_cells_per_bag
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        bags: CellBags,
        y: np.ndarray,
        control_bag: np.ndarray,
    ) -> "CloudPredDistributionRegressor":
        self.control_bag_ = control_bag.astype(np.float32)
        self.torch_version_ = torch.__version__
        self.device_ = _resolve_torch_device(self.device, torch)
        _set_torch_seed(self.random_state, torch)
        rng = np.random.default_rng(self.random_state)
        y_array = np.asarray(y, dtype=np.float32)
        train_idx, valid_idx = _inner_validation_split(
            y_array,
            validation_fraction=float(self.validation_fraction),
            validation_bins=int(self.validation_bins),
            random_state=int(self.random_state),
        )
        self.model_ = _CloudPredModule(
            input_dim=int(self.input_dim),
            n_components=int(self.n_components),
            view=str(self.view),
            initial_means=self.initial_means,
            initial_vars=self.initial_vars,
            hidden_units=tuple(int(value) for value in self.hidden_units),
        ).to(self.device_)
        control_tensor, _control_mask = _one_bag_tensor(
            control_bag,
            int(self.max_cells_per_bag),
            rng,
            self.device_,
        )
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        best_loss = np.inf
        best_state = copy.deepcopy(self.model_.state_dict())
        stale_epochs = 0
        indices = train_idx.copy()
        for epoch in range(int(self.max_epochs)):
            rng.shuffle(indices)
            for start in range(0, len(indices), int(self.batch_size)):
                batch_indices = indices[start : start + int(self.batch_size)]
                x_batch, mask_batch = _batch_tensors(
                    [bags[index] for index in batch_indices],
                    int(self.max_cells_per_bag),
                    rng,
                    self.device_,
                )
                target = torch.as_tensor(
                    y_array[batch_indices],
                    dtype=torch.float32,
                    device=self.device_,
                )
                pred = self.model_(x_batch, mask_batch, control_tensor)
                loss = ((pred - target) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.train_mse_loss_ = float(loss.detach().cpu())
            validation_loss = self._loss_on_indices(
                bags,
                y_array,
                valid_idx,
                control_tensor,
                rng,
            )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(self.patience):
                    self.n_epochs_run_ = epoch + 1
                    break
        else:
            self.n_epochs_run_ = int(self.max_epochs)
        self.best_epoch_ = int(self.n_epochs_run_ - stale_epochs)
        self.best_validation_loss_ = float(best_loss)
        self.model_.load_state_dict(best_state)
        return self

    def predict(self, bags: CellBags) -> np.ndarray:
        rng = np.random.default_rng(self.random_state + 1009)
        self.model_.eval()
        predictions: list[np.ndarray] = []
        control_tensor = self._default_control_tensor(rng)
        with torch.no_grad():
            for start in range(0, len(bags), int(self.batch_size)):
                batch = bags[start : start + int(self.batch_size)]
                x_batch, mask_batch = _batch_tensors(
                    batch,
                    int(self.max_cells_per_bag),
                    rng,
                    self.device_,
                )
                pred = self.model_(x_batch, mask_batch, control_tensor)
                predictions.append(pred.detach().cpu().numpy())
        return np.concatenate(predictions).astype(np.float64)

    def predict_source_weighted(self, data: DistributionBagData) -> np.ndarray:
        if self.view != "deltap" or not data.source_bags:
            return self.predict(data.bags)
        features = self._source_weighted_cloud_features(data)
        return self._predict_from_features(features)

    def _loss_on_indices(
        self,
        bags: CellBags,
        y: np.ndarray,
        indices: np.ndarray,
        control_tensor: object,
        rng: np.random.Generator,
    ) -> float:
        self.model_.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, len(indices), int(self.batch_size)):
                batch_indices = indices[start : start + int(self.batch_size)]
                x_batch, mask_batch = _batch_tensors(
                    [bags[index] for index in batch_indices],
                    int(self.max_cells_per_bag),
                    rng,
                    self.device_,
                )
                target = torch.as_tensor(
                    y[batch_indices],
                    dtype=torch.float32,
                    device=self.device_,
                )
                pred = self.model_(x_batch, mask_batch, control_tensor)
                losses.append(float(((pred - target) ** 2).mean().detach().cpu()))
        self.model_.train()
        return float(np.mean(losses)) if losses else np.inf

    def _default_control_tensor(self, rng: np.random.Generator) -> object:
        if not hasattr(self, "control_bag_"):
            msg = "CloudPredDistributionRegressor is missing fitted control_bag_"
            raise RuntimeError(msg)
        tensor, _mask = _one_bag_tensor(
            self.control_bag_,
            int(self.max_cells_per_bag),
            rng,
            self.device_,
        )
        return tensor

    def _source_weighted_cloud_features(self, data: DistributionBagData) -> np.ndarray:
        if data.source_genes is None or data.source_n_cells is None:
            msg = "External source metadata is required for source-weighted deltap"
            raise ValueError(msg)
        rng = np.random.default_rng(self.random_state + 7919)
        source_features = []
        self.model_.eval()
        with torch.no_grad():
            for bag, control in zip(
                data.source_bags,
                data.source_control_bags,
                strict=True,
            ):
                x_tensor, mask_tensor = _batch_tensors(
                    [bag],
                    int(self.max_cells_per_bag),
                    rng,
                    self.device_,
                )
                control_tensor, _control_mask = _one_bag_tensor(
                    control,
                    int(self.max_cells_per_bag),
                    rng,
                    self.device_,
                )
                feature = self.model_.features(
                    x_tensor,
                    mask_tensor,
                    control_tensor,
                )
                source_features.append(feature.detach().cpu().numpy()[0])
        return _weighted_gene_matrix(
            np.vstack(source_features),
            data.source_genes,
            data.source_n_cells,
            data.genes,
        )

    def _predict_from_features(self, features: np.ndarray) -> np.ndarray:
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, features.shape[0], int(self.batch_size)):
                x = torch.as_tensor(
                    features[start : start + int(self.batch_size)],
                    dtype=torch.float32,
                    device=self.device_,
                )
                preds.append(self.model_.predict_features(x).detach().cpu().numpy())
        return np.concatenate(preds).astype(np.float64)


class _CloudPredModule(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        n_components: int,
        view: str,
        initial_means: np.ndarray,
        initial_vars: np.ndarray,
        hidden_units: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_components = n_components
        self.view = view
        self.means = nn.Parameter(torch.as_tensor(initial_means, dtype=torch.float32))
        self.logvars = nn.Parameter(
            torch.log(
                torch.as_tensor(initial_vars, dtype=torch.float32).clamp_min(1e-4)
            )
        )
        feature_dim = n_components * (2 if view == "deltap" else 1)
        layers: list[nn.Module] = []
        previous = feature_dim
        for hidden in hidden_units:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        x: object,
        mask: object,
        control: object,
    ) -> object:
        return self.predict_features(self.features(x, mask, control))

    def features(self, x: object, mask: object, control: object) -> object:
        p = self._occupancy(x, mask)
        if self.view != "deltap":
            return p
        control_mask = torch.ones(
            (1, control.shape[1]),
            dtype=torch.bool,
            device=control.device,
        )
        control_p = self._occupancy(control, control_mask).expand_as(p)
        return torch.cat([p, p - control_p], dim=1)

    def predict_features(self, features: object) -> object:
        return self.head(features).squeeze(-1)

    def _occupancy(self, x: object, mask: object) -> object:
        logvar = self.logvars.clamp(-12.0, 8.0)
        var = logvar.exp()
        diff = x.unsqueeze(2) - self.means.view(1, 1, self.n_components, -1)
        scaled = diff.square() / var.view(1, 1, self.n_components, -1)
        log_prob = -0.5 * scaled.sum(-1)
        log_prob = log_prob - 0.5 * logvar.sum(-1).view(1, 1, self.n_components)
        log_prob = log_prob.masked_fill(~mask.unsqueeze(-1), -1e9)
        resp = log_prob.softmax(dim=-1) * mask.unsqueeze(-1).to(log_prob.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(log_prob.dtype)
        return resp.sum(dim=1) / denom


def run_distribution_cv(
    config: BaselineConfig,
    bags_npz: Path | None = None,
    *,
    run_id: str | None = None,
    resume: bool = False,
    selection: SelectionConfig | None = None,
    command: tuple[str, ...] = (),
    config_path: Path | None = None,
    log_file: Path | None = None,
) -> CvPaths:
    """Run distribution/prototype CV for observed single-cell bags."""
    set_seed(config.experiment.seed)
    merged_selection = merge_selection(config.selection, selection)
    feature_sets = (
        (_feature_set_from_bag_path(bags_npz),)
        if bags_npz is not None
        else filter_names(config.single_cell.feature_sets, merged_selection.features)
    )
    if not feature_sets:
        msg = "No distribution feature sets selected"
        raise ValueError(msg)
    bag_paths = (
        (bags_npz,)
        if bags_npz is not None
        else tuple(
            _resolve_cell_bags_npz(config.data.output_dir, fs)
            for fs in feature_sets
        )
    )
    context = create_run_context(
        config=config,
        features_npz=bag_paths[0],
        run_id=run_id,
        resume=resume,
        command=command,
        config_path=config_path,
    )
    actual_log_file, handler = _attach_run_log(context.run_dir, log_file)
    store = ArtifactStore(
        context.run_dir,
        config.experiment.human_result_tables,
        config.experiment.machine_result_format,
        config.experiment.topk_candidates,
        config.experiment.save_predictions,
        config.experiment.save_rankings,
    )
    manifest = manifest_base(config, context, "run-distribution-cv", resume)
    write_json(
        context.run_dir / "run_manifest.json",
        {**manifest, "status": "running", "log_file": str(actual_log_file)},
    )
    write_cv_config(config, context.feature_path, context.run_dir / "cv_config.json")
    try:
        for bag_path in bag_paths:
            data = load_distribution_bag_data(bag_path, config)
            _execute_distribution_cv(
                config,
                context.run_dir,
                store,
                data,
                selection,
                resume,
            )
        write_json(
            context.run_dir / "run_manifest.json",
            {
                **manifest,
                "status": "completed",
                "ended_at": _utc_now(),
                "log_file": str(actual_log_file),
            },
        )
    except Exception:
        write_json(
            context.run_dir / "run_manifest.json",
            {
                **manifest,
                "status": "failed",
                "ended_at": _utc_now(),
                "log_file": str(actual_log_file),
            },
        )
        raise
    finally:
        _detach_run_log(handler)
    return _cv_paths(context.run_dir)


def evaluate_distribution_external(
    config: BaselineConfig,
    *,
    run_dir: Path,
    external_bags_npz: Path,
    external_name: str,
    feature_set: str | None = None,
) -> tuple[Path, Path]:
    """Evaluate distribution/prototype CV checkpoints on external bags."""
    store = ArtifactStore(
        run_dir,
        config.experiment.human_result_tables,
        config.experiment.machine_result_format,
        config.experiment.topk_candidates,
        config.experiment.save_predictions,
        config.experiment.save_rankings,
    )
    external = load_distribution_bag_data(external_bags_npz, config)
    if feature_set is not None and external.feature_set != feature_set:
        msg = (
            f"External bags feature_set={external.feature_set!r} does not match "
            f"requested feature_set={feature_set!r}"
        )
        raise ValueError(msg)
    manifest = store.tables["model_manifest"]
    if manifest.empty:
        msg = f"Missing model_manifest under run directory: {run_dir}"
        raise FileNotFoundError(msg)
    keep = (
        (manifest["evaluation_scope"] == "internal_cv_all")
        & (manifest["feature_set"] == external.feature_set)
        & manifest["checkpoint_path"].notna()
    )
    selected = manifest.loc[keep].copy()
    if selected.empty:
        msg = "No distribution CV checkpoints found for external evaluation"
        raise ValueError(msg)
    for row in selected.itertuples(index=False):
        _evaluate_distribution_external_checkpoint(
            config,
            store,
            external,
            external_name,
            row,
        )
    _write_distribution_external_ensemble_results(config, store)
    return (
        run_dir / "artifacts" / "external_ensemble_metrics.parquet",
        run_dir / "artifacts" / "external_ensemble_predictions.parquet",
    )


def load_distribution_bag_data(
    path: Path,
    config: BaselineConfig,
) -> DistributionBagData:
    """Load bag arrays plus control/source-control arrays for distribution models."""
    base = load_cell_bag_data(path, config)
    payload = np.load(path, allow_pickle=True)
    source_bags: tuple[np.ndarray, ...] = ()
    source_control_bags: tuple[np.ndarray, ...] = ()
    source_genes: np.ndarray | None = None
    source_n_cells: np.ndarray | None = None
    if "source_cell_delta_pcs" in payload:
        source_bags = _bags_from_flat(
            payload["source_cell_delta_pcs"].astype(np.float32),
            payload["source_bag_offsets"].astype(np.int64),
        )
        source_control_bags = _bags_from_flat(
            payload["source_control_cell_delta_pcs"].astype(np.float32),
            payload["source_control_offsets"].astype(np.int64),
        )
        source_genes = payload["source_perturbation_gene"].astype(str)
        source_n_cells = payload["source_observed_n_cells"].astype(np.float64)
    if "control_cell_delta_pcs" in payload:
        control_bag = payload["control_cell_delta_pcs"].astype(np.float32)
    elif source_control_bags:
        control_bag = np.vstack(source_control_bags).astype(np.float32)
    else:
        msg = (
            f"Bag artifact {path} is missing control embeddings. "
            "Rebuild cell bags before running distribution regression."
        )
        raise ValueError(msg)
    return DistributionBagData(
        bags=base.bags,
        y=base.y,
        n_cells=base.n_cells,
        genes=base.genes,
        metadata=base.metadata,
        feature_set=base.feature_set,
        input_dim=base.n_pcs,
        control_bag=control_bag,
        source_bags=source_bags,
        source_control_bags=source_control_bags,
        source_genes=source_genes,
        source_n_cells=source_n_cells,
    )


def _execute_distribution_cv(
    config: BaselineConfig,
    run_dir: Path,
    store: ArtifactStore,
    data: DistributionBagData,
    selection: SelectionConfig | None,
    resume: bool,
) -> None:
    merged_selection = merge_selection(config.selection, selection)
    scopes = [
        ("internal_cv_all", np.arange(len(data.y), dtype=np.int64), (data.feature_set,))
    ]
    if merged_selection.scopes is not None:
        allowed_scopes = set(merged_selection.scopes)
        scopes = [scope for scope in scopes if scope[0] in allowed_scopes]
    if config.experiment.save_splits:
        store.write_splits(split_manifest(config, scopes, data.y, data.genes))
    selected_features = filter_names((data.feature_set,), merged_selection.features)
    selected_models = filter_names(
        _distribution_model_names(config, data),
        merged_selection.models,
    )
    selected_weightings = filter_names(
        tuple(config.distribution.weightings),
        merged_selection.weightings,
    )
    splitter = repeated_stratified_splitter(config)
    for evaluation_scope, row_indices, _allowed_features in scopes:
        y_bins = stratification_bins(data.y[row_indices], config.cv.stratify_bins)
        for fold_index, (train_local, test_local) in enumerate(
            splitter.split(row_indices, y_bins)
        ):
            if (
                merged_selection.folds is not None
                and fold_index not in merged_selection.folds
            ):
                continue
            train_idx = row_indices[train_local]
            test_idx = row_indices[test_local]
            for feature_name in selected_features:
                del feature_name
                for model_name in selected_models:
                    for weighting in selected_weightings:
                        job = job_key(
                            evaluation_scope,
                            fold_index,
                            data.feature_set,
                            model_name,
                            weighting,
                        )
                        if resume and job in store.completed_jobs:
                            continue
                        _fit_distribution_fold(
                            config,
                            run_dir,
                            store,
                            data,
                            evaluation_scope,
                            fold_index,
                            train_idx,
                            test_idx,
                            model_name,
                            weighting,
                            job,
                        )


def _fit_distribution_fold(
    config: BaselineConfig,
    run_dir: Path,
    store: ArtifactStore,
    data: DistributionBagData,
    evaluation_scope: str,
    fold_index: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    weighting: str,
    job: str,
) -> None:
    del weighting
    spec = _parse_distribution_model_name(model_name)
    train_bags = tuple(data.bags[index] for index in train_idx)
    test_bags = tuple(data.bags[index] for index in test_idx)
    gmm = _fit_fold_gmm(config, data, train_idx, spec["k"], fold_index)
    started = time.perf_counter()
    if spec["family"] == "cloudpred":
        model = CloudPredDistributionRegressor(
            input_dim=data.input_dim,
            n_components=spec["k"],
            view=spec["view"],
            initial_means=gmm.means_.astype(np.float32),
            initial_vars=gmm.covariances_.astype(np.float32),
            hidden_units=config.distribution.cloudpred_hidden_units,
            learning_rate=config.distribution.cloudpred_learning_rate,
            weight_decay=config.distribution.cloudpred_weight_decay,
            max_epochs=config.distribution.cloudpred_max_epochs,
            patience=config.distribution.cloudpred_patience,
            validation_fraction=config.distribution.cloudpred_validation_fraction,
            validation_bins=config.cv.stratify_bins,
            batch_size=config.distribution.cloudpred_batch_size,
            max_cells_per_bag=config.distribution.max_cells_per_bag,
            random_state=config.cv.random_state + fold_index,
            device=config.distribution.device,
        )
        model.control_bag_ = data.control_bag.astype(np.float32)
        model.fit(train_bags, data.y[train_idx], data.control_bag)
    else:
        estimator = _supervised_estimator(config, spec, fold_index)
        model = FrozenGMMDistributionRegressor(
            gmm=gmm,
            estimator=estimator,
            view=spec["view"],
            model_name=model_name,
        )
        model.fit(train_bags, data.y[train_idx], data.control_bag)
    fit_seconds = time.perf_counter() - started
    pred = model.predict(test_bags)
    path = checkpoint_path(
        run_dir,
        evaluation_scope,
        fold_index,
        data.feature_set,
        model_name,
        "unweighted",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    metric_row = {
        "job_key": job,
        "evaluation_scope": evaluation_scope,
        "fold": fold_index,
        "feature_set": data.feature_set,
        "model": model_name,
        "weighting": "unweighted",
        "fit_seconds": fit_seconds,
        **regression_metrics(data.y[test_idx], pred),
        **ranking_metrics(data.y[test_idx], pred, config.cv.essential_thresholds),
    }
    predictions = pd.DataFrame(
        {
            "job_key": job,
            "evaluation_scope": evaluation_scope,
            "fold": fold_index,
            "feature_set": data.feature_set,
            "model": model_name,
            "weighting": "unweighted",
            "perturbation_gene": data.genes[test_idx],
            "y_true": data.y[test_idx],
            "y_pred": pred,
        }
    ).merge(
        data.metadata[["perturbation_gene", "observed_n_cells"]],
        on="perturbation_gene",
        how="left",
    )
    store.append_fold_result(
        metric_row,
        predictions,
        {
            **metric_row,
            "checkpoint_path": str(path),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "distribution_family": spec["family"],
            "distribution_view": spec["view"],
            "distribution_components": spec["k"],
            "torch_device": getattr(model, "device_", None),
            "torch_version": getattr(model, "torch_version_", None),
            "torch_best_epoch": getattr(model, "best_epoch_", None),
            "torch_n_epochs_run": getattr(model, "n_epochs_run_", None),
            "torch_best_validation_loss": getattr(model, "best_validation_loss_", None),
        },
    )
    LOGGER.info("Completed distribution CV fold=%s model=%s", fold_index, model_name)


def _evaluate_distribution_external_checkpoint(
    config: BaselineConfig,
    store: ArtifactStore,
    external: DistributionBagData,
    external_name: str,
    row: object,
) -> None:
    model = joblib.load(Path(row.checkpoint_path))
    if hasattr(model, "predict_source_weighted"):
        pred = model.predict_source_weighted(external)
    else:
        pred = model.predict(external.bags)
    evaluation_scope = f"external:{external_name}"
    job = job_key(
        evaluation_scope,
        int(row.fold),
        str(row.feature_set),
        str(row.model),
        str(row.weighting),
    )
    metric_row = {
        "job_key": job,
        "evaluation_scope": evaluation_scope,
        "fold": int(row.fold),
        "feature_set": str(row.feature_set),
        "model": str(row.model),
        "weighting": str(row.weighting),
        "fit_seconds": float(row.fit_seconds),
        **regression_metrics(external.y, pred),
        **ranking_metrics(external.y, pred, config.cv.essential_thresholds),
    }
    predictions = pd.DataFrame(
        {
            "job_key": job,
            "evaluation_scope": evaluation_scope,
            "fold": int(row.fold),
            "feature_set": str(row.feature_set),
            "model": str(row.model),
            "weighting": str(row.weighting),
            "perturbation_gene": external.genes,
            "y_true": external.y,
            "y_pred": pred,
        }
    )
    metadata = external.metadata.copy()
    if "observed_n_cells" in metadata.columns:
        metadata["external_n_cells"] = metadata["observed_n_cells"].astype(float)
    predictions = predictions.merge(metadata, on="perturbation_gene", how="left")
    store.append_external_result(metric_row, predictions)


def _write_distribution_external_ensemble_results(
    config: BaselineConfig,
    store: ArtifactStore,
) -> None:
    predictions = store.tables["predictions"]
    if predictions.empty or "evaluation_scope" not in predictions:
        return
    external = predictions.loc[
        predictions["evaluation_scope"].astype(str).str.startswith("external:")
    ].copy()
    if external.empty:
        return
    train_lookup = _train_gene_lookup(store.tables["splits"])
    primary = _ensemble_predictions(external, train_lookup, target_heldout=False)
    heldout = _ensemble_predictions(external, train_lookup, target_heldout=True)
    ensemble_predictions = pd.concat([primary, heldout], ignore_index=True)
    metrics = _ensemble_metrics(config, ensemble_predictions)
    store.write_external_ensemble_results(metrics, ensemble_predictions)


def _fit_fold_gmm(
    config: BaselineConfig,
    data: DistributionBagData,
    train_idx: np.ndarray,
    n_components: int,
    fold_index: int,
) -> GaussianMixture:
    if config.distribution.covariance_type != "diag":
        msg = "Only diagonal covariance is supported for distribution regression"
        raise ValueError(msg)
    cells = [data.bags[index] for index in train_idx]
    if config.distribution.prototype_fit_scope == "train_genes_plus_controls":
        cells.append(data.control_bag)
    elif config.distribution.prototype_fit_scope != "train_genes_only":
        msg = (
            "Unsupported prototype_fit_scope="
            f"{config.distribution.prototype_fit_scope!r}"
        )
        raise ValueError(msg)
    matrix = np.vstack(cells).astype(np.float32)
    max_cells = config.distribution.max_gmm_fit_cells
    if max_cells is not None and matrix.shape[0] > max_cells:
        rng = np.random.default_rng(config.cv.random_state + fold_index)
        keep = rng.choice(matrix.shape[0], size=int(max_cells), replace=False)
        matrix = matrix[np.sort(keep)]
    gmm = GaussianMixture(
        n_components=int(n_components),
        covariance_type="diag",
        reg_covar=1e-4,
        random_state=config.cv.random_state + fold_index,
        max_iter=200,
        n_init=1,
    )
    gmm.fit(matrix)
    gmm.fit_gene_names_ = data.genes[train_idx].astype(str)
    gmm.fit_cell_count_ = int(matrix.shape[0])
    return gmm


def _supervised_estimator(
    config: BaselineConfig,
    spec: dict[str, object],
    fold_index: int,
) -> object:
    head = str(spec["head"])
    if head == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=float(spec["alpha"])),
        )
    if head == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=int(config.distribution.random_forest_n_estimators),
                min_samples_leaf=int(config.distribution.random_forest_min_samples_leaf),
                random_state=config.cv.random_state + fold_index,
                n_jobs=int(config.distribution.random_forest_n_jobs),
            ),
        )
    if head == "mlp":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=tuple(config.distribution.mlp_hidden_units),
                max_iter=int(config.distribution.mlp_max_epochs),
                early_stopping=True,
                n_iter_no_change=int(config.distribution.mlp_patience),
                random_state=config.cv.random_state + fold_index,
            ),
        )
    msg = f"Unsupported frozen distribution head: {head}"
    raise ValueError(msg)


def _distribution_model_names(
    config: BaselineConfig,
    data: DistributionBagData,
) -> tuple[str, ...]:
    token = _feature_token_for_distribution(data)
    primary_ks = tuple(int(value) for value in config.distribution.component_counts)
    sensitivity_ks = tuple(
        int(value) for value in config.distribution.sensitivity_component_counts
    )
    views = tuple(config.distribution.views)
    names: list[str] = []
    for k in (*primary_ks, *sensitivity_ks):
        for view in views:
            for alpha in config.distribution.ridge_alphas:
                names.append(
                    f"gmm_{token}_k{k}_{view}_ridge_alpha{_param_token(alpha)}"
                )
            names.append(f"gmm_{token}_k{k}_{view}_rf")
            if token.startswith("scvi") and k in primary_ks:
                names.append(f"gmm_{token}_k{k}_{view}_mlp")
    for k in primary_ks:
        for view in views:
            names.append(f"cloudpred_{token}_k{k}_{view}")
    return tuple(names)


def _parse_distribution_model_name(model_name: str) -> dict[str, object]:
    parts = model_name.split("_")
    if parts[0] == "cloudpred":
        return {
            "family": "cloudpred",
            "token": parts[1],
            "k": int(parts[2].replace("k", "")),
            "view": parts[3],
            "head": "cloudpred",
        }
    if parts[0] != "gmm":
        msg = f"Unsupported distribution model name: {model_name}"
        raise ValueError(msg)
    head = parts[4]
    alpha = None
    if head == "ridge":
        alpha = float(parts[5].replace("alpha", "").replace("p", "."))
    return {
        "family": "gmm",
        "token": parts[1],
        "k": int(parts[2].replace("k", "")),
        "view": parts[3],
        "head": head,
        "alpha": alpha,
    }


def _features_for_bags(
    gmm: GaussianMixture,
    bags: CellBags,
    view: str,
    control_feature: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        [
            _feature_from_occupancy(
                _occupancy_feature(gmm, bag),
                view,
                control_feature,
            )
            for bag in bags
        ]
    ).astype(np.float32)


def _source_weighted_features(
    gmm: GaussianMixture,
    data: DistributionBagData,
    view: str,
    control_feature: np.ndarray,
) -> np.ndarray:
    if data.source_genes is None or data.source_n_cells is None:
        msg = "External source metadata is required for source-weighted deltap"
        raise ValueError(msg)
    source_features = []
    for bag, control in zip(data.source_bags, data.source_control_bags, strict=True):
        control_occ = _occupancy_feature(gmm, control)
        source_features.append(
            _feature_from_occupancy(_occupancy_feature(gmm, bag), view, control_occ)
        )
    del control_feature
    return _weighted_gene_matrix(
        np.vstack(source_features),
        data.source_genes,
        data.source_n_cells,
        data.genes,
    )


def _feature_from_occupancy(
    feature: np.ndarray,
    view: str,
    control_feature: np.ndarray,
) -> np.ndarray:
    k = (feature.shape[0] - 4)
    p = feature[:k]
    stats = feature[k:]
    if view == "centered":
        return feature
    if view == "deltap":
        control_p = control_feature[:k]
        return np.concatenate([p, p - control_p, stats]).astype(np.float32)
    msg = f"Unsupported distribution view: {view}"
    raise ValueError(msg)


def _occupancy_feature(gmm: GaussianMixture, bag: np.ndarray | None) -> np.ndarray:
    if bag is None:
        return np.zeros(gmm.n_components + 4, dtype=np.float32)
    responsibilities = gmm.predict_proba(bag.astype(np.float32))
    p = responsibilities.mean(axis=0)
    entropy = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
    effective_k = float(np.exp(entropy))
    confidence = float(np.max(responsibilities, axis=1).mean())
    nll = -float(gmm.score_samples(bag.astype(np.float32)).mean())
    return np.concatenate(
        [
            p.astype(np.float32),
            np.asarray([entropy, effective_k, confidence, nll], dtype=np.float32),
        ]
    )


def _weighted_gene_matrix(
    source_features: np.ndarray,
    source_genes: np.ndarray,
    source_n_cells: np.ndarray,
    genes: np.ndarray,
) -> np.ndarray:
    rows = []
    for gene in genes.astype(str):
        mask = source_genes.astype(str) == gene
        if not np.any(mask):
            msg = f"External source rows are missing gene {gene!r}"
            raise ValueError(msg)
        weights = source_n_cells[mask].astype(np.float64)
        weights = weights / weights.sum()
        rows.append((source_features[mask] * weights[:, None]).sum(axis=0))
    return np.vstack(rows).astype(np.float32)


def _bags_from_flat(cells: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(
        cells[offsets[index] : offsets[index + 1]]
        for index in range(len(offsets) - 1)
    )


def _batch_tensors(
    bags: CellBags,
    max_cells_per_bag: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[object, object]:
    sampled = [_sample_bag(bag, max_cells_per_bag, rng) for bag in bags]
    max_len = max(bag.shape[0] for bag in sampled)
    input_dim = sampled[0].shape[1]
    x = np.zeros((len(sampled), max_len, input_dim), dtype=np.float32)
    mask = np.zeros((len(sampled), max_len), dtype=bool)
    for index, bag in enumerate(sampled):
        x[index, : bag.shape[0], :] = bag
        mask[index, : bag.shape[0]] = True
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(mask, dtype=torch.bool, device=device),
    )


def _one_bag_tensor(
    bag: np.ndarray,
    max_cells_per_bag: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[object, object]:
    return _batch_tensors([bag], max_cells_per_bag, rng, device)


def _sample_bag(
    bag: np.ndarray,
    max_cells_per_bag: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if bag.shape[0] <= max_cells_per_bag:
        return bag.astype(np.float32)
    indices = rng.choice(bag.shape[0], size=int(max_cells_per_bag), replace=False)
    return bag[np.sort(indices)].astype(np.float32)


def _feature_set_from_bag_path(path: Path | None) -> str:
    if path is None:
        msg = "Cannot infer feature set from a missing bag path"
        raise ValueError(msg)
    payload = np.load(path, allow_pickle=True)
    if "feature_set" not in payload:
        return "single_cell_pc_delta"
    value = payload["feature_set"]
    if np.asarray(value).shape == ():
        return str(value.item())
    return str(np.asarray(value).astype(str).reshape(-1)[0])


def _feature_token_for_distribution(data: DistributionBagData) -> str:
    proxy = type(
        "_Proxy",
        (),
        {"feature_set": data.feature_set, "n_pcs": data.input_dim},
    )()
    return _feature_token(proxy)


def _resolve_cell_bags_npz(output_dir: Path, feature_set: str) -> Path:
    if feature_set == "single_cell_pc_delta":
        return (
            output_dir
            / "features"
            / "single_cell_bags"
            / "replogle_k562_single_cell_bags.npz"
        )
    return (
        output_dir
        / "features"
        / "single_cell_bags"
        / feature_set
        / f"replogle_k562_{feature_set}_bags.npz"
    )


def _param_token(value: float | int) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "neg").replace(".", "p")
