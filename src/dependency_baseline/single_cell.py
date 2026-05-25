"""Deep Sets models and runners for observed single-cell bag baselines."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from collections.abc import Sequence
import copy

import joblib
import numpy as np
import pandas as pd
from torch import nn

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
from dependency_baseline.metrics import ranking_metrics, regression_metrics
from dependency_baseline.models import sample_weights


CellBags = Sequence[np.ndarray]
FEATURE_SET_NAME = "single_cell_pc_delta"
MODEL_NAME_TEMPLATE = "deepsets_pca{n_pcs}_meanpool"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellBagData:
    bags: tuple[np.ndarray, ...]
    y: np.ndarray
    n_cells: np.ndarray
    genes: np.ndarray
    metadata: pd.DataFrame
    n_pcs: int


class DeepSetsRegressor:
    """Small Deep Sets regressor for bag-level GeneEffect prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...] = (128, 64),
        bag_hidden_units: tuple[int, ...] = (64,),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        max_epochs: int = 500,
        patience: int = 40,
        validation_fraction: float = 0.15,
        validation_bins: int = 10,
        max_cells_per_bag: int = 256,
        batch_size: int = 32,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.bag_hidden_units = bag_hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.validation_bins = validation_bins
        self.max_cells_per_bag = max_cells_per_bag
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        bags: CellBags,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "DeepSetsRegressor":
        """Fit the regressor from ragged bags and bag-level labels."""
        import torch

        from dependency_baseline.models import (
            _inner_validation_split,
            _resolve_torch_device,
            _set_torch_seed,
        )

        self.torch_version_ = torch.__version__
        self.device_ = _resolve_torch_device(self.device, torch)
        _set_torch_seed(self.random_state, torch)
        rng = np.random.default_rng(self.random_state)
        y_array = np.asarray(y, dtype=np.float32)
        weights = (
            np.ones_like(y_array, dtype=np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32)
        )
        train_idx, valid_idx = _inner_validation_split(
            y_array,
            validation_fraction=float(self.validation_fraction),
            validation_bins=int(self.validation_bins),
            random_state=int(self.random_state),
        )
        self.training_indices_ = train_idx
        self.validation_indices_ = valid_idx
        self.model_ = _DeepSetsModule(
            input_dim=int(self.input_dim),
            hidden_units=tuple(int(value) for value in self.hidden_units),
            bag_hidden_units=tuple(int(value) for value in self.bag_hidden_units),
            dropout=float(self.dropout),
        ).to(self.device_)
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
            losses = []
            for start in range(0, len(indices), int(self.batch_size)):
                batch_indices = indices[start : start + int(self.batch_size)]
                x_batch, mask_batch = _batch_tensors(
                    [bags[index] for index in batch_indices],
                    max_cells_per_bag=int(self.max_cells_per_bag),
                    rng=rng,
                    observed_counts=None,
                )
                target = torch.as_tensor(
                    y_array[batch_indices],
                    dtype=torch.float32,
                    device=self.device_,
                )
                weight = torch.as_tensor(
                    weights[batch_indices],
                    dtype=torch.float32,
                    device=self.device_,
                )
                pred = self.model_(
                    x_batch.to(self.device_),
                    mask_batch.to(self.device_),
                )
                loss = ((pred - target) ** 2 * weight).sum() / weight.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            self.train_loss_ = float(np.mean(losses))
            validation_loss = self._loss_on_indices(bags, y_array, weights, valid_idx)
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
        self.best_validation_loss_ = best_loss
        self.best_epoch_ = int(self.n_epochs_run_ - stale_epochs)
        self.model_.load_state_dict(best_state)
        return self

    def _loss_on_indices(
        self,
        bags: CellBags,
        y: np.ndarray,
        sample_weight: np.ndarray,
        indices: np.ndarray,
    ) -> float:
        import torch

        self.model_.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, len(indices), int(self.batch_size)):
                batch_indices = indices[start : start + int(self.batch_size)]
                x_batch, mask_batch = _batch_tensors(
                    [bags[index] for index in batch_indices],
                    max_cells_per_bag=int(self.max_cells_per_bag),
                    rng=None,
                    observed_counts=None,
                )
                target = torch.as_tensor(
                    y[batch_indices],
                    dtype=torch.float32,
                    device=self.device_,
                )
                weight = torch.as_tensor(
                    sample_weight[batch_indices],
                    dtype=torch.float32,
                    device=self.device_,
                )
                pred = self.model_(
                    x_batch.to(self.device_),
                    mask_batch.to(self.device_),
                )
                loss = ((pred - target) ** 2 * weight).sum() / weight.sum()
                losses.append(float(loss.cpu()))
        self.model_.train()
        return float(np.mean(losses))

    def predict(
        self,
        bags: CellBags,
        observed_counts: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict bag-level labels from ragged bags."""
        import torch

        if not hasattr(self, "model_"):
            msg = "DeepSetsRegressor must be fitted before predict"
            raise ValueError(msg)
        self.model_.eval()
        predictions: list[np.ndarray] = []
        counts = None if observed_counts is None else np.asarray(observed_counts)
        with torch.no_grad():
            for start in range(0, len(bags), int(self.batch_size)):
                stop = min(start + int(self.batch_size), len(bags))
                x_batch, mask_batch = _batch_tensors(
                    bags[start:stop],
                    max_cells_per_bag=int(self.max_cells_per_bag),
                    rng=None,
                    observed_counts=None if counts is None else counts[start:stop],
                )
                pred = self.model_(
                    x_batch.to(self.device_),
                    mask_batch.to(self.device_),
                )
                predictions.append(pred.cpu().numpy())
        return np.concatenate(predictions).astype(np.float64)


def run_single_cell_cv(
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
    """Run repeated stratified CV for the observed single-cell Deep Sets baseline."""
    set_seed(config.experiment.seed)
    bag_path = bags_npz or resolve_cell_bags_npz(config.data.output_dir)
    context = create_run_context(
        config=config,
        features_npz=bag_path,
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
    manifest = manifest_base(config, context, "run-single-cell-cv", resume)
    write_json(
        context.run_dir / "run_manifest.json",
        {**manifest, "status": "running", "log_file": str(actual_log_file)},
    )
    write_cv_config(config, context.feature_path, context.run_dir / "cv_config.json")
    try:
        data = load_cell_bag_data(context.feature_path, config)
        _execute_single_cell_cv(config, context.run_dir, store, data, selection, resume)
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


def load_cell_bag_data(path: Path, config: BaselineConfig) -> CellBagData:
    """Load a single-cell bag artifact pack."""
    payload = np.load(path, allow_pickle=True)
    cells = payload["cell_delta_pcs"].astype(np.float32)
    offsets = payload["bag_offsets"].astype(np.int64)
    bags = tuple(
        cells[offsets[index] : offsets[index + 1]] for index in range(len(offsets) - 1)
    )
    metadata_path = path.parent / "feature_metadata.parquet"
    metadata = pd.read_parquet(metadata_path)
    return CellBagData(
        bags=bags,
        y=payload["y"].astype(np.float64),
        n_cells=payload["n_cells"].astype(np.float64),
        genes=payload["perturbation_gene"].astype(str),
        metadata=metadata,
        n_pcs=int(cells.shape[1]),
    )


def resolve_cell_bags_npz(output_dir: Path) -> Path:
    """Resolve the default single-cell bag NPZ path."""
    return (
        output_dir
        / "features"
        / "single_cell_bags"
        / "replogle_k562_single_cell_bags.npz"
    )


class _DeepSetsModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...],
        bag_hidden_units: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.phi = _mlp(input_dim, hidden_units, dropout)
        phi_dim = hidden_units[-1] if hidden_units else input_dim
        self.rho = _mlp(phi_dim, (*bag_hidden_units, 1), dropout)

    def forward(self, x: object, mask: object) -> object:
        encoded = self.phi(x)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (encoded * weights).sum(dim=1) / denom
        return self.rho(pooled).squeeze(-1)


def _mlp(
    input_dim: int,
    units: tuple[int, ...],
    dropout: float,
) -> object:
    layers = []
    last_dim = int(input_dim)
    for index, width in enumerate(units):
        layers.append(nn.Linear(last_dim, int(width)))
        if index < len(units) - 1:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
        last_dim = int(width)
    return nn.Sequential(*layers)


def _batch_tensors(
    bags: CellBags,
    *,
    max_cells_per_bag: int,
    rng: np.random.Generator | None,
    observed_counts: np.ndarray | None,
) -> tuple[object, object]:
    import torch

    bag_arrays = []
    masks = []
    width = int(max_cells_per_bag)
    input_dim = int(bags[0].shape[1])
    for bag_index, bag in enumerate(bags):
        array = np.asarray(bag, dtype=np.float32)
        observed = array.shape[0]
        if observed_counts is not None:
            observed = min(int(observed_counts[bag_index]), array.shape[0])
        real = array[:observed]
        if real.shape[0] > width:
            if rng is None:
                selected = np.linspace(0, real.shape[0] - 1, width, dtype=np.int64)
            else:
                selected = np.sort(rng.choice(real.shape[0], width, replace=False))
            real = real[selected]
        padded = np.zeros((width, input_dim), dtype=np.float32)
        mask = np.zeros(width, dtype=bool)
        padded[: real.shape[0]] = real
        mask[: real.shape[0]] = True
        bag_arrays.append(padded)
        masks.append(mask)
    return torch.as_tensor(np.stack(bag_arrays)), torch.as_tensor(np.stack(masks))


def _execute_single_cell_cv(
    config: BaselineConfig,
    run_dir: Path,
    store: ArtifactStore,
    data: CellBagData,
    selection: SelectionConfig | None,
    resume: bool,
) -> None:
    merged_selection = merge_selection(config.selection, selection)
    scopes = [
        ("internal_cv_all", np.arange(len(data.y), dtype=np.int64), (FEATURE_SET_NAME,))
    ]
    if merged_selection.scopes is not None:
        allowed_scopes = set(merged_selection.scopes)
        scopes = [scope for scope in scopes if scope[0] in allowed_scopes]
    if config.experiment.save_splits:
        store.write_splits(split_manifest(config, scopes, data.y, data.genes))
    selected_features = filter_names((FEATURE_SET_NAME,), merged_selection.features)
    selected_models = filter_names((_model_name(data),), merged_selection.models)
    selected_weightings = filter_names(
        ("unweighted", "sqrt_n_cells"),
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
                for model_name in selected_models:
                    for weighting in selected_weightings:
                        job = job_key(
                            evaluation_scope,
                            fold_index,
                            feature_name,
                            model_name,
                            weighting,
                        )
                        if resume and job in store.completed_jobs:
                            continue
                        _fit_single_cell_fold(
                            config,
                            run_dir,
                            store,
                            data,
                            evaluation_scope,
                            fold_index,
                            train_idx,
                            test_idx,
                            feature_name,
                            model_name,
                            weighting,
                            job,
                        )


def _fit_single_cell_fold(
    config: BaselineConfig,
    run_dir: Path,
    store: ArtifactStore,
    data: CellBagData,
    evaluation_scope: str,
    fold_index: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_name: str,
    model_name: str,
    weighting: str,
    job: str,
) -> None:
    train_bags = tuple(data.bags[index] for index in train_idx)
    test_bags = tuple(data.bags[index] for index in test_idx)
    weights = sample_weights(data.n_cells[train_idx])
    fit_weight = weights if weighting == "sqrt_n_cells" else None
    model = DeepSetsRegressor(
        input_dim=data.n_pcs,
        hidden_units=config.single_cell.hidden_units,
        bag_hidden_units=config.single_cell.bag_hidden_units,
        dropout=config.single_cell.dropout,
        learning_rate=config.single_cell.learning_rate,
        weight_decay=config.single_cell.weight_decay,
        max_epochs=config.single_cell.max_epochs,
        patience=config.single_cell.patience,
        validation_fraction=config.single_cell.validation_fraction,
        validation_bins=config.cv.stratify_bins,
        max_cells_per_bag=config.single_cell.max_cells_per_bag,
        batch_size=config.single_cell.batch_size,
        random_state=config.cv.random_state + fold_index,
        device=config.single_cell.device,
    )
    started = time.perf_counter()
    model.fit(train_bags, data.y[train_idx], sample_weight=fit_weight)
    fit_seconds = time.perf_counter() - started
    pred = model.predict(test_bags)
    path = checkpoint_path(
        run_dir,
        evaluation_scope,
        fold_index,
        feature_name,
        model_name,
        weighting,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    metric_row = {
        "job_key": job,
        "evaluation_scope": evaluation_scope,
        "fold": fold_index,
        "feature_set": feature_name,
        "model": model_name,
        "weighting": weighting,
        "fit_seconds": fit_seconds,
        **regression_metrics(data.y[test_idx], pred),
        **ranking_metrics(data.y[test_idx], pred, config.cv.essential_thresholds),
    }
    predictions = pd.DataFrame(
        {
            "job_key": job,
            "evaluation_scope": evaluation_scope,
            "fold": fold_index,
            "feature_set": feature_name,
            "model": model_name,
            "weighting": weighting,
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
            "torch_device": getattr(model, "device_", None),
            "torch_version": getattr(model, "torch_version_", None),
            "torch_best_epoch": getattr(model, "best_epoch_", None),
            "torch_n_epochs_run": getattr(model, "n_epochs_run_", None),
            "torch_best_validation_loss": getattr(model, "best_validation_loss_", None),
        },
    )
    LOGGER.info(
        "Completed single-cell CV fold=%s model=%s weighting=%s",
        fold_index,
        model_name,
        weighting,
    )


def _model_name(data: CellBagData) -> str:
    return MODEL_NAME_TEMPLATE.format(n_pcs=data.n_pcs)


def _cv_paths(run_dir: Path) -> CvPaths:
    return CvPaths(
        run_dir=run_dir,
        fold_metrics_path=run_dir / "artifacts" / "fold_metrics.parquet",
        summary_csv=run_dir / "results" / "summary_metrics.csv",
        predictions_path=run_dir / "artifacts" / "predictions.parquet",
        config_json=run_dir / "cv_config.json",
        manifest_json=run_dir / "run_manifest.json",
        splits_path=run_dir / "artifacts" / "splits.parquet",
        model_manifest_path=run_dir / "artifacts" / "model_manifest.parquet",
        topk_candidates_path=run_dir / "artifacts" / "topk_candidates.parquet",
        log_file=run_dir / "logs" / "run.log",
    )


def _attach_run_log(
    run_dir: Path,
    log_file: Path | None,
) -> tuple[Path, logging.Handler]:
    actual_log_file = log_file or run_dir / "logs" / "run.log"
    actual_log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(actual_log_file, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    return actual_log_file, handler


def _detach_run_log(handler: logging.Handler) -> None:
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


def _utc_now() -> str:
    from dependency_baseline.artifacts import utc_now

    return utc_now()
