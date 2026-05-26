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
import torch
from torch import nn

from dependency_baseline.cell_bags import PCA_FEATURE_SET
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
from dependency_baseline.evaluation import (
    _ensemble_metrics,
    _ensemble_predictions,
    _train_gene_lookup,
)


CellBags = Sequence[np.ndarray]
FEATURE_SET_NAME = PCA_FEATURE_SET
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellBagData:
    bags: tuple[np.ndarray, ...]
    y: np.ndarray
    n_cells: np.ndarray
    genes: np.ndarray
    metadata: pd.DataFrame
    n_pcs: int
    feature_set: str


class DeepSetsRegressor:
    """Small Deep Sets regressor for bag-level GeneEffect prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...] = (128, 64),
        bag_hidden_units: tuple[int, ...] = (64,),
        attention_heads: int = 4,
        attention_orthogonality_lambda: float = 0.01,
        attention_orthogonality: str = "cosine_squared_offdiag",
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
        self.attention_heads = attention_heads
        self.attention_orthogonality_lambda = attention_orthogonality_lambda
        self.attention_orthogonality = attention_orthogonality
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
        self.model_ = self._build_module(
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
                pred, penalty = self._forward_for_loss(
                    x_batch.to(self.device_),
                    mask_batch.to(self.device_),
                )
                mse_loss = ((pred - target) ** 2 * weight).sum() / weight.sum()
                loss = self._regularized_loss(mse_loss, penalty)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                self.train_mse_loss_ = float(mse_loss.detach().cpu())
                self.train_orthogonality_penalty_ = (
                    float(penalty.detach().cpu()) if penalty is not None else 0.0
                )

            self.train_loss_ = float(np.mean(losses))
            (
                validation_loss,
                validation_mse_loss,
                validation_penalty,
            ) = self._loss_components_on_indices(bags, y_array, weights, valid_idx)
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                self.best_validation_mse_loss_ = validation_mse_loss
                self.best_validation_orthogonality_penalty_ = validation_penalty
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

    def _forward_for_loss(
        self,
        x: object,
        mask: object,
    ) -> tuple[object, object | None]:
        return self.model_(x, mask), None

    def _regularized_loss(self, mse_loss: object, penalty: object | None) -> object:
        del penalty
        return mse_loss

    def _build_module(
        self,
        *,
        input_dim: int,
        hidden_units: tuple[int, ...],
        bag_hidden_units: tuple[int, ...],
        dropout: float,
    ) -> nn.Module:
        return _DeepSetsModule(
            input_dim=input_dim,
            hidden_units=hidden_units,
            bag_hidden_units=bag_hidden_units,
            dropout=dropout,
        )

    def _loss_components_on_indices(
        self,
        bags: CellBags,
        y: np.ndarray,
        sample_weight: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[float, float, float]:
        import torch

        self.model_.eval()
        losses = []
        mse_losses = []
        penalties = []
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
                pred, penalty = self._forward_for_loss(
                    x_batch.to(self.device_),
                    mask_batch.to(self.device_),
                )
                mse_loss = ((pred - target) ** 2 * weight).sum() / weight.sum()
                loss = self._regularized_loss(mse_loss, penalty)
                losses.append(float(loss.cpu()))
                mse_losses.append(float(mse_loss.cpu()))
                penalties.append(float(penalty.cpu()) if penalty is not None else 0.0)
        self.model_.train()
        return (
            float(np.mean(losses)),
            float(np.mean(mse_losses)),
            float(np.mean(penalties)),
        )

    def _loss_on_indices(
        self,
        bags: CellBags,
        y: np.ndarray,
        sample_weight: np.ndarray,
        indices: np.ndarray,
    ) -> float:
        return self._loss_components_on_indices(bags, y, sample_weight, indices)[0]

    def predict(
        self,
        bags: CellBags,
        observed_counts: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict bag-level labels from ragged bags."""
        import torch

        if not hasattr(self, "model_"):
            msg = f"{type(self).__name__} must be fitted before predict"
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


class AttentionMILRegressor(DeepSetsRegressor):
    """Single-head gated attention MIL regressor for bag-level labels."""

    def _build_module(
        self,
        *,
        input_dim: int,
        hidden_units: tuple[int, ...],
        bag_hidden_units: tuple[int, ...],
        dropout: float,
    ) -> nn.Module:
        return _AttentionMILModule(
            input_dim=input_dim,
            hidden_units=hidden_units,
            bag_hidden_units=bag_hidden_units,
            dropout=dropout,
        )

    def predict_with_attention(
        self,
        bags: CellBags,
        observed_counts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Predict labels and return attention weights over evaluated cell subsets."""
        import torch

        if not hasattr(self, "model_"):
            msg = "AttentionMILRegressor must be fitted before predict_with_attention"
            raise ValueError(msg)
        self.model_.eval()
        predictions: list[np.ndarray] = []
        attention_weights: list[np.ndarray] = []
        cell_indices: list[np.ndarray] = []
        counts = None if observed_counts is None else np.asarray(observed_counts)
        with torch.no_grad():
            for start in range(0, len(bags), int(self.batch_size)):
                stop = min(start + int(self.batch_size), len(bags))
                x_batch, mask_batch, index_batch = _batch_tensors(
                    bags[start:stop],
                    max_cells_per_bag=int(self.max_cells_per_bag),
                    rng=None,
                    observed_counts=None if counts is None else counts[start:stop],
                    return_indices=True,
                )
                pred, attn = self.model_(
                    x_batch.to(self.device_),
                    mask_batch.to(self.device_),
                    return_attention=True,
                )
                predictions.append(pred.cpu().numpy())
                attn_np = attn.cpu().numpy()
                mask_np = mask_batch.numpy()
                for row_index in range(attn_np.shape[0]):
                    valid = mask_np[row_index]
                    attention_weights.append(
                        attn_np[row_index, valid].astype(np.float64)
                    )
                    cell_indices.append(index_batch[row_index][valid].astype(np.int64))
        return (
            np.concatenate(predictions).astype(np.float64),
            attention_weights,
            cell_indices,
        )


class MultiHeadAttentionMILRegressor(AttentionMILRegressor):
    """Multi-head gated attention MIL regressor for bag-level labels."""

    def _build_module(
        self,
        *,
        input_dim: int,
        hidden_units: tuple[int, ...],
        bag_hidden_units: tuple[int, ...],
        dropout: float,
    ) -> nn.Module:
        return _MultiHeadAttentionMILModule(
            input_dim=input_dim,
            hidden_units=hidden_units,
            bag_hidden_units=bag_hidden_units,
            dropout=dropout,
            attention_heads=int(self.attention_heads),
        )

    def _forward_for_loss(
        self,
        x: object,
        mask: object,
    ) -> tuple[object, object | None]:
        prediction, attention = self.model_(x, mask, return_attention=True)
        return prediction, _attention_orthogonality_penalty(attention, mask)

    def _regularized_loss(self, mse_loss: object, penalty: object | None) -> object:
        if self.attention_orthogonality != "cosine_squared_offdiag":
            msg = (
                "Unsupported attention_orthogonality="
                f"{self.attention_orthogonality!r}"
            )
            raise ValueError(msg)
        if penalty is None:
            return mse_loss
        return mse_loss + float(self.attention_orthogonality_lambda) * penalty

    def predict_with_attention(
        self,
        bags: CellBags,
        observed_counts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Predict labels and return K attention maps per evaluated bag."""
        import torch

        if not hasattr(self, "model_"):
            msg = (
                "MultiHeadAttentionMILRegressor must be fitted before "
                "predict_with_attention"
            )
            raise ValueError(msg)
        self.model_.eval()
        predictions: list[np.ndarray] = []
        attention_weights: list[np.ndarray] = []
        cell_indices: list[np.ndarray] = []
        counts = None if observed_counts is None else np.asarray(observed_counts)
        with torch.no_grad():
            for start in range(0, len(bags), int(self.batch_size)):
                stop = min(start + int(self.batch_size), len(bags))
                x_batch, mask_batch, index_batch = _batch_tensors(
                    bags[start:stop],
                    max_cells_per_bag=int(self.max_cells_per_bag),
                    rng=None,
                    observed_counts=None if counts is None else counts[start:stop],
                    return_indices=True,
                )
                pred, attn = self.model_(
                    x_batch.to(self.device_),
                    mask_batch.to(self.device_),
                    return_attention=True,
                )
                predictions.append(pred.cpu().numpy())
                attn_np = attn.cpu().numpy()
                mask_np = mask_batch.numpy()
                for row_index in range(attn_np.shape[0]):
                    valid = mask_np[row_index]
                    attention_weights.append(
                        attn_np[row_index][:, valid].astype(np.float64)
                    )
                    cell_indices.append(index_batch[row_index][valid].astype(np.int64))
        return (
            np.concatenate(predictions).astype(np.float64),
            attention_weights,
            cell_indices,
        )


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
    """Run repeated stratified CV for observed single-cell set baselines."""
    set_seed(config.experiment.seed)
    merged_selection = merge_selection(config.selection, selection)
    feature_sets = (
        (_feature_set_from_bag_path(bags_npz),)
        if bags_npz is not None
        else filter_names(config.single_cell.feature_sets, merged_selection.features)
    )
    if not feature_sets:
        msg = "No single-cell feature sets selected"
        raise ValueError(msg)
    bag_paths = (
        (bags_npz,)
        if bags_npz is not None
        else tuple(
            resolve_cell_bags_npz_for_feature(config.data.output_dir, feature_set)
            for feature_set in feature_sets
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
    manifest = manifest_base(config, context, "run-single-cell-cv", resume)
    write_json(
        context.run_dir / "run_manifest.json",
        {**manifest, "status": "running", "log_file": str(actual_log_file)},
    )
    write_cv_config(config, context.feature_path, context.run_dir / "cv_config.json")
    try:
        for bag_path in bag_paths:
            data = load_cell_bag_data(bag_path, config)
            _execute_single_cell_cv(
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


def evaluate_single_cell_external(
    config: BaselineConfig,
    *,
    run_dir: Path,
    external_bags_npz: Path,
    external_name: str,
    feature_set: str | None = None,
) -> tuple[Path, Path]:
    """Evaluate existing single-cell CV checkpoints on an external bag artifact."""
    store = ArtifactStore(
        run_dir,
        config.experiment.human_result_tables,
        config.experiment.machine_result_format,
        config.experiment.topk_candidates,
        config.experiment.save_predictions,
        config.experiment.save_rankings,
    )
    external = load_cell_bag_data(external_bags_npz, config)
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
        msg = "No internal single-cell CV checkpoints found for external evaluation"
        raise ValueError(msg)

    for row in selected.itertuples(index=False):
        _evaluate_one_external_checkpoint(config, store, external, external_name, row)
    _write_single_cell_external_ensemble_results(config, store)
    return (
        run_dir / "artifacts" / "external_ensemble_metrics.parquet",
        run_dir / "artifacts" / "external_ensemble_predictions.parquet",
    )


def load_cell_bag_data(path: Path, config: BaselineConfig) -> CellBagData:
    """Load a single-cell bag artifact pack."""
    del config
    if not path.exists():
        msg = (
            "Missing single-cell bag artifact: "
            f"{path}. Build or sync the expected PCA/scVI/HVG bag NPZ before "
            "running single-cell MIL evaluation."
        )
        raise FileNotFoundError(msg)
    payload = np.load(path, allow_pickle=True)
    cells = payload["cell_delta_pcs"].astype(np.float32)
    offsets = payload["bag_offsets"].astype(np.int64)
    bags = tuple(
        cells[offsets[index] : offsets[index + 1]] for index in range(len(offsets) - 1)
    )
    metadata_path = path.parent / "feature_metadata.parquet"
    if not metadata_path.exists():
        msg = f"Missing single-cell bag metadata: {metadata_path}"
        raise FileNotFoundError(msg)
    metadata = pd.read_parquet(metadata_path)
    return CellBagData(
        bags=bags,
        y=payload["y"].astype(np.float64),
        n_cells=payload["n_cells"].astype(np.float64),
        genes=payload["perturbation_gene"].astype(str),
        metadata=metadata,
        n_pcs=int(cells.shape[1]),
        feature_set=_feature_set_from_payload(payload),
    )


def _evaluate_one_external_checkpoint(
    config: BaselineConfig,
    store: ArtifactStore,
    external: CellBagData,
    external_name: str,
    row: object,
) -> None:
    model = joblib.load(Path(row.checkpoint_path))
    model_name = str(row.model)
    if _is_attention_model(model_name):
        pred, attention_weights, cell_indices = model.predict_with_attention(
            external.bags
        )
    else:
        pred = model.predict(external.bags)
        attention_weights = []
        cell_indices = []
    evaluation_scope = f"external:{external_name}"
    job = job_key(
        evaluation_scope,
        int(row.fold),
        str(row.feature_set),
        model_name,
        str(row.weighting),
    )
    metric_row = {
        "job_key": job,
        "evaluation_scope": evaluation_scope,
        "fold": int(row.fold),
        "feature_set": str(row.feature_set),
        "model": model_name,
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
            "model": model_name,
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
    if _is_attention_model(model_name):
        _append_attention_weights(
            store.run_dir,
            _attention_weight_frame(
                job_key_value=job,
                evaluation_scope=evaluation_scope,
                fold=int(row.fold),
                feature_set=str(row.feature_set),
                model_name=model_name,
                weighting=str(row.weighting),
                data=external,
                bag_indices=np.arange(len(external.bags), dtype=np.int64),
                attention_weights=attention_weights,
                cell_indices=cell_indices,
            ),
        )
        _append_attention_head_diagnostics(
            store.run_dir,
            _attention_head_diagnostics_frame(
                job_key_value=job,
                evaluation_scope=evaluation_scope,
                fold=int(row.fold),
                feature_set=str(row.feature_set),
                model_name=model_name,
                weighting=str(row.weighting),
                data=external,
                bag_indices=np.arange(len(external.bags), dtype=np.int64),
                attention_weights=attention_weights,
            ),
        )


def _write_single_cell_external_ensemble_results(
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


def resolve_cell_bags_npz(output_dir: Path) -> Path:
    """Resolve the default single-cell bag NPZ path."""
    return resolve_cell_bags_npz_for_feature(output_dir, FEATURE_SET_NAME)


def resolve_cell_bags_npz_for_feature(output_dir: Path, feature_set: str) -> Path:
    """Resolve a single-cell bag NPZ path for one feature set."""
    if feature_set == FEATURE_SET_NAME:
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


def _feature_set_from_payload(payload: object) -> str:
    if "feature_set" not in payload:
        return FEATURE_SET_NAME
    value = payload["feature_set"]
    if np.asarray(value).shape == ():
        return str(value.item())
    return str(np.asarray(value).astype(str).reshape(-1)[0])


def _feature_set_from_bag_path(path: Path | None) -> str:
    if path is None:
        return FEATURE_SET_NAME
    payload = np.load(path, allow_pickle=True)
    try:
        return _feature_set_from_payload(payload)
    finally:
        payload.close()


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


class _AttentionMILModule(nn.Module):
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
        self.attention_v = nn.Linear(phi_dim, phi_dim)
        self.attention_u = nn.Linear(phi_dim, phi_dim)
        self.attention_w = nn.Linear(phi_dim, 1, bias=False)
        self.rho = _mlp(phi_dim, (*bag_hidden_units, 1), dropout)

    def forward(
        self,
        x: object,
        mask: object,
        *,
        return_attention: bool = False,
    ) -> object:
        encoded = self.phi(x)
        gated = self.attention_w(
            self.attention_v(encoded).tanh() * self.attention_u(encoded).sigmoid()
        ).squeeze(-1)
        scores = gated.masked_fill(~mask, -1.0e9)
        attention = scores.softmax(dim=1)
        attention = attention * mask.to(attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        pooled = (encoded * attention.unsqueeze(-1)).sum(dim=1)
        prediction = self.rho(pooled).squeeze(-1)
        if return_attention:
            return prediction, attention
        return prediction


class _MultiHeadAttentionMILModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...],
        bag_hidden_units: tuple[int, ...],
        dropout: float,
        attention_heads: int,
    ) -> None:
        super().__init__()
        if attention_heads < 2:
            msg = "attention_heads must be at least 2 for multi-head attention MIL"
            raise ValueError(msg)
        self.attention_heads = int(attention_heads)
        self.phi = _mlp(input_dim, hidden_units, dropout)
        phi_dim = hidden_units[-1] if hidden_units else input_dim
        self.attention_v = nn.Linear(phi_dim, self.attention_heads * phi_dim)
        self.attention_u = nn.Linear(phi_dim, self.attention_heads * phi_dim)
        self.attention_w = nn.Parameter(torch.empty(self.attention_heads, phi_dim))
        nn.init.xavier_uniform_(self.attention_w)
        self.rho = _mlp(
            self.attention_heads * phi_dim,
            (*bag_hidden_units, 1),
            dropout,
        )

    def forward(
        self,
        x: object,
        mask: object,
        *,
        return_attention: bool = False,
    ) -> object:
        encoded = self.phi(x)
        batch_size, n_cells, phi_dim = encoded.shape
        v = self.attention_v(encoded).view(
            batch_size,
            n_cells,
            self.attention_heads,
            phi_dim,
        )
        u = self.attention_u(encoded).view(
            batch_size,
            n_cells,
            self.attention_heads,
            phi_dim,
        )
        gated = v.tanh() * u.sigmoid()
        scores = (gated * self.attention_w.view(1, 1, self.attention_heads, phi_dim))
        scores = scores.sum(dim=-1).permute(0, 2, 1)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1.0e9)
        attention = scores.softmax(dim=-1)
        attention = attention * mask.unsqueeze(1).to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        pooled = (attention.unsqueeze(-1) * encoded.unsqueeze(1)).sum(dim=2)
        prediction = self.rho(pooled.flatten(start_dim=1)).squeeze(-1)
        if return_attention:
            return prediction, attention
        return prediction


def _attention_orthogonality_penalty(attention: object, mask: object) -> object:
    del mask
    heads = attention.shape[1]
    if heads < 2:
        return attention.new_tensor(0.0)
    normalized = attention / attention.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    gram = normalized @ normalized.transpose(1, 2)
    eye = torch.eye(heads, dtype=gram.dtype, device=gram.device).unsqueeze(0)
    off_diag = gram * (1.0 - eye)
    return off_diag.square().sum() / (attention.shape[0] * heads * (heads - 1))


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
    return_indices: bool = False,
) -> tuple[object, object] | tuple[object, object, list[np.ndarray]]:
    import torch

    bag_arrays = []
    masks = []
    selected_indices = []
    width = int(max_cells_per_bag)
    input_dim = int(bags[0].shape[1])
    for bag_index, bag in enumerate(bags):
        array = np.asarray(bag, dtype=np.float32)
        observed = array.shape[0]
        if observed_counts is not None:
            observed = min(int(observed_counts[bag_index]), array.shape[0])
        real = array[:observed]
        selected = np.arange(real.shape[0], dtype=np.int64)
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
        padded_indices = np.full(width, -1, dtype=np.int64)
        padded_indices[: selected.shape[0]] = selected
        selected_indices.append(padded_indices)
    tensors = torch.as_tensor(np.stack(bag_arrays)), torch.as_tensor(np.stack(masks))
    if return_indices:
        return (*tensors, selected_indices)
    return tensors


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
        (
            "internal_cv_all",
            np.arange(len(data.y), dtype=np.int64),
            (data.feature_set,),
        )
    ]
    if merged_selection.scopes is not None:
        allowed_scopes = set(merged_selection.scopes)
        scopes = [scope for scope in scopes if scope[0] in allowed_scopes]
    if config.experiment.save_splits:
        store.write_splits(split_manifest(config, scopes, data.y, data.genes))
    selected_features = filter_names((data.feature_set,), merged_selection.features)
    selected_models = filter_names(_model_names(data), merged_selection.models)
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
    model = _make_single_cell_model(
        model_name,
        input_dim=data.n_pcs,
        hidden_units=config.single_cell.hidden_units,
        bag_hidden_units=config.single_cell.bag_hidden_units,
        attention_heads=config.single_cell.attention_heads,
        attention_orthogonality_lambda=(
            config.single_cell.attention_orthogonality_lambda
        ),
        attention_orthogonality=config.single_cell.attention_orthogonality,
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
    if _is_attention_model(model_name):
        pred, attention_weights, cell_indices = model.predict_with_attention(test_bags)
    else:
        pred = model.predict(test_bags)
        attention_weights = []
        cell_indices = []
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
            "torch_train_mse_loss": getattr(model, "train_mse_loss_", None),
            "torch_train_orthogonality_penalty": getattr(
                model,
                "train_orthogonality_penalty_",
                None,
            ),
            "torch_best_validation_mse_loss": getattr(
                model,
                "best_validation_mse_loss_",
                None,
            ),
            "torch_best_validation_orthogonality_penalty": getattr(
                model,
                "best_validation_orthogonality_penalty_",
                None,
            ),
            "attention_heads": getattr(model, "attention_heads", None),
            "attention_orthogonality_lambda": getattr(
                model,
                "attention_orthogonality_lambda",
                None,
            ),
        },
    )
    if _is_attention_model(model_name):
        _append_attention_weights(
            run_dir,
            _attention_weight_frame(
                job_key_value=job,
                evaluation_scope=evaluation_scope,
                fold=fold_index,
                feature_set=feature_name,
                model_name=model_name,
                weighting=weighting,
                data=data,
                bag_indices=test_idx,
                attention_weights=attention_weights,
                cell_indices=cell_indices,
            ),
        )
        _append_attention_head_diagnostics(
            run_dir,
            _attention_head_diagnostics_frame(
                job_key_value=job,
                evaluation_scope=evaluation_scope,
                fold=fold_index,
                feature_set=feature_name,
                model_name=model_name,
                weighting=weighting,
                data=data,
                bag_indices=test_idx,
                attention_weights=attention_weights,
            ),
        )
    LOGGER.info(
        "Completed single-cell CV fold=%s model=%s weighting=%s",
        fold_index,
        model_name,
        weighting,
    )


def _model_names(data: CellBagData) -> tuple[str, ...]:
    token = _feature_token(data)
    return (
        f"deepsets_{token}_meanpool",
        f"attnmil_{token}_gated",
        f"mhattnmil_{token}_gated4_ortho001",
    )


def _feature_token(data: CellBagData) -> str:
    if data.feature_set == FEATURE_SET_NAME:
        return f"pca{data.n_pcs}"
    if data.feature_set == "single_cell_scvi_delta":
        return f"scvi{data.n_pcs}"
    if data.feature_set == "single_cell_hvg_delta":
        return f"hvg{data.n_pcs}"
    return data.feature_set.replace("single_cell_", "").replace("_delta", "")


def _is_attention_model(model_name: str) -> bool:
    return str(model_name).startswith(("attnmil_", "mhattnmil_"))


def _is_multihead_attention_model(model_name: str) -> bool:
    return str(model_name).startswith("mhattnmil_")


def _make_single_cell_model(model_name: str, **kwargs: object) -> DeepSetsRegressor:
    if _is_multihead_attention_model(model_name):
        model_cls = MultiHeadAttentionMILRegressor
    elif _is_attention_model(model_name):
        model_cls = AttentionMILRegressor
    else:
        model_cls = DeepSetsRegressor
    return model_cls(**kwargs)


def _attention_weight_frame(
    *,
    job_key_value: str,
    evaluation_scope: str,
    fold: int,
    feature_set: str,
    model_name: str,
    weighting: str,
    data: CellBagData,
    bag_indices: np.ndarray,
    attention_weights: list[np.ndarray],
    cell_indices: list[np.ndarray],
) -> pd.DataFrame:
    rows = []
    for local_index, bag_index in enumerate(bag_indices):
        metadata_row = data.metadata.iloc[int(bag_index)]
        feature_row = int(metadata_row.get("feature_row", int(bag_index)))
        gene = str(data.genes[int(bag_index)])
        weights = np.asarray(attention_weights[local_index], dtype=np.float64)
        if weights.ndim == 1:
            weights = weights[None, :]
        for head_index, head_weights in enumerate(weights):
            for position, weight in zip(
                cell_indices[local_index],
                head_weights,
                strict=True,
            ):
                rows.append(
                    {
                        "job_key": job_key_value,
                        "evaluation_scope": evaluation_scope,
                        "fold": int(fold),
                        "feature_set": feature_set,
                        "model": model_name,
                        "weighting": weighting,
                        "perturbation_gene": gene,
                        "feature_row": feature_row,
                        "attention_head": int(head_index),
                        "evaluated_cell_position": int(position),
                        "attention_weight": float(weight),
                    }
                )
    return pd.DataFrame(rows)


def _attention_head_diagnostics_frame(
    *,
    job_key_value: str,
    evaluation_scope: str,
    fold: int,
    feature_set: str,
    model_name: str,
    weighting: str,
    data: CellBagData,
    bag_indices: np.ndarray,
    attention_weights: list[np.ndarray],
) -> pd.DataFrame:
    rows = []
    for local_index, bag_index in enumerate(bag_indices):
        weights = np.asarray(attention_weights[local_index], dtype=np.float64)
        if weights.ndim == 1:
            weights = weights[None, :]
        clipped = np.clip(weights, 1.0e-12, 1.0)
        entropy = -(clipped * np.log(clipped)).sum(axis=1)
        effective_cells = 1.0 / np.square(weights).sum(axis=1).clip(min=1.0e-12)
        if weights.shape[0] > 1:
            norm = np.linalg.norm(weights, axis=1, keepdims=True).clip(min=1.0e-12)
            gram = (weights / norm) @ (weights / norm).T
            off_diag = gram[~np.eye(weights.shape[0], dtype=bool)]
            mean_similarity = float(off_diag.mean())
            penalty = float(np.square(off_diag).mean())
        else:
            mean_similarity = float("nan")
            penalty = 0.0
        metadata_row = data.metadata.iloc[int(bag_index)]
        feature_row = int(metadata_row.get("feature_row", int(bag_index)))
        gene = str(data.genes[int(bag_index)])
        for head_index in range(weights.shape[0]):
            rows.append(
                {
                    "job_key": job_key_value,
                    "evaluation_scope": evaluation_scope,
                    "fold": int(fold),
                    "feature_set": feature_set,
                    "model": model_name,
                    "weighting": weighting,
                    "perturbation_gene": gene,
                    "feature_row": feature_row,
                    "attention_head": int(head_index),
                    "attention_entropy": float(entropy[head_index]),
                    "attention_effective_cell_count": float(
                        effective_cells[head_index]
                    ),
                    "mean_offdiag_head_cosine": mean_similarity,
                    "orthogonality_penalty": penalty,
                }
            )
    return pd.DataFrame(rows)


def _append_attention_weights(run_dir: Path, weights: pd.DataFrame) -> None:
    if weights.empty:
        return
    path = run_dir / "artifacts" / "single_cell_attention_weights.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        if "attention_head" not in existing.columns:
            existing["attention_head"] = 0
        weights = (
            pd.concat([existing, weights], ignore_index=True)
            .drop_duplicates(
                [
                    "job_key",
                    "perturbation_gene",
                    "attention_head",
                    "evaluated_cell_position",
                ],
                keep="last",
            )
            .reset_index(drop=True)
        )
    weights.to_parquet(path, index=False)


def _append_attention_head_diagnostics(
    run_dir: Path,
    diagnostics: pd.DataFrame,
) -> None:
    if diagnostics.empty:
        return
    path = run_dir / "artifacts" / "single_cell_attention_head_diagnostics.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        diagnostics = (
            pd.concat([existing, diagnostics], ignore_index=True)
            .drop_duplicates(
                [
                    "job_key",
                    "perturbation_gene",
                    "attention_head",
                ],
                keep="last",
            )
            .reset_index(drop=True)
        )
    diagnostics.to_parquet(path, index=False)


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
