"""Fold-local linear A->B->C baselines for predicted perturbation bags."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.mixture import GaussianMixture
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
    write_formats,
    write_json,
)
from dependency_baseline.cell_bags import (
    HVG_FEATURE_SET,
    SCVI_FEATURE_SET,
    _dense_float32,
    _import_scvi,
    _select_hvg_indices,
)
from dependency_baseline.config import BaselineConfig, SelectionConfig
from dependency_baseline.datasets import (
    filter_names,
    merge_selection,
    repeated_stratified_splitter,
    split_manifest,
    stratification_bins,
)
from dependency_baseline.distribution import _feature_from_occupancy, _occupancy_feature
from dependency_baseline.features import _numeric_training_rows
from dependency_baseline.metrics import ranking_metrics, regression_metrics
from dependency_baseline.single_cell import (
    _attach_run_log,
    _cv_paths,
    _detach_run_log,
    _utc_now,
)

LOGGER = logging.getLogger(__name__)


MEAN_DELTA_METHOD = "mean_delta_ridge"
PSEUDO_PAIR_METHOD = "pseudo_pair_ridge"
SUPPORTED_METHODS = (MEAN_DELTA_METHOD, PSEUDO_PAIR_METHOD)


@dataclass(frozen=True)
class FoldExpressionData:
    genes: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    selected_indices: np.ndarray
    selected_symbols: list[str]
    selected_mean: np.ndarray
    selected_std: np.ndarray
    control_matrix: np.ndarray
    train_means: dict[str, np.ndarray]
    train_cells: dict[str, np.ndarray]
    control_obs: pd.DataFrame
    train_obs: dict[str, pd.DataFrame]
    target_indices: dict[str, int]


@dataclass(frozen=True)
class ProjectionResult:
    bags: tuple[np.ndarray, ...]
    control_bag: np.ndarray
    qa: dict[str, object]


@dataclass(frozen=True)
class PredictedBagResult:
    bags: tuple[np.ndarray, ...]
    metadata: pd.DataFrame


class FoldProjector:
    """Fold-local expression-to-B-space projector."""

    def __init__(
        self,
        *,
        feature_set: str,
        selected_mean: np.ndarray,
        selected_std: np.ndarray,
        control_reference: np.ndarray,
        scvi_model_dir: Path | None,
        selected_symbols: list[str],
        query_batch_cells: int,
    ) -> None:
        self.feature_set = feature_set
        self.selected_mean = selected_mean.astype(np.float32)
        self.selected_std = selected_std.astype(np.float32)
        self.control_reference = control_reference.astype(np.float32)
        self.scvi_model_dir = scvi_model_dir
        self.selected_symbols = selected_symbols
        self.query_batch_cells = int(query_batch_cells)

    def project(
        self,
        bags: tuple[np.ndarray, ...],
        control_matrix: np.ndarray,
    ) -> ProjectionResult:
        if self.feature_set == SCVI_FEATURE_SET:
            return self._project_scvi(bags, control_matrix)
        return self._project_hvg(bags, control_matrix)

    def _project_hvg(
        self,
        bags: tuple[np.ndarray, ...],
        control_matrix: np.ndarray,
    ) -> ProjectionResult:
        control_scaled = _scale_expression(
            control_matrix,
            self.selected_mean,
            self.selected_std,
        )
        control_centroid = control_scaled.mean(axis=0, dtype=np.float64)
        control_bag = (
            control_scaled.astype(np.float32) - control_centroid[None, :]
        ).astype(np.float32)
        projected = []
        for bag in bags:
            scaled = _scale_expression(bag, self.selected_mean, self.selected_std)
            projected.append(
                (scaled.astype(np.float32) - control_centroid[None, :]).astype(
                    np.float32
                )
            )
        return ProjectionResult(
            bags=tuple(projected),
            control_bag=control_bag,
            qa={
                "feature_set": self.feature_set,
                "projector": "fold_local_hvg_standardization",
                "control_cells": int(control_matrix.shape[0]),
                "embedding_dim": int(control_bag.shape[1]),
            },
        )

    def _project_scvi(
        self,
        bags: tuple[np.ndarray, ...],
        control_matrix: np.ndarray,
    ) -> ProjectionResult:
        if self.scvi_model_dir is None:
            msg = "scvi_model_dir is required for single_cell_scvi_delta projection"
            raise ValueError(msg)
        control_latent = self._encode_scvi(control_matrix)
        control_centroid = control_latent.mean(axis=0, dtype=np.float64)
        control_bag = (
            control_latent.astype(np.float32) - control_centroid[None, :]
        ).astype(np.float32)
        projected = []
        for bag in bags:
            latent = self._encode_scvi(bag)
            projected.append(
                (latent.astype(np.float32) - control_centroid[None, :]).astype(
                    np.float32
                )
            )
        return ProjectionResult(
            bags=tuple(projected),
            control_bag=control_bag,
            qa={
                "feature_set": self.feature_set,
                "projector": "fold_local_scvi",
                "scvi_model_dir": str(self.scvi_model_dir),
                "control_cells": int(control_matrix.shape[0]),
                "embedding_dim": int(control_bag.shape[1]),
            },
        )

    def _encode_scvi(self, matrix: np.ndarray) -> np.ndarray:
        scvi = _import_scvi()
        chunks = []
        for start in range(0, matrix.shape[0], self.query_batch_cells):
            stop = min(start + self.query_batch_cells, matrix.shape[0])
            query = ad.AnnData(matrix[start:stop].astype(np.float32))
            query.var_names = self.selected_symbols
            model = scvi.model.SCVI.load(str(self.scvi_model_dir), adata=query)
            chunks.append(
                np.asarray(model.get_latent_representation(), dtype=np.float32)
            )
        return np.vstack(chunks).astype(np.float32)


def run_predicted_b_cv(
    config: BaselineConfig,
    *,
    run_id: str | None = None,
    resume: bool = False,
    selection: SelectionConfig | None = None,
    command: tuple[str, ...] = (),
    config_path: Path | None = None,
    log_file: Path | None = None,
) -> CvPaths:
    """Run fold-local linear A->B->C baselines."""
    set_seed(config.experiment.seed)
    merged_selection = merge_selection(config.selection, selection)
    selected_methods = filter_names(
        tuple(config.predicted_b.methods),
        merged_selection.models,
    )
    if not selected_methods:
        msg = "No predicted-B methods selected"
        raise ValueError(msg)
    unsupported = set(selected_methods) - set(SUPPORTED_METHODS)
    if unsupported:
        msg = f"Unsupported predicted-B methods: {sorted(unsupported)}"
        raise ValueError(msg)
    if config.predicted_b.feature_set not in {SCVI_FEATURE_SET, HVG_FEATURE_SET}:
        msg = (
            "run-predicted-b-cv supports single_cell_scvi_delta or "
            "single_cell_hvg_delta"
        )
        raise ValueError(msg)

    context = create_run_context(
        config=config,
        features_npz=config.data.h5ad_path,
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
    manifest = manifest_base(config, context, "run-predicted-b-cv", resume)
    write_json(
        context.run_dir / "run_manifest.json",
        {
            **manifest,
            "status": "running",
            "log_file": str(actual_log_file),
            "leakage_scope": "fold-local A->B, featureizer, GMM, and C head",
        },
    )
    write_cv_config(config, context.feature_path, context.run_dir / "cv_config.json")
    try:
        _execute_predicted_b_cv(
            config,
            context.run_dir,
            store,
            selected_methods,
            merged_selection,
            resume,
        )
        write_json(
            context.run_dir / "run_manifest.json",
            {
                **manifest,
                "status": "completed",
                "ended_at": _utc_now(),
                "log_file": str(actual_log_file),
                "leakage_scope": "fold-local A->B, featureizer, GMM, and C head",
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


def _execute_predicted_b_cv(
    config: BaselineConfig,
    run_dir: Path,
    store: ArtifactStore,
    selected_methods: tuple[str, ...],
    selection: SelectionConfig,
    resume: bool,
) -> None:
    overlap = pd.read_csv(config.data.overlap_csv)
    metadata = _numeric_training_rows(overlap, config).copy().reset_index(drop=True)
    genes = metadata["perturbation_gene"].astype(str).to_numpy()
    y = metadata[config.data.depmap_label_col].to_numpy(dtype=np.float64)
    scope = ("internal_cv_all", np.arange(len(y), dtype=np.int64), selected_methods)
    if selection.scopes is not None and "internal_cv_all" not in selection.scopes:
        return
    if config.experiment.save_splits:
        store.write_splits(split_manifest(config, [scope], y, genes))

    splitter = repeated_stratified_splitter(config)
    y_bins = stratification_bins(y, config.cv.stratify_bins)
    adata = ad.read_h5ad(config.data.h5ad_path, backed="r")
    try:
        obs_labels = adata.obs[config.data.obs_perturbation_col].astype(str).to_numpy()
        var_symbols = _var_symbols(adata, config.data.var_gene_symbol_col)
        for fold_index, (train_idx, test_idx) in enumerate(
            splitter.split(genes, y_bins)
        ):
            if selection.folds is not None and fold_index not in selection.folds:
                continue
            for method in selected_methods:
                job = job_key(
                    "internal_cv_all",
                    fold_index,
                    config.predicted_b.feature_set,
                    method,
                    "unweighted",
                )
                if resume and job in store.completed_jobs:
                    continue
                _fit_one_predicted_b_fold(
                    config,
                    run_dir,
                    store,
                    adata,
                    obs_labels,
                    var_symbols,
                    metadata,
                    genes,
                    y,
                    train_idx.astype(np.int64),
                    test_idx.astype(np.int64),
                    fold_index,
                    method,
                    job,
                )
    finally:
        adata.file.close()


def _fit_one_predicted_b_fold(
    config: BaselineConfig,
    run_dir: Path,
    store: ArtifactStore,
    adata: ad.AnnData,
    obs_labels: np.ndarray,
    var_symbols: list[str],
    metadata: pd.DataFrame,
    genes: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_index: int,
    method: str,
    job: str,
) -> None:
    started = time.perf_counter()
    train_genes = genes[train_idx].astype(str)
    test_genes = genes[test_idx].astype(str)
    fold_data = _load_fold_expression_data(
        config,
        adata,
        obs_labels,
        var_symbols,
        metadata,
        train_genes,
        test_genes,
    )
    model = _fit_a_to_b_model(config, fold_data, train_genes, method, fold_index)
    control_panel = _control_panel(config, fold_data.control_matrix, fold_index)
    train_pred = _predict_bags(
        config,
        fold_data,
        model,
        train_genes,
        control_panel,
    )
    test_pred = _predict_bags(
        config,
        fold_data,
        model,
        test_genes,
        control_panel,
    )
    train_reconstruction = _reconstruction_metrics_for_predictions(
        train_genes,
        train_pred.bags,
        fold_data.train_means,
    )
    test_means = _load_observed_gene_means(
        adata,
        obs_labels,
        fold_data.selected_indices,
        test_genes,
        split_name="test",
    )
    test_reconstruction = _reconstruction_metrics_for_predictions(
        test_genes,
        test_pred.bags,
        test_means,
    )
    projector = _fit_fold_projector(
        config,
        run_dir,
        fold_data,
        train_genes,
        fold_index,
        method,
    )
    train_projection = projector.project(train_pred.bags, fold_data.control_matrix)
    test_projection = projector.project(test_pred.bags, fold_data.control_matrix)
    gmm = _fit_predicted_gmm(config, train_projection)
    x_train = _gmm_features(
        config,
        gmm,
        train_projection.bags,
        train_projection.control_bag,
    )
    x_test = _gmm_features(
        config,
        gmm,
        test_projection.bags,
        train_projection.control_bag,
    )
    c_model = Ridge(alpha=float(config.predicted_b.c_ridge_alpha))
    c_model.fit(x_train, y[train_idx])
    pred = c_model.predict(x_test)
    fit_seconds = time.perf_counter() - started

    path = checkpoint_path(
        run_dir,
        "internal_cv_all",
        fold_index,
        config.predicted_b.feature_set,
        method,
        "unweighted",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "a_to_b_model": model,
            "gmm": gmm,
            "c_model": c_model,
            "selected_symbols": fold_data.selected_symbols,
            "target_indices": fold_data.target_indices,
            "projection_qa": train_projection.qa,
        },
        path,
    )
    metric_row = {
        "job_key": job,
        "evaluation_scope": "internal_cv_all",
        "fold": fold_index,
        "feature_set": config.predicted_b.feature_set,
        "model": method,
        "weighting": "unweighted",
        "fit_seconds": fit_seconds,
        "a_to_b_train_mean_rmse": train_reconstruction["mean_rmse"],
        "a_to_b_train_mean_mae": train_reconstruction["mean_mae"],
        "a_to_b_test_mean_rmse": test_reconstruction["mean_rmse"],
        "a_to_b_test_mean_mae": test_reconstruction["mean_mae"],
        "predicted_cells_per_gene": int(control_panel.shape[0]),
        **regression_metrics(y[test_idx], pred),
        **ranking_metrics(y[test_idx], pred, config.cv.essential_thresholds),
    }
    predictions = pd.DataFrame(
        {
            "job_key": job,
            "evaluation_scope": "internal_cv_all",
            "fold": fold_index,
            "feature_set": config.predicted_b.feature_set,
            "model": method,
            "weighting": "unweighted",
            "perturbation_gene": test_genes,
            "y_true": y[test_idx],
            "y_pred": pred,
        }
    ).merge(
        metadata[["perturbation_gene", config.data.n_cells_col]],
        on="perturbation_gene",
        how="left",
    )
    predictions = predictions.rename(
        columns={config.data.n_cells_col: "observed_n_cells"}
    )
    store.append_fold_result(
        metric_row,
        predictions,
        {
            **metric_row,
            "checkpoint_path": str(path),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "gmm_components": int(config.predicted_b.gmm_components),
            "gmm_view": config.predicted_b.gmm_view,
            "c_ridge_alpha": float(config.predicted_b.c_ridge_alpha),
            "featureizer_scope": "fold_local_train_genes_plus_controls",
            "gmm_fit_source": "predicted_b_train_plus_controls",
        },
    )
    _write_fold_predicted_b_artifacts(
        config,
        run_dir,
        fold_index,
        method,
        train_pred,
        test_pred,
        train_reconstruction,
        test_reconstruction,
        train_projection.qa,
        gmm,
    )
    LOGGER.info("Completed predicted-B fold=%s method=%s", fold_index, method)


def _load_fold_expression_data(
    config: BaselineConfig,
    adata: ad.AnnData,
    obs_labels: np.ndarray,
    var_symbols: list[str],
    metadata: pd.DataFrame,
    train_genes: np.ndarray,
    test_genes: np.ndarray,
) -> FoldExpressionData:
    train_or_control = np.isin(obs_labels, [config.data.control_label, *train_genes])
    selected_indices = _fold_selected_gene_indices(adata.X, train_or_control, config)
    selected_symbols = [var_symbols[index] for index in selected_indices]
    selected_mean, selected_std = _selected_moments(
        adata.X,
        train_or_control,
        selected_indices,
        config.features.chunk_size,
    )
    target_indices = _target_indices([*train_genes, *test_genes], selected_symbols)
    control_mask = obs_labels == config.data.control_label
    control_matrix = _matrix_rows_cols(
        adata.X,
        np.flatnonzero(control_mask),
        selected_indices,
    )
    control_obs = adata.obs.loc[control_mask].reset_index(drop=True).copy()
    train_means: dict[str, np.ndarray] = {}
    train_cells: dict[str, np.ndarray] = {}
    train_obs: dict[str, pd.DataFrame] = {}
    for gene in train_genes.astype(str):
        mask = obs_labels == gene
        if not np.any(mask):
            msg = f"Train gene {gene!r} has no cells in h5ad"
            raise ValueError(msg)
        cells = _matrix_rows_cols(adata.X, np.flatnonzero(mask), selected_indices)
        train_cells[gene] = cells
        train_means[gene] = cells.mean(axis=0, dtype=np.float64).astype(np.float32)
        train_obs[gene] = adata.obs.loc[mask].reset_index(drop=True).copy()
    return FoldExpressionData(
        genes=np.asarray([*train_genes, *test_genes], dtype=object),
        y=metadata[config.data.depmap_label_col].to_numpy(dtype=np.float64),
        metadata=metadata,
        selected_indices=selected_indices,
        selected_symbols=selected_symbols,
        selected_mean=selected_mean,
        selected_std=selected_std,
        control_matrix=control_matrix.astype(np.float32),
        train_means=train_means,
        train_cells=train_cells,
        control_obs=control_obs,
        train_obs=train_obs,
        target_indices=target_indices,
    )


def _fit_a_to_b_model(
    config: BaselineConfig,
    data: FoldExpressionData,
    train_genes: np.ndarray,
    method: str,
    fold_index: int,
) -> Ridge:
    if method == MEAN_DELTA_METHOD:
        x, y = _mean_delta_training_matrix(config, data, train_genes)
    elif method == PSEUDO_PAIR_METHOD:
        x, y = _pseudo_pair_training_matrix(config, data, train_genes, fold_index)
    else:
        msg = f"Unsupported predicted-B method: {method}"
        raise ValueError(msg)
    model = Ridge(alpha=float(config.predicted_b.a_to_b_alpha))
    model.fit(x, y)
    return model


def _mean_delta_training_matrix(
    config: BaselineConfig,
    data: FoldExpressionData,
    train_genes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    control_mean = data.control_matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    rows = []
    targets = []
    train_gene_list = train_genes.astype(str).tolist()
    for gene in train_gene_list:
        rows.append(
            _a_to_b_features(
                config,
                control_mean[None, :],
                gene,
                data.target_indices,
            )[0]
        )
        targets.append(data.train_means[gene] - control_mean)
    return np.vstack(rows).astype(np.float32), np.vstack(targets).astype(np.float32)


def _pseudo_pair_training_matrix(
    config: BaselineConfig,
    data: FoldExpressionData,
    train_genes: np.ndarray,
    fold_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.cv.random_state + fold_index)
    rows = []
    targets = []
    train_gene_list = train_genes.astype(str).tolist()
    for gene in train_gene_list:
        perturbed = data.train_cells[gene]
        control_indices, perturbed_indices = _paired_indices(
            config,
            data,
            gene,
            rng,
        )
        control = data.control_matrix[control_indices]
        paired = perturbed[perturbed_indices]
        rows.append(
            _a_to_b_features(
                config,
                control,
                gene,
                data.target_indices,
            )
        )
        targets.append((paired - control).astype(np.float32))
    return np.vstack(rows).astype(np.float32), np.vstack(targets).astype(np.float32)


def _predict_bags(
    config: BaselineConfig,
    data: FoldExpressionData,
    model: Ridge,
    genes: np.ndarray,
    control_panel: np.ndarray,
) -> PredictedBagResult:
    bags = []
    rows = []
    for gene in genes.astype(str):
        features = _a_to_b_features(
            config,
            control_panel,
            gene,
            data.target_indices,
        )
        delta = model.predict(features).astype(np.float32)
        predicted = (control_panel + delta).astype(np.float32)
        if config.predicted_b.clip_min is not None:
            predicted = np.maximum(predicted, float(config.predicted_b.clip_min))
        bags.append(predicted)
        rows.append(
            {
                "perturbation_gene": gene,
                "predicted_n_cells": int(predicted.shape[0]),
                "target_gene_in_feature_space": int(
                    data.target_indices.get(gene, -1)
                )
                >= 0,
            }
        )
    return PredictedBagResult(
        bags=tuple(bags),
        metadata=pd.DataFrame(rows),
    )


def _a_to_b_features(
    config: BaselineConfig,
    expression: np.ndarray,
    gene: str,
    target_indices: dict[str, int],
) -> np.ndarray:
    masked = expression.astype(np.float32, copy=True)
    target_index = target_indices.get(gene, -1)
    if target_index >= 0:
        masked[:, target_index] = float(config.predicted_b.mask_value)
    return masked.astype(np.float32)


def _paired_indices(
    config: BaselineConfig,
    data: FoldExpressionData,
    gene: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    perturbed = data.train_cells[gene]
    n_samples = min(
        int(config.predicted_b.max_pair_samples_per_gene),
        int(perturbed.shape[0]),
    )
    strata_cols = [
        col
        for col in config.predicted_b.pairing_strata_cols
        if col in data.control_obs.columns and col in data.train_obs[gene].columns
    ]
    if not strata_cols:
        return (
            rng.choice(data.control_matrix.shape[0], size=n_samples, replace=True),
            rng.choice(perturbed.shape[0], size=n_samples, replace=True),
        )
    control_keys = _strata_keys(data.control_obs, strata_cols)
    perturbed_keys = _strata_keys(data.train_obs[gene], strata_cols)
    shared = np.intersect1d(np.unique(control_keys), np.unique(perturbed_keys))
    if shared.size == 0:
        return (
            rng.choice(data.control_matrix.shape[0], size=n_samples, replace=True),
            rng.choice(perturbed.shape[0], size=n_samples, replace=True),
        )
    control_rows = []
    perturbed_rows = []
    for _ in range(n_samples):
        key = str(rng.choice(shared))
        control_pool = np.flatnonzero(control_keys == key)
        perturbed_pool = np.flatnonzero(perturbed_keys == key)
        control_rows.append(int(rng.choice(control_pool)))
        perturbed_rows.append(int(rng.choice(perturbed_pool)))
    return (
        np.asarray(control_rows, dtype=np.int64),
        np.asarray(perturbed_rows, dtype=np.int64),
    )


def _control_panel(
    config: BaselineConfig,
    control_matrix: np.ndarray,
    fold_index: int,
) -> np.ndarray:
    if config.predicted_b.predicted_cells_per_gene != "capped_control_panel":
        msg = "Only predicted_cells_per_gene='capped_control_panel' is supported"
        raise ValueError(msg)
    n_cells = min(
        int(config.predicted_b.max_pred_cells_per_gene),
        int(control_matrix.shape[0]),
    )
    rng = np.random.default_rng(config.cv.random_state + 10_000 + fold_index)
    indices = rng.choice(control_matrix.shape[0], size=n_cells, replace=False)
    return control_matrix[np.sort(indices)].astype(np.float32)


def _fit_fold_projector(
    config: BaselineConfig,
    run_dir: Path,
    data: FoldExpressionData,
    train_genes: np.ndarray,
    fold_index: int,
    method: str,
) -> FoldProjector:
    feature_set = config.predicted_b.feature_set
    if feature_set == SCVI_FEATURE_SET:
        model_dir = _fit_fold_scvi(
            config,
            run_dir,
            data,
            train_genes,
            fold_index,
            method,
        )
    else:
        model_dir = None
    return FoldProjector(
        feature_set=feature_set,
        selected_mean=data.selected_mean,
        selected_std=data.selected_std,
        control_reference=data.control_matrix,
        scvi_model_dir=model_dir,
        selected_symbols=data.selected_symbols,
        query_batch_cells=config.predicted_b.scvi_query_batch_cells,
    )


def _fit_fold_scvi(
    config: BaselineConfig,
    run_dir: Path,
    data: FoldExpressionData,
    train_genes: np.ndarray,
    fold_index: int,
    method: str,
) -> Path:
    scvi = _import_scvi()
    train_matrix = np.vstack(
        [data.control_matrix, *(data.train_cells[gene] for gene in train_genes)]
    ).astype(np.float32)
    scvi_adata = ad.AnnData(train_matrix)
    scvi_adata.var_names = data.selected_symbols
    scvi.model.SCVI.setup_anndata(scvi_adata)
    model = scvi.model.SCVI(
        scvi_adata,
        n_latent=int(config.single_cell.scvi_latent_dim),
        n_hidden=int(config.single_cell.scvi_hidden_units),
        n_layers=int(config.single_cell.scvi_layers),
        dropout_rate=float(config.single_cell.dropout),
    )
    model.train(
        max_epochs=int(config.single_cell.scvi_max_epochs),
        batch_size=int(config.single_cell.scvi_batch_size),
        early_stopping=True,
    )
    model_dir = (
        run_dir
        / "artifacts"
        / "predicted_b"
        / method
        / f"fold_{fold_index}"
        / "scvi_model"
    )
    model.save(str(model_dir), overwrite=True, save_anndata=False)
    return model_dir


def _fit_predicted_gmm(
    config: BaselineConfig,
    projection: ProjectionResult,
) -> GaussianMixture:
    matrix = np.vstack([*projection.bags, projection.control_bag]).astype(np.float32)
    max_cells = config.distribution.max_gmm_fit_cells
    if max_cells is not None and matrix.shape[0] > max_cells:
        rng = np.random.default_rng(config.cv.random_state)
        indices = rng.choice(matrix.shape[0], size=int(max_cells), replace=False)
        matrix = matrix[np.sort(indices)]
    gmm = GaussianMixture(
        n_components=int(config.predicted_b.gmm_components),
        covariance_type="diag",
        random_state=int(config.cv.random_state),
        reg_covar=1e-5,
        max_iter=200,
    )
    gmm.fit(matrix)
    return gmm


def _gmm_features(
    config: BaselineConfig,
    gmm: GaussianMixture,
    bags: tuple[np.ndarray, ...],
    control_bag: np.ndarray,
) -> np.ndarray:
    control_feature = _occupancy_feature(gmm, control_bag)
    rows = [
        _feature_from_occupancy(
            _occupancy_feature(gmm, bag),
            config.predicted_b.gmm_view,
            control_feature,
        )
        for bag in bags
    ]
    return np.vstack(rows).astype(np.float32)


def _write_fold_predicted_b_artifacts(
    config: BaselineConfig,
    run_dir: Path,
    fold_index: int,
    method: str,
    train_pred: PredictedBagResult,
    test_pred: PredictedBagResult,
    train_reconstruction: dict[str, float],
    test_reconstruction: dict[str, float],
    projection_qa: dict[str, object],
    gmm: GaussianMixture,
) -> None:
    base = run_dir / "artifacts" / "predicted_b" / method / f"fold_{fold_index}"
    metadata = pd.concat(
        [
            train_pred.metadata.assign(split="train"),
            test_pred.metadata.assign(split="test"),
        ],
        ignore_index=True,
    )
    write_formats(
        base / "bag_metadata",
        metadata,
        (config.experiment.machine_result_format,),
    )
    reconstruction = pd.DataFrame(
        [
            {
                "fold": fold_index,
                "method": method,
                "a_to_b_train_mean_rmse": train_reconstruction["mean_rmse"],
                "a_to_b_train_mean_mae": train_reconstruction["mean_mae"],
                "a_to_b_test_mean_rmse": test_reconstruction["mean_rmse"],
                "a_to_b_test_mean_mae": test_reconstruction["mean_mae"],
            }
        ]
    )
    write_formats(
        base / "a_to_b_reconstruction_metrics",
        reconstruction,
        (config.experiment.machine_result_format,),
    )
    write_json(
        base / "feature_qa.json",
        {
            **projection_qa,
            "gmm_components": int(gmm.n_components),
            "gmm_view": config.predicted_b.gmm_view,
            "gmm_converged": bool(gmm.converged_),
            "gmm_n_iter": int(gmm.n_iter_),
        },
    )


def _fold_selected_gene_indices(
    matrix: object,
    row_mask: np.ndarray,
    config: BaselineConfig,
) -> np.ndarray:
    _mean, _std, variance = _chunked_moments(
        matrix,
        np.flatnonzero(row_mask),
        None,
        config.features.chunk_size,
    )
    return _select_hvg_indices(variance, config.single_cell.n_hvg)


def _selected_moments(
    matrix: object,
    row_mask: np.ndarray,
    selected_indices: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean, std, _variance = _chunked_moments(
        matrix,
        np.flatnonzero(row_mask),
        selected_indices,
        chunk_size,
    )
    return mean.astype(np.float32), std.astype(np.float32)


def _chunked_moments(
    matrix: object,
    row_indices: np.ndarray,
    col_indices: np.ndarray | None,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cols = matrix.shape[1] if col_indices is None else len(col_indices)
    sums = np.zeros(n_cols, dtype=np.float64)
    sums_sq = np.zeros(n_cols, dtype=np.float64)
    total = 0
    for start in range(0, len(row_indices), int(chunk_size)):
        stop = min(start + int(chunk_size), len(row_indices))
        block = _matrix_rows_cols(matrix, row_indices[start:stop], col_indices)
        block64 = block.astype(np.float64, copy=False)
        sums += block64.sum(axis=0)
        sums_sq += np.square(block64).sum(axis=0)
        total += block64.shape[0]
    if total == 0:
        msg = "Cannot compute moments from zero rows"
        raise ValueError(msg)
    mean = sums / float(total)
    variance = np.maximum(sums_sq / float(total) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std > 0, std, 1.0)
    return mean, std, variance


def _matrix_rows_cols(
    matrix: object,
    row_indices: np.ndarray,
    col_indices: np.ndarray | None,
) -> np.ndarray:
    if col_indices is None:
        return _dense_float32(matrix[row_indices])
    if sparse.issparse(matrix):
        return _dense_float32(matrix[row_indices][:, col_indices])
    return _dense_float32(matrix[row_indices][:, col_indices])


def _scale_expression(
    matrix: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((matrix - mean[None, :]) / std[None, :]).astype(np.float32)


def _var_symbols(adata: ad.AnnData, column: str) -> list[str]:
    if column in adata.var.columns:
        return adata.var[column].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _target_indices(
    genes: list[str] | np.ndarray,
    selected_symbols: list[str],
) -> dict[str, int]:
    symbol_to_index = {symbol: index for index, symbol in enumerate(selected_symbols)}
    return {str(gene): symbol_to_index.get(str(gene), -1) for gene in genes}


def _strata_keys(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].astype(str).agg("||".join, axis=1).to_numpy(dtype=str)


def _load_observed_gene_means(
    adata: ad.AnnData,
    obs_labels: np.ndarray,
    selected_indices: np.ndarray,
    genes: np.ndarray,
    split_name: str,
) -> dict[str, np.ndarray]:
    means: dict[str, np.ndarray] = {}
    for gene in genes.astype(str):
        mask = obs_labels == gene
        if not np.any(mask):
            msg = f"{split_name.capitalize()} gene {gene!r} has no cells in h5ad"
            raise ValueError(msg)
        cells = _matrix_rows_cols(adata.X, np.flatnonzero(mask), selected_indices)
        means[gene] = cells.mean(axis=0, dtype=np.float64).astype(np.float32)
    return means


def _reconstruction_metrics_for_predictions(
    genes: np.ndarray,
    predicted_bags: tuple[np.ndarray, ...],
    observed_means: dict[str, np.ndarray],
) -> dict[str, float]:
    rows = []
    for gene, predicted in zip(genes.astype(str), predicted_bags, strict=True):
        rows.append(_mean_reconstruction_row(gene, predicted, observed_means[gene]))
    return _aggregate_reconstruction(rows)


def _mean_reconstruction_row(
    gene: str,
    predicted: np.ndarray,
    observed_mean: np.ndarray,
) -> dict[str, float | str]:
    mean_pred = predicted.mean(axis=0, dtype=np.float64)
    error = mean_pred - observed_mean.astype(np.float64)
    return {
        "perturbation_gene": gene,
        "mean_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_mae": float(np.mean(np.abs(error))),
    }


def _aggregate_reconstruction(rows: list[dict[str, float | str]]) -> dict[str, float]:
    if not rows:
        return {"mean_rmse": np.nan, "mean_mae": np.nan}
    frame = pd.DataFrame(rows)
    return {
        "mean_rmse": float(frame["mean_rmse"].mean()),
        "mean_mae": float(frame["mean_mae"].mean()),
    }
