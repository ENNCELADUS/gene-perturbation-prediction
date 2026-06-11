"""Frozen-STATE feature ablation helpers for predicted-B->C experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from dataclasses import replace
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
import torch
from tqdm import tqdm
import yaml

from dependency_baseline.distribution import _feature_from_occupancy
from dependency_baseline.distribution import _occupancy_feature
from dependency_baseline.metrics import ranking_metrics, regression_metrics

from aivc_model.model import PerturbationVectorAdapter
from aivc_model.model import StateForwardAdapter
from aivc_model.model import load_state_model
from aivc_model.prepare import GeneBags
from aivc_model.prepare import GeneSplit
from aivc_model.prepare import encode_batch_labels
from aivc_model.prepare import fit_linear_projector
from aivc_model.prepare import load_config
from aivc_model.prepare import load_external_gene_bags
from aivc_model.prepare import load_gene_bags
from aivc_model.prepare import load_perturbation_vectors
from aivc_model.prepare import load_state_batch_lookup
from aivc_model.prepare import sample_indices
from aivc_model.prepare import with_cached_scvi_teacher_latents
from aivc_model.state_feature_ablation_tables import (
    EXTERNAL_ENSEMBLE_SCOPE,
    PRIMARY_EXTERNAL_SCOPE,
    adamson_heldout_ensemble_predictions,
    alpha_token,
    feature_qa_row,
    gmm_metadata_row,
    metric_rows_for_predictions,
    required_result_columns,
    validate_result_tables,
    write_ablation_artifacts,
    write_run_manifest,
)

INTERNAL_SCOPE = "internal_cv_all"
DEFAULT_ABLATION_ARMS = (
    "observed_scvi128_gmm_ridge_anchor",
    "state_output_scvi128_gmm_ridge",
    "state_output_hvg_gmm_ridge",
    "state_token_hidden_gmm_ridge",
)
TRAIN_LOG_COLUMNS = (
    "epoch",
    "fold",
    "elapsed_seconds",
    "fold_elapsed_seconds",
    "n_train_genes",
    "n_test_genes",
    "n_arms_completed",
    "n_fits_completed",
    "n_external_fits_completed",
    "n_views",
    "n_alphas",
    "internal_rmse_mean",
    "internal_spearman_mean",
    "internal_spearman_defined_rate",
    "external_ensemble_rmse_mean",
    "external_ensemble_spearman_mean",
    "external_ensemble_spearman_defined_rate",
    "external_heldout_rmse_mean",
    "external_heldout_spearman_mean",
    "external_heldout_spearman_defined_rate",
)
__all__ = (
    "EXTERNAL_ENSEMBLE_SCOPE",
    "PRIMARY_EXTERNAL_SCOPE",
    "FeatureArmData",
    "FoldFit",
    "StateFeatureAblationConfig",
    "TokenHiddenPair",
    "adamson_heldout_ensemble_predictions",
    "fit_fold_gmm",
    "fit_fold_ridge",
    "gmm_feature_matrix",
    "load_state_feature_ablation_config",
    "metric_rows_for_predictions",
    "required_result_columns",
    "same_path_token_hidden_bags",
    "state_output_bag",
    "state_output_feature_arm",
    "run_ablation_from_config",
    "state_token_hidden_feature_arm",
    "token_hidden_bag",
    "validate_result_tables",
    "write_ablation_artifacts",
)


@dataclass(frozen=True)
class StateFeatureAblationConfig:
    """Parsed config fields used by the frozen STATE feature ablation."""

    output_dir: Path
    run_id: str
    seed: int = 42
    n_splits: int = 5
    max_control_cells_per_gene: int = 512
    gmm_components: int = 64
    gmm_view: str = "centered"
    sensitivity_views: tuple[str, ...] = ("deltap",)
    ridge_alphas: tuple[float, ...] = (30.0, 300.0)
    arms: tuple[str, ...] = DEFAULT_ABLATION_ARMS
    external_name: str = "adamson_k562"
    primary_scope: str = PRIMARY_EXTERNAL_SCOPE
    interpretation: str = "adamson_guided_validation_sweep"
    device: str = "auto"


@dataclass(frozen=True)
class FeatureArmData:
    """One representation-space arm represented as gene-level cell bags."""

    feature_set: str
    arm: str
    genes: np.ndarray
    y: np.ndarray
    bags: tuple[np.ndarray, ...]
    control_bag: np.ndarray
    embedding_space: str
    gmm_fit_source: str = "B_hat_train_plus_controls"
    metadata: pd.DataFrame | None = None


@dataclass(frozen=True)
class FoldFit:
    """Fold-local GMM/Ridge outputs for one arm and one alpha."""

    metric_row: dict[str, object]
    predictions: pd.DataFrame
    qa_row: dict[str, object]
    gmm_row: dict[str, object]
    train_genes: tuple[str, ...]
    gmm: GaussianMixture
    ridge: Ridge
    view: str


@dataclass(frozen=True)
class TokenHiddenPair:
    """Same-path non-targeting and target token-hidden bags."""

    control_bag: np.ndarray
    perturbed_bag: np.ndarray


def load_state_feature_ablation_config(path: Path) -> StateFeatureAblationConfig:
    """Load the narrow ablation fields from a YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = raw.get("data", {})
    experiment = raw.get("experiment", {})
    cv = raw.get("cv", {})
    predicted_b = raw.get("predicted_b", {})
    distribution = raw.get("distribution", {})
    ablation = raw.get("state_feature_ablation", {})
    external = raw.get("external_test", {})
    output_dir = Path(data["output_dir"])
    run_id = str(
        ablation.get(
            "run_id",
            experiment.get("run_id", "state_frozen_feature_ablation"),
        )
    )
    return StateFeatureAblationConfig(
        output_dir=output_dir,
        run_id=run_id,
        seed=int(
            ablation.get("seed", experiment.get("seed", cv.get("random_state", 42)))
        ),
        n_splits=int(ablation.get("n_splits", cv.get("n_splits", 5))),
        max_control_cells_per_gene=int(
            ablation.get(
                "max_control_cells_per_gene",
                predicted_b.get("max_pred_cells_per_gene", 512),
            )
        ),
        gmm_components=int(
            ablation.get(
                "gmm_components",
                predicted_b.get(
                    "gmm_components",
                    next(iter(distribution.get("component_counts", [64]))),
                ),
            )
        ),
        gmm_view=str(ablation.get("gmm_view", predicted_b.get("gmm_view", "centered"))),
        sensitivity_views=tuple(
            str(value)
            for value in ablation.get(
                "sensitivity_views",
                distribution.get("views", ["centered", "deltap"])[1:],
            )
        ),
        ridge_alphas=tuple(
            float(value)
            for value in ablation.get(
                "ridge_alphas",
                distribution.get("ridge_alphas", [30.0, 300.0]),
            )
        ),
        arms=tuple(str(value) for value in ablation.get("arms", DEFAULT_ABLATION_ARMS)),
        external_name=str(
            external.get("name", ablation.get("external_name", "adamson_k562"))
        ),
        primary_scope=str(ablation.get("primary_scope", PRIMARY_EXTERNAL_SCOPE)),
        interpretation=str(
            ablation.get("interpretation", "adamson_guided_validation_sweep")
        ),
        device=str(ablation.get("device", raw.get("train", {}).get("device", "auto"))),
    )


def same_path_token_hidden_bags(
    adapter: StateForwardAdapter,
    control_cells: torch.Tensor,
    non_targeting_vector: torch.Tensor,
    target_vector: torch.Tensor,
    target_gene: str,
    *,
    control_gene: str = "non-targeting",
    batch_indices: torch.Tensor | None = None,
) -> TokenHiddenPair:
    """Embed control and target token-hidden bags through the same STATE path."""
    control = token_hidden_bag(
        adapter,
        control_cells,
        non_targeting_vector,
        control_gene,
        batch_indices=batch_indices,
    )
    perturbed = token_hidden_bag(
        adapter,
        control_cells,
        target_vector,
        target_gene,
        batch_indices=batch_indices,
    )
    return TokenHiddenPair(control_bag=control, perturbed_bag=perturbed)


def state_output_feature_arm(
    adapter: StateForwardAdapter,
    control_cells: torch.Tensor,
    perturbation_vectors: dict[str, torch.Tensor],
    genes: np.ndarray,
    y: np.ndarray,
    non_targeting_vector: torch.Tensor,
    *,
    feature_set: str,
    arm: str,
    embedding_space: str = "state_output",
    control_gene: str = "non-targeting",
    batch_indices: torch.Tensor | None = None,
) -> FeatureArmData:
    """Export a frozen STATE output-space feature arm for selected genes."""
    control_bag = state_output_bag(
        adapter,
        control_cells,
        non_targeting_vector,
        control_gene,
        batch_indices=batch_indices,
    )
    bags = tuple(
        state_output_bag(
            adapter,
            control_cells,
            perturbation_vectors[str(gene)],
            str(gene),
            batch_indices=batch_indices,
        )
        for gene in genes.astype(str)
    )
    return FeatureArmData(
        feature_set=feature_set,
        arm=arm,
        genes=genes.astype(str),
        y=np.asarray(y, dtype=np.float64),
        bags=bags,
        control_bag=control_bag,
        embedding_space=embedding_space,
    )


def state_token_hidden_feature_arm(
    adapter: StateForwardAdapter,
    control_cells: torch.Tensor,
    perturbation_vectors: dict[str, torch.Tensor],
    genes: np.ndarray,
    y: np.ndarray,
    non_targeting_vector: torch.Tensor,
    *,
    feature_set: str,
    arm: str = "state_token_hidden_gmm_ridge",
    control_gene: str = "non-targeting",
    batch_indices: torch.Tensor | None = None,
) -> FeatureArmData:
    """Export a frozen STATE token-hidden feature arm for selected genes."""
    control_bag = token_hidden_bag(
        adapter,
        control_cells,
        non_targeting_vector,
        control_gene,
        batch_indices=batch_indices,
    )
    bags = tuple(
        token_hidden_bag(
            adapter,
            control_cells,
            perturbation_vectors[str(gene)],
            str(gene),
            batch_indices=batch_indices,
        )
        for gene in genes.astype(str)
    )
    return FeatureArmData(
        feature_set=feature_set,
        arm=arm,
        genes=genes.astype(str),
        y=np.asarray(y, dtype=np.float64),
        bags=bags,
        control_bag=control_bag,
        embedding_space="state_token_hidden",
    )


def state_output_bag(
    adapter: StateForwardAdapter,
    control_cells: torch.Tensor,
    perturbation_vector: torch.Tensor,
    gene: str,
    *,
    batch_indices: torch.Tensor | None = None,
) -> np.ndarray:
    """Run frozen STATE once and return its output-space bag."""
    output = adapter(control_cells, perturbation_vector, gene, batch_indices)
    return output.detach().cpu().numpy().astype(np.float32)


def token_hidden_bag(
    adapter: StateForwardAdapter,
    control_cells: torch.Tensor,
    perturbation_vector: torch.Tensor,
    gene: str,
    *,
    batch_indices: torch.Tensor | None = None,
) -> np.ndarray:
    """Run STATE once and return the captured token hidden feature bag."""
    _ = adapter(control_cells, perturbation_vector, gene, batch_indices)
    hidden = adapter.last_token_features
    if hidden is None:
        msg = "STATE model did not expose _token_features during forward"
        raise ValueError(msg)
    return hidden.detach().cpu().numpy().astype(np.float32)


def fit_fold_gmm(
    data: FeatureArmData,
    train_idx: np.ndarray,
    *,
    n_components: int,
    random_state: int,
) -> GaussianMixture:
    """Fit a diagonal GMM on train-gene bags plus controls only."""
    matrix = np.vstack(
        [*(data.bags[index] for index in train_idx), data.control_bag]
    ).astype(np.float32)
    gmm = GaussianMixture(
        n_components=int(n_components),
        covariance_type="diag",
        reg_covar=1e-4,
        random_state=int(random_state),
        max_iter=200,
        n_init=1,
    )
    gmm.fit(matrix)
    gmm.fit_gene_names_ = data.genes[train_idx].astype(str)
    gmm.fit_cell_count_ = int(matrix.shape[0])
    return gmm


def fit_fold_ridge(
    data: FeatureArmData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    fold: int,
    alpha: float,
    n_components: int,
    view: str,
    random_state: int,
    weighting: str = "unweighted",
    sensitivity_views: tuple[str, ...] = ("deltap",),
) -> FoldFit:
    """Fit fold-local GMM features and a Ridge C head for one alpha."""
    gmm = fit_fold_gmm(
        data,
        train_idx,
        n_components=n_components,
        random_state=random_state + fold,
    )
    x_train = gmm_feature_matrix(gmm, data.bags, train_idx, data.control_bag, view)
    x_test = gmm_feature_matrix(gmm, data.bags, test_idx, data.control_bag, view)
    model = Ridge(alpha=float(alpha))
    model.fit(x_train, data.y[train_idx])
    y_pred = model.predict(x_test)
    alpha_name = alpha_token(alpha)
    model_name = f"{data.arm}_k{int(n_components)}_{view}_ridge_alpha{alpha_name}"
    job_key = f"{INTERNAL_SCOPE}__fold{fold}__{data.feature_set}__{model_name}"
    metric_row = {
        "job_key": job_key,
        "evaluation_scope": INTERNAL_SCOPE,
        "fold": int(fold),
        "feature_set": data.feature_set,
        "arm": data.arm,
        "model": model_name,
        "weighting": weighting,
        "primary_scope": PRIMARY_EXTERNAL_SCOPE,
        "secondary_scope": EXTERNAL_ENSEMBLE_SCOPE,
        **regression_metrics(data.y[test_idx], y_pred),
        **ranking_metrics(data.y[test_idx], y_pred, (-0.5, -1.0)),
    }
    predictions = pd.DataFrame(
        {
            "job_key": job_key,
            "evaluation_scope": INTERNAL_SCOPE,
            "fold": int(fold),
            "feature_set": data.feature_set,
            "arm": data.arm,
            "model": model_name,
            "weighting": weighting,
            "perturbation_gene": data.genes[test_idx].astype(str),
            "y_true": data.y[test_idx],
            "y_pred": y_pred,
            "primary_scope": PRIMARY_EXTERNAL_SCOPE,
            "secondary_scope": EXTERNAL_ENSEMBLE_SCOPE,
        }
    )
    qa_row = feature_qa_row(
        data,
        fold=fold,
        train_idx=train_idx,
        test_idx=test_idx,
        gmm=gmm,
        primary_view=view,
        sensitivity_views=sensitivity_views,
    )
    gmm_row = gmm_metadata_row(data, fold=fold, gmm=gmm)
    return FoldFit(
        metric_row=metric_row,
        predictions=predictions,
        qa_row=qa_row,
        gmm_row=gmm_row,
        train_genes=tuple(data.genes[train_idx].astype(str)),
        gmm=gmm,
        ridge=model,
        view=view,
    )


def gmm_feature_matrix(
    gmm: GaussianMixture,
    bags: tuple[np.ndarray, ...],
    indices: np.ndarray,
    control_bag: np.ndarray,
    view: str,
) -> np.ndarray:
    """Build centered or deltap GMM occupancy features for selected bags."""
    control_feature = _occupancy_feature(gmm, control_bag)
    rows = [
        _feature_from_occupancy(
            _occupancy_feature(gmm, bags[index]),
            view,
            control_feature,
        )
        for index in indices
    ]
    return np.vstack(rows).astype(np.float32)


def run_ablation_from_config(path: Path) -> Path:
    """Run the frozen STATE feature-ablation matrix from a YAML config."""
    config = load_state_feature_ablation_config(path)
    aivc_config = load_config(path)
    run_dir = config.output_dir / "runs" / config.run_id
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    train_log_path = run_dir / "train_log.csv"
    _initialize_train_log(train_log_path)
    write_run_manifest(
        run_dir,
        run_id=config.run_id,
        status="running",
        payload={
            "seed": config.seed,
            "n_splits": config.n_splits,
            "arms": list(config.arms),
            "max_control_cells_per_gene": config.max_control_cells_per_gene,
            "gmm_components": config.gmm_components,
            "gmm_view": config.gmm_view,
            "sensitivity_views": list(config.sensitivity_views),
            "ridge_alphas": list(config.ridge_alphas),
            "primary_metric": f"{config.primary_scope} Spearman",
            "interpretation": config.interpretation,
            "config_path": str(path),
            "checkpoint_path": str(aivc_config.state.checkpoint_path),
            "known_perturbation_vectors": str(
                aivc_config.state.known_perturbation_vectors
            ),
            "trusted_local_checkpoint_assets": True,
        },
    )
    try:
        data = load_gene_bags(aivc_config)
        external = load_external_gene_bags(
            aivc_config,
            data,
            artifacts_dir / "external_loader",
            project_scvi=False,
        )
        adapter, perturbations, batch_lookup, device = _load_frozen_state_runtime(
            aivc_config,
            data,
            external.data if external is not None else None,
            config.device,
        )
        fold_rows = []
        prediction_frames = []
        external_prediction_frames = []
        split_rows = []
        qa_rows = []
        gmm_rows = []
        folds = _stratified_folds(data.y, config.n_splits, config.seed)
        run_start = time.monotonic()
        fold_iterator = tqdm(
            enumerate(folds, start=1),
            desc="folds",
            total=len(folds),
            miniters=1,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for epoch, (train_idx, test_idx) in fold_iterator:
            fold = epoch - 1
            fold_start = time.monotonic()
            fold_iterator.set_postfix_str(f"fold={fold}", refresh=False)
            split = GeneSplit(
                train=np.sort(train_idx),
                val=np.sort(test_idx),
                test=np.asarray([], dtype=np.int64),
            )
            fold_split_rows = _split_rows(data, fold, train_idx, test_idx)
            split_rows.extend(fold_split_rows)
            primary_panel = _control_panel(data, config, fold, device, batch_lookup)
            external_panel = (
                _control_panel(
                    external.data,
                    config,
                    fold + 10_000,
                    device,
                    batch_lookup,
                )
                if external is not None
                else None
            )
            primary_arms, external_arms = _build_fold_arms(
                aivc_config=aivc_config,
                config=config,
                run_dir=run_dir,
                fold=fold,
                split=split,
                data=data,
                external_data=external.data if external is not None else None,
                adapter=adapter,
                perturbations=perturbations,
                primary_panel=primary_panel,
                external_panel=external_panel,
            )
            fit_tasks = [
                (arm_name, arm_data, view, alpha)
                for arm_name, arm_data in primary_arms.items()
                for view in (config.gmm_view, *config.sensitivity_views)
                for alpha in config.ridge_alphas
            ]
            fold_metric_rows = []
            fold_external_prediction_frames = []
            fit_iterator = tqdm(
                fit_tasks,
                desc=f"fold {epoch}/{len(folds)} fits",
                total=len(fit_tasks),
                miniters=1,
                dynamic_ncols=True,
                file=sys.stdout,
            )
            for arm_name, arm_data, view, alpha in fit_iterator:
                fit_iterator.set_postfix(
                    {
                        "arm": arm_name,
                        "view": view,
                        "alpha": alpha,
                    },
                    refresh=False,
                )
                fit = fit_fold_ridge(
                    arm_data,
                    train_idx,
                    test_idx,
                    fold=fold,
                    alpha=alpha,
                    n_components=config.gmm_components,
                    view=view,
                    random_state=config.seed,
                    sensitivity_views=config.sensitivity_views,
                )
                fold_rows.append(fit.metric_row)
                fold_metric_rows.append(fit.metric_row)
                prediction_frames.append(fit.predictions)
                qa_rows.append(fit.qa_row)
                gmm_rows.append(fit.gmm_row)
                external_arm = external_arms.get(arm_name)
                if external_arm is not None:
                    external_predictions_for_fit = _external_predictions_for_fit(
                        fit,
                        external_arm,
                        external_name=config.external_name,
                    )
                    external_prediction_frames.append(external_predictions_for_fit)
                    fold_external_prediction_frames.append(external_predictions_for_fit)
            fold_external_predictions = (
                pd.concat(fold_external_prediction_frames, ignore_index=True)
                if fold_external_prediction_frames
                else pd.DataFrame()
            )
            _append_train_log_row(
                train_log_path,
                _fold_train_log_row(
                    config,
                    epoch=epoch,
                    fold=fold,
                    elapsed_seconds=time.monotonic() - run_start,
                    fold_elapsed_seconds=time.monotonic() - fold_start,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    internal_metrics=pd.DataFrame(fold_metric_rows),
                    external_predictions=fold_external_predictions,
                    fold_splits=pd.DataFrame(fold_split_rows),
                    n_arms_completed=len(primary_arms),
                    n_fits_completed=len(fold_metric_rows),
                    n_external_fits_completed=len(fold_external_prediction_frames),
                ),
            )
        fold_metrics = pd.DataFrame(fold_rows)
        predictions = pd.concat(prediction_frames, ignore_index=True)
        splits = pd.DataFrame(split_rows)
        feature_qa = pd.DataFrame(qa_rows)
        gmm_metadata = pd.DataFrame(gmm_rows)
        external_predictions = (
            pd.concat(external_prediction_frames, ignore_index=True)
            if external_prediction_frames
            else pd.DataFrame(columns=predictions.columns)
        )
        external_ensemble_predictions = adamson_heldout_ensemble_predictions(
            external_predictions,
            splits,
            external_name=config.external_name,
        )
        external_ensemble_metrics = metric_rows_for_predictions(
            external_ensemble_predictions
        )
        write_ablation_artifacts(
            run_dir,
            fold_metrics=fold_metrics,
            predictions=pd.concat(
                [predictions, external_predictions],
                ignore_index=True,
            ),
            splits=splits,
            feature_qa=feature_qa,
            gmm_metadata=gmm_metadata,
            external_ensemble_predictions=external_ensemble_predictions,
            external_ensemble_metrics=external_ensemble_metrics,
        )
        write_run_manifest(
            run_dir,
            run_id=config.run_id,
            status="completed",
            payload={
                "seed": config.seed,
                "n_splits": config.n_splits,
                "arms": list(config.arms),
                "max_control_cells_per_gene": config.max_control_cells_per_gene,
                "primary_metric": f"{config.primary_scope} Spearman",
                "interpretation": config.interpretation,
                "config_path": str(path),
                "checkpoint_path": str(aivc_config.state.checkpoint_path),
                "known_perturbation_vectors": str(
                    aivc_config.state.known_perturbation_vectors
                ),
                "trusted_local_checkpoint_assets": True,
                "artifacts": {
                    "fold_metrics": str(artifacts_dir / "fold_metrics.parquet"),
                    "predictions": str(artifacts_dir / "predictions.parquet"),
                    "external_ensemble_metrics": str(
                        artifacts_dir / "external_ensemble_metrics.parquet"
                    ),
                },
            },
        )
    except Exception as exc:
        write_run_manifest(
            run_dir,
            run_id=config.run_id,
            status="failed",
            payload={
                "seed": config.seed,
                "n_splits": config.n_splits,
                "arms": list(config.arms),
                "primary_metric": f"{config.primary_scope} Spearman",
                "interpretation": config.interpretation,
                "config_path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    return run_dir


def _initialize_train_log(path: Path) -> None:
    """Create an empty fold-level train log before heavy ablation work starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=TRAIN_LOG_COLUMNS).to_csv(path, index=False)


def _append_train_log_row(path: Path, row: dict[str, object]) -> None:
    """Append one fold-level train-log row and flush it for partial run inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_LOG_COLUMNS)
        writer.writerow({column: row.get(column, "") for column in TRAIN_LOG_COLUMNS})
        handle.flush()


def _fold_train_log_row(
    config: StateFeatureAblationConfig,
    *,
    epoch: int,
    fold: int,
    elapsed_seconds: float,
    fold_elapsed_seconds: float,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    internal_metrics: pd.DataFrame,
    external_predictions: pd.DataFrame,
    fold_splits: pd.DataFrame,
    n_arms_completed: int,
    n_fits_completed: int,
    n_external_fits_completed: int,
) -> dict[str, object]:
    external_metrics = pd.DataFrame()
    if not external_predictions.empty:
        fold_external_ensembles = adamson_heldout_ensemble_predictions(
            external_predictions,
            fold_splits,
            external_name=config.external_name,
        )
        external_metrics = metric_rows_for_predictions(fold_external_ensembles)
    external_ensemble = _metrics_for_scope(external_metrics, EXTERNAL_ENSEMBLE_SCOPE)
    external_heldout = _metrics_for_scope(external_metrics, config.primary_scope)
    return {
        "epoch": int(epoch),
        "fold": int(fold),
        "elapsed_seconds": float(elapsed_seconds),
        "fold_elapsed_seconds": float(fold_elapsed_seconds),
        "n_train_genes": int(len(train_idx)),
        "n_test_genes": int(len(test_idx)),
        "n_arms_completed": int(n_arms_completed),
        "n_fits_completed": int(n_fits_completed),
        "n_external_fits_completed": int(n_external_fits_completed),
        "n_views": int(1 + len(config.sensitivity_views)),
        "n_alphas": int(len(config.ridge_alphas)),
        **_metric_summary(internal_metrics, "internal"),
        **_metric_summary(external_ensemble, "external_ensemble"),
        **_metric_summary(external_heldout, "external_heldout"),
    }


def _metrics_for_scope(metrics: pd.DataFrame, scope: str) -> pd.DataFrame:
    if metrics.empty or "evaluation_scope" not in metrics.columns:
        return pd.DataFrame()
    return metrics.loc[metrics["evaluation_scope"] == scope]


def _metric_summary(metrics: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_rmse_mean": _finite_mean(metrics, "rmse"),
        f"{prefix}_spearman_mean": _finite_mean(metrics, "spearman"),
        f"{prefix}_spearman_defined_rate": _finite_mean(metrics, "spearman_defined"),
    }


def _finite_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(finite.mean())


def _load_frozen_state_runtime(
    aivc_config: object,
    data: GeneBags,
    external_data: GeneBags | None,
    device_name: str,
) -> tuple[
    StateForwardAdapter,
    PerturbationVectorAdapter,
    dict[str, int],
    torch.device,
]:
    known_vectors = load_perturbation_vectors(
        aivc_config.state.known_perturbation_vectors
    )
    pert_dim = aivc_config.state.pert_dim or _infer_pert_dim(known_vectors)
    state_model = load_state_model(
        backend=aivc_config.state.backend,
        checkpoint_path=aivc_config.state.checkpoint_path,
        input_dim=aivc_config.state.input_dim or data.input_dim,
        output_dim=aivc_config.state.output_dim or data.input_dim,
        pert_dim=pert_dim,
        emit_checkpoint_output=False,
    )
    device = _resolve_device(device_name)
    adapter = StateForwardAdapter(state_model.to(device))
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()
    extra_genes: set[str] = set()
    if external_data is not None:
        extra_genes.update(str(gene) for gene in external_data.genes)
    perturbations = PerturbationVectorAdapter(
        sorted({*(str(gene) for gene in data.genes), *extra_genes, "non-targeting"}),
        known_vectors,
        pert_dim,
    ).to(device)
    for parameter in perturbations.parameters():
        parameter.requires_grad = False
    batch_lookup = load_state_batch_lookup(aivc_config.state.model_dir)
    return adapter, perturbations, batch_lookup, device


def _build_fold_arms(
    *,
    aivc_config: object,
    config: StateFeatureAblationConfig,
    run_dir: Path,
    fold: int,
    split: GeneSplit,
    data: GeneBags,
    external_data: GeneBags | None,
    adapter: StateForwardAdapter,
    perturbations: PerturbationVectorAdapter,
    primary_panel: tuple[torch.Tensor, torch.Tensor | None],
    external_panel: tuple[torch.Tensor, torch.Tensor | None] | None,
) -> tuple[dict[str, FeatureArmData], dict[str, FeatureArmData]]:
    fold_dir = run_dir / "artifacts" / "fold_local" / f"fold_{fold}"
    primary_expr = _state_expression_arms(
        data,
        adapter,
        perturbations,
        primary_panel,
        progress_desc=f"fold {fold} primary STATE",
    )
    external_expr = (
        _state_expression_arms(
            external_data,
            adapter,
            perturbations,
            external_panel,
            progress_desc=f"fold {fold} external STATE",
        )
        if external_data is not None and external_panel is not None
        else None
    )
    scvi_data = None
    scvi_external = None
    if _needs_scvi(config):
        scvi_data, scvi_external = with_cached_scvi_teacher_latents(
            aivc_config,
            data,
            split,
            fold_dir,
            external=None,
        )
        del scvi_external
    primary: dict[str, FeatureArmData] = {}
    external: dict[str, FeatureArmData] = {}
    if "observed_scvi128_gmm_ridge_anchor" in config.arms:
        if scvi_data is None:
            msg = "observed scVI anchor requires projector.teacher='scvi'"
            raise ValueError(msg)
        primary["observed_scvi128_gmm_ridge_anchor"] = replace(
            _arm_from_bags(
                data=scvi_data,
                bags=scvi_data.latent_bags,
                control_bag=scvi_data.control_latent,
                arm="observed_scvi128_gmm_ridge_anchor",
                feature_set="observed_scvi128",
                embedding_space="scvi_latent",
            ),
            gmm_fit_source="observed_train_plus_controls",
        )
    if "state_output_hvg_gmm_ridge" in config.arms:
        primary["state_output_hvg_gmm_ridge"] = _arm_from_bags(
            data=data,
            bags=primary_expr["output_bags"],
            control_bag=primary_expr["output_control"],
            arm="state_output_hvg_gmm_ridge",
            feature_set="state_output_hvg",
            embedding_space="state_output",
        )
        if external_data is not None and external_expr is not None:
            external["state_output_hvg_gmm_ridge"] = _arm_from_bags(
                data=external_data,
                bags=external_expr["output_bags"],
                control_bag=external_expr["output_control"],
                arm="state_output_hvg_gmm_ridge",
                feature_set="state_output_hvg",
                embedding_space="state_output",
            )
    if "state_token_hidden_gmm_ridge" in config.arms:
        primary["state_token_hidden_gmm_ridge"] = _arm_from_bags(
            data=data,
            bags=primary_expr["hidden_bags"],
            control_bag=primary_expr["hidden_control"],
            arm="state_token_hidden_gmm_ridge",
            feature_set="state_token_hidden",
            embedding_space="state_token_hidden",
        )
        if external_data is not None and external_expr is not None:
            external["state_token_hidden_gmm_ridge"] = _arm_from_bags(
                data=external_data,
                bags=external_expr["hidden_bags"],
                control_bag=external_expr["hidden_control"],
                arm="state_token_hidden_gmm_ridge",
                feature_set="state_token_hidden",
                embedding_space="state_token_hidden",
            )
    if "state_output_scvi128_gmm_ridge" in config.arms:
        if scvi_data is None:
            msg = "state output scVI arm requires projector.teacher='scvi'"
            raise ValueError(msg)
        weight, bias = _fit_projector(
            scvi_data,
            split,
            aivc_config.projector.ridge_alpha,
        )
        primary["state_output_scvi128_gmm_ridge"] = _arm_from_bags(
            data=data,
            bags=_project_bags(primary_expr["output_bags"], weight, bias),
            control_bag=_project_matrix(primary_expr["output_control"], weight, bias),
            arm="state_output_scvi128_gmm_ridge",
            feature_set="state_output_scvi128",
            embedding_space="fold_local_scvi_latent",
        )
        if external_data is not None and external_expr is not None:
            external["state_output_scvi128_gmm_ridge"] = _arm_from_bags(
                data=external_data,
                bags=_project_bags(external_expr["output_bags"], weight, bias),
                control_bag=_project_matrix(
                    external_expr["output_control"],
                    weight,
                    bias,
                ),
                arm="state_output_scvi128_gmm_ridge",
                feature_set="state_output_scvi128",
                embedding_space="fold_local_scvi_latent",
            )
    return primary, external


def _state_expression_arms(
    data: GeneBags,
    adapter: StateForwardAdapter,
    perturbations: PerturbationVectorAdapter,
    panel: tuple[torch.Tensor, torch.Tensor | None],
    *,
    progress_desc: str,
) -> dict[str, object]:
    control_cells, batch_indices = panel
    non_targeting = perturbations("non-targeting")
    output_control = state_output_bag(
        adapter,
        control_cells,
        non_targeting,
        "non-targeting",
        batch_indices=batch_indices,
    )
    hidden_control = token_hidden_bag(
        adapter,
        control_cells,
        non_targeting,
        "non-targeting",
        batch_indices=batch_indices,
    )
    output_bags = []
    hidden_bags = []
    with torch.no_grad():
        for gene in tqdm(
            data.genes.astype(str),
            desc=progress_desc,
            total=len(data.genes),
            miniters=max(1, len(data.genes) // 10),
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            vector = perturbations(str(gene))
            output_bags.append(
                state_output_bag(
                    adapter,
                    control_cells,
                    vector,
                    str(gene),
                    batch_indices=batch_indices,
                )
            )
            hidden_bags.append(
                token_hidden_bag(
                    adapter,
                    control_cells,
                    vector,
                    str(gene),
                    batch_indices=batch_indices,
                )
            )
    return {
        "output_control": output_control,
        "hidden_control": hidden_control,
        "output_bags": tuple(output_bags),
        "hidden_bags": tuple(hidden_bags),
    }


def _control_panel(
    data: GeneBags,
    config: StateFeatureAblationConfig,
    fold_seed: int,
    device: torch.device,
    batch_lookup: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    rng = np.random.default_rng(config.seed + int(fold_seed))
    n_rows = min(config.max_control_cells_per_gene, data.control_input.shape[0])
    indices = np.sort(sample_indices(data.control_input.shape[0], n_rows, rng))
    cells = torch.as_tensor(
        data.control_input[indices],
        dtype=torch.float32,
        device=device,
    )
    batch = None
    if data.control_batch is not None:
        encoded = encode_batch_labels(data.control_batch[indices], batch_lookup)
        batch = torch.as_tensor(encoded, dtype=torch.long, device=device)
    return cells, batch


def _fit_projector(
    data: GeneBags,
    split: GeneSplit,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_expr = np.vstack(
        [data.control_input, *(data.input_bags[i] for i in split.train)]
    )
    train_latent = np.vstack(
        [data.control_latent, *(data.latent_bags[i] for i in split.train)]
    )
    return fit_linear_projector(train_expr, train_latent, ridge_alpha)


def _project_bags(
    bags: tuple[np.ndarray, ...],
    weight: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, ...]:
    return tuple(_project_matrix(bag, weight, bias) for bag in bags)


def _project_matrix(
    matrix: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    return (np.asarray(matrix, dtype=np.float32) @ weight + bias).astype(np.float32)


def _arm_from_bags(
    *,
    data: GeneBags,
    bags: tuple[np.ndarray, ...],
    control_bag: np.ndarray,
    arm: str,
    feature_set: str,
    embedding_space: str,
) -> FeatureArmData:
    return FeatureArmData(
        feature_set=feature_set,
        arm=arm,
        genes=data.genes.astype(str),
        y=np.asarray(data.y, dtype=np.float64),
        bags=tuple(np.asarray(bag, dtype=np.float32) for bag in bags),
        control_bag=np.asarray(control_bag, dtype=np.float32),
        embedding_space=embedding_space,
    )


def _external_predictions_for_fit(
    fit: FoldFit,
    data: FeatureArmData,
    *,
    external_name: str,
    weighting: str = "unweighted",
) -> pd.DataFrame:
    y_pred = fit.ridge.predict(
        gmm_feature_matrix(
            fit.gmm,
            data.bags,
            np.arange(len(data.genes), dtype=np.int64),
            data.control_bag,
            fit.view,
        )
    )
    return pd.DataFrame(
        {
            "job_key": [
                f"external:{external_name}__fold{fit.metric_row['fold']}__"
                f"{data.feature_set}__{fit.metric_row['model']}"
            ]
            * len(data.genes),
            "evaluation_scope": f"external:{external_name}",
            "fold": int(fit.metric_row["fold"]),
            "feature_set": data.feature_set,
            "arm": data.arm,
            "model": fit.metric_row["model"],
            "weighting": weighting,
            "perturbation_gene": data.genes.astype(str),
            "y_true": data.y,
            "y_pred": y_pred,
            "primary_scope": PRIMARY_EXTERNAL_SCOPE,
            "secondary_scope": EXTERNAL_ENSEMBLE_SCOPE,
        }
    )


def _split_rows(
    data: GeneBags,
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for split_name, indices in (("train", train_idx), ("test", test_idx)):
        for index in indices:
            rows.append(
                {
                    "evaluation_scope": INTERNAL_SCOPE,
                    "fold": int(fold),
                    "split": split_name,
                    "perturbation_gene": str(data.genes[index]),
                }
            )
    return rows


def _stratified_folds(
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=np.float64)
    bins = min(int(n_splits), max(2, int(np.unique(y).shape[0])))
    try:
        labels = pd.qcut(y, q=bins, labels=False, duplicates="drop")
        strata = np.asarray(labels, dtype=np.int64)
    except ValueError:
        strata = np.zeros(len(y), dtype=np.int64)
    counts = np.bincount(strata)
    if counts.size == 0 or counts.min(initial=0) < int(n_splits):
        strata = np.zeros(len(y), dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=int(n_splits),
        shuffle=True,
        random_state=int(seed),
    )
    return [
        (train.astype(np.int64), test.astype(np.int64))
        for train, test in splitter.split(np.arange(len(y)), strata)
    ]


def _needs_scvi(config: StateFeatureAblationConfig) -> bool:
    return any("scvi" in arm for arm in config.arms)


def _infer_pert_dim(known_vectors: dict[str, np.ndarray]) -> int:
    if not known_vectors:
        msg = "state.pert_dim is required when no perturbation vectors are loaded"
        raise ValueError(msg)
    return int(next(iter(known_vectors.values())).shape[0])


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_from_config(path: Path) -> Path:
    """Run a frozen STATE feature-ablation from a YAML config."""
    return run_ablation_from_config(path)


def main(argv: list[str] | None = None) -> int:
    """CLI for running the frozen STATE feature ablation."""
    parser = argparse.ArgumentParser(description="Run a frozen STATE feature ablation.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    run_from_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
