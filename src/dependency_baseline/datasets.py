"""Feature loading, split, and dataset helpers for dependency experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from dependency_baseline.artifacts import job_key
from dependency_baseline.config import BaselineConfig, SelectionConfig
from dependency_baseline.models import (
    ModelSpec,
    compatible_model_feature_shape,
    filter_models,
)

PREDICTION_META_COLUMNS = ["perturbation_gene", "observed_n_cells", "target_gene_index"]


@dataclass(frozen=True)
class ExternalEvaluationData:
    name: str
    feature_sets: dict[str, np.ndarray]
    y: np.ndarray
    genes: np.ndarray
    metadata: pd.DataFrame


def load_feature_arrays(feature_path: Path) -> dict[str, np.ndarray]:
    """Load dependency baseline feature arrays from NPZ."""
    feature_data = np.load(feature_path, allow_pickle=True)
    arrays = {
        "delta": feature_data["delta"].astype(np.float32),
        "response_burden": feature_data["response_burden"].astype(np.float32),
        "program_scores": feature_data["program_scores"].astype(np.float32),
        "y": feature_data["y"].astype(np.float64),
        "n_cells": feature_data["n_cells"].astype(np.float64),
        "target_gene_index": feature_data["target_gene_index"].astype(np.int64),
        "genes": feature_data["perturbation_gene"].astype(str),
        "program_score_columns": feature_data["program_score_columns"].astype(str),
    }
    if "nar_viability_scores" in feature_data:
        arrays["nar_viability_scores"] = feature_data["nar_viability_scores"].astype(
            np.float32
        )
        arrays["nar_viability_score_columns"] = feature_data[
            "nar_viability_score_columns"
        ].astype(str)
    return arrays


def feature_sets(
    delta: np.ndarray,
    burden: np.ndarray,
    program_scores: np.ndarray,
    n_cells: np.ndarray,
    target_indices: np.ndarray,
    nar_viability_scores: np.ndarray | None = None,
    nar_viability_score_columns: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build named feature matrices used by the model ladder."""
    target_delta = np.full((delta.shape[0], 1), np.nan, dtype=np.float32)
    valid = target_indices >= 0
    target_delta[valid, 0] = delta[np.where(valid)[0], target_indices[valid]]

    delta_masked = delta.copy()
    delta_masked[valid, target_indices[valid]] = 0.0
    sets = {
        "delta_all": delta,
        "delta_mask_target": delta_masked,
        "response_burden": burden,
        "program_scores": program_scores,
        "program_scores_plus_burden": np.hstack([program_scores, burden]).astype(
            np.float32
        ),
        "target_knockdown_only": target_delta,
        "n_cells_only": n_cells.reshape(-1, 1).astype(np.float32),
    }
    if nar_viability_scores is not None:
        residual_scores = _residualizer_score_columns(
            nar_viability_scores,
            nar_viability_score_columns,
        )
        nuisance_scores = np.hstack([residual_scores, burden]).astype(np.float32)
        sets["nar_viability_scores"] = nar_viability_scores
        sets["nar_viability_scores_plus_burden"] = np.hstack(
            [nar_viability_scores, burden]
        ).astype(np.float32)
        sets["nar_resid_delta_all"] = np.hstack([delta, residual_scores]).astype(
            np.float32
        )
        sets["nar_resid_delta_mask_target"] = np.hstack(
            [delta_masked, residual_scores]
        ).astype(np.float32)
        sets["nuisance_scores"] = nuisance_scores
        sets["nuisance_resid_delta_all"] = np.hstack([delta, nuisance_scores]).astype(
            np.float32
        )
        sets["nuisance_resid_delta_mask_target"] = np.hstack(
            [delta_masked, nuisance_scores]
        ).astype(np.float32)
    return sets


def selected_scopes(
    y: np.ndarray,
    target_indices: np.ndarray,
    feature_names: tuple[str, ...],
    selection: SelectionConfig,
) -> list[tuple[str, np.ndarray, tuple[str, ...]]]:
    """Build internal evaluation scopes after optional selection filtering."""
    scopes = [
        ("internal_cv_all", np.arange(len(y), dtype=np.int64), feature_names),
        (
            "internal_cv_target_index_valid",
            np.flatnonzero(target_indices >= 0),
            ("delta_all", "delta_mask_target"),
        ),
    ]
    if selection.scopes is None:
        return scopes
    allowed = set(selection.scopes)
    return [scope for scope in scopes if scope[0] in allowed]


def split_manifest(
    config: BaselineConfig,
    scopes: list[tuple[str, np.ndarray, tuple[str, ...]]],
    y: np.ndarray,
    genes: np.ndarray,
) -> pd.DataFrame:
    """Create train/test split manifest for selected CV scopes."""
    rows: list[dict[str, object]] = []
    for evaluation_scope, row_indices, _allowed_features in scopes:
        splitter = repeated_stratified_splitter(config)
        y_bins = stratification_bins(y[row_indices], config.cv.stratify_bins)
        for fold_index, (train_local, test_local) in enumerate(
            splitter.split(row_indices, y_bins)
        ):
            train_genes = genes[row_indices[train_local]]
            test_genes = genes[row_indices[test_local]]
            rows.extend(_split_rows(evaluation_scope, fold_index, train_genes, "train"))
            rows.extend(_split_rows(evaluation_scope, fold_index, test_genes, "test"))
    return pd.DataFrame(rows)


def repeated_stratified_splitter(config: BaselineConfig) -> RepeatedStratifiedKFold:
    """Create the configured repeated stratified splitter."""
    return RepeatedStratifiedKFold(
        n_splits=config.cv.n_splits,
        n_repeats=config.cv.n_repeats,
        random_state=config.cv.random_state,
    )


def load_external_evaluations(
    config: BaselineConfig,
) -> tuple[ExternalEvaluationData, ...]:
    """Load configured external feature packs for sanity evaluation."""
    datasets: list[ExternalEvaluationData] = []
    for external in config.data.external_evaluations:
        feature_data = np.load(external.features_npz, allow_pickle=True)
        delta = feature_data["delta"].astype(np.float32)
        burden = feature_data["response_burden"].astype(np.float32)
        program_scores = feature_data["program_scores"].astype(np.float32)
        n_cells = feature_data["n_cells"].astype(np.float64)
        target_indices = feature_data["target_gene_index"].astype(np.int64)
        nar_scores = (
            feature_data["nar_viability_scores"].astype(np.float32)
            if "nar_viability_scores" in feature_data
            else None
        )
        nar_score_columns = (
            feature_data["nar_viability_score_columns"].astype(str)
            if "nar_viability_score_columns" in feature_data
            else None
        )
        datasets.append(
            ExternalEvaluationData(
                name=external.name,
                feature_sets=feature_sets(
                    delta,
                    burden,
                    program_scores,
                    n_cells,
                    target_indices,
                    nar_scores,
                    nar_score_columns,
                ),
                y=feature_data["y"].astype(np.float64),
                genes=feature_data["perturbation_gene"].astype(str),
                metadata=_external_metadata_frame(feature_data),
            )
        )
    return tuple(datasets)


def _external_metadata_frame(feature_data: np.lib.npyio.NpzFile) -> pd.DataFrame:
    metadata = pd.DataFrame(
        {
            "perturbation_gene": feature_data["perturbation_gene"].astype(str),
        }
    )
    if "source_dataset" in feature_data:
        metadata["source_dataset"] = feature_data["source_dataset"].astype(str)
    if "external_row_count" in feature_data:
        metadata["external_row_count"] = feature_data["external_row_count"].astype(int)
    if "n_cells" in feature_data:
        metadata["external_n_cells"] = feature_data["n_cells"].astype(float)
    return metadata


def _residualizer_score_columns(
    nar_viability_scores: np.ndarray,
    nar_viability_score_columns: np.ndarray | None,
) -> np.ndarray:
    if nar_viability_score_columns is None:
        return nar_viability_scores
    keep = nar_viability_score_columns != "nar_mean_score"
    if not keep.any():
        return nar_viability_scores
    return nar_viability_scores[:, keep]


def count_internal_fit_steps(
    *,
    config: BaselineConfig,
    evaluation_scopes: list[tuple[str, np.ndarray, tuple[str, ...]]],
    all_feature_sets: dict[str, np.ndarray],
    model_specs: list[ModelSpec],
    selection: SelectionConfig,
    completed_jobs: set[str],
) -> int:
    """Count pending internal CV fit steps for progress logging."""
    steps = 0
    selected_models = filter_models(model_specs, selection.models)
    selected_weightings = filter_names(
        ("unweighted", "sqrt_n_cells"),
        selection.weightings,
    )
    for scope, row_indices, allowed_features in evaluation_scopes:
        if row_indices.size < config.cv.n_splits:
            continue
        fold_train_rows = row_indices.size - math.ceil(
            row_indices.size / config.cv.n_splits
        )
        for fold_index in range(config.cv.n_splits * config.cv.n_repeats):
            if selection.folds is not None and fold_index not in selection.folds:
                continue
            steps += _count_scope_fold_steps(
                scope,
                fold_index,
                fold_train_rows,
                allowed_features,
                all_feature_sets,
                selected_models,
                selected_weightings,
                selection,
                completed_jobs,
            )
    return steps


def filter_names(
    names: tuple[str, ...],
    selected: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Apply optional name selection while preserving configured order."""
    if selected is None:
        return names
    selected_set = set(selected)
    return tuple(name for name in names if name in selected_set)


def merge_selection(
    config_selection: SelectionConfig,
    override: SelectionConfig | None,
) -> SelectionConfig:
    """Merge CLI selection overrides over config defaults."""
    if override is None:
        return config_selection
    return SelectionConfig(
        scopes=override.scopes or config_selection.scopes,
        features=override.features or config_selection.features,
        models=override.models or config_selection.models,
        folds=override.folds or config_selection.folds,
        weightings=override.weightings or config_selection.weightings,
    )


def stratification_bins(y: np.ndarray, requested_bins: int) -> np.ndarray:
    """Bin continuous labels for repeated stratified CV."""
    bins = min(requested_bins, max(2, len(y) // 10))
    while bins >= 2:
        try:
            labels = pd.qcut(y, q=bins, labels=False, duplicates="drop")
            values = np.asarray(labels, dtype=np.int64)
            if len(np.unique(values)) >= 2:
                return values
        except ValueError:
            bins -= 1
    return np.zeros_like(y, dtype=np.int64)


def _split_rows(
    evaluation_scope: str,
    fold_index: int,
    genes: np.ndarray,
    split: str,
) -> list[dict[str, object]]:
    return [
        {
            "evaluation_scope": evaluation_scope,
            "fold": fold_index,
            "perturbation_gene": gene,
            "split": split,
        }
        for gene in genes
    ]


def _count_scope_fold_steps(
    scope: str,
    fold_index: int,
    fold_train_rows: int,
    allowed_features: tuple[str, ...],
    all_feature_sets: dict[str, np.ndarray],
    selected_models: list[ModelSpec],
    selected_weightings: tuple[str, ...],
    selection: SelectionConfig,
    completed_jobs: set[str],
) -> int:
    steps = 0
    for feature_name in filter_names(allowed_features, selection.features):
        x_train_shape = (fold_train_rows, all_feature_sets[feature_name].shape[1])
        for spec in selected_models:
            if not compatible_model_feature_shape(
                spec.name,
                feature_name,
                x_train_shape,
            ):
                continue
            steps += _count_weighting_steps(
                scope,
                fold_index,
                feature_name,
                spec,
                selected_weightings,
                completed_jobs,
            )
    return steps


def _count_weighting_steps(
    scope: str,
    fold_index: int,
    feature_name: str,
    spec: ModelSpec,
    selected_weightings: tuple[str, ...],
    completed_jobs: set[str],
) -> int:
    steps = 0
    for weighting in selected_weightings:
        if weighting == "sqrt_n_cells" and not spec.supports_weight:
            continue
        job = job_key(scope, fold_index, feature_name, spec.name, weighting)
        if job not in completed_jobs:
            steps += 1
    return steps
