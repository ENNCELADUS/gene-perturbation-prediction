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


def load_feature_arrays(feature_path: Path) -> dict[str, np.ndarray]:
    """Load dependency baseline feature arrays from NPZ."""
    feature_data = np.load(feature_path, allow_pickle=True)
    return {
        "delta": feature_data["delta"].astype(np.float32),
        "response_burden": feature_data["response_burden"].astype(np.float32),
        "y": feature_data["y"].astype(np.float64),
        "n_cells": feature_data["n_cells"].astype(np.float64),
        "target_gene_index": feature_data["target_gene_index"].astype(np.int64),
        "genes": feature_data["perturbation_gene"].astype(str),
    }


def feature_sets(
    delta: np.ndarray,
    burden: np.ndarray,
    n_cells: np.ndarray,
    target_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build named feature matrices used by the model ladder."""
    target_delta = np.full((delta.shape[0], 1), np.nan, dtype=np.float32)
    valid = target_indices >= 0
    target_delta[valid, 0] = delta[np.where(valid)[0], target_indices[valid]]

    delta_masked = delta.copy()
    delta_masked[valid, target_indices[valid]] = 0.0
    return {
        "delta_all": delta,
        "delta_mask_target": delta_masked,
        "response_burden": burden,
        "target_knockdown_only": target_delta,
        "n_cells_only": n_cells.reshape(-1, 1).astype(np.float32),
    }


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
        n_cells = feature_data["n_cells"].astype(np.float64)
        target_indices = feature_data["target_gene_index"].astype(np.int64)
        datasets.append(
            ExternalEvaluationData(
                name=external.name,
                feature_sets=feature_sets(delta, burden, n_cells, target_indices),
                y=feature_data["y"].astype(np.float64),
                genes=feature_data["perturbation_gene"].astype(str),
            )
        )
    return tuple(datasets)


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
