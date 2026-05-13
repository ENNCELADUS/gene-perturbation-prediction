"""Public experiment orchestration API for dependency baselines."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dependency_baseline.artifacts import (
    ArtifactStore,
    CvPaths,
    FinalFitPaths,
    checkpoint_path,
    create_run_context,
    job_key,
    manifest_base,
    set_seed,
    summarize_results as summarize_results,
    utc_now,
    write_cv_config,
    write_json,
)
from dependency_baseline.config import BaselineConfig, SelectionConfig
from dependency_baseline.datasets import (
    PREDICTION_META_COLUMNS,
    ExternalEvaluationData,
    count_internal_fit_steps,
    feature_sets,
    filter_names,
    load_external_evaluations,
    load_feature_arrays,
    merge_selection,
    repeated_stratified_splitter,
    selected_scopes,
    split_manifest,
    stratification_bins,
)
from dependency_baseline.metrics import (
    rank_predictions,
    ranking_metrics,
    regression_metrics as regression_metrics,
    summarize_metrics as summarize_metrics,
)
from dependency_baseline.models import (
    ModelSpec,
    build_model_specs,
    compatible_model_feature,
    filter_models,
    fit_estimator,
    sample_weights,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CvPaths",
    "FinalFitPaths",
    "fit_final",
    "regression_metrics",
    "run_cv",
    "summarize_metrics",
    "summarize_results",
]


def run_cv(
    config: BaselineConfig,
    features_npz: Path | None = None,
    *,
    run_id: str | None = None,
    resume: bool = False,
    selection: SelectionConfig | None = None,
    command: tuple[str, ...] = (),
    config_path: Path | None = None,
) -> CvPaths:
    """Run repeated stratified CV with incremental artifacts and checkpoints."""
    set_seed(config.experiment.seed)
    context = create_run_context(
        config=config,
        features_npz=features_npz,
        run_id=run_id,
        resume=resume,
        command=command,
        config_path=config_path,
    )
    store = _artifact_store(config, context.run_dir)
    manifest = manifest_base(config, context, "run-cv", resume)
    write_json(
        context.run_dir / "run_manifest.json",
        {**manifest, "status": "running"},
    )
    write_cv_config(config, context.feature_path, context.run_dir / "cv_config.json")

    try:
        execute_cv(config, context.feature_path, store, selection, resume)
        write_json(
            context.run_dir / "run_manifest.json",
            {**manifest, "status": "completed", "ended_at": utc_now()},
        )
    except Exception:
        write_json(
            context.run_dir / "run_manifest.json",
            {**manifest, "status": "failed", "ended_at": utc_now()},
        )
        raise
    return _cv_paths(context.run_dir)


def fit_final(
    config: BaselineConfig,
    features_npz: Path | None = None,
    *,
    run_id: str | None = None,
    selection: SelectionConfig | None = None,
    command: tuple[str, ...] = (),
    config_path: Path | None = None,
) -> FinalFitPaths:
    """Fit selected models on all numeric Replogle rows and save checkpoints."""
    set_seed(config.experiment.seed)
    context = create_run_context(
        config=config,
        features_npz=features_npz,
        run_id=run_id,
        resume=True,
        command=command,
        config_path=config_path,
    )
    store = _artifact_store(config, context.run_dir)
    manifest = manifest_base(config, context, "fit-final", True)
    write_json(
        context.run_dir / "final_manifest.json",
        {**manifest, "status": "running"},
    )

    try:
        execute_final_fit(
            config,
            context.feature_path,
            context.run_dir,
            store,
            selection,
        )
        write_json(
            context.run_dir / "final_manifest.json",
            {**manifest, "status": "completed", "ended_at": utc_now()},
        )
    except Exception:
        write_json(
            context.run_dir / "final_manifest.json",
            {**manifest, "status": "failed", "ended_at": utc_now()},
        )
        raise
    return FinalFitPaths(
        run_dir=context.run_dir,
        final_model_manifest_csv=context.run_dir / "final_model_manifest.csv",
        final_rankings_csv=context.run_dir / "final_rankings.csv",
        manifest_json=context.run_dir / "final_manifest.json",
    )


def _artifact_store(config: BaselineConfig, run_dir: Path) -> ArtifactStore:
    return ArtifactStore(
        run_dir,
        config.experiment.result_formats,
        config.experiment.topk_candidates,
        config.experiment.save_predictions,
        config.experiment.save_rankings,
    )


def _cv_paths(run_dir: Path) -> CvPaths:
    return CvPaths(
        run_dir=run_dir,
        fold_metrics_csv=run_dir / "fold_metrics.csv",
        summary_csv=run_dir / "summary_metrics.csv",
        predictions_csv=run_dir / "predictions.csv",
        config_json=run_dir / "cv_config.json",
        manifest_json=run_dir / "run_manifest.json",
        splits_csv=run_dir / "splits.csv",
        model_manifest_csv=run_dir / "model_manifest.csv",
        topk_candidates_csv=run_dir / "topk_candidates.csv",
    )


def execute_cv(
    config: BaselineConfig,
    feature_path: Path,
    store: ArtifactStore,
    selection: SelectionConfig | None,
    resume: bool,
) -> None:
    """Run all selected internal CV jobs and optional external evaluations."""
    feature_data = load_feature_arrays(feature_path)
    metadata = pd.read_csv(
        config.data.output_dir / "replogle_k562_feature_metadata.csv"
    )
    all_feature_sets = feature_sets(
        feature_data["delta"],
        feature_data["response_burden"],
        feature_data["n_cells"],
        feature_data["target_gene_index"],
    )
    merged_selection = merge_selection(config.selection, selection)
    scopes = selected_scopes(
        feature_data["y"],
        feature_data["target_gene_index"],
        tuple(all_feature_sets.keys()),
        merged_selection,
    )
    if config.experiment.save_splits:
        store.write_splits(
            split_manifest(config, scopes, feature_data["y"], feature_data["genes"])
        )
    model_specs = build_model_specs(config)
    progress = _progress_state(
        config,
        scopes,
        all_feature_sets,
        model_specs,
        merged_selection,
        store.completed_jobs if resume else set(),
    )
    LOGGER.info(
        "Starting CV run: n_splits=%s n_repeats=%s pending_fit_steps=%s",
        config.cv.n_splits,
        config.cv.n_repeats,
        progress["total"],
    )
    external_evaluations = load_external_evaluations(config)
    for evaluation_scope, row_indices, allowed_features in scopes:
        _run_internal_cv_scope(
            config=config,
            store=store,
            evaluation_scope=evaluation_scope,
            row_indices=row_indices,
            allowed_features=allowed_features,
            feature_sets=all_feature_sets,
            model_specs=model_specs,
            external_evaluations=external_evaluations,
            feature_data=feature_data,
            metadata=metadata,
            progress=progress,
            selection=merged_selection,
            resume=resume,
        )


def execute_final_fit(
    config: BaselineConfig,
    feature_path: Path,
    run_dir: Path,
    store: ArtifactStore,
    selection: SelectionConfig | None,
) -> None:
    """Fit selected final models over all numeric Replogle rows."""
    feature_data = load_feature_arrays(feature_path)
    all_feature_sets = feature_sets(
        feature_data["delta"],
        feature_data["response_burden"],
        feature_data["n_cells"],
        feature_data["target_gene_index"],
    )
    merged_selection = merge_selection(config.selection, selection)
    selected_features = filter_names(
        tuple(all_feature_sets.keys()),
        merged_selection.features,
    )
    selected_models = filter_models(build_model_specs(config), merged_selection.models)
    weightings = filter_names(
        ("unweighted", "sqrt_n_cells"),
        merged_selection.weightings,
    )
    weights = sample_weights(feature_data["n_cells"])
    for feature_name in selected_features:
        _fit_final_feature(
            config,
            run_dir,
            store,
            feature_data,
            all_feature_sets[feature_name],
            feature_name,
            selected_models,
            weightings,
            weights,
        )


def _run_internal_cv_scope(
    *,
    config: BaselineConfig,
    store: ArtifactStore,
    evaluation_scope: str,
    row_indices: np.ndarray,
    allowed_features: tuple[str, ...],
    feature_sets: dict[str, np.ndarray],
    model_specs: list[ModelSpec],
    external_evaluations: tuple[ExternalEvaluationData, ...],
    feature_data: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    progress: dict[str, int],
    selection: SelectionConfig,
    resume: bool,
) -> None:
    _validate_scope_size(evaluation_scope, row_indices, config.cv.n_splits)
    splitter = repeated_stratified_splitter(config)
    y_bins = stratification_bins(
        feature_data["y"][row_indices],
        config.cv.stratify_bins,
    )
    selected_features = filter_names(allowed_features, selection.features)
    selected_models = filter_models(model_specs, selection.models)
    selected_weightings = filter_names(
        ("unweighted", "sqrt_n_cells"),
        selection.weightings,
    )
    for fold_index, (train_local, test_local) in enumerate(
        splitter.split(row_indices, y_bins)
    ):
        if selection.folds is not None and fold_index not in selection.folds:
            continue
        _run_fold_models(
            config=config,
            store=store,
            evaluation_scope=evaluation_scope,
            fold_index=fold_index,
            train_idx=row_indices[train_local],
            test_idx=row_indices[test_local],
            selected_features=selected_features,
            selected_models=selected_models,
            selected_weightings=selected_weightings,
            feature_sets=feature_sets,
            external_evaluations=external_evaluations,
            feature_data=feature_data,
            metadata=metadata,
            progress=progress,
            resume=resume,
        )


def _run_fold_models(
    *,
    config: BaselineConfig,
    store: ArtifactStore,
    evaluation_scope: str,
    fold_index: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    selected_features: tuple[str, ...],
    selected_models: list[ModelSpec],
    selected_weightings: tuple[str, ...],
    feature_sets: dict[str, np.ndarray],
    external_evaluations: tuple[ExternalEvaluationData, ...],
    feature_data: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    progress: dict[str, int],
    resume: bool,
) -> None:
    LOGGER.info(
        "Starting scope=%s fold=%s/%s train_rows=%s test_rows=%s",
        evaluation_scope,
        fold_index + 1,
        config.cv.n_splits * config.cv.n_repeats,
        len(train_idx),
        len(test_idx),
    )
    weights = sample_weights(feature_data["n_cells"][train_idx])
    for feature_name in selected_features:
        x_train = feature_sets[feature_name][train_idx]
        x_test = feature_sets[feature_name][test_idx]
        for spec in selected_models:
            if compatible_model_feature(spec.name, feature_name, x_train):
                _run_model_weightings(
                    config=config,
                    store=store,
                    evaluation_scope=evaluation_scope,
                    fold_index=fold_index,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    feature_name=feature_name,
                    x_train=x_train,
                    x_test=x_test,
                    spec=spec,
                    selected_weightings=selected_weightings,
                    weights=weights,
                    external_evaluations=external_evaluations,
                    feature_data=feature_data,
                    metadata=metadata,
                    progress=progress,
                    resume=resume,
                )


def _run_model_weightings(
    *,
    config: BaselineConfig,
    store: ArtifactStore,
    evaluation_scope: str,
    fold_index: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    spec: ModelSpec,
    selected_weightings: tuple[str, ...],
    weights: np.ndarray,
    external_evaluations: tuple[ExternalEvaluationData, ...],
    feature_data: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    progress: dict[str, int],
    resume: bool,
) -> None:
    for weighting in selected_weightings:
        if weighting == "sqrt_n_cells" and not spec.supports_weight:
            continue
        job = job_key(evaluation_scope, fold_index, feature_name, spec.name, weighting)
        if resume and job in store.completed_jobs:
            LOGGER.info("Skipping completed job=%s", job)
            continue
        fitted, fit_seconds = fit_estimator(
            spec,
            x_train,
            feature_data["y"][train_idx],
            weights,
            weighting,
        )
        pred = fitted.predict(x_test)
        _persist_cv_fit(
            config=config,
            store=store,
            fitted=fitted,
            pred=pred,
            evaluation_scope=evaluation_scope,
            fold_index=fold_index,
            feature_name=feature_name,
            spec=spec,
            weighting=weighting,
            fit_seconds=fit_seconds,
            test_idx=test_idx,
            train_idx=train_idx,
            feature_data=feature_data,
            metadata=metadata,
            job=job,
        )
        progress["completed"] += 1
        _log_completed(
            progress,
            evaluation_scope,
            fold_index,
            feature_name,
            spec,
            weighting,
        )
        if evaluation_scope == "internal_cv_all":
            _evaluate_external_datasets(
                config=config,
                store=store,
                external_evaluations=external_evaluations,
                fold_index=fold_index,
                feature_name=feature_name,
                model_name=spec.name,
                weighting=weighting,
                fitted=fitted,
                fit_seconds=fit_seconds,
            )


def _persist_cv_fit(
    *,
    config: BaselineConfig,
    store: ArtifactStore,
    fitted: object,
    pred: np.ndarray,
    evaluation_scope: str,
    fold_index: int,
    feature_name: str,
    spec: ModelSpec,
    weighting: str,
    fit_seconds: float,
    test_idx: np.ndarray,
    train_idx: np.ndarray,
    feature_data: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    job: str,
) -> None:
    path = checkpoint_path(
        store.run_dir,
        evaluation_scope,
        fold_index,
        feature_name,
        spec.name,
        weighting,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, path)
    metric_row = _metric_row(
        config,
        job,
        evaluation_scope,
        fold_index,
        feature_name,
        spec.name,
        weighting,
        fit_seconds,
        feature_data["y"][test_idx],
        pred,
    )
    predictions = _prediction_frame(
        metric_row,
        feature_data["genes"][test_idx],
        feature_data["y"][test_idx],
        pred,
        metadata,
    )
    store.append_fold_result(
        metric_row,
        predictions,
        {
            **metric_row,
            "checkpoint_path": str(path),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        },
    )


def _evaluate_external_datasets(
    *,
    config: BaselineConfig,
    store: ArtifactStore,
    external_evaluations: tuple[ExternalEvaluationData, ...],
    fold_index: int,
    feature_name: str,
    model_name: str,
    weighting: str,
    fitted: object,
    fit_seconds: float,
) -> None:
    for external in external_evaluations:
        if feature_name not in external.feature_sets:
            continue
        pred = fitted.predict(external.feature_sets[feature_name])
        evaluation_scope = f"external:{external.name}"
        job = job_key(evaluation_scope, fold_index, feature_name, model_name, weighting)
        metric_row = _metric_row(
            config,
            job,
            evaluation_scope,
            fold_index,
            feature_name,
            model_name,
            weighting,
            fit_seconds,
            external.y,
            pred,
        )
        predictions = pd.DataFrame(
            {
                "job_key": job,
                "evaluation_scope": evaluation_scope,
                "fold": fold_index,
                "feature_set": feature_name,
                "model": model_name,
                "weighting": weighting,
                "perturbation_gene": external.genes,
                "y_true": external.y,
                "y_pred": pred,
            }
        )
        store.append_external_result(metric_row, predictions)


def _fit_final_feature(
    config: BaselineConfig,
    run_dir: Path,
    store: ArtifactStore,
    feature_data: dict[str, np.ndarray],
    x_train: np.ndarray,
    feature_name: str,
    selected_models: list[ModelSpec],
    weightings: tuple[str, ...],
    weights: np.ndarray,
) -> None:
    for spec in selected_models:
        if not compatible_model_feature(spec.name, feature_name, x_train):
            continue
        for weighting in weightings:
            if weighting == "sqrt_n_cells" and not spec.supports_weight:
                continue
            _fit_one_final_model(
                config,
                run_dir,
                store,
                feature_data,
                x_train,
                feature_name,
                spec,
                weighting,
                weights,
            )


def _fit_one_final_model(
    config: BaselineConfig,
    run_dir: Path,
    store: ArtifactStore,
    feature_data: dict[str, np.ndarray],
    x_train: np.ndarray,
    feature_name: str,
    spec: ModelSpec,
    weighting: str,
    weights: np.ndarray,
) -> None:
    job = job_key("final", -1, feature_name, spec.name, weighting)
    fitted, fit_seconds = fit_estimator(
        spec,
        x_train,
        feature_data["y"],
        weights,
        weighting,
    )
    pred = fitted.predict(x_train)
    path = checkpoint_path(run_dir, "final", -1, feature_name, spec.name, weighting)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, path)
    manifest_row = {
        "job_key": job,
        "feature_set": feature_name,
        "model": spec.name,
        "weighting": weighting,
        "fit_seconds": fit_seconds,
        "checkpoint_path": str(path),
        "n_train": len(feature_data["y"]),
        **regression_metrics(feature_data["y"], pred),
        **ranking_metrics(feature_data["y"], pred, config.cv.essential_thresholds),
    }
    ranking = rank_predictions(
        pd.DataFrame(
            {
                "job_key": job,
                "evaluation_scope": "final",
                "fold": -1,
                "feature_set": feature_name,
                "model": spec.name,
                "weighting": weighting,
                "perturbation_gene": feature_data["genes"],
                "y_true": feature_data["y"],
                "y_pred": pred,
            }
        )
    )
    store.append_final_result(manifest_row, ranking)
    LOGGER.info(
        "Completed final fit feature=%s model=%s weighting=%s",
        feature_name,
        spec.name,
        weighting,
    )


def _metric_row(
    config: BaselineConfig,
    job: str,
    evaluation_scope: str,
    fold_index: int,
    feature_name: str,
    model_name: str,
    weighting: str,
    fit_seconds: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    return {
        "job_key": job,
        "evaluation_scope": evaluation_scope,
        "fold": fold_index,
        "feature_set": feature_name,
        "model": model_name,
        "weighting": weighting,
        "fit_seconds": fit_seconds,
        **regression_metrics(y_true, y_pred),
        **ranking_metrics(y_true, y_pred, config.cv.essential_thresholds),
    }


def _prediction_frame(
    metric_row: dict[str, object],
    genes: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "job_key": metric_row["job_key"],
            "evaluation_scope": metric_row["evaluation_scope"],
            "fold": metric_row["fold"],
            "feature_set": metric_row["feature_set"],
            "model": metric_row["model"],
            "weighting": metric_row["weighting"],
            "perturbation_gene": genes,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    return predictions.merge(
        metadata[PREDICTION_META_COLUMNS],
        on="perturbation_gene",
        how="left",
    )


def _progress_state(
    config: BaselineConfig,
    scopes: list[tuple[str, np.ndarray, tuple[str, ...]]],
    all_feature_sets: dict[str, np.ndarray],
    model_specs: list[ModelSpec],
    selection: SelectionConfig,
    completed_jobs: set[str],
) -> dict[str, int]:
    return {
        "completed": 0,
        "total": count_internal_fit_steps(
            config=config,
            evaluation_scopes=scopes,
            all_feature_sets=all_feature_sets,
            model_specs=model_specs,
            selection=selection,
            completed_jobs=completed_jobs,
        ),
    }


def _validate_scope_size(
    evaluation_scope: str,
    row_indices: np.ndarray,
    n_splits: int,
) -> None:
    if row_indices.size >= n_splits:
        return
    msg = (
        f"Evaluation scope {evaluation_scope!r} has {row_indices.size} rows, "
        f"fewer than n_splits={n_splits}"
    )
    raise ValueError(msg)


def _log_completed(
    progress: dict[str, int],
    evaluation_scope: str,
    fold_index: int,
    feature_name: str,
    spec: ModelSpec,
    weighting: str,
) -> None:
    LOGGER.info(
        "Completed fit %s/%s scope=%s fold=%s feature=%s model=%s weighting=%s",
        progress["completed"],
        progress["total"],
        evaluation_scope,
        fold_index + 1,
        feature_name,
        spec.name,
        weighting,
    )
