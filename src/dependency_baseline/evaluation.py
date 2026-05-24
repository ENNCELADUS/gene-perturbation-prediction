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
    read_feature_metadata,
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
    log_file: Path | None = None,
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
    actual_log_file, log_handler = _attach_run_log(context.run_dir, log_file)
    store = _artifact_store(config, context.run_dir)
    manifest = manifest_base(config, context, "run-cv", resume)
    write_json(
        context.run_dir / "run_manifest.json",
        {**manifest, "status": "running", "log_file": str(actual_log_file)},
    )
    write_cv_config(config, context.feature_path, context.run_dir / "cv_config.json")

    try:
        execute_cv(config, context.feature_path, store, selection, resume)
        write_json(
            context.run_dir / "run_manifest.json",
            {
                **manifest,
                "status": "completed",
                "ended_at": utc_now(),
                "log_file": str(actual_log_file),
            },
        )
    except Exception:
        write_json(
            context.run_dir / "run_manifest.json",
            {
                **manifest,
                "status": "failed",
                "ended_at": utc_now(),
                "log_file": str(actual_log_file),
            },
        )
        raise
    finally:
        _detach_run_log(log_handler)
    return _cv_paths(context.run_dir)


def fit_final(
    config: BaselineConfig,
    features_npz: Path | None = None,
    *,
    run_id: str | None = None,
    selection: SelectionConfig | None = None,
    command: tuple[str, ...] = (),
    config_path: Path | None = None,
    log_file: Path | None = None,
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
    actual_log_file, log_handler = _attach_run_log(context.run_dir, log_file)
    store = _artifact_store(config, context.run_dir)
    manifest = manifest_base(config, context, "fit-final", True)
    write_json(
        context.run_dir / "final_manifest.json",
        {**manifest, "status": "running", "log_file": str(actual_log_file)},
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
            {
                **manifest,
                "status": "completed",
                "ended_at": utc_now(),
                "log_file": str(actual_log_file),
            },
        )
    except Exception:
        write_json(
            context.run_dir / "final_manifest.json",
            {
                **manifest,
                "status": "failed",
                "ended_at": utc_now(),
                "log_file": str(actual_log_file),
            },
        )
        raise
    finally:
        _detach_run_log(log_handler)
    return FinalFitPaths(
        run_dir=context.run_dir,
        final_model_manifest_path=context.run_dir
        / "artifacts"
        / "final_model_manifest.parquet",
        final_rankings_path=context.run_dir / "artifacts" / "final_rankings.parquet",
        manifest_json=context.run_dir / "final_manifest.json",
        log_file=actual_log_file,
    )


def _artifact_store(config: BaselineConfig, run_dir: Path) -> ArtifactStore:
    return ArtifactStore(
        run_dir,
        config.experiment.human_result_tables,
        config.experiment.machine_result_format,
        config.experiment.topk_candidates,
        config.experiment.save_predictions,
        config.experiment.save_rankings,
    )


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


def execute_cv(
    config: BaselineConfig,
    feature_path: Path,
    store: ArtifactStore,
    selection: SelectionConfig | None,
    resume: bool,
) -> None:
    """Run all selected internal CV jobs and optional external evaluations."""
    feature_data = load_feature_arrays(feature_path)
    metadata = read_feature_metadata(config.data.output_dir)
    all_feature_sets = feature_sets(
        feature_data["delta"],
        feature_data["response_burden"],
        feature_data["program_scores"],
        feature_data["n_cells"],
        feature_data["target_gene_index"],
        feature_data.get("nar_viability_scores"),
        feature_data.get("nar_viability_score_columns"),
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
    _write_external_ensemble_results(config, store)


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
        feature_data["program_scores"],
        feature_data["n_cells"],
        feature_data["target_gene_index"],
        feature_data.get("nar_viability_scores"),
        feature_data.get("nar_viability_score_columns"),
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
        predictions = predictions.merge(
            external.metadata,
            on="perturbation_gene",
            how="left",
        )
        store.append_external_result(metric_row, predictions)


def _write_external_ensemble_results(
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
    selected = _selected_cv_models(store.tables["summary_metrics"])
    if selected:
        external = external.loc[external["model"].isin(selected)].copy()
    if external.empty:
        return

    train_lookup = _train_gene_lookup(store.tables["splits"])
    primary_predictions = _ensemble_predictions(
        external,
        train_lookup,
        target_heldout=False,
    )
    heldout_predictions = _ensemble_predictions(
        external,
        train_lookup,
        target_heldout=True,
    )
    ensemble_predictions = pd.concat(
        [primary_predictions, heldout_predictions],
        ignore_index=True,
    )
    metrics = _ensemble_metrics(config, ensemble_predictions)
    store.write_external_ensemble_results(metrics, ensemble_predictions)


def _selected_cv_models(summary: pd.DataFrame) -> set[str]:
    if summary.empty:
        return set()
    required = {
        "evaluation_scope",
        "feature_set",
        "model",
        "weighting",
        "spearman_mean",
    }
    if not required.issubset(summary.columns):
        return set()
    scoped = summary.loc[
        (summary["evaluation_scope"] == "internal_cv_all")
        & (summary["feature_set"] == "delta_all")
        & (summary["weighting"] == "unweighted")
    ].copy()
    if scoped.empty:
        return set()
    scoped["model_family"] = scoped["model"].map(_model_family)
    sort_columns = ["model_family", "spearman_mean"]
    ascending = [True, False]
    if "auroc_lt_neg1p0_mean" in scoped.columns:
        sort_columns.append("auroc_lt_neg1p0_mean")
        ascending.append(False)
    if "rmse_mean" in scoped.columns:
        sort_columns.append("rmse_mean")
        ascending.append(True)
    ranked = scoped.sort_values(sort_columns, ascending=ascending, na_position="last")
    return set(ranked.groupby("model_family", sort=False).head(1)["model"].astype(str))


def _model_family(model_name: str) -> str:
    if model_name == "ridge" or model_name.startswith("ridge_alpha"):
        return "ridge"
    if "random_forest" in model_name and model_name.startswith("pca"):
        return "pca_random_forest"
    if "_ridge" in model_name and model_name.startswith("pca"):
        return "pca_ridge"
    if model_name.startswith("xgboost"):
        return "xgboost"
    return model_name


def _train_gene_lookup(splits: pd.DataFrame) -> dict[int, set[str]]:
    if splits.empty:
        return {}
    train = splits.loc[
        (splits["evaluation_scope"] == "internal_cv_all") & (splits["split"] == "train")
    ]
    return {
        int(fold): set(group["perturbation_gene"].astype(str))
        for fold, group in train.groupby("fold")
    }


def _ensemble_predictions(
    predictions: pd.DataFrame,
    train_lookup: dict[int, set[str]],
    *,
    target_heldout: bool,
) -> pd.DataFrame:
    rows = []
    for key, group in predictions.groupby(
        ["evaluation_scope", "feature_set", "model", "weighting", "perturbation_gene"],
        dropna=False,
    ):
        evaluation_scope, feature_set, model, weighting, gene = key
        eligible = group
        if target_heldout:
            keep = [
                str(gene) not in train_lookup.get(int(row.fold), set())
                for row in group.itertuples(index=False)
            ]
            eligible = group.loc[keep]
            ensemble_scope = str(evaluation_scope).replace(
                "external:",
                "external_ensemble_target_heldout:",
                1,
            )
        else:
            ensemble_scope = str(evaluation_scope).replace(
                "external:",
                "external_ensemble:",
                1,
            )
        if eligible.empty:
            continue
        row = {
            "job_key": "__".join(
                [
                    ensemble_scope,
                    str(feature_set),
                    str(model),
                    str(weighting),
                    str(gene),
                ]
            ),
            "evaluation_scope": ensemble_scope,
            "fold": -1,
            "feature_set": feature_set,
            "model": model,
            "model_family": _model_family(str(model)),
            "weighting": weighting,
            "perturbation_gene": gene,
            "y_true": float(eligible["y_true"].mean()),
            "y_pred": float(eligible["y_pred"].mean()),
            "ensemble_size": int(len(eligible)),
        }
        for column in ("source_dataset", "external_row_count", "external_n_cells"):
            if column in eligible.columns:
                values = eligible[column].dropna().astype(str).unique()
                row[column] = ";".join(sorted(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _ensemble_metrics(
    config: BaselineConfig,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    grouped = predictions.groupby(
        ["evaluation_scope", "feature_set", "model", "model_family", "weighting"],
        dropna=False,
    )
    for key, group in grouped:
        evaluation_scope, feature_set, model, model_family, weighting = key
        row = {
            "evaluation_scope": evaluation_scope,
            "feature_set": feature_set,
            "model": model,
            "model_family": model_family,
            "weighting": weighting,
            "n_genes": int(group["perturbation_gene"].nunique()),
            "mean_ensemble_size": float(group["ensemble_size"].mean()),
            "min_ensemble_size": int(group["ensemble_size"].min()),
            "max_ensemble_size": int(group["ensemble_size"].max()),
            **regression_metrics(
                group["y_true"].to_numpy(),
                group["y_pred"].to_numpy(),
            ),
            **ranking_metrics(
                group["y_true"].to_numpy(),
                group["y_pred"].to_numpy(),
                config.cv.essential_thresholds,
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["evaluation_scope", "spearman", "pearson"],
        ascending=[True, False, False],
        na_position="last",
    )


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
