"""Cross-validation evaluation for dependency baselines."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vcc_dependency_baseline.config import BaselineConfig


@dataclass(frozen=True)
class CvPaths:
    fold_metrics_csv: Path
    summary_csv: Path
    predictions_csv: Path
    config_json: Path


@dataclass(frozen=True)
class ExternalEvaluationData:
    name: str
    feature_sets: dict[str, np.ndarray]
    y: np.ndarray
    genes: np.ndarray


def run_cv(config: BaselineConfig, features_npz: Path | None = None) -> CvPaths:
    """Run repeated stratified cross-validation on built features."""
    output_dir = config.data.output_dir / "cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = (
        features_npz or config.data.output_dir / "replogle_k562_delta_features.npz"
    )
    feature_data = np.load(feature_path, allow_pickle=True)
    metadata = pd.read_csv(
        config.data.output_dir / "replogle_k562_feature_metadata.csv"
    )

    delta = feature_data["delta"].astype(np.float32)
    burden = feature_data["response_burden"].astype(np.float32)
    y = feature_data["y"].astype(np.float64)
    n_cells = feature_data["n_cells"].astype(np.float64)
    target_indices = feature_data["target_gene_index"].astype(np.int64)
    genes = feature_data["perturbation_gene"].astype(str)

    feature_sets = _feature_sets(delta, burden, n_cells, target_indices)
    model_specs = _model_specs(config)
    external_evaluations = _load_external_evaluations(config)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    evaluation_scopes = [
        (
            "internal_cv_all",
            np.arange(len(y), dtype=np.int64),
            tuple(feature_sets.keys()),
        ),
        (
            "internal_cv_target_index_valid",
            np.flatnonzero(target_indices >= 0),
            ("delta_all", "delta_mask_target"),
        ),
    ]
    for evaluation_scope, row_indices, allowed_features in evaluation_scopes:
        scope_metrics, scope_predictions = _run_internal_cv_scope(
            config=config,
            evaluation_scope=evaluation_scope,
            row_indices=row_indices,
            allowed_features=allowed_features,
            feature_sets=feature_sets,
            model_specs=model_specs,
            external_evaluations=external_evaluations,
            y=y,
            n_cells=n_cells,
            genes=genes,
        )
        metric_rows.extend(scope_metrics)
        prediction_rows.extend(scope_predictions)

    fold_metrics = pd.DataFrame(metric_rows)
    summary = summarize_metrics(fold_metrics)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metadata_cols = ["perturbation_gene", "observed_n_cells", "target_gene_index"]
    predictions = predictions.merge(
        metadata[metadata_cols], on="perturbation_gene", how="left"
    )

    fold_metrics_csv = output_dir / "fold_metrics.csv"
    summary_csv = output_dir / "summary_metrics.csv"
    predictions_csv = output_dir / "predictions.csv"
    config_json = output_dir / "cv_config.json"
    fold_metrics.to_csv(fold_metrics_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    predictions.to_csv(predictions_csv, index=False)
    config_json.write_text(
        json.dumps(
            {
                "n_splits": config.cv.n_splits,
                "n_repeats": config.cv.n_repeats,
                "random_state": config.cv.random_state,
                "stratify_bins": config.cv.stratify_bins,
                "model_set": config.cv.model_set,
                "features_npz": str(feature_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return CvPaths(fold_metrics_csv, summary_csv, predictions_csv, config_json)


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold-level metrics by feature set and model."""
    metric_cols = [
        col
        for col in fold_metrics.columns
        if col not in {"evaluation_scope", "fold", "feature_set", "model", "weighting"}
        and pd.api.types.is_numeric_dtype(fold_metrics[col])
    ]
    rows = []
    grouped = fold_metrics.groupby(
        ["evaluation_scope", "feature_set", "model", "weighting"],
        dropna=False,
    )
    for (evaluation_scope, feature_set, model, weighting), group in grouped:
        row: dict[str, object] = {
            "evaluation_scope": evaluation_scope,
            "feature_set": feature_set,
            "model": model,
            "weighting": weighting,
            "n_folds": len(group),
        }
        for metric in metric_cols:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = values.mean() if not values.empty else np.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["evaluation_scope", "spearman_mean", "pearson_mean"],
        ascending=[True, False, False],
        na_position="last",
    )


def _run_internal_cv_scope(
    *,
    config: BaselineConfig,
    evaluation_scope: str,
    row_indices: np.ndarray,
    allowed_features: tuple[str, ...],
    feature_sets: dict[str, np.ndarray],
    model_specs: list[tuple[str, object, bool]],
    external_evaluations: tuple[ExternalEvaluationData, ...],
    y: np.ndarray,
    n_cells: np.ndarray,
    genes: np.ndarray,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    if row_indices.size < config.cv.n_splits:
        msg = (
            f"Evaluation scope {evaluation_scope!r} has {row_indices.size} rows, "
            f"fewer than n_splits={config.cv.n_splits}"
        )
        raise ValueError(msg)

    scope_y = y[row_indices]
    y_bins = _stratification_bins(scope_y, config.cv.stratify_bins)
    splitter = RepeatedStratifiedKFold(
        n_splits=config.cv.n_splits,
        n_repeats=config.cv.n_repeats,
        random_state=config.cv.random_state,
    )
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold_index, (train_local, test_local) in enumerate(
        splitter.split(row_indices, y_bins)
    ):
        train_idx = row_indices[train_local]
        test_idx = row_indices[test_local]
        sample_weight = _sample_weights(n_cells[train_idx])
        for feature_name in allowed_features:
            x = feature_sets[feature_name]
            x_train = x[train_idx]
            x_test = x[test_idx]
            for model_name, model, supports_weight in model_specs:
                if not _compatible_model_feature(model_name, feature_name, x_train):
                    continue
                weighting_modes = ["unweighted"]
                if supports_weight:
                    weighting_modes.append("sqrt_n_cells")
                for weighting in weighting_modes:
                    fit_params = {}
                    if weighting == "sqrt_n_cells":
                        fit_params = _fit_params_for_sample_weight(model, sample_weight)
                    fitted = clone(model)
                    fitted.fit(x_train, y[train_idx], **fit_params)
                    pred = fitted.predict(x_test)
                    metric_rows.append(
                        {
                            "evaluation_scope": evaluation_scope,
                            "fold": fold_index,
                            "feature_set": feature_name,
                            "model": model_name,
                            "weighting": weighting,
                            **regression_metrics(y[test_idx], pred),
                            **ranking_metrics(
                                y[test_idx],
                                pred,
                                config.cv.essential_thresholds,
                            ),
                        }
                    )
                    prediction_rows.append(
                        pd.DataFrame(
                            {
                                "evaluation_scope": evaluation_scope,
                                "fold": fold_index,
                                "feature_set": feature_name,
                                "model": model_name,
                                "weighting": weighting,
                                "perturbation_gene": genes[test_idx],
                                "y_true": y[test_idx],
                                "y_pred": pred,
                            }
                        )
                    )
                    if evaluation_scope == "internal_cv_all":
                        external_metrics, external_predictions = (
                            _evaluate_external_datasets(
                                config=config,
                                external_evaluations=external_evaluations,
                                fold_index=fold_index,
                                feature_name=feature_name,
                                model_name=model_name,
                                weighting=weighting,
                                fitted=fitted,
                            )
                        )
                        metric_rows.extend(external_metrics)
                        prediction_rows.extend(external_predictions)
    return metric_rows, prediction_rows


def _load_external_evaluations(
    config: BaselineConfig,
) -> tuple[ExternalEvaluationData, ...]:
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
                feature_sets=_feature_sets(delta, burden, n_cells, target_indices),
                y=feature_data["y"].astype(np.float64),
                genes=feature_data["perturbation_gene"].astype(str),
            )
        )
    return tuple(datasets)


def _evaluate_external_datasets(
    *,
    config: BaselineConfig,
    external_evaluations: tuple[ExternalEvaluationData, ...],
    fold_index: int,
    feature_name: str,
    model_name: str,
    weighting: str,
    fitted: object,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for external in external_evaluations:
        if feature_name not in external.feature_sets:
            continue
        pred = fitted.predict(external.feature_sets[feature_name])
        metric_rows.append(
            {
                "evaluation_scope": f"external:{external.name}",
                "fold": fold_index,
                "feature_set": feature_name,
                "model": model_name,
                "weighting": weighting,
                **regression_metrics(external.y, pred),
                **ranking_metrics(
                    external.y,
                    pred,
                    config.cv.essential_thresholds,
                ),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "evaluation_scope": f"external:{external.name}",
                    "fold": fold_index,
                    "feature_set": feature_name,
                    "model": model_name,
                    "weighting": weighting,
                    "perturbation_gene": external.genes,
                    "y_true": external.y,
                    "y_pred": pred,
                }
            )
        )
    return metric_rows, prediction_rows


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics with constant-input guards."""
    return {
        "spearman": _corr(spearmanr, y_true, y_pred),
        "pearson": _corr(pearsonr, y_true, y_pred),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[str, float]:
    """Compute binary essentiality diagnostics from continuous predictions."""
    metrics: dict[str, float] = {}
    # More negative GeneEffect means more essential; invert predictions for ranking.
    score = -y_pred
    for threshold in thresholds:
        labels = y_true < threshold
        suffix = str(threshold).replace("-", "neg").replace(".", "p")
        if labels.any() and (~labels).any():
            metrics[f"auroc_lt_{suffix}"] = float(roc_auc_score(labels, score))
            metrics[f"auprc_lt_{suffix}"] = float(
                average_precision_score(labels, score)
            )
        else:
            metrics[f"auroc_lt_{suffix}"] = np.nan
            metrics[f"auprc_lt_{suffix}"] = np.nan
        metrics[f"top5pct_enrichment_lt_{suffix}"] = _top_enrichment(
            labels, score, 0.05
        )
    return metrics


def _feature_sets(
    delta: np.ndarray,
    burden: np.ndarray,
    n_cells: np.ndarray,
    target_indices: np.ndarray,
) -> dict[str, np.ndarray]:
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


def _model_specs(config: BaselineConfig) -> list[tuple[str, object, bool]]:
    models: list[tuple[str, object, bool]] = [
        ("mean_label", DummyRegressor(strategy="mean"), False),
        (
            "ridge",
            make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
            ),
            True,
        ),
        (
            "elastic_net",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=5000, random_state=0),
            ),
            True,
        ),
    ]
    for n_components in config.cv.pca_components:
        models.append(
            (
                f"pca{n_components}_ridge",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    PCA(n_components=n_components, random_state=config.cv.random_state),
                    Ridge(alpha=10.0),
                ),
                True,
            )
        )
    if config.cv.model_set == "quick":
        return models[:2]

    models.append(
        (
            "random_forest",
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=5,
                    random_state=config.cv.random_state,
                    n_jobs=-1,
                ),
            ),
            True,
        )
    )
    try:
        from xgboost import XGBRegressor

        models.append(
            (
                "xgboost",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=config.cv.random_state,
                        n_jobs=4,
                    ),
                ),
                True,
            )
        )
    except ImportError:
        pass
    return models


def _compatible_model_feature(
    model_name: str, feature_name: str, x_train: np.ndarray
) -> bool:
    if not model_name.startswith("pca"):
        return True
    if feature_name not in {"delta_all", "delta_mask_target"}:
        return False
    n_components = int(model_name.removeprefix("pca").removesuffix("_ridge"))
    return n_components <= min(x_train.shape[0], x_train.shape[1])


def _fit_params_for_sample_weight(
    model: object, sample_weight: np.ndarray
) -> dict[str, np.ndarray]:
    if hasattr(model, "steps"):
        final_name = model.steps[-1][0]
        return {f"{final_name}__sample_weight": sample_weight}
    return {"sample_weight": sample_weight}


def _sample_weights(n_cells: np.ndarray) -> np.ndarray:
    weights = np.sqrt(np.maximum(n_cells, 1.0))
    return weights / np.mean(weights)


def _stratification_bins(y: np.ndarray, requested_bins: int) -> np.ndarray:
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


def _corr(func: object, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if _is_constant(y_true) or _is_constant(y_pred):
        return np.nan
    result = func(y_true, y_pred)
    statistic = result.statistic if hasattr(result, "statistic") else result[0]
    return float(statistic)


def _is_constant(values: np.ndarray) -> bool:
    if values.size <= 1:
        return True
    return bool(np.allclose(values, values[0]))


def _top_enrichment(labels: np.ndarray, score: np.ndarray, fraction: float) -> float:
    if not labels.any():
        return np.nan
    k = max(1, int(math.ceil(len(labels) * fraction)))
    top = np.argsort(score)[-k:]
    observed = labels[top].mean()
    expected = labels.mean()
    return float(observed / expected) if expected > 0 else np.nan
