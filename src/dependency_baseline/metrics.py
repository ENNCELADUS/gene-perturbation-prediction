"""Metrics and ranking helpers for dependency prediction experiments."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

SUMMARY_GROUP_COLUMNS = ["evaluation_scope", "feature_set", "model", "weighting"]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics with constant-input guards."""
    spearman = _corr(spearmanr, y_true, y_pred)
    pearson = _corr(pearsonr, y_true, y_pred)
    return {
        "spearman": spearman,
        "spearman_defined": not np.isnan(spearman),
        "pearson": pearson,
        "pearson_defined": not np.isnan(pearson),
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
            labels,
            score,
            0.05,
        )
    return metrics


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold-level metrics by feature set and model."""
    metric_cols = [
        col
        for col in fold_metrics.columns
        if col
        not in {
            "job_key",
            "evaluation_scope",
            "fold",
            "feature_set",
            "model",
            "weighting",
            "checkpoint_path",
        }
        and pd.api.types.is_numeric_dtype(fold_metrics[col])
    ]
    rows = []
    grouped = fold_metrics.groupby(SUMMARY_GROUP_COLUMNS, dropna=False)
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
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["evaluation_scope", "spearman_mean", "pearson_mean"],
        ascending=[True, False, False],
        na_position="last",
    )


def summarize_rankings(topk_candidates: pd.DataFrame) -> pd.DataFrame:
    """Summarize top-k candidate lists by model group."""
    if topk_candidates.empty:
        return pd.DataFrame()
    grouped = topk_candidates.groupby(
        [*SUMMARY_GROUP_COLUMNS, "top_k"],
        dropna=False,
    )
    rows = []
    for key, group in grouped:
        evaluation_scope, feature_set, model, weighting, top_k = key
        rows.append(
            {
                "evaluation_scope": evaluation_scope,
                "feature_set": feature_set,
                "model": model,
                "weighting": weighting,
                "top_k": top_k,
                "n_rows": len(group),
                "mean_y_true": float(group["y_true"].mean()),
                "min_y_true": float(group["y_true"].min()),
                "mean_predicted_dependency_score": float(
                    group["predicted_dependency_score"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def rank_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rank predictions by predicted dependency strength within each job."""
    rankings = []
    for _key, group in predictions.groupby(
        [*SUMMARY_GROUP_COLUMNS, "fold"],
        dropna=False,
    ):
        ranked = group.copy()
        ranked["predicted_dependency_score"] = -ranked["y_pred"]
        ranked = ranked.sort_values(
            ["predicted_dependency_score", "perturbation_gene"],
            ascending=[False, True],
        )
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        rankings.append(ranked)
    if not rankings:
        return pd.DataFrame()
    columns = [
        "job_key",
        "evaluation_scope",
        "fold",
        "feature_set",
        "model",
        "weighting",
        "rank",
        "perturbation_gene",
        "y_true",
        "y_pred",
        "predicted_dependency_score",
    ]
    extra_cols = [col for col in predictions.columns if col not in columns]
    return pd.concat(rankings, ignore_index=True)[columns + extra_cols]


def topk_candidates(
    predictions: pd.DataFrame,
    topk_values: tuple[int, ...],
) -> pd.DataFrame:
    """Collect top-k ranked candidates for each model group."""
    rankings = rank_predictions(predictions)
    if rankings.empty:
        return rankings
    topk_rows = []
    for top_k in topk_values:
        top = rankings.loc[rankings["rank"] <= top_k].copy()
        top["top_k"] = top_k
        topk_rows.append(top)
    return pd.concat(topk_rows, ignore_index=True) if topk_rows else pd.DataFrame()


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
    top = np.argsort(score)[::-1][:k]
    observed = labels[top].mean()
    expected = labels.mean()
    if expected == 0:
        return np.nan
    return float(observed / expected)
