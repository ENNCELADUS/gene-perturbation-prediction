"""Markdown reporting for NAR viability-axis audit runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from dependency_baseline.artifacts import read_named_table, read_table


def write_viability_axis_report(
    *,
    run_dir: Path,
    features_npz: Path,
    output_path: Path,
    baseline_predictions: Path | None = None,
) -> Path:
    """Write a compact markdown report for a viability-axis audit run."""
    summary = read_named_table(run_dir, "summary_metrics")
    feature_data = np.load(features_npz, allow_pickle=True)
    contrasts = _key_contrasts(summary)
    correlations = _feature_correlations(feature_data, baseline_predictions)

    lines = [
        "# Replogle K562 NAR Viability-Axis Audit",
        "",
        "## Key Contrasts",
        "",
        contrasts.to_markdown(index=False),
        "",
        "## Model Comparison",
        "",
        _model_comparison_table(summary),
        "",
        "## Feature Correlations",
        "",
        correlations.to_markdown(index=False),
        "",
        "## Reading Guide",
        "",
        "- If `nar_score_ridge` approaches the full transcriptome baseline, the "
        "B->C signal is strongly explained by the generic NAR viability axis.",
        "- If residualized PCA models retain most performance, the signal is not "
        "only the generic NAR cell-death/proliferation axis.",
        "- Compare Spearman/AUROC for ranking and RMSE/R2 for calibration.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _key_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    scoped = summary.loc[
        (summary["evaluation_scope"] == "internal_cv_all")
        & (summary["weighting"] == "unweighted")
    ].copy()
    if scoped.empty:
        return pd.DataFrame()

    score_only = scoped.loc[
        (scoped["feature_set"] == "nar_viability_scores")
        & (scoped["model"] == "nar_score_ridge")
    ]
    score_plus_burden = scoped.loc[
        (scoped["feature_set"] == "nar_viability_scores_plus_burden")
        & (scoped["model"] == "nar_score_plus_burden_ridge")
    ]
    baseline = scoped.loc[
        scoped["feature_set"].isin(["delta_all", "delta_mask_target"])
        & scoped["model"].isin(["pca50_ridge", "pca50_random_forest"])
    ]
    residualized = scoped.loc[
        scoped["feature_set"].isin(
            ["nar_resid_delta_all", "nar_resid_delta_mask_target"]
        )
        & scoped["model"].isin(
            ["nar_resid_pca50_ridge", "nar_resid_pca50_random_forest"]
        )
    ]
    nuisance_score = scoped.loc[
        (scoped["feature_set"] == "nuisance_scores")
        & (scoped["model"] == "nuisance_score_ridge")
    ]
    nuisance_residualized = scoped.loc[
        scoped["feature_set"].isin(
            ["nuisance_resid_delta_all", "nuisance_resid_delta_mask_target"]
        )
        & scoped["model"].str.startswith("nuisance_resid_pca")
    ]
    program_score = scoped.loc[
        scoped["feature_set"].isin(["program_scores", "program_scores_plus_burden"])
        & scoped["model"].str.startswith("program_score")
    ]

    rows = [
        _best_row(score_only, "NAR score only"),
        _best_row(score_plus_burden, "NAR score + burden"),
        _best_row(nuisance_score, "NAR + burden nuisance scores"),
        _best_row(program_score, "Program scores"),
        _best_row(baseline, "Best transcriptome baseline"),
        _best_row(residualized, "Best NAR-residualized transcriptome"),
        _best_row(nuisance_residualized, "Best NAR+burden-residualized transcriptome"),
    ]
    table = pd.DataFrame([row for row in rows if row])
    if table.empty:
        return table
    best_baseline = table.loc[
        table["contrast"].eq("Best transcriptome baseline"),
        "spearman",
    ]
    if not best_baseline.empty:
        table["spearman_delta_vs_best_baseline"] = table["spearman"] - float(
            best_baseline.iloc[0]
        )
    return table


def _best_row(rows: pd.DataFrame, label: str) -> dict[str, object]:
    if rows.empty:
        return {}
    rows = rows.sort_values("spearman_mean", ascending=False, na_position="last")
    row = rows.iloc[0]
    return {
        "contrast": label,
        "feature_set": row["feature_set"],
        "model": row["model"],
        "spearman": row.get("spearman_mean"),
        "auroc_ge_lt_-1": row.get("auroc_lt_neg1p0_mean"),
        "rmse": row.get("rmse_mean"),
    }


def _model_comparison_table(summary: pd.DataFrame) -> str:
    keys = [
        ("nar_viability_scores", "nar_score_ridge"),
        ("nar_viability_scores_plus_burden", "nar_score_plus_burden_ridge"),
        ("delta_all", "pca50_ridge"),
        ("delta_all", "pca50_random_forest"),
        ("nar_resid_delta_all", "nar_resid_pca50_ridge"),
        ("nar_resid_delta_all", "nar_resid_pca50_random_forest"),
        ("nuisance_scores", "nuisance_score_ridge"),
        ("nuisance_resid_delta_all", "nuisance_resid_pca50_ridge"),
        ("nuisance_resid_delta_all", "nuisance_resid_pca50_random_forest"),
        ("nuisance_resid_delta_all", "nuisance_resid_pca50_plus_scores_ridge"),
        (
            "nuisance_resid_delta_all",
            "nuisance_resid_pca50_plus_scores_random_forest",
        ),
        ("program_scores", "program_score_ridge"),
        ("program_scores", "program_score_elastic_net"),
        ("program_scores", "program_score_random_forest"),
        ("program_scores_plus_burden", "program_score_ridge"),
        ("program_scores_plus_burden", "program_score_elastic_net"),
        ("program_scores_plus_burden", "program_score_random_forest"),
        ("nar_resid_delta_mask_target", "nar_resid_pca50_ridge"),
        ("nar_resid_delta_mask_target", "nar_resid_pca50_random_forest"),
    ]
    rows = []
    for feature_set, model in keys:
        match = summary.loc[
            (summary["evaluation_scope"] == "internal_cv_all")
            & (summary["feature_set"] == feature_set)
            & (summary["model"] == model)
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "spearman": row.get("spearman_mean"),
                "pearson": row.get("pearson_mean"),
                "rmse": row.get("rmse_mean"),
                "r2": row.get("r2_mean"),
                "auroc_ge_lt_-1": row.get("auroc_lt_neg1p0_mean"),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def _feature_correlations(
    feature_data: np.lib.npyio.NpzFile,
    baseline_predictions: Path | None,
) -> pd.DataFrame:
    y = feature_data["y"].astype(float)
    values: dict[str, np.ndarray] = {"y_true": y}
    if "nar_viability_scores" in feature_data:
        score_columns = feature_data["nar_viability_score_columns"].astype(str)
        for index, column in enumerate(score_columns):
            values[column] = feature_data["nar_viability_scores"][:, index].astype(
                float
            )
    burden_columns = feature_data["response_burden_columns"].astype(str)
    burden = feature_data["response_burden"].astype(float)
    for column in ("delta_l2", "delta_l1_mean", "delta_top500_abs_mean"):
        matches = np.flatnonzero(burden_columns == column)
        if matches.size:
            values[f"response_burden_{column}"] = burden[:, matches[0]]
    if "program_scores" in feature_data:
        program_columns = feature_data["program_score_columns"].astype(str)
        program_scores = feature_data["program_scores"].astype(float)
        for index, column in enumerate(program_columns):
            values[column] = program_scores[:, index]
    if baseline_predictions is not None:
        baseline = _load_prediction_table(baseline_predictions)
        pred = _best_baseline_prediction_by_gene(baseline)
        genes = feature_data["perturbation_gene"].astype(str)
        aligned = pd.DataFrame({"perturbation_gene": genes}).merge(
            pred,
            on="perturbation_gene",
            how="left",
        )
        values["baseline_y_pred"] = aligned["baseline_y_pred"].to_numpy()

    rows = []
    names = list(values)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            mask = np.isfinite(values[left]) & np.isfinite(values[right])
            if mask.sum() < 3:
                continue
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "spearman": float(
                        spearmanr(
                            values[left][mask],
                            values[right][mask],
                        ).statistic
                    ),
                    "pearson": float(
                        pearsonr(
                            values[left][mask],
                            values[right][mask],
                        ).statistic
                    ),
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def _load_prediction_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return read_table(path)


def _best_baseline_prediction_by_gene(predictions: pd.DataFrame) -> pd.DataFrame:
    filtered = predictions.loc[
        (predictions["evaluation_scope"] == "internal_cv_all")
        & (predictions["feature_set"] == "delta_all")
        & (predictions["model"].isin(["pca50_random_forest", "pca50_ridge"]))
        & (predictions["weighting"] == "unweighted")
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["perturbation_gene", "baseline_y_pred"])
    if "pca50_random_forest" in set(filtered["model"]):
        filtered = filtered.loc[filtered["model"] == "pca50_random_forest"]
    grouped = filtered.groupby("perturbation_gene", as_index=False)["y_pred"].mean()
    return grouped.rename(columns={"y_pred": "baseline_y_pred"})
