"""Result-table helpers for frozen STATE feature ablations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from dependency_baseline.artifacts import write_formats
from dependency_baseline.metrics import ranking_metrics, regression_metrics


PRIMARY_EXTERNAL_SCOPE = "external_ensemble_target_heldout:adamson_k562"
EXTERNAL_ENSEMBLE_SCOPE = "external_ensemble:adamson_k562"
REQUIRED_QA_COLUMNS = (
    "fold",
    "feature_set",
    "arm",
    "embedding_space",
    "control_embedding_path",
    "n_train_genes",
    "n_test_genes",
    "gmm_components",
    "gmm_converged",
    "gmm_n_iter",
    "gmm_fit_source",
    "primary_view",
    "sensitivity_views",
)


def adamson_heldout_ensemble_predictions(
    external_predictions: pd.DataFrame,
    fold_membership: pd.DataFrame,
    *,
    external_name: str = "adamson_k562",
) -> pd.DataFrame:
    """Aggregate external fold predictions, excluding train-target overlaps."""
    primary = _ensemble_by_target(
        external_predictions,
        evaluation_scope=f"external_ensemble:{external_name}",
    )
    train_lookup = _train_gene_lookup(fold_membership)
    heldout_rows = []
    for row in external_predictions.itertuples(index=False):
        train_genes = train_lookup.get(int(row.fold), set())
        if str(row.perturbation_gene) not in train_genes:
            heldout_rows.append(row._asdict())
    heldout_source = pd.DataFrame(heldout_rows, columns=external_predictions.columns)
    heldout = _ensemble_by_target(
        heldout_source,
        evaluation_scope=f"external_ensemble_target_heldout:{external_name}",
    )
    return pd.concat([primary, heldout], ignore_index=True)


def metric_rows_for_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute metric rows for prediction tables with experiment scopes."""
    rows = []
    group_cols = ["evaluation_scope", "feature_set", "arm", "model", "weighting"]
    for key, group in predictions.groupby(group_cols, dropna=False):
        y_true = group["y_true"].to_numpy(dtype=np.float64)
        y_pred = group["y_pred"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "evaluation_scope": key[0],
                "feature_set": key[1],
                "arm": key[2],
                "model": key[3],
                "weighting": key[4],
                "n_predictions": int(len(group)),
                "primary_scope": PRIMARY_EXTERNAL_SCOPE,
                "secondary_scope": EXTERNAL_ENSEMBLE_SCOPE,
                **regression_metrics(y_true, y_pred),
                **ranking_metrics(y_true, y_pred, (-0.5, -1.0)),
            }
        )
    return pd.DataFrame(rows)


def feature_qa_row(
    data: object,
    *,
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    gmm: GaussianMixture,
    primary_view: str,
    sensitivity_views: tuple[str, ...],
) -> dict[str, object]:
    """Build required QA metadata for one fold/arm."""
    return {
        "fold": int(fold),
        "feature_set": data.feature_set,
        "arm": data.arm,
        "embedding_space": data.embedding_space,
        "control_embedding_path": "same_path_non_targeting"
        if data.embedding_space == "state_token_hidden"
        else "shared_control_panel",
        "n_train_genes": int(len(train_idx)),
        "n_test_genes": int(len(test_idx)),
        "gmm_components": int(gmm.n_components),
        "gmm_converged": bool(gmm.converged_),
        "gmm_n_iter": int(gmm.n_iter_),
        "gmm_fit_source": getattr(data, "gmm_fit_source", "B_hat_train_plus_controls"),
        "primary_view": primary_view,
        "sensitivity_views": ",".join(sensitivity_views),
    }


def gmm_metadata_row(
    data: object,
    *,
    fold: int,
    gmm: GaussianMixture,
) -> dict[str, object]:
    """Build GMM convergence metadata for one fold/arm."""
    return {
        "fold": int(fold),
        "feature_set": data.feature_set,
        "arm": data.arm,
        "gmm_components": int(gmm.n_components),
        "gmm_converged": bool(gmm.converged_),
        "gmm_n_iter": int(gmm.n_iter_),
        "gmm_lower_bound": float(gmm.lower_bound_),
        "fit_cell_count": int(getattr(gmm, "fit_cell_count_", 0)),
        "fit_gene_names": ",".join(getattr(gmm, "fit_gene_names_", ())),
    }


def required_result_columns() -> dict[str, tuple[str, ...]]:
    """Return result table columns required by the ablation contract."""
    common = (
        "evaluation_scope",
        "feature_set",
        "arm",
        "model",
        "weighting",
        "primary_scope",
        "secondary_scope",
    )
    return {
        "fold_metrics": (
            "job_key",
            "fold",
            "arm",
            "spearman",
            "spearman_defined",
            *common,
        ),
        "predictions": (
            "job_key",
            "fold",
            "arm",
            "perturbation_gene",
            "y_true",
            "y_pred",
            *common,
        ),
        "feature_qa": REQUIRED_QA_COLUMNS,
    }


def validate_result_tables(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_qa: pd.DataFrame,
) -> None:
    """Validate required ablation result columns."""
    tables = {
        "fold_metrics": fold_metrics,
        "predictions": predictions,
        "feature_qa": feature_qa,
    }
    for name, required in required_result_columns().items():
        missing = set(required) - set(tables[name].columns)
        if missing:
            msg = f"{name} missing required columns: {sorted(missing)}"
            raise ValueError(msg)


def write_ablation_artifacts(
    run_dir: Path,
    *,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    feature_qa: pd.DataFrame,
    gmm_metadata: pd.DataFrame,
    external_ensemble_predictions: pd.DataFrame | None = None,
    external_ensemble_metrics: pd.DataFrame | None = None,
) -> None:
    """Write the frozen STATE ablation artifact table set."""
    validate_result_tables(fold_metrics, predictions, feature_qa)
    write_formats(run_dir / "artifacts" / "fold_metrics", fold_metrics, ("parquet",))
    write_formats(run_dir / "artifacts" / "predictions", predictions, ("parquet",))
    write_formats(run_dir / "artifacts" / "fold_membership", splits, ("parquet",))
    write_formats(run_dir / "artifacts" / "feature_qa", feature_qa, ("parquet",))
    write_formats(
        run_dir / "artifacts" / "gmm_convergence_metadata",
        gmm_metadata,
        ("parquet",),
    )
    if external_ensemble_predictions is not None:
        write_formats(
            run_dir / "artifacts" / "external_ensemble_predictions",
            external_ensemble_predictions,
            ("parquet",),
        )
    if external_ensemble_metrics is not None:
        write_formats(
            run_dir / "artifacts" / "external_ensemble_metrics",
            external_ensemble_metrics,
            ("parquet",),
        )


def write_run_manifest(
    run_dir: Path,
    *,
    run_id: str,
    status: str,
    payload: dict[str, object],
) -> None:
    """Write the ablation run manifest."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"entrypoint": "state_feature_ablation", "status": status, **payload}
    manifest["run_id"] = run_id
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _ensemble_by_target(
    predictions: pd.DataFrame,
    *,
    evaluation_scope: str,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=[*predictions.columns, "ensemble_size"])
    group_cols = ["feature_set", "arm", "model", "weighting", "perturbation_gene"]
    rows = []
    for key, group in predictions.groupby(group_cols, dropna=False):
        rows.append(
            {
                "evaluation_scope": evaluation_scope,
                "feature_set": key[0],
                "arm": key[1],
                "model": key[2],
                "weighting": key[3],
                "perturbation_gene": key[4],
                "y_true": float(group["y_true"].iloc[0]),
                "y_pred": float(group["y_pred"].mean()),
                "ensemble_size": int(len(group)),
                "primary_scope": PRIMARY_EXTERNAL_SCOPE,
                "secondary_scope": EXTERNAL_ENSEMBLE_SCOPE,
            }
        )
    return pd.DataFrame(rows)


def _train_gene_lookup(splits: pd.DataFrame) -> dict[int, set[str]]:
    lookup: dict[int, set[str]] = {}
    if splits.empty:
        return lookup
    train = splits.loc[splits["split"] == "train"]
    for fold, group in train.groupby("fold"):
        lookup[int(fold)] = set(group["perturbation_gene"].astype(str))
    return lookup


def alpha_token(alpha: float) -> str:
    value = float(alpha)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")
