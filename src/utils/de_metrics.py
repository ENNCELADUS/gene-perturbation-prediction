"""
Differential Expression metrics for perturbation prediction.

Metrics (per docs/eval_metrics.md):
- PDS: Perturbation Discrimination Score (cosine similarity ranking)
- DES: Differential Expression Score (DE gene overlap)
- MAE_top2k: MAE on top 2000 genes by |log2FC|
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

try:  # Optional dependency; fallback to pandas if unavailable
    import polars as pl
except ImportError:  # pragma: no cover - exercised when polars not installed
    pl = None
from pdex import parallel_differential_expression


def _ensure_dense_2d(x: np.ndarray) -> np.ndarray:
    """Convert sparse/1D array to dense 2D (n_samples, n_features)."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def _build_single_perturbation_adata(
    control_expr: np.ndarray, perturbed_expr: np.ndarray, gene_names: np.ndarray
) -> ad.AnnData:
    """Construct a simple AnnData with control and perturbed cells for DE."""
    control_expr = _ensure_dense_2d(control_expr)
    perturbed_expr = _ensure_dense_2d(perturbed_expr)
    gene_names = np.asarray(gene_names, dtype=object)

    n_genes = control_expr.shape[1]
    if perturbed_expr.shape[1] != n_genes:
        raise ValueError(f"Gene count mismatch: {n_genes} vs {perturbed_expr.shape[1]}")
    if len(gene_names) != n_genes:
        raise ValueError(f"gene_names length mismatch: {len(gene_names)} vs {n_genes}")

    x = np.vstack([control_expr, perturbed_expr])
    obs = pd.DataFrame(
        {
            "perturbation": ["control"] * control_expr.shape[0]
            + ["perturbed"] * perturbed_expr.shape[0]
        }
    )
    var = pd.DataFrame(index=gene_names)
    return ad.AnnData(X=x, obs=obs, var=var)


def _run_de(
    adata: ad.AnnData,
    control: str,
    pert_col: str,
    method: str,
    num_workers: int,
    batch_size: int,
) -> pd.DataFrame | "pl.DataFrame":
    """Run DE with pdex and ensure |log2FC| sorting columns exist."""
    de_kwargs = dict(
        adata=adata,
        reference=control,
        groupby_key=pert_col,
        metric=method,
        num_workers=num_workers,
        batch_size=batch_size,
        is_log1p=True,
        as_polars=pl is not None,
    )
    try:
        df = parallel_differential_expression(**de_kwargs)
    except (PermissionError, OSError):
        from pdex._single_cell import parallel_differential_expression_vec_wrapper

        de_kwargs["num_workers"] = 1
        df = parallel_differential_expression_vec_wrapper(**de_kwargs)

    if pl is not None and isinstance(df, pl.DataFrame):
        if "log2_fold_change" not in df.columns:
            df = df.with_columns(
                pl.col("fold_change")
                .log(base=2)
                .fill_nan(0.0)
                .alias("log2_fold_change")
            )
        if "abs_log2_fold_change" not in df.columns:
            df = df.with_columns(
                pl.col("log2_fold_change").abs().alias("abs_log2_fold_change")
            )
        return df

    # Pandas fallback if polars not installed
    if "log2_fold_change" not in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["log2_fold_change"] = np.log2(df.get("fold_change", np.nan))
    if "abs_log2_fold_change" not in df.columns:
        df["abs_log2_fold_change"] = df["log2_fold_change"].abs()
    return df.reset_index(drop=True)


def _get_sorted_genes(
    df: pd.DataFrame | "pl.DataFrame", fdr: float, topk: int | None
) -> dict[str, np.ndarray]:
    """Filter by FDR and sort DE genes by |log2FC| descending."""
    if "fdr" not in df.columns:
        return {}
    lists: dict[str, np.ndarray] = {}

    if pl is not None and isinstance(df, pl.DataFrame):
        sig = df.filter(pl.col("fdr") < fdr)
        for pert in sig["target"].unique().to_list():
            genes = (
                sig.filter(pl.col("target") == pert)
                .sort("abs_log2_fold_change", descending=True)
                .select("feature")
                .to_numpy()
                .ravel()
            )
            if topk is not None:
                genes = genes[:topk]
            lists[str(pert)] = genes
        return lists

    # Pandas path
    sig = df[df["fdr"] < fdr]
    for pert, sub in sig.groupby("target"):
        genes = (
            sub.sort_values("abs_log2_fold_change", ascending=False)["feature"]
            .to_numpy()
            .ravel()
        )
        if topk is not None:
            genes = genes[:topk]
        lists[str(pert)] = genes
    return lists


def compute_de_comparison_metrics(
    control_expr: np.ndarray,
    pred_expr: np.ndarray,
    truth_expr: np.ndarray,
    gene_names: np.ndarray,
    fdr_threshold: float = 0.05,
    threads: int = -1,
    topk: int | None = None,
) -> dict:
    """
    Compute DE-based metrics comparing predicted vs ground truth perturbation.

    DES calculation is aligned with `csc286_eval.py`:
    - Compute DE on control vs. perturbed using pdex (Wilcoxon, BH FDR)
    - Filter by FDR and sort by |log2FC| descending
    - DES = |intersection(real[:k], pred[:k])| / k, where k = min(|real|, topk)

    Args:
        control_expr: Control cell expression (n_ctrl, n_genes),
            log1p-normalized
        pred_expr: Predicted perturbed expression (n_pred, n_genes),
            log1p-normalized
        truth_expr: Ground truth perturbed expression (n_truth, n_genes),
            log1p-normalized
        gene_names: Gene names array
        fdr_threshold: FDR threshold for significant DE genes (default 0.05)
        threads: Worker count hint for pdex (uses 1 if <= 0)
        topk: Optional top-K for DES overlap; default uses |truth DE genes|

    Returns:
        dict with keys: des, n_de_truth, n_de_pred, n_intersect
    """
    gene_names = np.asarray(gene_names, dtype=object)
    adata_truth = _build_single_perturbation_adata(control_expr, truth_expr, gene_names)
    adata_pred = _build_single_perturbation_adata(control_expr, pred_expr, gene_names)

    num_workers = max(1, threads) if threads > 0 else 1
    real_de = _run_de(
        adata_truth,
        control="control",
        pert_col="perturbation",
        method="wilcoxon",
        num_workers=num_workers,
        batch_size=100,
    )
    pred_de = _run_de(
        adata_pred,
        control="control",
        pert_col="perturbation",
        method="wilcoxon",
        num_workers=num_workers,
        batch_size=100,
    )
    des_score, n_de_truth, n_de_pred, n_intersect = compute_des_from_de_frames(
        pred_df=pred_de,
        real_df=real_de,
        fdr=fdr_threshold,
        topk=topk,
        control="control",
    )

    return {
        "des": float(des_score),
        "n_de_truth": int(n_de_truth),
        "n_de_pred": int(n_de_pred),
        "n_intersect": int(n_intersect),
    }


def compute_des(
    truth_de_genes: np.ndarray,
    pred_de_genes: np.ndarray,
    topk: int | None = None,
) -> tuple[float, int]:
    """
    Compute Differential Expression Score (DES) per docs/eval_metrics.md §3.

    This is a direct port of the DES overlap calculation used in `csc286_eval.py`:
    - Both truth and pred gene lists are assumed sorted by |log2FC| descending
    - k_eff = len(truth) if topk is None, else min(len(truth), topk)
    - DES = |intersection(truth[:k_eff], pred[:k_eff])| / k_eff

    Args:
        truth_de_genes: Ground truth DE gene names, sorted by |log2FC| descending
        pred_de_genes: Predicted DE gene names, sorted by |log2FC| descending
        topk: Optional top-K limit; default uses |truth|

    Returns:
        Tuple of (des_score, n_intersect)
    """
    truth_de_genes = np.asarray(truth_de_genes)
    pred_de_genes = np.asarray(pred_de_genes)

    # Determine effective K (aligned with csc286_eval.py)
    k_eff = len(truth_de_genes) if topk is None else min(len(truth_de_genes), topk)

    if k_eff == 0:
        return 0.0, 0

    n_intersect = np.intersect1d(truth_de_genes[:k_eff], pred_de_genes[:k_eff]).size
    return float(n_intersect / k_eff), int(n_intersect)


def compute_des_per_pert(
    pred_df: pd.DataFrame | "pl.DataFrame",
    real_df: pd.DataFrame | "pl.DataFrame",
    *,
    fdr: float,
    topk: int | None,
) -> dict[str, float]:
    """
    Compute DES per perturbation from DE result frames.

    Ported from `csc286_eval.py::compute_des_per_pert` (uses FDR filtering and
    overlap@K on |log2FC|-sorted DE genes).
    """
    real_lists = _get_sorted_genes(real_df, fdr, topk)
    pred_lists = _get_sorted_genes(pred_df, fdr, topk)
    perts = set(real_lists) & set(pred_lists)
    scores: dict[str, float] = {}
    for pert in perts:
        real_genes = real_lists[pert]
        pred_genes = pred_lists[pert]
        k_eff = len(real_genes) if topk is None else min(len(real_genes), topk)
        if k_eff == 0:
            scores[pert] = 0.0
            continue
        overlap = np.intersect1d(real_genes[:k_eff], pred_genes[:k_eff]).size / k_eff
        scores[pert] = float(overlap)
    return scores


def compute_des_from_de_frames(
    pred_df: pd.DataFrame | "pl.DataFrame",
    real_df: pd.DataFrame | "pl.DataFrame",
    *,
    fdr: float,
    topk: int | None,
    control: str | None = None,
) -> tuple[float, int, int, int]:
    """
    Compute DES using DE outputs (aligned with `csc286_eval.py`).

    Returns:
        des_score, n_de_truth, n_de_pred, n_intersect
    """
    real_lists = _get_sorted_genes(real_df, fdr, topk)
    pred_lists = _get_sorted_genes(pred_df, fdr, topk)
    perts = set(real_lists) & set(pred_lists)
    if control is not None:
        perts.discard(control)
    if not perts:
        return 0.0, 0, 0, 0

    pert = next(iter(perts))
    truth_genes = real_lists[pert]
    pred_genes = pred_lists[pert]
    k_eff = len(truth_genes) if topk is None else min(len(truth_genes), topk)
    if k_eff == 0:
        return 0.0, len(truth_genes), len(pred_genes), 0

    n_intersect = np.intersect1d(truth_genes[:k_eff], pred_genes[:k_eff]).size
    return (
        float(n_intersect / k_eff),
        len(truth_genes),
        len(pred_genes),
        int(n_intersect),
    )


def compute_pds(
    pred_deltas: dict[str, np.ndarray],
    truth_deltas: dict[str, np.ndarray],
    target_gene_indices: dict[str, int] | None = None,  # kept for API compat
) -> dict:
    """
    Compute Perturbation Discrimination Score (PDS) per docs/eval_metrics.md §1.

    For each perturbation k, compute cosine similarity S_{k,j} between predicted
    delta and all true deltas. R_k = rank of true perturbation (1 = best).

    Returns: mean_rank, npds (= mean_rank / N), ranks, cosine_self
    """
    targets = list(pred_deltas.keys())
    n = len(targets)
    if n == 0:
        return {"mean_rank": np.nan, "npds": np.nan, "ranks": {}, "cosine_self": {}}

    # Build matrices: (N, G) for pred and truth deltas
    pred_mat = np.stack([pred_deltas[t] for t in targets])
    truth_mat = np.stack([truth_deltas[t] for t in targets])

    # Normalize rows (epsilon to avoid division by zero)
    eps = 1e-12
    pred_norm = pred_mat / (np.linalg.norm(pred_mat, axis=1, keepdims=True) + eps)
    truth_norm = truth_mat / (np.linalg.norm(truth_mat, axis=1, keepdims=True) + eps)

    # Cosine similarity matrix S[k,j] = cos(pred_k, truth_j)
    S = pred_norm @ truth_norm.T  # (N, N)

    # Compute ranks: for each row k, rank of diagonal element S[k,k]
    # Rank = 1 + count of elements > S[k,k] in row k
    diag = np.diag(S)
    ranks_arr = 1 + (S > diag[:, None]).sum(axis=1)

    ranks = {t: int(ranks_arr[i]) for i, t in enumerate(targets)}
    cosine_self = {t: float(diag[i]) for i, t in enumerate(targets)}

    mean_rank = float(ranks_arr.mean())
    npds = mean_rank / n

    return {
        "mean_rank": mean_rank,
        "npds": npds,
        "ranks": ranks,
        "cosine_self": cosine_self,
    }


def compute_mae_top2k(
    pred_expr: np.ndarray,
    truth_expr: np.ndarray,
    control_mean: np.ndarray,
    top_k: int = 2000,
) -> float:
    """
    Compute MAE on top K genes using ground truth perturbation effect magnitude.

    Per docs/eval_metrics.md:
    1. Compute |delta| between perturbed and control pseudobulk (log1p space).
       Data is already log1p-normalized, so the delta is a log-ratio up to a
       constant base change; no extra log2 conversion is needed.
    2. Select top K genes by |delta|.
    3. Compute MAE on log1p-normalized expression for the selected genes.

    Args:
        pred_expr: Predicted expression (n_cells, n_genes), log1p-normalized
        truth_expr: Ground truth expression (n_cells, n_genes), log1p-normalized
        control_mean: Control mean expression (n_genes,), log1p-normalized
        top_k: Number of top genes to select (default 2000)

    Returns:
        MAE on top K genes
    """
    # Compute pseudobulk means
    pred_mean = pred_expr.mean(axis=0).flatten()
    truth_mean = truth_expr.mean(axis=0).flatten()
    control_mean = np.asarray(control_mean).flatten()

    # Data is log1p-normalized; |truth_mean - control_mean| is proportional
    # to |log2 fold change| and suffices for selecting the top genes.
    truth_delta = np.abs(truth_mean - control_mean)

    # Select top K genes by |delta|
    n_genes = len(truth_delta)
    k = min(top_k, n_genes)
    top_indices = np.argsort(truth_delta)[-k:]

    # Compute MAE on selected genes (on log1p-normalized values)
    mae = np.mean(np.abs(pred_mean[top_indices] - truth_mean[top_indices]))

    return float(mae)


def compute_pseudobulk_delta(
    perturbed_expr: np.ndarray,
    control_mean: np.ndarray,
) -> np.ndarray:
    """
    Compute perturbation delta: pseudobulk(perturbed) - pseudobulk(control).

    Assumes log1p-normalized expression.
    """
    perturbed_mean = perturbed_expr.mean(axis=0)
    return np.asarray(perturbed_mean).flatten() - np.asarray(control_mean).flatten()


# Baseline scores for normalization (pre-calculated on Training dataset)
# Per docs/eval_metrics.md Section 4.2.2
BASELINE_PDS = 0.5167  # nPDS (mean_rank / N), lower is better
BASELINE_MAE_TOP2000 = 0.1258  # lower is better
BASELINE_DES = 0.0442  # higher is better


def compute_overall_score(pds: float, mae: float, des: float) -> dict:
    """
    Compute scaled metrics and overall score per eval_metrics.md Section 4.2.2.

    Args:
        pds: Normalized PDS (mean_rank / N); lower is better.
        mae: MAE on top 2000 genes (lower is better)
        des: DES score (higher is better, range 0-1)

    Returns:
        dict with pds_scaled, mae_scaled, des_scaled, overall_score
    """
    # Scale and clip to [0, 1]
    pds_scaled = (
        max(0.0, (BASELINE_PDS - pds) / BASELINE_PDS) if not np.isnan(pds) else np.nan
    )
    mae_scaled = (
        max(0.0, (BASELINE_MAE_TOP2000 - mae) / BASELINE_MAE_TOP2000)
        if not np.isnan(mae)
        else np.nan
    )
    des_scaled = (
        max(0.0, (des - BASELINE_DES) / (1 - BASELINE_DES))
        if not np.isnan(des)
        else np.nan
    )

    # Overall score = mean of valid scaled scores * 100
    scaled = [pds_scaled, mae_scaled, des_scaled]
    valid = [s for s in scaled if not np.isnan(s)]
    overall_score = float(np.mean(valid) * 100) if valid else 0.0

    return {
        "pds_scaled": float(pds_scaled) if not np.isnan(pds_scaled) else np.nan,
        "mae_scaled": float(mae_scaled) if not np.isnan(mae_scaled) else np.nan,
        "des_scaled": float(des_scaled) if not np.isnan(des_scaled) else np.nan,
        "overall_score": overall_score,
    }
