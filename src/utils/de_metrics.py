"""
Differential Expression metrics for perturbation prediction.

Metrics (per docs/eval_metrics.md):
- PDS: Perturbation Discrimination Score (cosine similarity ranking)
- DES: Differential Expression Score (DE gene overlap)
- MAE_top2k: MAE on top 2000 genes by |log2FC|
"""

import numpy as np
from pdex import parallel_differential_expression


def _ensure_dense_2d(x: np.ndarray) -> np.ndarray:
    """Convert sparse/1D array to dense 2D (n_samples, n_features)."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def _pdex_de(
    control_expr: np.ndarray,
    perturbed_expr: np.ndarray,
    gene_names: np.ndarray,
    *,
    fdr_threshold: float,
    threads: int = -1,
) -> tuple[set[int], np.ndarray]:
    """Run DE analysis using pdex. Returns (de_gene_indices, log2fc_array)."""
    import anndata as ad
    import pandas as pd

    control_expr = _ensure_dense_2d(control_expr)
    perturbed_expr = _ensure_dense_2d(perturbed_expr)
    gene_names = np.asarray(gene_names, dtype=object)
    n_genes = control_expr.shape[1]

    if perturbed_expr.shape[1] != n_genes:
        raise ValueError(f"Gene count mismatch: {n_genes} vs {perturbed_expr.shape[1]}")
    if len(gene_names) != n_genes:
        raise ValueError(f"gene_names length mismatch: {len(gene_names)} vs {n_genes}")

    # Build AnnData for DE analysis
    x = np.vstack([control_expr, perturbed_expr])
    obs = pd.DataFrame(
        {
            "__group": ["control"] * control_expr.shape[0]
            + ["perturbed"] * perturbed_expr.shape[0]
        }
    )
    adata = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=gene_names))

    # Common DE parameters
    de_params = dict(
        adata=adata,
        groups=["perturbed"],
        reference="control",
        groupby_key="__group",
        batch_size=100,
        metric="wilcoxon",
        tie_correct=True,
        is_log1p=True,
        as_polars=False,
    )
    num_workers = max(1, threads) if threads > 0 else 1

    try:
        df = parallel_differential_expression(**de_params, num_workers=num_workers)
    except OSError as e:
        if getattr(e, "errno", None) != 13:
            raise
        # Permission error fallback to single-threaded
        from pdex._single_cell import parallel_differential_expression_vec_wrapper

        df = parallel_differential_expression_vec_wrapper(**de_params, num_workers=1)

    if df is None or len(df) == 0:
        return set(), np.zeros(n_genes, dtype=float)

    if "target" in df.columns:
        df = df[df["target"] == "perturbed"]

    # Ensure log2_fold_change column exists
    if "log2_fold_change" not in df.columns and "fold_change" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df = df.assign(log2_fold_change=np.log2(df["fold_change"].to_numpy()))

    # Map gene names to indices
    name_to_idx = {str(g): i for i, g in enumerate(gene_names)}

    # Extract log2FC values
    log2fc = np.zeros(n_genes, dtype=float)
    for feat, val in zip(
        df["feature"].astype(str), df["log2_fold_change"], strict=False
    ):
        if (idx := name_to_idx.get(feat)) is not None and np.isfinite(val):
            log2fc[idx] = val

    # Extract DE genes (FDR < threshold)
    de_genes: set[int] = set()
    if "fdr" in df.columns:
        for feat, fdr in zip(df["feature"].astype(str), df["fdr"], strict=False):
            if np.isfinite(fdr) and fdr < fdr_threshold:
                if (idx := name_to_idx.get(feat)) is not None:
                    de_genes.add(idx)

    return de_genes, log2fc


def compute_de_comparison_metrics(
    control_expr: np.ndarray,
    pred_expr: np.ndarray,
    truth_expr: np.ndarray,
    gene_names: np.ndarray,
    fdr_threshold: float = 0.05,
    threads: int = -1,
) -> dict:
    """
    Compute DE-based metrics comparing predicted vs ground truth perturbation.

    Uses the official `pdex.parallel_differential_expression` to call DE genes
    (Wilcoxon rank-sum / Mann–Whitney U, BH FDR), then computes DES per
    docs/eval_metrics.md §3.

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

    Returns:
        dict with keys: des, n_de_truth, n_de_pred, n_intersect
    """
    control_expr = _ensure_dense_2d(control_expr)
    pred_expr = _ensure_dense_2d(pred_expr)
    truth_expr = _ensure_dense_2d(truth_expr)
    gene_names = np.asarray(gene_names, dtype=object)

    truth_de_genes, _ = _pdex_de(
        control_expr,
        truth_expr,
        gene_names,
        fdr_threshold=fdr_threshold,
        threads=threads,
    )
    pred_de_genes, pred_log2fc = _pdex_de(
        control_expr,
        pred_expr,
        gene_names,
        fdr_threshold=fdr_threshold,
        threads=threads,
    )

    des_score, n_intersect = compute_des(truth_de_genes, pred_de_genes, pred_log2fc)
    return {
        "des": float(des_score),
        "n_de_truth": int(len(truth_de_genes)),
        "n_de_pred": int(len(pred_de_genes)),
        "n_intersect": int(n_intersect),
    }


def compute_des(
    truth_de_genes: set,
    pred_de_genes: set,
    pred_log2fc: np.ndarray,
) -> tuple[float, int]:
    """
    Compute Differential Expression Score (DES) per docs/eval_metrics.md §3.

    If |pred| > |truth|, truncate pred to top |truth| genes by |log2FC|.
    DES = |intersection| / |truth|
    """
    n_true = len(truth_de_genes)
    if n_true == 0:
        return 0.0, 0

    # Truncate predicted set if larger than truth set (§3.3)
    if len(pred_de_genes) <= n_true:
        pred_eval = pred_de_genes
    else:
        log2fc = np.asarray(pred_log2fc, dtype=float).ravel()
        ranked = sorted(
            pred_de_genes,
            key=lambda i: abs(log2fc[i]) if 0 <= i < len(log2fc) else 0.0,
            reverse=True,
        )
        pred_eval = set(ranked[:n_true])

    n_intersect = len(pred_eval & truth_de_genes)
    return n_intersect / n_true, n_intersect


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
