"""
Differential Expression-based evaluation metrics for perturbation prediction.

Implements metrics from docs/eval_metrics.md:
- PDS (Perturbation Discrimination Score) - cosine similarity ranking
- DES (Differential Expression Score) - DE gene overlap [PLACEHOLDER]
- MAE_top2k (MAE on top 2000 genes by |log2FC|)

NOTE: DES metric implementation has been temporarily removed and returns placeholder values.
"""

import numpy as np


def wilcoxon_test_per_gene(
    group1: np.ndarray,
    group2: np.ndarray,
) -> np.ndarray:
    """
    Placeholder for per-gene differential expression statistical testing.

    Args:
        group1: Expression matrix (n_cells_1, n_genes)
        group2: Expression matrix (n_cells_2, n_genes)

    Returns:
        p_values: Array of placeholder p-values (n_genes,)
    """
    group1 = np.atleast_2d(group1)
    n_genes = group1.shape[1] if group1.ndim == 2 else 0
    return np.full(n_genes, np.nan, dtype=float)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """
    Placeholder for Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of raw p-values

    Returns:
        p_adj: Array of placeholder adjusted p-values
    """
    return np.full_like(np.asarray(p_values, dtype=float), np.nan, dtype=float)


def compute_log2fc(
    perturbed_expr: np.ndarray,
    control_expr: np.ndarray,
) -> np.ndarray:
    """
    Placeholder for log2 fold-change computation.

    Args:
        perturbed_expr: Perturbed expression (n_cells, n_genes), log1p-normalized
        control_expr: Control expression (n_cells, n_genes), log1p-normalized

    Returns:
        log2fc: Array of placeholder log2 fold changes (n_genes,)
    """
    perturbed_expr = np.atleast_2d(perturbed_expr)
    n_genes = perturbed_expr.shape[1] if perturbed_expr.ndim == 2 else 0
    return np.full(n_genes, np.nan, dtype=float)


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

    Placeholder implementation (DES temporarily disabled).

    Args:
        control_expr: Control cell expression (n_ctrl, n_genes), log1p-normalized
        pred_expr: Predicted perturbed expression (n_pred, n_genes), log1p-normalized
        truth_expr: Ground truth perturbed expression (n_truth, n_genes), log1p-normalized
        gene_names: Gene names array
        fdr_threshold: FDR threshold for significant DE genes (default 0.05)
        threads: Unused, kept for API compatibility

    Returns:
        dict with keys: des, n_de_truth, n_de_pred, n_intersect
    """
    return {
        "des": np.nan,
        "n_de_truth": 0,
        "n_de_pred": 0,
        "n_intersect": 0,
    }


def compute_des(
    truth_de_genes: set,
    pred_de_genes: set,
    pred_log2fc: np.ndarray,
) -> tuple[float, int]:
    """
    Placeholder for Differential Expression Score (DES).

    Args:
        truth_de_genes: Set of gene indices that are truly DE
        pred_de_genes: Set of gene indices predicted as DE
        pred_log2fc: Array of predicted log2 fold changes (n_genes,)

    Returns:
        (des_score, n_intersect) placeholder values
    """
    return np.nan, 0


def compute_pds(
    pred_deltas: dict[str, np.ndarray],
    truth_deltas: dict[str, np.ndarray],
    target_gene_indices: dict[str, int] | None = None,
) -> dict:
    """
    Compute Perturbation Discrimination Score using cosine similarity ranking.

    For each predicted perturbation k, compute cosine similarity between
    predicted delta and all true deltas:
        S_{k,j} = (pred_delta_k · truth_delta_j) / (|pred_delta_k| * |truth_delta_j|)

    Rank = position of true perturbation among all similarities (descending).

    Per eval_metrics.md Section 4.2.1:
    - MeanRank = (1/N) * sum(R_k)
    - nPDS = (1/N^2) * sum(R_k) = MeanRank / N

    Args:
        pred_deltas: {target_gene: delta_vector} for predictions
        truth_deltas: {target_gene: delta_vector} for ground truth
        target_gene_indices: {target_gene: gene_index} - not used in cosine version

    Returns:
        dict with:
            - mean_rank: mean of all ranks
            - npds: normalized PDS (MeanRank / N)
            - ranks: {target_gene: rank}
            - cosine_self: {target_gene: cos(δ̂_k, δ_k)} self-similarity
    """
    targets = list(pred_deltas.keys())
    n = len(targets)

    if n == 0:
        return {"mean_rank": np.nan, "npds": np.nan, "ranks": {}, "cosine_self": {}}

    ranks = {}
    cosine_self = {}

    for p in targets:
        pred_delta = pred_deltas[p]
        truth_delta_self = truth_deltas[p]

        # Compute cosine similarity to all true deltas
        similarities = []
        pred_norm = np.linalg.norm(pred_delta)
        truth_self_norm = np.linalg.norm(truth_delta_self)

        # Store self-similarity (cos(δ̂_k, δ_k))
        if pred_norm > 0 and truth_self_norm > 0:
            cosine_self[p] = float(
                np.dot(pred_delta, truth_delta_self) / (pred_norm * truth_self_norm)
            )
        else:
            cosine_self[p] = 0.0

        for t in targets:
            truth_delta = truth_deltas[t]
            truth_norm = np.linalg.norm(truth_delta)

            # Cosine similarity
            if pred_norm > 0 and truth_norm > 0:
                cos_sim = np.dot(pred_delta, truth_delta) / (pred_norm * truth_norm)
            else:
                cos_sim = 0.0

            similarities.append((t, cos_sim))

        # Sort by similarity (descending - higher similarity = better match)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Find rank of correct perturbation (1-indexed)
        rank = next(i + 1 for i, (t, _) in enumerate(similarities) if t == p)
        ranks[p] = rank

    # Mean rank and normalized PDS per eval_metrics.md
    mean_rank = sum(ranks.values()) / n
    npds = mean_rank / n  # nPDS = (1/N^2) * sum(R_k) = MeanRank / N

    return {
        "mean_rank": float(mean_rank),
        "npds": float(npds),
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
    Compute MAE on top K genes by ground truth |log2 fold change|.

    Per docs/eval_metrics.md:
    1. Compute |LFC| = |log2(c_k + 1) - log2(c_ntc + 1)| for ground truth
       Since data is already log1p-normalized, we use it directly.
    2. Select top K genes by |LFC|
    3. Compute MAE on log1p-normalized expression for selected genes

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

    # Since data is log1p-normalized, compute log2 fold change:
    # log2FC = log2(exp(log1p_k) / exp(log1p_ntc))
    #        = (log1p_k - log1p_ntc) / log(2)
    # We just need |delta| for ranking, so use |truth_mean - control_mean|
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
BASELINE_PDS = 0.4833
BASELINE_MAE_TOP2000 = 0.1258
BASELINE_DES = 0.2534


def compute_overall_score(pds: float, mae: float, des: float) -> dict:
    """
    Compute scaled metrics and overall score per eval_metrics.md Section 4.2.2.

    Args:
        pds: PDS score (1 - npds, higher is better, range 0-1)
        mae: MAE on top 2000 genes (lower is better)
        des: DES score (higher is better, range 0-1)

    Returns:
        dict with pds_scaled, mae_scaled, des_scaled, overall_score
    """
    # Scale and clip to [0, 1]
    pds_scaled = (
        max(0.0, (pds - BASELINE_PDS) / (1 - BASELINE_PDS))
        if not np.isnan(pds)
        else np.nan
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
