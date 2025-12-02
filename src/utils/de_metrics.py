"""
Differential Expression-based evaluation metrics for perturbation prediction.

Implements metrics from docs/eval_metrics.md:
- PDS (Perturbation Discrimination Score) - cosine similarity ranking
- DES (Differential Expression Score) - DE gene overlap
- MAE_top2k (MAE on top 2000 genes by |log2FC|)
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse


def build_de_adata(
    control_expr: np.ndarray,
    perturbed_expr: np.ndarray,
    gene_names: np.ndarray,
) -> ad.AnnData:
    """Build AnnData with control and perturbed cells labeled for DE analysis."""
    n_ctrl = control_expr.shape[0]
    n_pert = perturbed_expr.shape[0]

    # Stack expression matrices
    X = np.vstack([control_expr, perturbed_expr])

    # Create obs with perturbation labels
    obs = pd.DataFrame({"perturbation": ["control"] * n_ctrl + ["perturbed"] * n_pert})
    obs["perturbation"] = obs["perturbation"].astype("category")

    # Create var
    var = pd.DataFrame(index=gene_names)

    return ad.AnnData(X=scipy.sparse.csr_matrix(X), obs=obs, var=var)


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

    Args:
        control_expr: Control cell expression (n_ctrl, n_genes)
        pred_expr: Predicted perturbed expression (n_pred, n_genes)
        truth_expr: Ground truth perturbed expression (n_truth, n_genes)
        gene_names: Gene names array
        fdr_threshold: FDR threshold for significant DE genes (for DES)
        threads: Number of threads for hpdex

    Returns:
        dict with keys: des, n_de_truth, n_de_pred
    """
    from hpdex import pden  # numba backend (no C++ required)

    # Build AnnData for truth DE
    adata_truth = build_de_adata(control_expr, truth_expr, gene_names)
    # Build AnnData for predicted DE
    adata_pred = build_de_adata(control_expr, pred_expr, gene_names)

    # Determine number of workers
    num_workers = threads if threads > 0 else 1

    # Run DE analysis using numba backend
    de_truth = pden(
        adata_truth,
        groupby_key="perturbation",
        reference="control",
        num_workers=num_workers,
    )
    de_pred = pden(
        adata_pred,
        groupby_key="perturbation",
        reference="control",
        num_workers=num_workers,
    )

    # Merge DE results on feature (gene)
    de_truth = de_truth.set_index("feature")
    de_pred = de_pred.set_index("feature")

    # DES (Differential Expression Score)
    truth_sig = set(de_truth.index[de_truth["fdr"] < fdr_threshold])
    pred_sig = set(de_pred.index[de_pred["fdr"] < fdr_threshold])

    n_de_truth = len(truth_sig)
    n_de_pred = len(pred_sig)

    des, n_intersect = compute_des(
        truth_sig,
        pred_sig,
        de_pred.loc[list(pred_sig), "log2_fold_change"] if pred_sig else pd.Series(),
    )

    return {
        "des": des,
        "n_de_truth": n_de_truth,
        "n_de_pred": n_de_pred,
        "n_intersect": n_intersect,
    }


def compute_des(
    truth_sig: set,
    pred_sig: set,
    pred_log2fc: pd.Series,
) -> tuple[float, int]:
    """
    Compute Differential Expression Score.

    If n_pred <= n_true: DES = |intersection| / n_true
    If n_pred > n_true: select top n_true genes by |log2FC| from pred_sig,
    then compute overlap.

    Returns:
        (des_score, n_intersect)
    """
    n_true = len(truth_sig)
    n_pred = len(pred_sig)

    if n_true == 0:
        return np.nan, 0

    if n_pred <= n_true:
        intersection = truth_sig & pred_sig
        n_intersect = len(intersection)
        return n_intersect / n_true, n_intersect
    else:
        # Select top n_true genes by absolute log2FC
        top_pred = set(pred_log2fc.abs().nlargest(n_true).index)
        intersection = truth_sig & top_pred
        n_intersect = len(intersection)
        return n_intersect / n_true, n_intersect


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
