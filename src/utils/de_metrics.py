"""
Differential Expression-based evaluation metrics for perturbation prediction.

Implements metrics from https://virtualcellchallenge.org/evaluation:
- DES (Differential Expression Score)
- PDS (Perturbation Discrimination Score)
Plus standard metrics on DE log2FC:
- Pearson correlation
- MSE
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
from scipy.stats import pearsonr


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
        fdr_threshold: FDR threshold for significant DE genes
        threads: Number of threads for hpdex

    Returns:
        dict with keys: pearson_log2fc, mse_log2fc, des, n_de_truth, n_de_pred
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

    # Align genes
    common_genes = de_truth.index.intersection(de_pred.index)
    truth_log2fc = de_truth.loc[common_genes, "log2_fold_change"].values
    pred_log2fc = de_pred.loc[common_genes, "log2_fold_change"].values

    # Handle NaN/Inf in log2FC
    valid_mask = np.isfinite(truth_log2fc) & np.isfinite(pred_log2fc)
    truth_log2fc_valid = truth_log2fc[valid_mask]
    pred_log2fc_valid = pred_log2fc[valid_mask]

    # Pearson correlation
    if len(truth_log2fc_valid) > 2:
        pearson_corr, _ = pearsonr(truth_log2fc_valid, pred_log2fc_valid)
    else:
        pearson_corr = np.nan

    # MSE
    mse = np.mean((truth_log2fc_valid - pred_log2fc_valid) ** 2)

    # DES (Differential Expression Score)
    truth_sig = set(de_truth.index[de_truth["fdr"] < fdr_threshold])
    pred_sig = set(de_pred.index[de_pred["fdr"] < fdr_threshold])

    n_de_truth = len(truth_sig)
    n_de_pred = len(pred_sig)

    des = compute_des(
        truth_sig,
        pred_sig,
        de_pred.loc[list(pred_sig), "log2_fold_change"] if pred_sig else pd.Series(),
    )

    return {
        "pearson_log2fc": pearson_corr,
        "mse_log2fc": mse,
        "des": des,
        "n_de_truth": n_de_truth,
        "n_de_pred": n_de_pred,
    }


def compute_des(
    truth_sig: set,
    pred_sig: set,
    pred_log2fc: pd.Series,
) -> float:
    """
    Compute Differential Expression Score.

    If n_pred <= n_true: DES = |intersection| / n_true
    If n_pred > n_true: select top n_true genes by |log2FC| from pred_sig,
    then compute overlap.
    """
    n_true = len(truth_sig)
    n_pred = len(pred_sig)

    if n_true == 0:
        return np.nan

    if n_pred <= n_true:
        intersection = truth_sig & pred_sig
        return len(intersection) / n_true
    else:
        # Select top n_true genes by absolute log2FC
        top_pred = set(pred_log2fc.abs().nlargest(n_true).index)
        intersection = truth_sig & top_pred
        return len(intersection) / n_true


def compute_pds(
    pred_deltas: dict[str, np.ndarray],
    truth_deltas: dict[str, np.ndarray],
    target_gene_indices: dict[str, int],
) -> tuple[float, dict[str, float]]:
    """
    Compute Perturbation Discrimination Score.

    For each predicted perturbation, rank all true perturbation deltas by L1 distance.
    PDS = 1 - (rank - 1) / N, where rank is position of correct perturbation.

    Args:
        pred_deltas: {target_gene: delta_vector} for predictions
        truth_deltas: {target_gene: delta_vector} for ground truth
        target_gene_indices: {target_gene: gene_index} to exclude from distance

    Returns:
        (mean_pds, {target_gene: pds_score})
    """
    targets = list(pred_deltas.keys())
    n = len(targets)

    if n == 0:
        return np.nan, {}

    pds_scores = {}

    for p in targets:
        pred_delta = pred_deltas[p].copy()
        target_idx = target_gene_indices.get(p, -1)

        # Compute L1 distance to all true deltas
        distances = []
        for t in targets:
            truth_delta = truth_deltas[t].copy()

            # Exclude target gene from distance calculation
            if target_idx >= 0:
                pred_delta_masked = np.delete(pred_delta, target_idx)
                truth_delta_masked = np.delete(truth_delta, target_idx)
            else:
                pred_delta_masked = pred_delta
                truth_delta_masked = truth_delta

            dist = np.sum(np.abs(pred_delta_masked - truth_delta_masked))
            distances.append((t, dist))

        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])

        # Find rank of correct perturbation (1-indexed)
        rank = next(i + 1 for i, (t, _) in enumerate(distances) if t == p)

        # PDS = 1 - (rank - 1) / N
        pds = 1 - (rank - 1) / n
        pds_scores[p] = pds

    mean_pds = np.mean(list(pds_scores.values()))
    return mean_pds, pds_scores


def compute_pseudobulk_delta(
    perturbed_expr: np.ndarray,
    control_mean: np.ndarray,
) -> np.ndarray:
    """
    Compute perturbation delta: pseudobulk(perturbed) - pseudobulk(control).

    Assumes log1p-normalized expression.
    """
    perturbed_mean = perturbed_expr.mean(axis=0)
    return perturbed_mean - control_mean
