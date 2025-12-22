"""
Preprocessing utilities for reverse perturbation prediction.

Includes:
- Perturbed gene masking (anti-cheat)
- Prototype library construction
"""

from typing import List, Optional, Tuple

import numpy as np
import anndata as ad


def mask_perturbed_genes(
    adata: ad.AnnData,
    condition_col: str = "condition",
    gene_name_col: str = "gene_name",
    mask_value: float = 0.0,
    layer: Optional[str] = None,
) -> ad.AnnData:
    """
    Mask expression of perturbed genes to prevent information leakage.

    This is the anti-cheat protocol: the model should not be able to
    identify perturbations by looking at the perturbed gene's own
    expression (which may be knocked out/down).

    Args:
        adata: AnnData object
        condition_col: Column with perturbation condition
        gene_name_col: Column in var with gene names
        mask_value: Value to set for masked genes
        layer: AnnData layer key to mask (defaults to X)

    Returns:
        AnnData with perturbed genes masked
    """
    adata = adata.copy()

    # Get gene names from var
    gene_names = (
        adata.var[gene_name_col].values
        if gene_name_col in adata.var
        else adata.var.index.values
    )
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Select the matrix to mask
    if layer is None or layer == "X":
        X = adata.X.copy()
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData.layers")
        X = adata.layers[layer].copy()
    if hasattr(X, "toarray"):
        X = X.toarray()

    for i, condition in enumerate(adata.obs[condition_col]):
        # Parse perturbed genes from condition (format: GENE1+GENE2 or GENE+ctrl)
        genes = condition.split("+")
        for gene in genes:
            if gene != "ctrl" and gene in gene_to_idx:
                X[i, gene_to_idx[gene]] = mask_value

    if layer is None or layer == "X":
        adata.X = X
    else:
        adata.layers[layer] = X
    return adata


def build_prototype_library(
    adata: ad.AnnData,
    conditions: List[str],
    n_prototypes: int = 30,
    m_cells_per_prototype: Optional[int] = None,
    method: str = "bootstrap",
    seed: int = 42,
    condition_col: str = "condition",
    layer: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build reference library with multiple prototype profiles per condition.

    Creates diversity via bootstrap sampling for similarity-based retrieval.

    Args:
        adata: AnnData object with single-cell data
        conditions: List of condition names to include
        n_prototypes: Number of prototypes per condition (default 30)
        m_cells_per_prototype: Cells sampled per prototype (None = all cells)
        method: 'bootstrap' (resample with replacement) or 'sample' (split cells)
        seed: Random seed for reproducibility
        condition_col: Column name with condition labels
        layer: Layer to use (None for X)

    Returns:
        Tuple of (profiles array, condition labels list)
        - profiles: (n_conditions * n_prototypes, n_genes)
        - labels: condition name for each row
    """
    rng = np.random.default_rng(seed)

    all_profiles = []
    all_labels = []

    for cond in conditions:
        # Skip control
        if cond == "ctrl":
            continue

        # Get cells for this condition
        mask = adata.obs[condition_col] == cond
        cells = adata[mask]
        n_cells = mask.sum()

        if n_cells == 0:
            continue

        # Get expression matrix
        if layer:
            expr = cells.layers[layer]
        else:
            expr = cells.X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()

        # Generate prototypes
        if method == "bootstrap":
            # Bootstrap: sample with replacement and average
            for _ in range(n_prototypes):
                sample_size = m_cells_per_prototype or n_cells
                indices = rng.choice(n_cells, size=sample_size, replace=True)
                sampled = expr[indices]
                prototype = np.mean(sampled, axis=0).flatten()
                all_profiles.append(prototype)
                all_labels.append(cond)

        elif method == "sample":
            # Sample: split cells into groups and average each
            if n_cells < n_prototypes:
                group_indices = np.array_split(
                    rng.choice(n_cells, size=n_prototypes, replace=True), n_prototypes
                )
            else:
                indices = rng.permutation(n_cells)
                group_indices = np.array_split(indices, n_prototypes)

            for group in group_indices:
                if len(group) > 0:
                    if m_cells_per_prototype:
                        group = rng.choice(
                            group,
                            size=min(len(group), m_cells_per_prototype),
                            replace=False,
                        )
                    prototype = np.mean(expr[group], axis=0).flatten()
                    all_profiles.append(prototype)
                    all_labels.append(cond)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bootstrap' or 'sample'.")

    if not all_profiles:
        return np.array([]), []

    profiles = np.vstack(all_profiles)
    return profiles, all_labels
