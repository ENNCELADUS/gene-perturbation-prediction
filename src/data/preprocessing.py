"""
Preprocessing utilities for reverse perturbation prediction.

Includes:
- Pseudo-bulk aggregation
- Perturbed gene masking (anti-cheat)
- Reference library construction
"""

from typing import Optional

import numpy as np
import anndata as ad


def create_pseudo_bulk(
    adata: ad.AnnData,
    groupby: str = "condition",
    layer: Optional[str] = None,
) -> ad.AnnData:
    """
    Create pseudo-bulk profiles by averaging cells per condition.

    Args:
        adata: AnnData object with single-cell data
        groupby: Column in obs to group by
        layer: Layer to use (None for X)

    Returns:
        AnnData with pseudo-bulk profiles (one per condition)
    """
    conditions = adata.obs[groupby].unique()
    profiles = []
    obs_data = []

    for cond in conditions:
        mask = adata.obs[groupby] == cond
        cells = adata[mask]

        # Get expression matrix
        if layer:
            expr = cells.layers[layer]
        else:
            expr = cells.X

        # Average expression
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        mean_expr = np.mean(expr, axis=0).flatten()
        profiles.append(mean_expr)

        # Collect metadata
        obs_data.append({
            groupby: cond,
            "n_cells": mask.sum(),
            "control": cells.obs["control"].iloc[0] if "control" in cells.obs else 0,
        })

    # Create new AnnData
    X = np.vstack(profiles)
    import pandas as pd
    obs = pd.DataFrame(obs_data)

    pseudo_bulk = ad.AnnData(X=X, obs=obs, var=adata.var.copy())
    return pseudo_bulk


def mask_perturbed_genes(
    adata: ad.AnnData,
    condition_col: str = "condition",
    gene_name_col: str = "gene_name",
    mask_value: float = 0.0,
) -> ad.AnnData:
    """
    Mask expression of perturbed genes to prevent information leakage.

    This is the anti-cheat protocol from the roadmap: the model should
    not be able to identify perturbations by looking at the perturbed
    gene's own expression (which may be knocked out/down).

    Args:
        adata: AnnData object
        condition_col: Column with perturbation condition
        gene_name_col: Column in var with gene names
        mask_value: Value to set for masked genes

    Returns:
        AnnData with perturbed genes masked
    """
    adata = adata.copy()

    # Get gene names from var
    gene_names = adata.var[gene_name_col].values if gene_name_col in adata.var else adata.var.index.values
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Process each cell
    X = adata.X.copy()
    if hasattr(X, "toarray"):
        X = X.toarray()

    for i, condition in enumerate(adata.obs[condition_col]):
        # Parse perturbed genes from condition (format: GENE1+GENE2 or GENE+ctrl)
        genes = condition.split("+")
        for gene in genes:
            if gene != "ctrl" and gene in gene_to_idx:
                X[i, gene_to_idx[gene]] = mask_value

    adata.X = X
    return adata


def build_reference_library(
    adata: ad.AnnData,
    condition_col: str = "condition",
    exclude_control: bool = True,
) -> dict:
    """
    Build reference library of perturbation signatures.

    Args:
        adata: Pseudo-bulk AnnData (one row per condition)
        condition_col: Column with condition names
        exclude_control: Whether to exclude control from library

    Returns:
        Dictionary mapping condition -> expression vector
    """
    library = {}

    for i, row in adata.obs.iterrows():
        condition = row[condition_col]

        # Skip control if requested
        if exclude_control and condition == "ctrl":
            continue

        expr = adata[i].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        library[condition] = expr.flatten()

    return library
