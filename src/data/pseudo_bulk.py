"""
Pseudo-bulk analysis for retrieval stability.

Implements:
- Pseudo-bulk generation with configurable size
- Performance vs pseudo-bulk size curves
- Data efficiency analysis
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import numpy as np
import anndata as ad


def create_pseudo_bulk(
    adata: ad.AnnData,
    cells_per_bulk: int = 50,
    n_bulks: int = 1,
    condition_col: str = "condition",
    seed: int = 42,
    layer: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Create pseudo-bulk profiles by averaging cells.

    Args:
        adata: AnnData with cells
        cells_per_bulk: Number of cells to average per pseudo-bulk
        n_bulks: Number of pseudo-bulks to create per condition
        condition_col: Column with condition labels
        seed: Random seed
        layer: Optional layer to use instead of X

    Returns:
        Tuple of (bulk_profiles, condition_labels)
    """
    rng = np.random.default_rng(seed)

    if layer and layer != "X":
        if layer not in adata.layers:
            if layer == "counts":
                print(
                    "  [PseudoBulk] Layer 'counts' not found in AnnData.layers; using X."
                )
                X = adata.X
            else:
                raise ValueError(f"Layer '{layer}' not found in AnnData.layers")
        else:
            X = adata.layers[layer]
    else:
        X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    conditions = adata.obs[condition_col].unique()
    conditions = [c for c in conditions if c != "ctrl"]

    all_bulks = []
    all_labels = []

    for cond in conditions:
        mask = adata.obs[condition_col] == cond
        cond_X = X[mask]
        n_cells = cond_X.shape[0]

        if n_cells < cells_per_bulk:
            # Use all cells if fewer than requested
            bulk = np.mean(cond_X, axis=0)
            all_bulks.append(bulk.flatten())
            all_labels.append(cond)
        else:
            # Create n_bulks pseudo-bulks
            for _ in range(n_bulks):
                indices = rng.choice(n_cells, size=cells_per_bulk, replace=False)
                bulk = np.mean(cond_X[indices], axis=0)
                all_bulks.append(bulk.flatten())
                all_labels.append(cond)

    if not all_bulks:
        return np.array([]), []

    return np.vstack(all_bulks), all_labels


def compute_pseudo_bulk_stability(
    adata: ad.AnnData,
    encoder,
    cells_per_bulk_values: List[int] = [10, 25, 50, 100, 200],
    n_bulks: int = 10,
    n_repeats: int = 5,
    condition_col: str = "condition",
    seed: int = 42,
) -> Dict[int, Dict[str, float]]:
    """
    Compute embedding stability across different pseudo-bulk sizes.

    Stability is measured as average pairwise cosine similarity between
    pseudo-bulks of the same condition.

    Args:
        adata: AnnData with cells
        encoder: Fitted encoder with encode() method
        cells_per_bulk_values: List of cells-per-bulk to test
        n_bulks: Number of bulks per condition per repeat
        n_repeats: Number of random repeats
        condition_col: Column with condition labels
        seed: Random seed

    Returns:
        Dict mapping cells_per_bulk -> {mean_stability, std_stability}
    """
    rng = np.random.default_rng(seed)
    results = {}

    for cells_per_bulk in cells_per_bulk_values:
        stabilities = []

        for repeat in range(n_repeats):
            # Create pseudo-bulks
            bulks, labels = create_pseudo_bulk(
                adata,
                cells_per_bulk=cells_per_bulk,
                n_bulks=n_bulks,
                condition_col=condition_col,
                seed=seed + repeat,
            )

            if len(bulks) == 0:
                continue

            # Encode
            embeddings = encoder.encode(bulks)

            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            embeddings = embeddings / norms

            # Compute within-condition similarity
            unique_conds = list(set(labels))
            for cond in unique_conds:
                mask = [l == cond for l in labels]
                cond_embs = embeddings[mask]

                if cond_embs.shape[0] < 2:
                    continue

                # Pairwise cosine similarity
                sim_matrix = cond_embs @ cond_embs.T
                # Get upper triangle (excluding diagonal)
                n = sim_matrix.shape[0]
                upper_tri = sim_matrix[np.triu_indices(n, k=1)]
                stabilities.extend(upper_tri.tolist())

        if stabilities:
            results[cells_per_bulk] = {
                "mean_stability": float(np.mean(stabilities)),
                "std_stability": float(np.std(stabilities)),
                "n_samples": len(stabilities),
            }
        else:
            results[cells_per_bulk] = {
                "mean_stability": 0.0,
                "std_stability": 0.0,
                "n_samples": 0,
            }

    return results


def generate_stability_curve_data(
    stability_results: Dict[int, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert stability results to plottable arrays.

    Args:
        stability_results: Output from compute_pseudo_bulk_stability

    Returns:
        Tuple of (cells_per_bulk, mean_stability, std_stability)
    """
    sorted_keys = sorted(stability_results.keys())

    x = np.array(sorted_keys)
    means = np.array([stability_results[k]["mean_stability"] for k in sorted_keys])
    stds = np.array([stability_results[k]["std_stability"] for k in sorted_keys])

    return x, means, stds
