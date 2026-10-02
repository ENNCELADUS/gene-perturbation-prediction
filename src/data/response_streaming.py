"""Memory-bounded assembly of per-gene X-Atlas-Orion reservoirs into CSR.

A genome-scale line keeps millions of reservoir cells resident at once, so the
CSR arrays are built with vectorized numpy and every reservoir slot is
released as soon as it is consumed. Cells unpack to
``(genes, values, barcode, sample)``.
"""

from __future__ import annotations

import logging
from typing import Mapping, MutableSequence, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

_LOGGER = logging.getLogger(__name__)


def resolve_total_budget_keep_mask(
    n_cells: int, total_cells: int | None, seed: int
) -> np.ndarray:
    """Seeded keep-mask over ``n_cells`` rows; ``None`` keeps every row."""
    if total_cells is None or total_cells >= n_cells:
        return np.ones(n_cells, dtype=bool)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_cells, size=total_cells, replace=False)
    mask = np.zeros(n_cells, dtype=bool)
    mask[chosen] = True
    return mask


def drain_gene_reservoirs_to_matrix(
    reservoirs: Mapping[str, MutableSequence[Sequence[object] | None]],
    metadata: pd.DataFrame,
    *,
    metadata_var_columns: Sequence[str],
    total_cells: int | None = None,
    seed: int = 0,
) -> tuple[csr_matrix, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build CSR from ``{gene: [cell, ...]}``, setting each slot to ``None``.

    Rows follow ``sorted(reservoirs)`` then slot order; ``total_cells``
    optionally downsamples them. Tokens absent from ``metadata`` are dropped
    with a warning. Returns ``(matrix, var, perturbation_genes, barcodes,
    samples)``.
    """
    genes_order = sorted(reservoirs)
    n_cells_before_budget = sum(len(reservoirs[gene]) for gene in genes_order)
    keep_mask = resolve_total_budget_keep_mask(n_cells_before_budget, total_cells, seed)

    valid_lengths: list[int] = []
    perturbation_genes: list[str] = []
    barcodes: list[str] = []
    samples: list[str] = []
    metadata_index = metadata.index
    metadata_token_used = np.zeros(len(metadata_index), dtype=bool)
    missing_tokens: set[int] = set()
    global_index = 0
    for gene in genes_order:
        for cell in reservoirs[gene]:
            if keep_mask[global_index]:
                cell_genes = cell[0]
                perturbation_genes.append(gene)
                barcodes.append(cell[2])
                samples.append(cell[3])
                positions = metadata_index.get_indexer(cell_genes)
                valid_entries = positions >= 0
                metadata_token_used[positions[valid_entries]] = True
                valid_lengths.append(int(valid_entries.sum()))
                if not valid_entries.all():
                    missing_tokens.update(
                        int(token) for token in np.unique(cell_genes[~valid_entries])
                    )
            global_index += 1

    n_cells = len(valid_lengths)
    token_lookup = np.sort(
        np.asarray(metadata_index[metadata_token_used], dtype=np.int64)
    )
    n_dropped = len(missing_tokens)
    if n_dropped:
        n_distinct_tokens = len(token_lookup) + n_dropped
        _LOGGER.warning(
            "dropped %d/%d (%.1f%%) otherwise-valid gene tokens missing from "
            "gene metadata index",
            n_dropped,
            n_distinct_tokens,
            100.0 * n_dropped / n_distinct_tokens,
        )
    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(valid_lengths, out=indptr[1:])
    total_nnz = int(indptr[-1])
    out_cols = np.empty(total_nnz, dtype=np.int64)
    out_values = np.empty(total_nnz, dtype=np.float32)

    offset = 0
    global_index = 0
    for gene in genes_order:
        bucket = reservoirs[gene]
        for slot in range(len(bucket)):
            cell = bucket[slot]
            # Drain unconditionally -- an unselected cell's memory is freed
            # exactly like a selected one's, since keep_mask only decides
            # what feeds the OUTPUT, not what stays resident.
            bucket[slot] = None
            keep = keep_mask[global_index]
            global_index += 1
            if not keep:
                continue
            cell_genes, cell_values = cell[0], cell[1]
            if token_lookup.size:
                positions = np.searchsorted(token_lookup, cell_genes)
                positions_clipped = np.clip(positions, 0, len(token_lookup) - 1)
                valid_entries = token_lookup[positions_clipped] == cell_genes
            else:
                valid_entries = np.zeros(len(cell_genes), dtype=bool)
            n_valid = int(valid_entries.sum())
            if n_valid:
                out_cols[offset : offset + n_valid] = positions_clipped[valid_entries]
                out_values[offset : offset + n_valid] = cell_values[valid_entries]
            offset += n_valid

    matrix = csr_matrix(
        (out_values, out_cols, indptr), shape=(n_cells, len(token_lookup))
    )
    matrix.sum_duplicates()
    var = metadata.loc[token_lookup, list(metadata_var_columns)].copy()
    var.index = var[metadata_var_columns[0]].astype(str)
    return (
        matrix,
        var,
        np.asarray(perturbation_genes, dtype=object),
        np.asarray(barcodes, dtype=object),
        np.asarray(samples, dtype=object),
    )


__all__ = [
    "drain_gene_reservoirs_to_matrix",
    "resolve_total_budget_keep_mask",
]
