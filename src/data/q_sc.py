"""Per-gene basal expression summaries ``q_sc`` of one cell line."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from src.data.basal import align_columns


@dataclass(frozen=True)
class QScFeatures:
    """Log-space mean, detected fraction and log-space variance per gene.

    ``values`` is ``[genes, 3]`` float32; rows of genes absent from the
    line's source are NaN with ``available`` False.
    """

    symbols: tuple[str, ...]
    values: np.ndarray
    available: np.ndarray


def compute_q_sc(
    matrix: object,
    symbols: Sequence[str],
    genes: Sequence[str],
    library_size: np.ndarray,
    target_sum: float,
) -> QScFeatures:
    """Summarise ``log1p(x * target_sum / library_size)`` over every cell.

    ``matrix`` holds raw counts over the source's genes (named by
    ``symbols``); columns sharing a symbol are summed before the transform.
    The detected fraction is the fraction of cells with a nonzero count.
    """
    counts, available = align_columns(matrix, symbols, genes)
    counts.eliminate_zeros()
    n_cells = counts.shape[0]
    detected = np.diff(counts.tocsc().indptr) / n_cells
    scale = np.zeros(n_cells, dtype=np.float64)
    nonzero = library_size > 0
    scale[nonzero] = target_sum / library_size[nonzero]
    logged = sparse.csr_matrix(sparse.diags(scale) @ counts)
    logged.data = np.log1p(logged.data)
    mean = np.asarray(logged.sum(axis=0)).ravel() / n_cells
    square = np.asarray(logged.multiply(logged).sum(axis=0)).ravel() / n_cells
    variance = np.maximum(square - mean**2, 0.0)
    values = np.column_stack([mean, detected, variance]).astype(np.float32)
    values[~available] = np.nan
    return QScFeatures(symbols=tuple(genes), values=values, available=available)
