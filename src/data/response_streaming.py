"""Memory-bounded streaming of X-Atlas-Orion perturbed cells.

Shards are read ahead on a few threads but consumed in sorted order through
one seeded Algorithm-R reservoir per perturbation. Only rows a reservoir
keeps are converted, and each is reduced at once to its whole-library UMI
total and its counts in STATE's HVG order, so no cell's full transcriptome
outlives its shard.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Final, NamedTuple, Sequence, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.sparse import csr_matrix

from src.data.basal import (
    XATLAS_GENE_METADATA_TOKEN_COL,
    ResponseCells,
    align_columns,
    read_xatlas_shard,
    require_raw_counts,
    xatlas_token_rows,
)
from src.data.expression import library_sizes

_LOGGER = logging.getLogger(__name__)

#: Parquet shards read ahead of the reservoir, one thread each.
PARQUET_PREFETCH_THREADS: Final[int] = 8

_T = TypeVar("_T")


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


def prefetched(
    read: Callable[[Path], _T], paths: Iterable[Path], threads: int
) -> Iterator[_T]:
    """``read(path)`` of each path in order, with at most ``threads`` reads ahead."""
    with ThreadPoolExecutor(max_workers=threads) as pool:
        pending: deque = deque()
        for path in paths:
            pending.append(pool.submit(read, path))
            if len(pending) == threads:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


class _ReducedCell(NamedTuple):
    """One reservoir cell: library size, HVG counts, tokens absent from metadata."""

    library_size: np.float64
    hvg_columns: np.ndarray
    hvg_counts: np.ndarray
    missing_tokens: np.ndarray


class _TokenSpace(NamedTuple):
    """Gene metadata tokens (ascending) and the symbol of each."""

    tokens: np.ndarray
    symbols: pd.Series


def _token_space(gene_metadata_path: Path, symbol_col: str) -> _TokenSpace:
    metadata = pd.read_parquet(gene_metadata_path).set_index(
        XATLAS_GENE_METADATA_TOKEN_COL
    )
    if metadata.empty or not metadata.index.is_unique:
        raise ValueError(
            f"{gene_metadata_path}: {XATLAS_GENE_METADATA_TOKEN_COL} must be "
            "non-empty and unique"
        )
    tokens = np.sort(np.asarray(metadata.index, dtype=np.int64))
    return _TokenSpace(tokens, metadata.loc[tokens, symbol_col].astype(str))


def _reduce_rows(
    table: pa.Table,
    rows: np.ndarray,
    space: _TokenSpace,
    gene_order: Sequence[str],
    label: str,
) -> list[_ReducedCell]:
    """Reduce ``table`` rows to library sizes and counts in ``gene_order``.

    The rows form one CSR over every metadata token, ascending, so each row
    holds the entries (and duplicate-token sums) a CSR over only the tokens
    in use would hold, in the same order.
    """
    cell, token, value = xatlas_token_rows(table, rows)
    position = np.minimum(np.searchsorted(space.tokens, token), len(space.tokens) - 1)
    known = space.tokens[position] == token
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell[known], minlength=len(rows)), out=indptr[1:])
    matrix = csr_matrix(
        (value[known], position[known], indptr),
        shape=(len(rows), len(space.tokens)),
    )
    matrix.sum_duplicates()
    require_raw_counts(matrix, label)
    sizes = library_sizes(matrix)
    hvg, _ = align_columns(matrix, space.symbols, gene_order)
    missing_cell, missing_token = cell[~known], token[~known]
    missing_bounds = np.searchsorted(missing_cell, np.arange(len(rows) + 1))
    return [
        _ReducedCell(
            library_size=sizes[index],
            hvg_columns=hvg.indices[hvg.indptr[index] : hvg.indptr[index + 1]].copy(),
            hvg_counts=hvg.data[hvg.indptr[index] : hvg.indptr[index + 1]].copy(),
            missing_tokens=missing_token[
                missing_bounds[index] : missing_bounds[index + 1]
            ].copy(),
        )
        for index in range(len(rows))
    ]


def _drain(
    reservoirs: dict[str, list[_ReducedCell]],
    n_genes: int,
    *,
    total_cells: int | None,
    seed: int,
    model_id: str,
) -> ResponseCells:
    """Cells in ``sorted(reservoirs)`` then slot order, ``total_cells`` applied.

    Every slot is released as it is consumed.
    """
    genes_order = sorted(reservoirs)
    keep = resolve_total_budget_keep_mask(
        sum(len(reservoirs[gene]) for gene in genes_order), total_cells, seed
    )
    labels: list[str] = []
    kept: list[_ReducedCell | None] = []
    index = 0
    for gene in genes_order:
        for cell in reservoirs.pop(gene):
            if keep[index]:
                labels.append(gene)
                kept.append(cell)
            index += 1
    indptr = np.zeros(len(kept) + 1, dtype=np.int64)
    np.cumsum([len(cell.hvg_columns) for cell in kept], out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    data = np.empty(int(indptr[-1]), dtype=np.float64)
    sizes = np.empty(len(kept), dtype=np.float64)
    missing: set[int] = set()
    for row, cell in enumerate(kept):
        indices[indptr[row] : indptr[row + 1]] = cell.hvg_columns
        data[indptr[row] : indptr[row + 1]] = cell.hvg_counts
        sizes[row] = cell.library_size
        missing.update(cell.missing_tokens.tolist())
        kept[row] = None
    if missing:
        _LOGGER.warning(
            "%s: dropped %d distinct gene tokens missing from the gene metadata index",
            model_id,
            len(missing),
        )
    hvg = csr_matrix((data, indices, indptr), shape=(len(labels), n_genes))
    return ResponseCells(
        labels=np.asarray(labels, dtype=object),
        library_sizes=sizes,
        hvg_counts=hvg,
        present=np.bincount(indices, minlength=n_genes) > 0,
    )


def read_xatlas_response_cells(
    shard_dir: Path,
    gene_metadata_path: Path,
    *,
    model_id: str,
    shard_glob: str,
    control_label: str,
    pass_guide_filter_value: int,
    symbol_col: str,
    gene_order: Sequence[str],
    max_cells_per_gene: int | None = None,
    total_cells_per_line: int | None = None,
    seed: int = 0,
) -> ResponseCells:
    """Perturbed, guide-QC-passing X-Atlas-Orion cells, capped per gene.

    One seeded Algorithm-R reservoir per ``gene_target`` runs over all shards
    in sorted order; tokens absent from the gene metadata are dropped. An HVG
    is ``present`` when a kept cell has a positive count for it.
    """
    paths = sorted(Path(shard_dir).glob(shard_glob))
    if not paths:
        raise ValueError(
            f"No parquet shards matching {shard_glob!r} found under {shard_dir}"
        )
    space = _token_space(gene_metadata_path, symbol_col)
    label = f"{model_id} response source"
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, list] = {}
    seen: dict[str, int] = {}
    counts = np.zeros(3, dtype=np.int64)
    read = partial(
        read_xatlas_shard,
        control_label=control_label,
        pass_guide_filter_value=pass_guide_filter_value,
    )
    for labels, rows, table, shard_counts in prefetched(
        read, paths, PARQUET_PREFETCH_THREADS
    ):
        counts += shard_counts
        # The slot each accepted row takes; a later row of the same shard
        # replacing it means the earlier one is never converted.
        accepted: dict[tuple[str, int], int] = {}
        for index, gene in enumerate(labels):
            bucket = reservoirs.setdefault(gene, [])
            seen[gene] = seen.get(gene, 0) + 1
            if max_cells_per_gene is None or len(bucket) < max_cells_per_gene:
                bucket.append(None)
                accepted[gene, len(bucket) - 1] = index
                continue
            replacement = int(rng.integers(0, seen[gene]))
            if replacement < max_cells_per_gene:
                accepted[gene, replacement] = index
        if accepted:
            cells = _reduce_rows(
                table, rows[list(accepted.values())], space, gene_order, label
            )
            for (gene, slot), cell in zip(accepted, cells, strict=True):
                reservoirs[gene][slot] = cell
    total, perturbed, passing = (int(value) for value in counts)
    _LOGGER.info(
        "X-Atlas-Orion %s (%s): %d rows, %d perturbed, %d pass guide QC",
        shard_dir,
        shard_glob,
        total,
        perturbed,
        passing,
    )
    if not reservoirs:
        raise ValueError(
            f"No perturbed cells found matching gene_target!={control_label!r} "
            f"under {shard_dir} ({shard_glob!r})"
        )
    return _drain(
        reservoirs,
        len(gene_order),
        total_cells=total_cells_per_line,
        seed=seed,
        model_id=model_id,
    )


__all__ = [
    "PARQUET_PREFETCH_THREADS",
    "prefetched",
    "read_xatlas_response_cells",
    "resolve_total_budget_keep_mask",
]
