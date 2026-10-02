"""Raw-count helpers and memory-bounded readers of Perturb-seq sources.

The readers return raw non-negative UMI counts: whole-library totals and
counts in a given gene order, reduced one row chunk (or one parquet shard) at
a time so a full-transcriptome matrix is never held. Expression-space
transforms happen in ``src.data.expression``, never here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, NamedTuple, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from scipy.sparse import csr_matrix

from src.data.expression import library_sizes

_LOGGER = logging.getLogger(__name__)

_ENSEMBL_ID_PATTERN = re.compile(r"^ENSG\d+(\.\d+)?$")
_SYMBOL_COLUMNS: Final[tuple[str, ...]] = ("gene_symbol", "gene_symbols", "gene_name")

# X-Atlas-Orion parquet schema: one row per cell, parallel token/value arrays.
_XATLAS_GENE_TOKEN_COL: Final[str] = "gene_token_id"
_XATLAS_EXPRESSION_COL: Final[str] = "gene_expression"
_XATLAS_CELL_BARCODE_COL: Final[str] = "cell_barcode"
_XATLAS_SAMPLE_COL: Final[str] = "sample"
_XATLAS_PERTURBATION_COL: Final[str] = "gene_target"
_XATLAS_PASS_GUIDE_FILTER_COL: Final[str] = "pass_guide_filter"
_XATLAS_READ_COLUMNS: Final[tuple[str, ...]] = (
    _XATLAS_GENE_TOKEN_COL,
    _XATLAS_EXPRESSION_COL,
    _XATLAS_CELL_BARCODE_COL,
    _XATLAS_SAMPLE_COL,
    _XATLAS_PERTURBATION_COL,
    _XATLAS_PASS_GUIDE_FILTER_COL,
)
XATLAS_GENE_METADATA_TOKEN_COL: Final[str] = "gene_token_id"

#: Rows per backed read; one huge fancy index over a dense unchunked HDF5
#: dataset can spin for tens of minutes before reading a byte.
_MATERIALIZE_CHUNK_ROWS: Final[int] = 2000


@dataclass(frozen=True)
class ResponseCells:
    """Perturbed cells of one anchor, reduced to what response targets need.

    ``labels`` are the raw perturbation labels, ``library_sizes`` each cell's
    UMI total over every gene of its source (float64) and ``hvg_counts`` its
    raw counts in a given gene order (float64 CSR; source columns sharing a
    symbol summed, absent genes zero and ``False`` in ``present``).
    """

    labels: np.ndarray
    library_sizes: np.ndarray
    hvg_counts: csr_matrix
    present: np.ndarray


# --- raw-count helpers -------------------------------------------------------


def require_raw_counts(matrix: object, label: str) -> None:
    """Raise unless ``matrix`` holds finite, non-negative integer UMI counts.

    An already normalised or log-transformed source would otherwise be
    transformed a second time without any error.
    """
    data = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if data.size and (
        not np.isfinite(data).all()
        or (data < 0).any()
        or not np.equal(data, np.floor(data)).all()
    ):
        raise ValueError(
            f"{label}: expression must be raw integer UMI counts (found non-integer, "
            "negative or non-finite values; the source may already be normalised)"
        )


def symbol_column(var: pd.DataFrame, configured: str) -> str:
    """Resolve the ``var`` gene-symbol column; ``auto`` picks the one present."""
    if configured == "auto":
        candidates = [name for name in _SYMBOL_COLUMNS if name in var.columns]
        if len(candidates) != 1:
            raise ValueError(
                f"var needs exactly one gene-symbol column of {_SYMBOL_COLUMNS}, "
                f"found {candidates}"
            )
        return candidates[0]
    if configured not in var.columns:
        raise ValueError(f"var is missing gene-symbol column {configured!r}")
    return configured


def align_columns(
    matrix: object, symbols: Sequence[str], order: Sequence[str]
) -> tuple[csr_matrix, np.ndarray]:
    """Re-express columns in ``order`` by exact symbol match.

    Columns sharing a symbol are summed; genes of ``order`` absent from
    ``symbols`` are zero columns and ``False`` in the returned mask.
    """
    position = {str(gene): index for index, gene in enumerate(order)}
    target = np.asarray([position.get(str(symbol), -1) for symbol in symbols])
    keep = np.flatnonzero(target >= 0)
    mapper = csr_matrix(
        (np.ones(len(keep), dtype=np.float64), (keep, target[keep])),
        shape=(len(target), len(position)),
    )
    source = (
        csr_matrix(matrix)
        if sparse.issparse(matrix)
        else csr_matrix(np.asarray(matrix))
    )
    present = np.zeros(len(position), dtype=bool)
    present[target[keep]] = True
    return (source @ mapper).tocsr(), present


def assert_tx1_input_contract(adata: ad.AnnData) -> None:
    """Raw non-negative finite ``.X``, Ensembl ``var`` index, ``cell_type`` set."""
    matrix = adata.X
    data = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if data.size and not np.all(np.isfinite(data)):
        raise ValueError(
            "Tx1 input contract violation: .X contains non-finite values (NaN or inf)"
        )
    if data.size and np.any(data < 0):
        raise ValueError("Tx1 input contract violation: .X contains negative values")
    if "cell_type" not in adata.obs.columns:
        raise ValueError("Tx1 input contract violation: obs is missing 'cell_type'")
    index = adata.var.index.astype(str)
    if not len(index) or not all(_ENSEMBL_ID_PATTERN.match(value) for value in index):
        raise ValueError(
            "Tx1 input contract violation: var.index must be Ensembl gene ids "
            f"matching {_ENSEMBL_ID_PATTERN.pattern!r}"
        )
    if "ensembl_id" not in adata.var.columns:
        raise ValueError(
            "Tx1 input contract violation: var is missing the 'ensembl_id' column "
            "that the Tx1 encoder reads"
        )


# --- backed h5ad reads ---------------------------------------------------------


def _require_ensembl_source(
    var: pd.DataFrame, var_ensembl_col: str, h5ad_path: Path
) -> None:
    """Ensembl ids must be a ``var`` column or the named ``var`` index."""
    if var_ensembl_col in var.columns or var.index.name == var_ensembl_col:
        return
    raise ValueError(
        f"{h5ad_path} var has neither a column nor an index named "
        f"{var_ensembl_col!r}; available columns: {sorted(var.columns)}, "
        f"index name: {var.index.name!r}"
    )


def _select_indices_deterministic(
    candidate_indices: np.ndarray, max_cells: int | None, seed: int
) -> np.ndarray:
    """Seeded subsample of row indices (sorted), or all when uncapped."""
    if max_cells is None or max_cells >= len(candidate_indices):
        return candidate_indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidate_indices), size=max_cells, replace=False)
    return np.sort(candidate_indices[chosen])


def _row_chunks(
    matrix: object,
    sorted_indices: np.ndarray,
    *,
    chunk_size: int = _MATERIALIZE_CHUNK_ROWS,
    sparsify_chunks: bool = False,
) -> Iterator[object]:
    """Rows ``sorted_indices`` (ascending) of a backed matrix, one chunk at a time.

    ``sparsify_chunks`` converts each dense chunk to CSR before the next is read.
    """
    for start in range(0, len(sorted_indices), chunk_size):
        chunk = matrix[sorted_indices[start : start + chunk_size], :]
        if sparsify_chunks and not sparse.issparse(chunk):
            chunk = csr_matrix(chunk)
        yield chunk


def _materialize_rows(
    matrix: object,
    row_indices: np.ndarray,
    *,
    chunk_size: int = _MATERIALIZE_CHUNK_ROWS,
    sparsify_chunks: bool = False,
) -> np.ndarray | csr_matrix:
    """Read ``row_indices`` (any order) from a backed matrix in sorted chunks.

    Returns the rows in the caller's order. ``sparsify_chunks`` converts each
    dense chunk to CSR as it is read, bounding the dense footprint.
    """
    order = np.argsort(row_indices)
    sorted_indices = row_indices[order]
    inverse = np.argsort(order)
    if not sorted_indices.size:
        stacked = matrix[sorted_indices, :]
    else:
        chunks = list(
            _row_chunks(
                matrix,
                sorted_indices,
                chunk_size=chunk_size,
                sparsify_chunks=sparsify_chunks,
            )
        )
        if sparse.issparse(chunks[0]):
            stacked = sparse.vstack(chunks, format="csr")
        else:
            stacked = np.concatenate([np.asarray(chunk) for chunk in chunks], axis=0)
    if sparse.issparse(stacked):
        return stacked.tocsr()[inverse, :]
    return np.asarray(stacked)[inverse]


def _group_candidate_indices_by_label(
    candidate_indices: np.ndarray, labels: np.ndarray
) -> dict[str, np.ndarray]:
    """Group co-indexed candidate rows by label; each group stays ascending."""
    frame = pd.DataFrame(
        {"index": candidate_indices, "label": np.asarray(labels).astype(str)}
    )
    return {
        str(label): group["index"].to_numpy(dtype=np.int64)
        for label, group in frame.groupby("label", sort=True)
    }


def _cap_indices_by_group(
    grouped_indices: Mapping[str, np.ndarray], max_per_group: int | None, seed: int
) -> np.ndarray:
    """Cap each group in sorted-key order, one shared sub-seed stream."""
    if not grouped_indices:
        return np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = [
        _select_indices_deterministic(
            grouped_indices[key], max_per_group, int(rng.integers(0, 2**32))
        )
        for key in sorted(grouped_indices)
    ]
    return np.concatenate(selected)


def perturbseq_control_library_sizes(
    h5ad_path: Path,
    *,
    control_label: str,
    perturbation_col: str,
    var_ensembl_col: str,
    label: str,
) -> np.ndarray:
    """Whole-library UMI totals of a Perturb-seq h5ad's control cells, in row order."""
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        _require_ensembl_source(backed.var, var_ensembl_col, h5ad_path)
        labels = backed.obs[perturbation_col].astype(str).to_numpy()
        control = np.flatnonzero(labels == control_label)
        if not control.size:
            raise ValueError(
                f"No control cells found for {perturbation_col}={control_label!r} "
                f"in {h5ad_path}"
            )
        sizes = []
        for chunk in _row_chunks(backed.X, control, sparsify_chunks=True):
            chunk = csr_matrix(chunk)
            require_raw_counts(chunk, label)
            sizes.append(library_sizes(chunk))
    finally:
        backed.file.close()
    return np.concatenate(sizes)


def read_perturbseq_response_cells(
    h5ad_path: Path,
    *,
    control_label: str,
    perturbation_col: str,
    var_ensembl_col: str,
    symbol_col: str,
    gene_order: Sequence[str],
    label: str,
    max_cells_per_gene: int | None = None,
    total_cells_per_line: int | None = None,
    seed: int = 0,
) -> ResponseCells:
    """Perturbed cells of a Perturb-seq h5ad, columns aligned to ``gene_order``.

    The per-gene cap is applied before any expression value is read; the
    optional total cap follows it. Rows come out grouped by sorted label.
    Each row chunk is reduced to library sizes and aligned columns before the
    next is read.
    """
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        _require_ensembl_source(backed.var, var_ensembl_col, h5ad_path)
        labels = backed.obs[perturbation_col].astype(str).to_numpy()
        candidate = np.flatnonzero(labels != control_label)
        if not candidate.size:
            raise ValueError(
                f"No perturbed cells found for {perturbation_col}!={control_label!r} "
                f"in {h5ad_path}"
            )
        grouped = _group_candidate_indices_by_label(candidate, labels[candidate])
        selected = _cap_indices_by_group(grouped, max_cells_per_gene, seed)
        selected = _select_indices_deterministic(selected, total_cells_per_line, seed)
        symbols = backed.var[symbol_col].astype(str)
        order = np.argsort(selected)
        sizes, counts = [], []
        for chunk in _row_chunks(backed.X, selected[order], sparsify_chunks=True):
            chunk = csr_matrix(chunk)
            require_raw_counts(chunk, label)
            sizes.append(library_sizes(chunk))
            aligned, present = align_columns(chunk, symbols, gene_order)
            counts.append(aligned)
    finally:
        backed.file.close()
    inverse = np.argsort(order)
    return ResponseCells(
        labels=labels[selected],
        library_sizes=np.concatenate(sizes)[inverse],
        hvg_counts=sparse.vstack(counts, format="csr")[inverse, :],
        present=present,
    )


# --- X-Atlas-Orion parquet shards ----------------------------------------------


class _XatlasCell(NamedTuple):
    """One X-Atlas-Orion cell, filtered to positive-value gene tokens."""

    genes: np.ndarray
    values: np.ndarray
    cell_barcode: str
    sample: str


def _filter_xatlas_shard(
    path: Path,
    control_label: str,
    pass_guide_filter_value: int,
    *,
    perturbed: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Read one shard; keep control (or perturbed) rows passing guide QC.

    Returns the frame and ``[total, after_label_filter, after_guide_filter]``.
    """
    frame = pd.read_parquet(path, columns=list(_XATLAS_READ_COLUMNS))
    total = len(frame)
    is_control = frame[_XATLAS_PERTURBATION_COL].astype(str) == control_label
    frame = frame[~is_control if perturbed else is_control]
    after_label = len(frame)
    frame = frame[
        frame[_XATLAS_PASS_GUIDE_FILTER_COL].astype(int) == pass_guide_filter_value
    ]
    return frame, np.array([total, after_label, len(frame)], dtype=np.int64)


def _row_to_xatlas_cell(row: object) -> _XatlasCell:
    """X-Atlas tokens reserve no special ids; only positive values are kept."""
    genes = np.asarray(getattr(row, _XATLAS_GENE_TOKEN_COL), dtype=np.int64)
    values = np.asarray(getattr(row, _XATLAS_EXPRESSION_COL), dtype=np.float32)
    valid = values > 0
    return _XatlasCell(
        genes=genes[valid],
        values=values[valid],
        cell_barcode=str(row.cell_barcode),
        sample=str(getattr(row, _XATLAS_SAMPLE_COL)),
    )


def _assemble_tahoe_matrix(
    reservoir: list[tuple[np.ndarray, np.ndarray]], metadata: pd.DataFrame
) -> tuple[csr_matrix, pd.DataFrame]:
    """CSR matrix and Ensembl ``var`` for Tahoe-100M token/value cells."""
    return _assemble_token_matrix(
        reservoir, metadata, metadata_var_columns=("ensembl_id", "gene_symbol")
    )


def _assemble_token_matrix(
    reservoir: list[tuple[np.ndarray, np.ndarray]],
    metadata: pd.DataFrame,
    *,
    metadata_var_columns: Sequence[str],
) -> tuple[csr_matrix, pd.DataFrame]:
    """CSR matrix from per-cell token/value arrays via a token-indexed table.

    Tokens absent from ``metadata`` are dropped with a warning; the first of
    ``metadata_var_columns`` (the Ensembl id) becomes ``var.index``.
    """
    valid_tokens = {int(token) for genes, _ in reservoir for token in genes.tolist()}
    tokens = sorted(token for token in valid_tokens if token in metadata.index)
    n_dropped = len(valid_tokens) - len(tokens)
    if n_dropped:
        _LOGGER.warning(
            "dropped %d/%d (%.1f%%) otherwise-valid gene tokens missing from "
            "gene metadata index",
            n_dropped,
            len(valid_tokens),
            100.0 * n_dropped / len(valid_tokens),
        )
    positions = {token: index for index, token in enumerate(tokens)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row_index, (genes, counts) in enumerate(reservoir):
        for token, count in zip(genes.tolist(), counts.tolist(), strict=True):
            position = positions.get(int(token))
            if position is not None:
                rows.append(row_index)
                columns.append(position)
                values.append(count)
    matrix = csr_matrix((values, (rows, columns)), shape=(len(reservoir), len(tokens)))
    var = metadata.loc[tokens, list(metadata_var_columns)].copy()
    var.index = var[metadata_var_columns[0]].astype(str)
    return matrix, var


def read_xatlas_shard(
    path: Path, control_label: str, pass_guide_filter_value: int
) -> tuple[list[str], np.ndarray, pa.Table, np.ndarray]:
    """Perturbed, guide-QC-passing rows of one shard, token lists left in Arrow.

    Applies :func:`_filter_xatlas_shard`'s filters to the label and QC columns
    only. Returns ``(labels, rows, table, [total, after_label_filter,
    after_guide_filter])``: ``rows[i]`` is the ``table`` row (token and value
    lists) of ``labels[i]``.
    """
    table = pq.read_table(
        path,
        columns=[
            _XATLAS_PERTURBATION_COL,
            _XATLAS_PASS_GUIDE_FILTER_COL,
            _XATLAS_GENE_TOKEN_COL,
            _XATLAS_EXPRESSION_COL,
        ],
    )
    frame = table.select(
        [_XATLAS_PERTURBATION_COL, _XATLAS_PASS_GUIDE_FILTER_COL]
    ).to_pandas()
    target = frame[_XATLAS_PERTURBATION_COL]
    perturbed = np.flatnonzero((target.astype(str) != control_label).to_numpy())
    guide = frame[_XATLAS_PASS_GUIDE_FILTER_COL].iloc[perturbed].astype(int)
    rows = perturbed[(guide == pass_guide_filter_value).to_numpy()]
    labels = [str(value) for value in target.iloc[rows]]
    counts = np.array([len(frame), len(perturbed), len(rows)], dtype=np.int64)
    return (
        labels,
        rows,
        table.select([_XATLAS_GENE_TOKEN_COL, _XATLAS_EXPRESSION_COL]),
        counts,
    )


def xatlas_token_rows(table: pa.Table, rows: np.ndarray) -> tuple[np.ndarray, ...]:
    """``(cell, token, value)`` of every positive entry of ``table`` rows ``rows``.

    ``cell`` indexes ``rows``; entries keep their order within each cell.
    X-Atlas tokens reserve no special ids; only positive values are kept.
    """
    subset = table.take(pa.array(rows, type=pa.int64()))
    tokens = subset.column(_XATLAS_GENE_TOKEN_COL).combine_chunks()
    values = subset.column(_XATLAS_EXPRESSION_COL).combine_chunks()
    lengths = np.asarray(tokens.value_lengths(), dtype=np.int64)
    if not np.array_equal(lengths, np.asarray(values.value_lengths(), dtype=np.int64)):
        raise ValueError("X-Atlas-Orion token and value lists differ in length")
    cell = np.repeat(np.arange(len(rows)), lengths)
    token = np.asarray(tokens.flatten().to_numpy(zero_copy_only=False), dtype=np.int64)
    value = np.asarray(
        values.flatten().to_numpy(zero_copy_only=False), dtype=np.float32
    )
    positive = value > 0
    return cell[positive], token[positive], value[positive]
