"""Raw-count AnnData builders for Perturb-seq sources, and raw-count helpers.

Every builder returns raw non-negative UMI counts with ``var`` indexed by
Ensembl id (the Tx1 input contract). Expression-space transforms happen in
``src.data.expression``, never here.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final, Mapping, NamedTuple, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

from src.data.response_streaming import drain_gene_reservoirs_to_matrix

_LOGGER = logging.getLogger(__name__)

_PERTURBSEQ_BASAL_SOURCE = "Perturb-seq non-targeting control"
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
_XATLAS_CONTROL_LABEL: Final[str] = "Non-Targeting"
_XATLAS_PASS_GUIDE_FILTER_VALUE: Final[int] = 1
_XATLAS_GENE_METADATA_TOKEN_COL: Final[str] = "gene_token_id"
_XATLAS_GENE_METADATA_VAR_COLUMNS: Final[tuple[str, str]] = ("ensembl_id", "gene_name")

#: Rows per backed read; one huge fancy index over a dense unchunked HDF5
#: dataset can spin for tens of minutes before reading a byte.
_MATERIALIZE_CHUNK_ROWS: Final[int] = 2000


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


def _ensembl_var(var: pd.DataFrame, var_ensembl_col: str) -> pd.DataFrame:
    var = var.copy()
    if var_ensembl_col in var.columns:
        var.index = var[var_ensembl_col].astype(str)
    var["ensembl_id"] = var.index.astype(str)
    return var


def _select_indices_deterministic(
    candidate_indices: np.ndarray, max_cells: int | None, seed: int
) -> np.ndarray:
    """Seeded subsample of row indices (sorted), or all when uncapped."""
    if max_cells is None or max_cells >= len(candidate_indices):
        return candidate_indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidate_indices), size=max_cells, replace=False)
    return np.sort(candidate_indices[chosen])


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
        chunks = [
            matrix[sorted_indices[start : start + chunk_size], :]
            for start in range(0, len(sorted_indices), chunk_size)
        ]
        if sparsify_chunks:
            chunks = [
                chunk if sparse.issparse(chunk) else csr_matrix(chunk)
                for chunk in chunks
            ]
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


def _labelled_adata(
    matrix: csr_matrix,
    var: pd.DataFrame,
    obs_names: Sequence[str],
    *,
    model_id: str,
    columns: Mapping[str, object],
) -> ad.AnnData:
    obs = pd.DataFrame(
        {"cell_type": model_id, "model_id": model_id, **columns}, index=list(obs_names)
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    assert_tx1_input_contract(adata)
    return adata


def build_perturbseq_basal_adata(
    h5ad_path: Path,
    *,
    control_label: str,
    perturbation_col: str,
    model_id: str,
    var_ensembl_col: str,
    max_cells: int | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Non-targeting control cells of a Perturb-seq h5ad, over every gene."""
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
        selected = _select_indices_deterministic(control, max_cells, seed)
        matrix = csr_matrix(_materialize_rows(backed.X, selected, sparsify_chunks=True))
        obs_names = backed.obs_names.to_numpy()[selected].astype(str)
        var = _ensembl_var(backed.var, var_ensembl_col)
    finally:
        backed.file.close()
    return _labelled_adata(
        matrix,
        var,
        obs_names,
        model_id=model_id,
        columns={"basal_source": _PERTURBSEQ_BASAL_SOURCE},
    )


def build_perturbseq_response_adata(
    h5ad_path: Path,
    *,
    control_label: str,
    perturbation_col: str,
    model_id: str,
    var_ensembl_col: str,
    max_cells_per_gene: int | None = None,
    total_cells_per_line: int | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Perturbed cells of a Perturb-seq h5ad with ``obs["perturbation_gene"]``.

    The per-gene cap is applied before any expression value is read; the
    optional total cap follows it. Rows come out grouped by sorted label.
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
        matrix = csr_matrix(_materialize_rows(backed.X, selected, sparsify_chunks=True))
        obs_names = backed.obs_names.to_numpy()[selected].astype(str)
        var = _ensembl_var(backed.var, var_ensembl_col)
    finally:
        backed.file.close()
    adata = _labelled_adata(
        matrix,
        var,
        obs_names,
        model_id=model_id,
        columns={"perturbation_gene": labels[selected]},
    )
    _LOGGER.info(
        "%s: %d perturbed cells, %d perturbations",
        model_id,
        adata.n_obs,
        len(set(labels[selected])),
    )
    return adata


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


def build_xatlas_orion_response_adata(
    shard_dir: Path,
    gene_metadata_path: Path,
    *,
    model_id: str,
    shard_glob: str = "*.parquet",
    control_label: str = _XATLAS_CONTROL_LABEL,
    pass_guide_filter_value: int = _XATLAS_PASS_GUIDE_FILTER_VALUE,
    max_cells_per_gene: int | None = None,
    total_cells_per_line: int | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Perturbed, guide-QC-passing X-Atlas-Orion cells, capped per gene.

    Shards are streamed through one Algorithm-R reservoir per perturbation;
    the reservoir is drained straight into CSR to bound peak memory.
    """
    reservoirs = _stream_xatlas_response_cells(
        shard_dir,
        shard_glob,
        control_label=control_label,
        pass_guide_filter_value=pass_guide_filter_value,
        max_cells_per_gene=max_cells_per_gene,
        seed=seed,
    )
    metadata = pd.read_parquet(gene_metadata_path).set_index(
        _XATLAS_GENE_METADATA_TOKEN_COL
    )
    matrix, var, perturbation_genes, barcodes, samples = (
        drain_gene_reservoirs_to_matrix(
            reservoirs,
            metadata,
            metadata_var_columns=_XATLAS_GENE_METADATA_VAR_COLUMNS,
            total_cells=total_cells_per_line,
            seed=seed,
        )
    )
    adata = _labelled_adata(
        matrix,
        var,
        [f"{sample}:{barcode}" for sample, barcode in zip(samples, barcodes)],
        model_id=model_id,
        columns={"perturbation_gene": perturbation_genes, "sample": samples},
    )
    _LOGGER.info(
        "%s: %d perturbed cells, %d perturbations",
        model_id,
        adata.n_obs,
        len(set(perturbation_genes.tolist())),
    )
    return adata


def _stream_xatlas_response_cells(
    shard_dir: Path,
    shard_glob: str,
    *,
    control_label: str,
    pass_guide_filter_value: int,
    max_cells_per_gene: int | None = None,
    seed: int = 0,
) -> dict[str, list[_XatlasCell]]:
    """One seeded Algorithm-R reservoir per ``gene_target`` over all shards."""
    paths = sorted(Path(shard_dir).glob(shard_glob))
    if not paths:
        raise ValueError(
            f"No parquet shards matching {shard_glob!r} found under {shard_dir}"
        )
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, list[_XatlasCell]] = {}
    seen: dict[str, int] = {}
    counts = np.zeros(3, dtype=np.int64)
    for path in paths:
        frame, shard_counts = _filter_xatlas_shard(
            path, control_label, pass_guide_filter_value, perturbed=True
        )
        counts += shard_counts
        for row in frame.itertuples(index=False):
            gene = str(getattr(row, _XATLAS_PERTURBATION_COL))
            cell = _row_to_xatlas_cell(row)
            bucket = reservoirs.setdefault(gene, [])
            seen[gene] = seen.get(gene, 0) + 1
            if max_cells_per_gene is None or len(bucket) < max_cells_per_gene:
                bucket.append(cell)
                continue
            replacement = int(rng.integers(0, seen[gene]))
            if replacement < max_cells_per_gene:
                bucket[replacement] = cell
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
            f"No perturbed cells found matching {_XATLAS_PERTURBATION_COL}!="
            f"{control_label!r} under {shard_dir} ({shard_glob!r})"
        )
    return reservoirs
