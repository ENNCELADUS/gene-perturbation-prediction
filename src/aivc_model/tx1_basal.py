"""Assemble per-cell-line basal AnnData objects for Tahoe-x1 (Tx1) inference.

Turns the two Phase-A basal data sources -- Tahoe-100M DMSO parquet shards and
Perturb-seq non-targeting-control h5ad files -- into AnnData objects that
satisfy the verified Tx1 input contract: raw non-negative counts in ``.X``,
``var`` indexed by Ensembl gene id, and ``obs["cell_type"]`` set. This module
is pure and CPU-only; it performs no model loading or GPU inference (Wave 1
Phase B, Task 1: `.superpowers/sdd/briefs/task-1-brief.md`).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final, NamedTuple, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

_LOGGER = logging.getLogger(__name__)

_REQUIRED_MANIFEST_COLUMNS = (
    "model_id",
    "cellosaurus_id",
    "cell_line_name",
    "lineage",
    "dmso_cells",
    "basal_source",
    "tx1_pretraining_exposure",
    "role",
    "omics_expression_available",
)
_VALID_ROLES = frozenset({"test", "train_head", "train_response_and_head"})
_VALID_BASAL_SOURCES = frozenset(
    {"Tahoe-100M DMSO", "Perturb-seq non-targeting control"}
)
_TAHOE_BASAL_SOURCE = "Tahoe-100M DMSO"
_PERTURBSEQ_BASAL_SOURCE = "Perturb-seq non-targeting control"
_ENSEMBL_ID_PATTERN = re.compile(r"^ENSG\d+(\.\d+)?$")
_MIN_VALID_TOKEN_ID = 3  # token ids < 3 are special (pad/mask/cls) tokens
_SHARD_COLUMNS = ("genes", "expressions", "cell_line_id")

# --- X-Atlas-Orion parquet schema (audited, see task-3-brief.md) -----------
#
# Structurally the same shape as the Tahoe-100M DMSO shards -- a per-cell
# gene-token-id array + expression-value array, resolved to Ensembl ids via
# a token-indexed metadata parquet -- differing only in column names, the
# control-label column, and an extra guide-QC filter. Column names are the
# audited schema's own fixed contract (like ``_SHARD_COLUMNS`` above), not
# something that varies per line, so they are module constants rather than
# per-line config; ``control_label`` and ``pass_guide_filter_value`` are
# still explicit function parameters (not magic literals) since a caller
# may need to override them in tests.
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


def load_line_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the frozen Phase-A cell-line manifest.

    Args:
        path: Path to ``cell_line_manifest.csv``.

    Returns:
        The manifest as loaded, unmodified beyond validation. Row counts and
        the specific set of lines are read from the file, never asserted
        here -- the manifest is the source of truth, not this code.

    Raises:
        ValueError: A required column is missing, or ``role`` /
            ``basal_source`` contain values outside the contracted
            enumerations.
    """
    manifest = pd.read_csv(path)
    missing_columns = [
        column
        for column in _REQUIRED_MANIFEST_COLUMNS
        if column not in manifest.columns
    ]
    if missing_columns:
        raise ValueError(f"cell line manifest is missing columns: {missing_columns}")
    bad_roles = sorted(set(manifest["role"].astype(str)) - _VALID_ROLES)
    if bad_roles:
        raise ValueError(f"cell line manifest has invalid role values: {bad_roles}")
    bad_sources = sorted(
        set(manifest["basal_source"].astype(str)) - _VALID_BASAL_SOURCES
    )
    if bad_sources:
        raise ValueError(
            f"cell line manifest has invalid basal_source values: {bad_sources}"
        )
    return manifest


def build_tahoe_basal_adata(
    shard_dir: Path,
    gene_metadata_path: Path,
    cellosaurus_id: str,
    *,
    cell_line_name: str,
    model_id: str,
    max_cells: int | None = None,
    seed: int,
) -> ad.AnnData:
    """Assemble one cell line's basal AnnData from Tahoe-100M DMSO shards.

    Streams every ``*.parquet`` shard under ``shard_dir``, keeps only rows
    whose ``cell_line_id`` equals ``cellosaurus_id``, drops special token ids
    (``< 3``) and non-positive expression values, and maps surviving token
    ids to Ensembl gene ids via ``gene_metadata_path``. When ``max_cells``
    bounds the result, cells are chosen by Algorithm-R reservoir sampling
    seeded with ``seed``, so a single streaming pass over the shard directory
    yields a deterministic subset without knowing the matching row count in
    advance: the same seed and inputs reproduce the identical cell set, and a
    different seed generally selects a different one.

    Args:
        shard_dir: Directory of Tahoe-100M DMSO parquet shards.
        gene_metadata_path: Parquet file mapping ``token_id`` to
            ``ensembl_id`` and ``gene_symbol``.
        cellosaurus_id: Cellosaurus id to select, e.g. ``"CVCL_0334"``.
        cell_line_name: Human-readable line name written to
            ``obs["cell_type"]``.
        model_id: DepMap model id written to ``obs["model_id"]``.
        max_cells: Optional cap on the number of cells returned. ``None``
            returns every matching cell.
        seed: Seed for deterministic reservoir sampling.

    Returns:
        An AnnData satisfying ``assert_tx1_input_contract``, with
        ``obs["cellosaurus_id"]``, ``obs["model_id"]``, and
        ``obs["basal_source"]`` additionally set for batch-confound grouping.

    Raises:
        ValueError: No shards are found, no cell matches ``cellosaurus_id``,
            or the assembled AnnData fails the Tx1 input contract.
    """
    reservoir = _reservoir_sample_tahoe_cells(
        shard_dir, cellosaurus_id, max_cells, seed
    )
    metadata = pd.read_parquet(gene_metadata_path).set_index("token_id")
    matrix, var = _assemble_tahoe_matrix(reservoir, metadata)
    n_cells = matrix.shape[0]
    obs = pd.DataFrame(
        {
            "cell_type": [cell_line_name] * n_cells,
            "cellosaurus_id": [cellosaurus_id] * n_cells,
            "model_id": [model_id] * n_cells,
            "basal_source": [_TAHOE_BASAL_SOURCE] * n_cells,
        },
        index=[f"{cellosaurus_id}-{index}" for index in range(n_cells)],
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    assert_tx1_input_contract(adata)
    _LOGGER.info(
        "built Tahoe basal AnnData for %s: %d cells, %d genes",
        cellosaurus_id,
        n_cells,
        adata.n_vars,
    )
    return adata


def _reservoir_sample_tahoe_cells(
    shard_dir: Path,
    cellosaurus_id: str,
    max_cells: int | None,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stream matching (genes, values) pairs with Algorithm-R reservoir sampling."""
    paths = sorted(shard_dir.glob("*.parquet"))
    if not paths:
        raise ValueError(f"No parquet shards found under {shard_dir}")
    rng = np.random.default_rng(seed)
    reservoir: list[tuple[np.ndarray, np.ndarray]] = []
    seen = 0
    for path in paths:
        frame = pd.read_parquet(path, columns=list(_SHARD_COLUMNS))
        frame = frame[frame["cell_line_id"].astype(str) == cellosaurus_id]
        for row in frame.itertuples(index=False):
            genes = np.asarray(row.genes, dtype=np.int32)
            values = np.asarray(row.expressions, dtype=np.float32)
            valid = (genes >= _MIN_VALID_TOKEN_ID) & (values > 0)
            cell = (genes[valid], values[valid])
            seen += 1
            if max_cells is None or len(reservoir) < max_cells:
                reservoir.append(cell)
                continue
            replacement = int(rng.integers(0, seen))
            if replacement < max_cells:
                reservoir[replacement] = cell
    if not reservoir:
        raise ValueError(
            f"No cells found for cellosaurus_id={cellosaurus_id!r} in {shard_dir}"
        )
    return reservoir


def _assemble_tahoe_matrix(
    reservoir: list[tuple[np.ndarray, np.ndarray]],
    metadata: pd.DataFrame,
) -> tuple[csr_matrix, pd.DataFrame]:
    """Build the CSR matrix and Ensembl-indexed var frame for sampled cells.

    Otherwise-valid token ids (id >= 3, positive expression value) that are
    absent from ``metadata.index`` are dropped from the gene panel. A stale
    or partial ``gene_metadata.parquet`` would otherwise silently truncate
    the panel, so the count and fraction dropped are logged as a warning.
    """
    return _assemble_token_matrix(
        reservoir, metadata, metadata_var_columns=("ensembl_id", "gene_symbol")
    )


def _assemble_token_matrix(
    reservoir: list[tuple[np.ndarray, np.ndarray]],
    metadata: pd.DataFrame,
    *,
    metadata_var_columns: Sequence[str],
) -> tuple[csr_matrix, pd.DataFrame]:
    """Build a CSR matrix + Ensembl-indexed var frame from a token/value reservoir.

    Shared by the Tahoe-100M DMSO (:func:`_assemble_tahoe_matrix`) and
    X-Atlas-Orion (:func:`_assemble_xatlas_matrix`) parquet basal builders:
    both represent basal cells as already-validity-filtered parallel
    gene-token-id/expression-value arrays and both map surviving tokens to
    Ensembl gene ids via a token-indexed metadata parquet. Otherwise-valid
    tokens absent from ``metadata.index`` are dropped from the gene panel; a
    stale or partial metadata file would otherwise silently truncate the
    panel, so the count and fraction dropped are logged as a warning.

    Args:
        reservoir: Per-cell ``(gene_token_ids, expression_values)`` pairs,
            already filtered to valid tokens/positive values.
        metadata: Token-indexed metadata, e.g. ``token_id`` ->
            ``ensembl_id``/``gene_symbol`` (Tahoe) or ``gene_token_id`` ->
            ``ensembl_id``/``gene_name`` (X-Atlas-Orion).
        metadata_var_columns: The ``metadata`` columns to carry into the
            returned ``var``; the first must be the Ensembl id column, which
            becomes ``var.index``.

    Returns:
        ``(matrix, var)``: a CSR matrix of shape ``(len(reservoir),
        n_surviving_tokens)`` and the matching Ensembl-indexed ``var`` frame.
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


def build_perturbseq_basal_adata(
    h5ad_path: Path,
    *,
    control_label: str,
    perturbation_col: str,
    cell_line_name: str,
    model_id: str,
    cellosaurus_id: str,
    var_ensembl_col: str,
    max_cells: int | None = None,
    seed: int,
) -> ad.AnnData:
    """Assemble one cell line's basal AnnData from a Perturb-seq h5ad.

    Opens ``h5ad_path`` backed (these files run 5-66 GB) and materializes
    only the rows whose ``obs[perturbation_col]`` exactly equals
    ``control_label``. When ``max_cells`` bounds the result, the row subset
    is chosen by deterministic index sampling seeded with ``seed`` before
    anything is materialized: unlike the streaming Tahoe shards, backed
    ``obs`` already exposes the full control population size, so no
    reservoir is needed. The same seed and inputs reproduce the identical
    cell set.

    Args:
        h5ad_path: Path to the Perturb-seq h5ad file.
        control_label: Exact ``obs[perturbation_col]`` value identifying
            non-targeting controls.
        perturbation_col: ``obs`` column carrying the perturbation label.
        cell_line_name: Human-readable line name written to
            ``obs["cell_type"]``.
        model_id: DepMap model id written to ``obs["model_id"]``.
        cellosaurus_id: Cellosaurus id written to ``obs["cellosaurus_id"]``.
        var_ensembl_col: ``var`` column carrying Ensembl gene ids.
        max_cells: Optional cap on the number of cells returned. ``None``
            returns every control cell.
        seed: Seed for deterministic index sampling.

    Returns:
        An AnnData satisfying ``assert_tx1_input_contract``, with
        ``obs["cellosaurus_id"]``, ``obs["model_id"]``, and
        ``obs["basal_source"]`` additionally set for batch-confound grouping.

    Raises:
        ValueError: ``var_ensembl_col`` names neither a ``var`` column nor
            ``var.index.name``, no control cells are found, or the assembled
            AnnData fails the Tx1 input contract.
    """
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        _require_ensembl_source(backed.var, var_ensembl_col, h5ad_path)
        control_mask = (
            backed.obs[perturbation_col].astype(str) == control_label
        ).to_numpy()
        control_indices = np.flatnonzero(control_mask)
        if not control_indices.size:
            raise ValueError(
                f"No control cells found for {perturbation_col}={control_label!r} "
                f"in {h5ad_path}"
            )
        selected = _select_indices_deterministic(control_indices, max_cells, seed)
        matrix = csr_matrix(_materialize_rows(backed.X, selected))
        obs_names = backed.obs_names.to_numpy()[selected].astype(str).tolist()
        var = backed.var.copy()
    finally:
        backed.file.close()

    if var_ensembl_col in var.columns:
        var.index = var[var_ensembl_col].astype(str)
    # else: _require_ensembl_source already confirmed var.index.name ==
    # var_ensembl_col, so var.index (copied from backed.var) already holds
    # the Ensembl ids -- no reassignment needed.
    n_cells = matrix.shape[0]
    obs = pd.DataFrame(
        {
            "cell_type": [cell_line_name] * n_cells,
            "cellosaurus_id": [cellosaurus_id] * n_cells,
            "model_id": [model_id] * n_cells,
            "basal_source": [_PERTURBSEQ_BASAL_SOURCE] * n_cells,
        },
        index=obs_names,
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    assert_tx1_input_contract(adata)
    _LOGGER.info(
        "built Perturb-seq basal AnnData for %s: %d cells, %d genes",
        cellosaurus_id,
        n_cells,
        adata.n_vars,
    )
    return adata


def _require_ensembl_source(
    var: pd.DataFrame, var_ensembl_col: str, h5ad_path: Path
) -> None:
    """Validate that ``var_ensembl_col`` names a ``var`` column or its index.

    Perturb-seq h5ad sources vary in how they carry Ensembl ids: some (e.g.
    the local Adamson UPR file, whose ``var.index`` is gene symbols) carry
    Ensembl ids as an explicit column alongside a non-Ensembl index; others
    (e.g. the Replogle-lineage K562 essential h5ad already in this repo,
    whose ``var`` has no such column at all) carry Ensembl ids directly as
    the index, with ``var.index.name`` set accordingly. Both are valid
    inputs to this builder; only neither is an error.

    Args:
        var: The candidate h5ad's ``var`` frame.
        var_ensembl_col: The caller-supplied Ensembl column/index name.
        h5ad_path: Source path, for the error message.

    Raises:
        ValueError: ``var_ensembl_col`` is neither a column of ``var`` nor
            the name of ``var.index``.
    """
    if var_ensembl_col in var.columns or var.index.name == var_ensembl_col:
        return
    raise ValueError(
        f"{h5ad_path} var has neither a column nor an index named "
        f"{var_ensembl_col!r}; available columns: {sorted(var.columns)}, "
        f"index name: {var.index.name!r}"
    )


def _select_indices_deterministic(
    candidate_indices: np.ndarray,
    max_cells: int | None,
    seed: int,
) -> np.ndarray:
    """Deterministically subsample row indices, or keep all when uncapped."""
    if max_cells is None or max_cells >= len(candidate_indices):
        return candidate_indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidate_indices), size=max_cells, replace=False)
    return np.sort(candidate_indices[chosen])


def _materialize_rows(matrix: object, row_indices: np.ndarray) -> np.ndarray:
    """Read selected backed rows, sorting first for backends requiring it."""
    order = np.argsort(row_indices)
    chunk = matrix[row_indices[order], :]
    inverse = np.argsort(order)
    if sparse.issparse(chunk):
        return chunk.tocsr()[inverse, :]
    return np.asarray(chunk)[inverse]


class _XatlasCell(NamedTuple):
    """One reservoir-sampled X-Atlas-Orion cell, pre-filtered to valid genes."""

    genes: np.ndarray
    values: np.ndarray
    cell_barcode: str
    sample: str


def build_xatlas_orion_basal_adata(
    shard_dir: Path,
    gene_metadata_path: Path,
    *,
    cell_line_name: str,
    model_id: str,
    cellosaurus_id: str,
    shard_glob: str = "*.parquet",
    control_label: str = _XATLAS_CONTROL_LABEL,
    pass_guide_filter_value: int = _XATLAS_PASS_GUIDE_FILTER_VALUE,
    max_cells: int | None = None,
    seed: int,
) -> ad.AnnData:
    """Assemble one cell line's basal AnnData from X-Atlas-Orion parquet shards.

    Structurally the same shape as :func:`build_tahoe_basal_adata` (a
    gene-token-id + expression-value array per cell, reservoir-sampled and
    resolved to Ensembl ids via a token-indexed metadata parquet), differing
    in the shard schema's column names, filtering by ``gene_target`` instead
    of a Cellosaurus id (each shard directory holds one line's batches
    already, selected via ``shard_glob``), and the extra ``pass_guide_filter``
    guide-QC filter. Control/guide-QC filtering happens while streaming,
    before reservoir sampling; the cells each filter removes are logged.

    Args:
        shard_dir: Directory holding this line's X-Atlas-Orion parquet
            shards (may also hold other cell lines' shards; ``shard_glob``
            restricts to this line's files).
        gene_metadata_path: Parquet file mapping ``gene_token_id`` to
            ``ensembl_id`` and ``gene_name``.
        cell_line_name: Human-readable line name written to
            ``obs["cell_type"]``.
        model_id: DepMap model id written to ``obs["model_id"]``.
        cellosaurus_id: Cellosaurus id written to ``obs["cellosaurus_id"]``.
        shard_glob: Glob pattern (relative to ``shard_dir``) selecting this
            line's shard files, e.g. ``"HCT116_Batch*.parquet"``.
        control_label: Exact ``gene_target`` value identifying
            non-targeting-control cells.
        pass_guide_filter_value: Required ``pass_guide_filter`` value (an
            explicit, named parameter rather than a magic literal).
        max_cells: Optional cap on the number of cells returned. ``None``
            returns every matching cell.
        seed: Seed for deterministic reservoir sampling.

    Returns:
        An AnnData satisfying ``assert_tx1_input_contract``, with
        ``obs["cellosaurus_id"]``, ``obs["model_id"]``,
        ``obs["basal_source"]``, and ``obs["sample"]`` (the X-Atlas-Orion
        batch key, needed for the design's basal-source batch-confound
        audit) additionally set.

    Raises:
        ValueError: No shard matches ``shard_glob``, no cell survives the
            control-label and guide-filter criteria, or the assembled
            AnnData fails the Tx1 input contract.
    """
    reservoir = _reservoir_sample_xatlas_cells(
        shard_dir,
        shard_glob,
        control_label=control_label,
        pass_guide_filter_value=pass_guide_filter_value,
        max_cells=max_cells,
        seed=seed,
    )
    metadata = pd.read_parquet(gene_metadata_path).set_index(
        _XATLAS_GENE_METADATA_TOKEN_COL
    )
    matrix, var = _assemble_token_matrix(
        [(cell.genes, cell.values) for cell in reservoir],
        metadata,
        metadata_var_columns=_XATLAS_GENE_METADATA_VAR_COLUMNS,
    )
    n_cells = matrix.shape[0]
    obs = pd.DataFrame(
        {
            "cell_type": [cell_line_name] * n_cells,
            "cellosaurus_id": [cellosaurus_id] * n_cells,
            "model_id": [model_id] * n_cells,
            "basal_source": [_PERTURBSEQ_BASAL_SOURCE] * n_cells,
            "sample": [cell.sample for cell in reservoir],
        },
        index=[f"{cell.sample}:{cell.cell_barcode}" for cell in reservoir],
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    assert_tx1_input_contract(adata)
    _LOGGER.info(
        "built X-Atlas-Orion basal AnnData for %s: %d cells, %d genes",
        cellosaurus_id,
        n_cells,
        adata.n_vars,
    )
    return adata


def _reservoir_sample_xatlas_cells(
    shard_dir: Path,
    shard_glob: str,
    *,
    control_label: str,
    pass_guide_filter_value: int,
    max_cells: int | None,
    seed: int,
) -> list[_XatlasCell]:
    """Stream X-Atlas-Orion parquet shards with Algorithm-R reservoir sampling.

    Mirrors :func:`_reservoir_sample_tahoe_cells`'s single streaming pass and
    deterministic Algorithm-R selection, applied to a shard schema that
    requires two row filters (non-targeting control, guide-QC pass) instead
    of one (``cell_line_id`` equality) before the positive-expression-value
    filter (see :func:`_row_to_xatlas_cell` for why this source's gene
    tokens are not also filtered by id, unlike Tahoe's).

    Args:
        shard_dir: Directory of X-Atlas-Orion parquet shards.
        shard_glob: Glob pattern selecting this line's shard files.
        control_label: Exact ``gene_target`` value identifying
            non-targeting-control cells.
        pass_guide_filter_value: Required ``pass_guide_filter`` value.
        max_cells: Optional cap on the number of cells returned.
        seed: Seed for deterministic reservoir sampling.

    Returns:
        Reservoir-sampled cells, each already filtered to positive-value
        gene tokens.

    Raises:
        ValueError: No shard matches ``shard_glob``, or no cell survives the
            control-label and guide-filter criteria.
    """
    paths = sorted(shard_dir.glob(shard_glob))
    if not paths:
        raise ValueError(
            f"No parquet shards matching {shard_glob!r} found under {shard_dir}"
        )
    rng = np.random.default_rng(seed)
    reservoir: list[_XatlasCell] = []
    seen = 0
    counts = np.zeros(3, dtype=np.int64)  # total, after-control, after-guide-filter
    for path in paths:
        frame, shard_counts = _filter_xatlas_shard(
            path, control_label, pass_guide_filter_value
        )
        counts += shard_counts
        for row in frame.itertuples(index=False):
            seen += 1
            cell = _row_to_xatlas_cell(row)
            if max_cells is None or len(reservoir) < max_cells:
                reservoir.append(cell)
                continue
            replacement = int(rng.integers(0, seen))
            if replacement < max_cells:
                reservoir[replacement] = cell
    _log_xatlas_filter_counts(
        shard_dir, shard_glob, counts, control_label, pass_guide_filter_value
    )
    if not reservoir:
        raise ValueError(
            f"No cells found matching {_XATLAS_PERTURBATION_COL}={control_label!r} "
            f"and {_XATLAS_PASS_GUIDE_FILTER_COL}={pass_guide_filter_value} under "
            f"{shard_dir} ({shard_glob!r})"
        )
    return reservoir


def _filter_xatlas_shard(
    path: Path, control_label: str, pass_guide_filter_value: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Read one shard and apply the control-label and guide-QC row filters.

    Returns:
        The filtered frame and ``[total_rows, after_control, after_guide]``
        row counts, for the caller to accumulate and log across shards.
    """
    frame = pd.read_parquet(path, columns=list(_XATLAS_READ_COLUMNS))
    total = len(frame)
    frame = frame[frame[_XATLAS_PERTURBATION_COL].astype(str) == control_label]
    after_control = len(frame)
    frame = frame[
        frame[_XATLAS_PASS_GUIDE_FILTER_COL].astype(int) == pass_guide_filter_value
    ]
    after_guide = len(frame)
    return frame, np.array([total, after_control, after_guide], dtype=np.int64)


def _row_to_xatlas_cell(row: object) -> _XatlasCell:
    """Build a validity-filtered ``_XatlasCell`` from one filtered shard row.

    Unlike Tahoe-100M's shard schema, X-Atlas-Orion's ``gene_token_id``
    vocabulary does **not** reserve low ids (0, 1, 2, ...) as special
    pad/mask/cls tokens -- the real ``gene_metadata.parquet`` audited for
    this source maps token ids 0-4 to ordinary genes (e.g. ``DDX11L2``,
    ``OR4F5``). So only the positive-expression-value filter applies here;
    :func:`_reservoir_sample_tahoe_cells`'s ``id >= _MIN_VALID_TOKEN_ID``
    filter must not be reused for this source.
    """
    genes = np.asarray(getattr(row, _XATLAS_GENE_TOKEN_COL), dtype=np.int64)
    values = np.asarray(getattr(row, _XATLAS_EXPRESSION_COL), dtype=np.float32)
    valid = values > 0
    return _XatlasCell(
        genes=genes[valid],
        values=values[valid],
        cell_barcode=str(row.cell_barcode),
        sample=str(getattr(row, _XATLAS_SAMPLE_COL)),
    )


def _log_xatlas_filter_counts(
    shard_dir: Path,
    shard_glob: str,
    counts: np.ndarray,
    control_label: str,
    pass_guide_filter_value: int,
) -> None:
    """Log how many cells the control-label and guide-QC filters each removed."""
    total, after_control, after_guide = (int(value) for value in counts)
    _LOGGER.info(
        "X-Atlas-Orion cell filtering under %s (%s): %d total rows, %d after "
        "%s==%r (removed %d), %d after %s==%d (removed %d)",
        shard_dir,
        shard_glob,
        total,
        after_control,
        _XATLAS_PERTURBATION_COL,
        control_label,
        total - after_control,
        after_guide,
        _XATLAS_PASS_GUIDE_FILTER_COL,
        pass_guide_filter_value,
        after_control - after_guide,
    )


# --- RESPONSE (perturbed-cell) builders ------------------------------------
#
# Counterparts to the basal (control-only) builders above, added for Wave 2
# Phase C Task 5 (multi-line observed-response assembly:
# `.superpowers/sdd/phase-c/task-5-brief.md`). ST's observed-response
# supervision target is post-perturbation gene expression from *perturbed*
# cells -- CPU-only raw-count extraction, never Tx1 inference (that stays
# basal-only, Phase B). These reuse the same per-source parsing/assembly
# primitives as their basal siblings (`_require_ensembl_source`,
# `_materialize_rows`, `_assemble_token_matrix`, `_row_to_xatlas_cell`)
# rather than duplicating them, and deliberately do not reservoir-sample or
# cap cell counts: unlike a fixed-size basal panel, the observed-response
# target needs every available perturbed cell for the requested gene(s).


def build_perturbseq_response_adata(
    h5ad_path: Path,
    *,
    control_label: str,
    perturbation_col: str,
    cell_line_name: str,
    model_id: str,
    cellosaurus_id: str,
    var_ensembl_col: str,
    genes: Sequence[str] | None = None,
) -> ad.AnnData:
    """Assemble one cell line's observed post-perturbation response AnnData.

    Counterpart to :func:`build_perturbseq_basal_adata` for RESPONSE
    (perturbed, not control) cells: selects every cell whose
    ``obs[perturbation_col]`` is **not** ``control_label``, optionally
    further restricted to ``genes``, and keeps each cell's own perturbation
    label as ``obs["perturbation_gene"]`` so a caller can group cells into
    per-gene bags.

    Args:
        h5ad_path: Path to the Perturb-seq h5ad file.
        control_label: Exact ``obs[perturbation_col]`` value identifying
            non-targeting controls, excluded from the result.
        perturbation_col: ``obs`` column carrying the perturbation label.
        cell_line_name: Human-readable line name written to
            ``obs["cell_type"]``.
        model_id: DepMap model id written to ``obs["model_id"]``.
        cellosaurus_id: Cellosaurus id written to ``obs["cellosaurus_id"]``.
        var_ensembl_col: ``var`` column or index name carrying Ensembl gene
            ids (see :func:`_require_ensembl_source`).
        genes: Optional subset of perturbation-gene labels to keep (exact
            ``obs[perturbation_col]`` string match). ``None`` keeps every
            non-control cell.

    Returns:
        An AnnData satisfying ``assert_tx1_input_contract``, with
        ``obs["perturbation_gene"]`` set to each cell's own perturbation
        label plus the same ``cellosaurus_id``/``model_id`` provenance
        columns as :func:`build_perturbseq_basal_adata`.

    Raises:
        ValueError: ``var_ensembl_col`` names neither a ``var`` column nor
            ``var.index.name``, no perturbed cell survives the requested
            filters, or the assembled AnnData fails the Tx1 input contract.
    """
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        _require_ensembl_source(backed.var, var_ensembl_col, h5ad_path)
        obs_labels = backed.obs[perturbation_col].astype(str)
        mask = obs_labels != control_label
        if genes is not None:
            allowed = {str(gene) for gene in genes}
            mask = mask & obs_labels.isin(allowed)
        selected = np.flatnonzero(mask.to_numpy())
        if not selected.size:
            restriction = f" restricted to {sorted(set(genes))}" if genes else ""
            raise ValueError(
                f"No perturbed cells found for {perturbation_col}!={control_label!r}"
                f"{restriction} in {h5ad_path}"
            )
        matrix = csr_matrix(_materialize_rows(backed.X, selected))
        obs_names = backed.obs_names.to_numpy()[selected].astype(str).tolist()
        perturbation_gene = obs_labels.to_numpy()[selected]
        var = backed.var.copy()
    finally:
        backed.file.close()

    if var_ensembl_col in var.columns:
        var.index = var[var_ensembl_col].astype(str)
    # else: _require_ensembl_source already confirmed var.index.name ==
    # var_ensembl_col, matching build_perturbseq_basal_adata's convention.
    n_cells = matrix.shape[0]
    obs = pd.DataFrame(
        {
            "cell_type": [cell_line_name] * n_cells,
            "cellosaurus_id": [cellosaurus_id] * n_cells,
            "model_id": [model_id] * n_cells,
            "perturbation_gene": perturbation_gene,
        },
        index=obs_names,
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    assert_tx1_input_contract(adata)
    _LOGGER.info(
        "built Perturb-seq response AnnData for %s: %d cells, %d genes, %d "
        "distinct perturbations",
        cellosaurus_id,
        n_cells,
        adata.n_vars,
        len(set(perturbation_gene.tolist())),
    )
    return adata


def build_xatlas_orion_response_adata(
    shard_dir: Path,
    gene_metadata_path: Path,
    *,
    cell_line_name: str,
    model_id: str,
    cellosaurus_id: str,
    shard_glob: str = "*.parquet",
    control_label: str = _XATLAS_CONTROL_LABEL,
    pass_guide_filter_value: int = _XATLAS_PASS_GUIDE_FILTER_VALUE,
    genes: Sequence[str] | None = None,
) -> ad.AnnData:
    """Assemble one line's observed post-perturbation response AnnData from
    X-Atlas-Orion parquet shards.

    Counterpart to :func:`build_xatlas_orion_basal_adata` for RESPONSE
    cells: selects every guide-QC-passing cell whose ``gene_target`` is
    **not** ``control_label``, optionally further restricted to ``genes``,
    keeping each cell's own ``gene_target`` as ``obs["perturbation_gene"]``.

    Args:
        shard_dir: Directory holding this line's X-Atlas-Orion parquet
            shards.
        gene_metadata_path: Parquet file mapping ``gene_token_id`` to
            ``ensembl_id`` and ``gene_name``.
        cell_line_name: Human-readable line name written to
            ``obs["cell_type"]``.
        model_id: DepMap model id written to ``obs["model_id"]``.
        cellosaurus_id: Cellosaurus id written to ``obs["cellosaurus_id"]``.
        shard_glob: Glob pattern (relative to ``shard_dir``) selecting this
            line's shard files.
        control_label: Exact ``gene_target`` value identifying
            non-targeting-control cells, excluded from the result.
        pass_guide_filter_value: Required ``pass_guide_filter`` value.
        genes: Optional subset of ``gene_target`` labels to keep. ``None``
            keeps every perturbed cell.

    Returns:
        An AnnData satisfying ``assert_tx1_input_contract``, with
        ``obs["perturbation_gene"]``/``obs["sample"]`` set alongside the
        same provenance columns as :func:`build_xatlas_orion_basal_adata`.

    Raises:
        ValueError: No shard matches ``shard_glob``, no cell survives the
            requested filters, or the assembled AnnData fails the Tx1 input
            contract.
    """
    cells, perturbation_genes = _stream_xatlas_response_cells(
        shard_dir,
        shard_glob,
        control_label=control_label,
        pass_guide_filter_value=pass_guide_filter_value,
        genes=genes,
    )
    metadata = pd.read_parquet(gene_metadata_path).set_index(
        _XATLAS_GENE_METADATA_TOKEN_COL
    )
    matrix, var = _assemble_token_matrix(
        [(cell.genes, cell.values) for cell in cells],
        metadata,
        metadata_var_columns=_XATLAS_GENE_METADATA_VAR_COLUMNS,
    )
    n_cells = matrix.shape[0]
    obs = pd.DataFrame(
        {
            "cell_type": [cell_line_name] * n_cells,
            "cellosaurus_id": [cellosaurus_id] * n_cells,
            "model_id": [model_id] * n_cells,
            "perturbation_gene": perturbation_genes,
            "sample": [cell.sample for cell in cells],
        },
        index=[f"{cell.sample}:{cell.cell_barcode}" for cell in cells],
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    assert_tx1_input_contract(adata)
    _LOGGER.info(
        "built X-Atlas-Orion response AnnData for %s: %d cells, %d genes, %d "
        "distinct perturbations",
        cellosaurus_id,
        n_cells,
        adata.n_vars,
        len(set(perturbation_genes.tolist())),
    )
    return adata


def _stream_xatlas_response_cells(
    shard_dir: Path,
    shard_glob: str,
    *,
    control_label: str,
    pass_guide_filter_value: int,
    genes: Sequence[str] | None,
) -> tuple[list[_XatlasCell], np.ndarray]:
    """Stream every perturbed, guide-QC-passing X-Atlas-Orion cell (no cap).

    Mirrors :func:`_reservoir_sample_xatlas_cells`'s shard-streaming shape
    but keeps every matching row instead of reservoir-sampling a fixed-size
    subset, and filters to ``gene_target != control_label`` (perturbed)
    rather than ``== control_label`` (basal).
    """
    paths = sorted(shard_dir.glob(shard_glob))
    if not paths:
        raise ValueError(
            f"No parquet shards matching {shard_glob!r} found under {shard_dir}"
        )
    allowed = {str(gene) for gene in genes} if genes is not None else None
    cells: list[_XatlasCell] = []
    perturbation_genes: list[str] = []
    counts = np.zeros(3, dtype=np.int64)  # total, after-perturbed, after-guide-filter
    for path in paths:
        frame = pd.read_parquet(path, columns=list(_XATLAS_READ_COLUMNS))
        total = len(frame)
        frame = frame[frame[_XATLAS_PERTURBATION_COL].astype(str) != control_label]
        if allowed is not None:
            frame = frame[frame[_XATLAS_PERTURBATION_COL].astype(str).isin(allowed)]
        after_perturbed = len(frame)
        frame = frame[
            frame[_XATLAS_PASS_GUIDE_FILTER_COL].astype(int) == pass_guide_filter_value
        ]
        after_guide = len(frame)
        counts += np.array([total, after_perturbed, after_guide], dtype=np.int64)
        for row in frame.itertuples(index=False):
            cells.append(_row_to_xatlas_cell(row))
            perturbation_genes.append(str(getattr(row, _XATLAS_PERTURBATION_COL)))
    _log_xatlas_response_filter_counts(
        shard_dir, shard_glob, counts, control_label, pass_guide_filter_value
    )
    if not cells:
        raise ValueError(
            f"No perturbed cells found matching {_XATLAS_PERTURBATION_COL}!="
            f"{control_label!r} under {shard_dir} ({shard_glob!r})"
        )
    return cells, np.asarray(perturbation_genes, dtype=object)


def _log_xatlas_response_filter_counts(
    shard_dir: Path,
    shard_glob: str,
    counts: np.ndarray,
    control_label: str,
    pass_guide_filter_value: int,
) -> None:
    """Log how many cells the perturbed-label and guide-QC filters each removed.

    A separate function from :func:`_log_xatlas_filter_counts` rather than a
    reuse: that one's message hardcodes an ``==`` comparison against
    ``control_label``, which would misreport this builder's ``!=`` (plus
    optional gene-subset) selection as an equality filter.
    """
    total, after_perturbed, after_guide = (int(value) for value in counts)
    _LOGGER.info(
        "X-Atlas-Orion response cell filtering under %s (%s): %d total rows, "
        "%d after selecting %s!=%r (optionally gene-restricted) (removed %d), "
        "%d after %s==%d (removed %d)",
        shard_dir,
        shard_glob,
        total,
        after_perturbed,
        _XATLAS_PERTURBATION_COL,
        control_label,
        total - after_perturbed,
        after_guide,
        _XATLAS_PASS_GUIDE_FILTER_COL,
        pass_guide_filter_value,
        after_perturbed - after_guide,
    )


def assert_tx1_input_contract(adata: ad.AnnData) -> None:
    """Enforce the verified Tx1 input contract on a candidate basal AnnData.

    This is Global Constraint 4 (raw non-negative ``.X``; ``var`` indexed by
    Ensembl gene id; ``obs["cell_type"]`` present) made executable.

    Args:
        adata: Candidate basal AnnData.

    Raises:
        ValueError: ``.X`` contains negative or non-finite (NaN/Inf) values,
            ``var.index`` does not look like Ensembl gene ids, or ``obs`` is
            missing ``cell_type``.
    """
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
