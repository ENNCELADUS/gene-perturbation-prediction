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
    var = metadata.loc[tokens, ["ensembl_id", "gene_symbol"]].copy()
    var.index = var["ensembl_id"].astype(str)
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
        ValueError: ``var_ensembl_col`` is absent from ``var``, no control
            cells are found, or the assembled AnnData fails the Tx1 input
            contract.
    """
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if var_ensembl_col not in backed.var.columns:
            raise ValueError(
                f"{h5ad_path} var is missing Ensembl column {var_ensembl_col!r}; "
                f"available columns: {sorted(backed.var.columns)}"
            )
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

    var.index = var[var_ensembl_col].astype(str)
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
