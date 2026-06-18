"""Cross-cell-line Selectivity engine over DepMap GeneEffect + omics."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ENTREZ_RE = re.compile(r"\((\d+)\)\s*$")
_OMICS_META_COLUMNS: tuple[str, ...] = (
    "Unnamed: 0",
    "SequencingID",
    "ModelID",
    "ModelConditionID",
    "IsDefaultEntryForModel",
    "IsDefaultEntryForMC",
)


def parse_entrez_columns(columns: list[str]) -> dict[int, str]:
    """Map Entrez id -> column label for ``SYMBOL (ENTREZ)`` headers."""
    mapping: dict[int, str] = {}
    for col in columns:
        match = _ENTREZ_RE.search(str(col))
        if match is not None:
            mapping[int(match.group(1))] = col
    return mapping


def load_gene_effect_matrix(path: Path) -> tuple[pd.Index, dict[int, np.ndarray]]:
    """Load the GeneEffect matrix as ``(cell_line_index, {entrez: vector})``."""
    frame = pd.read_csv(path, index_col=0)
    entrez_map = parse_entrez_columns(list(frame.columns))
    vectors = {
        entrez: pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        for entrez, col in entrez_map.items()
    }
    return frame.index, vectors


def load_ach_indexed_matrix(
    path: Path, cell_line_index: pd.Index
) -> dict[int, np.ndarray]:
    """Load an ``ACH-``-indexed omics matrix reindexed onto the GeneEffect lines."""
    frame = pd.read_csv(path, index_col=0)
    frame = frame.reindex(cell_line_index)
    entrez_map = parse_entrez_columns(list(frame.columns))
    return {
        entrez: pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        for entrez, col in entrez_map.items()
    }


def load_modelid_column_matrix(
    path: Path, cell_line_index: pd.Index
) -> dict[int, np.ndarray]:
    """Load an omics matrix whose ``ModelID`` is a column, reindexed onto lines."""
    frame = pd.read_csv(path)
    if "ModelID" not in frame.columns:
        raise ValueError(f"{path} has no ModelID column")
    frame = frame.drop(
        columns=[c for c in _OMICS_META_COLUMNS if c in frame.columns],
        errors="ignore",
    ).set_index(frame["ModelID"])
    frame = frame.reindex(cell_line_index)
    entrez_map = parse_entrez_columns(list(frame.columns))
    return {
        entrez: pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        for entrez, col in entrez_map.items()
    }


def build_defective_mask(
    entrez: int,
    n_lines: int,
    damaging: dict[int, np.ndarray],
    hotspot: dict[int, np.ndarray],
    cn_log2: dict[int, np.ndarray],
    expr: dict[int, np.ndarray],
    cn_loss_thr: float,
    expr_low_quantile: float,
) -> np.ndarray:
    """Composite-OR defective call for one gene across the cell-line panel.

    A line is defective if any available channel fires: damaging mutation,
    hotspot mutation, copy-number loss (``cn_log2 < cn_loss_thr``), or low
    expression (``expr <= per-gene quantile``). Missing channels and NaN values
    never fire.

    Args:
        entrez: Entrez id of the anchor gene.
        n_lines: Number of cell lines (length of the output mask).
        damaging: Entrez -> binary damaging-mutation vector.
        hotspot: Entrez -> binary hotspot-mutation vector.
        cn_log2: Entrez -> log2 copy-number vector.
        expr: Entrez -> log2(TPM+1) expression vector.
        cn_loss_thr: Copy-number loss threshold (strictly less than).
        expr_low_quantile: Quantile defining low expression (<=).

    Returns:
        Boolean mask of shape ``(n_lines,)``.
    """
    mask = np.zeros(n_lines, dtype=bool)
    if entrez in damaging:
        mask |= np.nan_to_num(damaging[entrez], nan=0.0) >= 1.0
    if entrez in hotspot:
        mask |= np.nan_to_num(hotspot[entrez], nan=0.0) >= 1.0
    if entrez in cn_log2:
        vec = cn_log2[entrez]
        mask |= np.where(np.isnan(vec), False, vec < cn_loss_thr)
    if entrez in expr:
        vec = expr[entrez]
        finite = vec[~np.isnan(vec)]
        if finite.size > 0:
            thr = float(np.quantile(finite, expr_low_quantile))
            mask |= np.where(np.isnan(vec), False, vec <= thr)
    return mask


@dataclass(frozen=True)
class SelectivityTable:
    """Full directional Selectivity matrix over a gene universe.

    Attributes:
        entrez_order: Universe gene Entrez ids in row/column order.
        sel_matrix: ``(n, n)`` where ``sel_matrix[i, j] = sel(gene_i -> gene_j)``.
        pan_essential: ``(n,)`` mean GeneEffect per gene across all lines.
        coverage_flag: ``(n,)`` 1 if anchor had >= n_min defective and intact
            lines, else 0 (and its ``sel`` row is zeroed).
        essential_fraction: ``(n,)`` fraction of finite lines with GeneEffect
            below ``essential_effect_thr`` (used by the non-pan-essential
            diagnostic slice).
        index_by_entrez: Entrez id -> row index.
    """

    entrez_order: tuple[int, ...]
    sel_matrix: np.ndarray
    pan_essential: np.ndarray
    coverage_flag: np.ndarray
    essential_fraction: np.ndarray
    index_by_entrez: dict[int, int]


def build_selectivity_table(
    entrez_order: tuple[int, ...],
    gene_effect_vectors: dict[int, np.ndarray],
    damaging: dict[int, np.ndarray],
    hotspot: dict[int, np.ndarray],
    cn_log2: dict[int, np.ndarray],
    expr: dict[int, np.ndarray],
    cn_loss_thr: float,
    expr_low_quantile: float,
    n_min: int,
    essential_effect_thr: float = -0.5,
) -> SelectivityTable:
    """Build the directional Selectivity matrix for a gene universe."""
    n_gene = len(entrez_order)
    sample = next(iter(gene_effect_vectors.values()))
    n_lines = sample.shape[0]

    ge = np.full((n_lines, n_gene), np.nan, dtype=np.float32)
    for col, entrez in enumerate(entrez_order):
        if entrez in gene_effect_vectors:
            ge[:, col] = gene_effect_vectors[entrez]
    valid = (~np.isnan(ge)).astype(np.float32)
    ge0 = np.nan_to_num(ge, nan=0.0).astype(np.float32)
    with np.errstate(invalid="ignore"):
        pan_essential = np.nanmean(ge, axis=0)
    pan_essential = np.nan_to_num(pan_essential, nan=0.0)
    below = ((ge < essential_effect_thr) & ~np.isnan(ge)).sum(axis=0)
    line_counts = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        essential_fraction = np.where(line_counts > 0, below / line_counts, 0.0)

    defective = np.zeros((n_gene, n_lines), dtype=np.float32)
    for row, entrez in enumerate(entrez_order):
        defective[row] = build_defective_mask(
            entrez,
            n_lines,
            damaging,
            hotspot,
            cn_log2,
            expr,
            cn_loss_thr,
            expr_low_quantile,
        )

    def_sum = defective @ ge0
    def_cnt = defective @ valid
    total_sum = ge0.sum(axis=0, keepdims=True)
    total_cnt = valid.sum(axis=0, keepdims=True)
    intact_sum = total_sum - def_sum
    intact_cnt = total_cnt - def_cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        dep_def = np.where(def_cnt > 0, def_sum / def_cnt, 0.0)
        dep_intact = np.where(intact_cnt > 0, intact_sum / intact_cnt, 0.0)
    sel_matrix = (dep_intact - dep_def).astype(np.float32)

    n_def_lines = defective.sum(axis=1)
    n_intact_lines = n_lines - n_def_lines
    coverage_flag = ((n_def_lines >= n_min) & (n_intact_lines >= n_min)).astype(int)
    sel_matrix[coverage_flag == 0, :] = 0.0

    return SelectivityTable(
        entrez_order=tuple(entrez_order),
        sel_matrix=sel_matrix,
        pan_essential=pan_essential.astype(float),
        coverage_flag=coverage_flag,
        essential_fraction=essential_fraction.astype(float),
        index_by_entrez={e: i for i, e in enumerate(entrez_order)},
    )


STANDARD_FILES: dict[str, str] = {
    "gene_effect": "CRISPRGeneEffect.csv",
    "damaging": "OmicsSomaticMutationsMatrixDamaging.csv",
    "hotspot": "OmicsSomaticMutationsMatrixHotspot.csv",
    "cn": "PortalOmicsCNGeneLog2.csv",
    "expr": "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
}


def build_selectivity_table_from_depmap(
    depmap_dir: Path,
    entrez_order: tuple[int, ...],
    config: object,
) -> SelectivityTable:
    """Load the 5 DepMap CSVs and build the Selectivity table for a universe.

    Args:
        depmap_dir: Directory containing the standard DepMap files.
        entrez_order: Universe gene Entrez ids in canonical order.
        config: Object exposing ``cn_loss_thr``, ``expr_low_quantile``,
            ``sel_n_min`` (e.g. ``SLBaselineConfig``).

    Returns:
        A :class:`SelectivityTable`.
    """
    depmap_dir = Path(depmap_dir)
    lines, gene_effect = load_gene_effect_matrix(
        depmap_dir / STANDARD_FILES["gene_effect"]
    )
    damaging = load_modelid_column_matrix(
        depmap_dir / STANDARD_FILES["damaging"], lines
    )
    hotspot = load_modelid_column_matrix(depmap_dir / STANDARD_FILES["hotspot"], lines)
    cn_log2 = load_ach_indexed_matrix(depmap_dir / STANDARD_FILES["cn"], lines)
    expr = load_modelid_column_matrix(depmap_dir / STANDARD_FILES["expr"], lines)
    return build_selectivity_table(
        entrez_order=entrez_order,
        gene_effect_vectors=gene_effect,
        damaging=damaging,
        hotspot=hotspot,
        cn_log2=cn_log2,
        expr=expr,
        cn_loss_thr=config.cn_loss_thr,
        expr_low_quantile=config.expr_low_quantile,
        n_min=config.sel_n_min,
    )


@dataclass(frozen=True)
class UniverseSelectivity:
    """Selectivity arrays gathered into the benchmark gene-universe order."""

    sel_matrix: np.ndarray
    pan_essential: np.ndarray
    coverage_flag: np.ndarray
    essential_fraction: np.ndarray


def align_selectivity_to_universe(
    table: SelectivityTable,
    entrez_universe: list[int],
    sel_lambda: float,
) -> UniverseSelectivity:
    """Gather a SelectivityTable into universe order and apply the lambda penalty.

    Genes whose Entrez id is absent from ``table`` get zero sel rows/cols, zero
    pan-essentiality, zero essential-fraction, and coverage 0. The penalty
    subtracts ``sel_lambda * max(0, -pan_essential[j])`` from every entry in
    column ``j``.

    Args:
        table: Source SelectivityTable.
        entrez_universe: Universe gene Entrez ids in canonical order.
        sel_lambda: Pan-essentiality soft-penalty coefficient (0 disables it).

    Returns:
        A :class:`UniverseSelectivity`.
    """
    rows = np.array(
        [table.index_by_entrez.get(int(e), -1) for e in entrez_universe], dtype=int
    )
    present = rows >= 0
    safe_rows = np.where(present, rows, 0)
    sel = table.sel_matrix[np.ix_(safe_rows, safe_rows)].astype(float)
    sel[~present, :] = 0.0
    sel[:, ~present] = 0.0
    pan = np.where(present, table.pan_essential[safe_rows], 0.0)
    cov = np.where(present, table.coverage_flag[safe_rows], 0).astype(int)
    ess = np.where(present, table.essential_fraction[safe_rows], 0.0)
    if sel_lambda != 0.0:
        penalty = sel_lambda * np.maximum(0.0, -pan)
        sel = sel - penalty[np.newaxis, :]
    return UniverseSelectivity(
        sel_matrix=sel,
        pan_essential=pan,
        coverage_flag=cov,
        essential_fraction=ess,
    )
