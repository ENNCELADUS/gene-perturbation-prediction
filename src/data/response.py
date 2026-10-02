"""Response sources and observed response targets in STATE's log expression space.

Each anchor line's perturbed cells are read as raw counts, normalised over
the whole library of each cell (``log1p(x * T / L_cell)``), sliced to STATE's
HVG order and grouped into one bag per perturbed gene.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from src.data.basal import (
    align_columns,
    build_perturbseq_basal_adata,
    build_perturbseq_response_adata,
    build_xatlas_orion_response_adata,
    require_raw_counts,
)
from src.data.expression import library_sizes, log_normalize

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerturbseqSource:
    """A Perturb-seq h5ad response source."""

    h5ad_path: Path
    control_label: str
    perturbation_col: str
    var_ensembl_col: str
    target_gene_symbol_col: str = "gene_name"


@dataclass(frozen=True)
class XatlasOrionSource:
    """An X-Atlas-Orion parquet response source."""

    shard_dir: Path
    gene_metadata_path: Path
    control_label: str
    shard_glob: str = "*.parquet"
    pass_guide_filter_value: int = 1
    target_gene_symbol_col: str = "gene_name"


ResponseSource = PerturbseqSource | XatlasOrionSource


def load_response_sources(path: Path) -> dict[str, ResponseSource]:
    """Parse ``perturbseq_sources.json``: one entry per anchor ModelID."""
    sources: dict[str, ResponseSource] = {}
    for model_id, entry in json.loads(Path(path).read_text()).items():
        symbol_col = str(entry.get("target_gene_symbol_col", "gene_name"))
        source_type = entry.get("source_type", "h5ad")
        if source_type == "h5ad":
            sources[str(model_id)] = PerturbseqSource(
                h5ad_path=Path(entry["h5ad_path"]),
                control_label=str(entry["control_label"]),
                perturbation_col=str(entry["perturbation_col"]),
                var_ensembl_col=str(entry["var_ensembl_col"]),
                target_gene_symbol_col=symbol_col,
            )
        elif source_type == "xatlas_orion_parquet":
            sources[str(model_id)] = XatlasOrionSource(
                shard_dir=Path(entry["shard_dir"]),
                gene_metadata_path=Path(entry["gene_metadata_path"]),
                control_label=str(entry["control_label"]),
                shard_glob=str(entry.get("shard_glob", "*.parquet")),
                pass_guide_filter_value=int(entry.get("pass_guide_filter_value", 1)),
                target_gene_symbol_col=symbol_col,
            )
        else:
            raise ValueError(
                f"{path}: {model_id} has unknown source_type {source_type!r}"
            )
    return sources


def control_library_sizes(source: ResponseSource, model_id: str) -> np.ndarray:
    """Whole-library UMI totals of a source's non-targeting cells."""
    if not isinstance(source, PerturbseqSource):
        raise ValueError(f"{model_id}: control library sizes need an h5ad source")
    adata = build_perturbseq_basal_adata(
        source.h5ad_path,
        control_label=source.control_label,
        perturbation_col=source.perturbation_col,
        model_id=model_id,
        var_ensembl_col=source.var_ensembl_col,
    )
    require_raw_counts(adata.X, f"{model_id} response source controls")
    return library_sizes(adata.X)


def response_cells(
    source: ResponseSource,
    model_id: str,
    *,
    max_cells_per_gene: int | None,
    total_cells_per_line: int | None,
    seed: int,
) -> ad.AnnData:
    """Raw-count perturbed cells of one anchor, ``obs["perturbation_gene"]`` set."""
    if isinstance(source, PerturbseqSource):
        return build_perturbseq_response_adata(
            source.h5ad_path,
            control_label=source.control_label,
            perturbation_col=source.perturbation_col,
            model_id=model_id,
            var_ensembl_col=source.var_ensembl_col,
            max_cells_per_gene=max_cells_per_gene,
            total_cells_per_line=total_cells_per_line,
            seed=seed,
        )
    return build_xatlas_orion_response_adata(
        source.shard_dir,
        source.gene_metadata_path,
        model_id=model_id,
        shard_glob=source.shard_glob,
        control_label=source.control_label,
        pass_guide_filter_value=source.pass_guide_filter_value,
        max_cells_per_gene=max_cells_per_gene,
        total_cells_per_line=total_cells_per_line,
        seed=seed,
    )


def build_response_targets(
    sources: dict[str, ResponseSource],
    hvg_order: Sequence[str],
    target_sum: float,
    *,
    max_cells_per_gene: int | None,
    total_cells_per_line: int | None,
    seed: int,
) -> tuple[list[tuple[str, str]], list[np.ndarray]]:
    """Log-space HVG bags for every perturbed gene of every anchor.

    Keys are ``(ModelID, upper-case gene)`` with anchors and genes sorted.
    STATE HVGs absent from a source are zero columns.
    """
    keys: list[tuple[str, str]] = []
    bags: list[np.ndarray] = []
    for model_id in sorted(sources):
        source = sources[model_id]
        _LOGGER.info("Reading perturbed cells of response anchor %s", model_id)
        adata = response_cells(
            source,
            model_id,
            max_cells_per_gene=max_cells_per_gene,
            total_cells_per_line=total_cells_per_line,
            seed=seed,
        )
        require_raw_counts(adata.X, f"{model_id} response source")
        sizes = library_sizes(adata.X)
        symbols = adata.var[source.target_gene_symbol_col].astype(str)
        hvg, present = align_columns(adata.X, symbols, hvg_order)
        if not present.all():
            _LOGGER.warning(
                "%s: %d/%d STATE HVGs absent from the response source, zero-filled",
                model_id,
                int((~present).sum()),
                len(present),
            )
        labels = adata.obs["perturbation_gene"].astype(str).str.strip().str.upper()
        groups = pd.Series(np.arange(len(labels))).groupby(labels.to_numpy()).indices
        for gene in sorted(groups):
            rows = groups[gene]
            keys.append((model_id, str(gene)))
            bags.append(log_normalize(hvg[rows], sizes[rows], target_sum))
    return keys, bags


__all__ = [
    "PerturbseqSource",
    "ResponseSource",
    "XatlasOrionSource",
    "build_response_targets",
    "control_library_sizes",
    "load_response_sources",
    "response_cells",
]
