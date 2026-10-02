"""Response sources and observed response targets in STATE's log expression space.

Each anchor line's perturbed cells are read as raw counts and reduced to the
whole-library UMI total of each cell and its counts in STATE's HVG order;
targets are ``log1p(x * T / L_cell)`` of those counts, one bag per perturbed
gene. An anchor's reduced cells can be written to a part file by one process
and turned into bags by another once ``T`` is known.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data.basal import (
    ResponseCells,
    perturbseq_control_library_sizes,
    read_perturbseq_response_cells,
)
from src.data.expression import log_normalize
from src.data.response_streaming import read_xatlas_response_cells

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
    return perturbseq_control_library_sizes(
        source.h5ad_path,
        control_label=source.control_label,
        perturbation_col=source.perturbation_col,
        var_ensembl_col=source.var_ensembl_col,
        label=f"{model_id} response source controls",
    )


def response_cells(
    source: ResponseSource,
    model_id: str,
    hvg_order: Sequence[str],
    *,
    max_cells_per_gene: int | None,
    total_cells_per_line: int | None,
    seed: int,
) -> ResponseCells:
    """Raw-count perturbed cells of one anchor, reduced to STATE's HVG order.

    STATE HVGs absent from the source are zero columns.
    """
    _LOGGER.info("Reading perturbed cells of response anchor %s", model_id)
    caps = dict(
        max_cells_per_gene=max_cells_per_gene,
        total_cells_per_line=total_cells_per_line,
        seed=seed,
    )
    if isinstance(source, PerturbseqSource):
        cells = read_perturbseq_response_cells(
            source.h5ad_path,
            control_label=source.control_label,
            perturbation_col=source.perturbation_col,
            var_ensembl_col=source.var_ensembl_col,
            symbol_col=source.target_gene_symbol_col,
            gene_order=hvg_order,
            label=f"{model_id} response source",
            **caps,
        )
    else:
        cells = read_xatlas_response_cells(
            source.shard_dir,
            source.gene_metadata_path,
            model_id=model_id,
            shard_glob=source.shard_glob,
            control_label=source.control_label,
            pass_guide_filter_value=source.pass_guide_filter_value,
            symbol_col=source.target_gene_symbol_col,
            gene_order=hvg_order,
            **caps,
        )
    _LOGGER.info(
        "%s: %d perturbed cells, %d perturbations",
        model_id,
        len(cells.labels),
        len(set(cells.labels.tolist())),
    )
    if not cells.present.all():
        _LOGGER.warning(
            "%s: %d/%d STATE HVGs absent from the response source, zero-filled",
            model_id,
            int((~cells.present).sum()),
            len(cells.present),
        )
    return cells


def condition_rows(labels: Sequence[str]) -> dict[str, np.ndarray]:
    """Rows of each perturbed gene (labels stripped, upper-cased), genes sorted."""
    genes = pd.Series(labels).astype(str).str.strip().str.upper()
    groups = pd.Series(np.arange(len(genes))).groupby(genes.to_numpy()).indices
    return {str(gene): groups[gene] for gene in sorted(groups)}


@dataclass(frozen=True)
class ResponsePart:
    """One anchor's reduced perturbed cells, grouped by perturbed gene.

    Rows ``offsets[i]:offsets[i + 1]`` are the cells of ``genes[i]``.
    """

    genes: tuple[str, ...]
    offsets: np.ndarray
    library_sizes: np.ndarray
    hvg_counts: csr_matrix

    def bag(self, index: int, target_sum: float) -> np.ndarray:
        """Log-space HVG bag of ``genes[index]``."""
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return log_normalize(
            self.hvg_counts[start:stop], self.library_sizes[start:stop], target_sum
        )


def response_part(cells: ResponseCells) -> ResponsePart:
    """Group reduced cells by perturbed gene; each group keeps its row order."""
    rows = condition_rows(cells.labels)
    order = np.concatenate([np.zeros(0, dtype=np.int64), *rows.values()])
    return ResponsePart(
        genes=tuple(rows),
        offsets=np.cumsum([0, *(len(group) for group in rows.values())]),
        library_sizes=cells.library_sizes[order],
        hvg_counts=cells.hvg_counts[order, :],
    )


def write_response_part(
    path: Path,
    source: ResponseSource,
    model_id: str,
    hvg_order: Sequence[str],
    *,
    max_cells_per_gene: int | None,
    total_cells_per_line: int | None,
    seed: int,
) -> Path:
    """Read one anchor and write its :class:`ResponsePart` to ``path`` (``.npz``)."""
    part = response_part(
        response_cells(
            source,
            model_id,
            hvg_order,
            max_cells_per_gene=max_cells_per_gene,
            total_cells_per_line=total_cells_per_line,
            seed=seed,
        )
    )
    with Path(path).open("wb") as handle:
        np.savez(
            handle,
            genes=np.asarray(part.genes, dtype=str),
            offsets=np.asarray(part.offsets, dtype=np.int64),
            library_sizes=part.library_sizes,
            hvg_data=part.hvg_counts.data,
            hvg_indices=part.hvg_counts.indices,
            hvg_indptr=part.hvg_counts.indptr,
            hvg_shape=np.asarray(part.hvg_counts.shape, dtype=np.int64),
        )
    return Path(path)


def read_response_part(path: Path) -> ResponsePart:
    with np.load(path) as payload:
        return ResponsePart(
            genes=tuple(str(gene) for gene in payload["genes"]),
            offsets=payload["offsets"],
            library_sizes=payload["library_sizes"],
            hvg_counts=csr_matrix(
                (payload["hvg_data"], payload["hvg_indices"], payload["hvg_indptr"]),
                shape=tuple(int(n) for n in payload["hvg_shape"]),
            ),
        )


def part_conditions(path: Path) -> list[tuple[str, int]]:
    """``(gene, n_cells)`` of a part file, without reading its cells."""
    with np.load(path) as payload:
        genes, offsets = payload["genes"], payload["offsets"]
    return [(str(gene), int(n)) for gene, n in zip(genes, np.diff(offsets))]


def response_bags(
    parts: Mapping[str, Path], keys: Sequence[tuple[str, str]], target_sum: float
) -> Iterator[np.ndarray]:
    """Log-space bag of each ``(ModelID, gene)`` key, in order.

    Keys of one anchor must be consecutive; one part is resident at a time.
    """
    model_id, part, index = None, None, {}
    for key_model_id, gene in keys:
        if key_model_id != model_id:
            model_id, part = key_model_id, read_response_part(parts[key_model_id])
            index = {name: position for position, name in enumerate(part.genes)}
        yield part.bag(index[gene], target_sum)


__all__ = [
    "PerturbseqSource",
    "ResponsePart",
    "ResponseSource",
    "XatlasOrionSource",
    "condition_rows",
    "control_library_sizes",
    "load_response_sources",
    "part_conditions",
    "read_response_part",
    "response_bags",
    "response_cells",
    "response_part",
    "write_response_part",
]
