"""The response anchors' observed perturbation responses, from source to prepared
targets: Perturb-seq and X-Atlas-Orion response cells capped per gene (the X-Atlas
shards streamed through per-gene reservoirs), per-anchor parts, and the prepared
targets cache every condition is read from."""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, NamedTuple, TypeVar

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse
from scipy.sparse import csr_matrix

from src.data.basal import (
    XATLAS_GENE_METADATA_TOKEN_COL,
    align_columns,
    read_xatlas_shard,
    require_ensembl_source,
    require_raw_counts,
    row_chunks,
    xatlas_token_rows,
)
from src.data.expression import library_sizes, log_normalize

_LOGGER = logging.getLogger(__name__)


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


def _select_indices_deterministic(
    candidate_indices: np.ndarray, max_cells: int, seed: int
) -> np.ndarray:
    """Seeded subsample of at most ``max_cells`` row indices (sorted)."""
    if max_cells >= len(candidate_indices):
        return candidate_indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidate_indices), size=max_cells, replace=False)
    return np.sort(candidate_indices[chosen])


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
    grouped_indices: Mapping[str, np.ndarray], max_per_group: int, seed: int
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
        require_ensembl_source(backed.var, var_ensembl_col, h5ad_path)
        labels = backed.obs[perturbation_col].astype(str).to_numpy()
        control = np.flatnonzero(labels == control_label)
        if not control.size:
            raise ValueError(
                f"No control cells found for {perturbation_col}={control_label!r} "
                f"in {h5ad_path}"
            )
        sizes = []
        for chunk in row_chunks(backed.X, control, sparsify_chunks=True):
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
    max_cells_per_gene: int,
    seed: int = 0,
) -> ResponseCells:
    """Perturbed cells of a Perturb-seq h5ad, columns aligned to ``gene_order``.

    The per-gene cap is applied before any expression value is read. Rows come
    out grouped by sorted label.
    Each row chunk is reduced to library sizes and aligned columns before the
    next is read.
    """
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        require_ensembl_source(backed.var, var_ensembl_col, h5ad_path)
        labels = backed.obs[perturbation_col].astype(str).to_numpy()
        candidate = np.flatnonzero(labels != control_label)
        if not candidate.size:
            raise ValueError(
                f"No perturbed cells found for {perturbation_col}!={control_label!r} "
                f"in {h5ad_path}"
            )
        grouped = _group_candidate_indices_by_label(candidate, labels[candidate])
        selected = _cap_indices_by_group(grouped, max_cells_per_gene, seed)
        symbols = backed.var[symbol_col].astype(str)
        order = np.argsort(selected)
        sizes, counts = [], []
        for chunk in row_chunks(backed.X, selected[order], sparsify_chunks=True):
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
    max_cells_per_gene: int,
    seed: int,
) -> ResponseCells:
    """Raw-count perturbed cells of one anchor, reduced to STATE's HVG order.

    STATE HVGs absent from the source are zero columns.
    """
    _LOGGER.info("Reading perturbed cells of response anchor %s", model_id)
    caps = dict(max_cells_per_gene=max_cells_per_gene, seed=seed)
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
    max_cells_per_gene: int,
    seed: int,
) -> Path:
    """Read one anchor and write its :class:`ResponsePart` to ``path`` (``.npz``)."""
    part = response_part(
        response_cells(
            source,
            model_id,
            hvg_order,
            max_cells_per_gene=max_cells_per_gene,
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


#: Parquet shards read ahead of the reservoir, one thread each.
PARQUET_PREFETCH_THREADS: Final[int] = 8

_T = TypeVar("_T")


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
    model_id: str,
) -> ResponseCells:
    """Cells in ``sorted(reservoirs)`` then slot order.

    Every slot is released as it is consumed.
    """
    labels: list[str] = []
    kept: list[_ReducedCell | None] = []
    for gene in sorted(reservoirs):
        for cell in reservoirs.pop(gene):
            labels.append(gene)
            kept.append(cell)
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
    max_cells_per_gene: int,
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
            if len(bucket) < max_cells_per_gene:
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
    return _drain(reservoirs, len(gene_order), model_id=model_id)


@dataclass(frozen=True)
class ResponseTargetsCache:
    """Memory-mapped log-space target cells keyed by ``(ModelID, gene)``."""

    model_ids: tuple[str, ...]
    genes: tuple[str, ...]
    target_cells: np.ndarray
    offsets: np.ndarray

    @property
    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.model_ids, self.genes, strict=True))

    def target_bag(self, index: int) -> np.ndarray:
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return np.asarray(self.target_cells[start:stop])


def write_response_targets(
    directory: Path,
    keys: Sequence[tuple[str, str]],
    lengths: Sequence[int],
    width: int,
    bags: Iterable[np.ndarray],
) -> Path:
    """Atomically write one ``[lengths[i], width]`` bag per key.

    ``bags`` is consumed in key order and written straight to disk, so only
    one bag needs to be resident at a time.
    """
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    lengths = [int(length) for length in lengths]
    if len(lengths) != len(keys):
        raise ValueError(f"{len(keys)} keys but {len(lengths)} bag lengths")
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tmp_dir = directory.parent / f".tmp-{directory.name}-{uuid.uuid4().hex}"
    tmp_dir.mkdir()
    try:
        cells = np.lib.format.open_memmap(
            tmp_dir / "target_cells.npy",
            mode="w+",
            dtype=np.float32,
            shape=(int(offsets[-1]), int(width)),
        )
        for key, start, length, bag in zip(
            keys, offsets[:-1], lengths, bags, strict=True
        ):
            if bag.shape != (length, width):
                raise ValueError(f"{key}: bag shape {bag.shape} != {(length, width)}")
            cells[start : start + length] = bag
        cells.flush()
        del cells
        np.save(tmp_dir / "offsets.npy", offsets)
        pd.DataFrame(
            {
                "model_id": [model_id for model_id, _ in keys],
                "gene": [gene for _, gene in keys],
                "n_cells": lengths,
            }
        ).to_parquet(tmp_dir / "conditions.parquet")
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(tmp_dir, directory)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return directory


def open_response_targets(directory: Path) -> ResponseTargetsCache:
    """Open a prepared response cache read-only."""
    directory = Path(directory)
    conditions = pd.read_parquet(directory / "conditions.parquet")
    return ResponseTargetsCache(
        model_ids=tuple(conditions["model_id"].astype(str)),
        genes=tuple(conditions["gene"].astype(str)),
        target_cells=np.load(directory / "target_cells.npy", mmap_mode="r"),
        offsets=np.load(directory / "offsets.npy"),
    )
