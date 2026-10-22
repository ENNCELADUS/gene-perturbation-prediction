"""Build the extra single-cell lines' raw-UMI h5ad from their public deposits.

One run takes the sources of one wave of the spec
(``configs/data/extra_single_cell_sources_26Q1.json``). For every library of a
source it reads the deposited counts, calls cells, names each cell's line (the
whole library, the authors' per-cell labels, hashtags, or, for a pool without
labels, the member whose DepMap bulk RNA its cluster matches), keeps the basal
arms and applies the QC floor. Line names become ModelIDs only through accepted
rows of the identity map; a name the map lacks raises. It then writes one
``<output-dir>/<ModelID>.h5ad`` for each line the map's choice takes from this
source (never a line of the 226), and ``<output-dir>/qc/<source>.json`` last. A
source whose QC report exists is skipped; no h5ad is overwritten.

uv run python -m src.data.prepare.build_extra_single_cell_sources \
    --spec configs/data/extra_single_cell_sources_26Q1.json --wave 1 --print-downloads
uv run python -m src.data.prepare.build_extra_single_cell_sources \
    --spec configs/data/extra_single_cell_sources_26Q1.json --wave 1 \
    --output-dir data/sl_dependency_v0/processed/basal_raw_umi
"""  # noqa: E501

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import tarfile
import zipfile
from array import array
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from src.data.basal import align_columns, assert_tx1_input_contract
from src.data.depmap import read_depmap_matrix, read_model_ids, read_models
from src.data.prepare.build_extra_single_cell_lines import Admission, admit, read_map
from src.data.prepare.prepare_cell_line_atlas_raw_umi import (
    assert_raw_integer,
    load_gene_mapping,
    map_and_collapse_genes,
    probe_tx1_loader,
)
from src.data.splits import FixedSplit, load_geneeffect_split

#: Keys each reader and each identity kind takes (all required).
READERS = {
    "tenx_mtx": ("matrix", "features", "barcodes", "header", "gene_id_prefix"),
    "tenx_h5": ("h5",),
    "tenx_bam": ("bam", "cell_prefix"),
    "table": ("path", "sep", "skiprows", "genes_by_cells"),
    "triplets": ("matrix", "cell_table", "gene_table"),
    "mtx_names": ("matrix", "gene_table", "cell_table"),
    "anndata": ("path", "counts", "gene_column", "symbol_column"),
    "tahoe": ("shards", "gene_metadata", "cell_lines", "drug"),
}
IDENTITIES = {
    "line": ("identifier",),
    "labels": (
        "path",
        "sep",
        "cell_column",
        "label_column",
        "arm_column",
        "keep",
        "cell_prefix",
        "cell_suffix",
        "skiprows",
    ),
    "barcode": ("pattern", "identifier", "rename"),
    "hashtag": ("counts", "cell_suffix", "hashtags", "min_count", "min_ratio"),
    "pool": ("members", "clusters", "min_margin"),
    "rt_hash": ("rt_barcodes", "rt_map", "hash_table", "min_count", "min_ratio"),
}


@dataclass
class Library:
    """One library as deposited: integer counts of cells × genes."""

    counts: sparse.csr_matrix
    cells: list[str]
    genes: list[str]
    symbols: list[str] | None
    tags: sparse.csr_matrix | None
    tag_names: list[str]
    obs: pd.DataFrame | None


def _require(spec: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    missing = sorted(set(keys) - set(spec))
    if missing:
        raise ValueError(f"{where}: missing keys {missing}")


def _expression_and_tags(
    counts: sparse.csr_matrix,
    cells: list[str],
    features: pd.DataFrame,
) -> Library:
    """Split 10x features (``id``, ``name``, ``type``) into expression and tags."""
    types = features["type"].fillna("Gene Expression").to_numpy()
    expression = types == "Gene Expression"
    tags = ~expression
    return Library(
        counts=counts[:, np.flatnonzero(expression)].tocsr(),
        cells=cells,
        genes=features.loc[expression, "id"].tolist(),
        symbols=features.loc[expression, "name"].tolist(),
        tags=counts[:, np.flatnonzero(tags)].tocsr() if tags.any() else None,
        tag_names=features.loc[tags, "name"].tolist(),
        obs=None,
    )


def _read_tenx_mtx(root: Path, spec: Mapping[str, Any]) -> Library:
    """10x mtx (genes x cells). ``header``: the features and barcodes files start
    with a header row. ``gene_id_prefix`` (e.g. ``hg19_`` of a mixed-species
    reference): keep only features carrying it, without it."""
    counts = sparse.csr_matrix(mmread(root / spec["matrix"]).T)
    skip = 1 if spec["header"] else 0

    def table(name: str) -> pd.DataFrame:
        return pd.read_csv(
            root / spec[name], sep="\t", header=None, skiprows=skip, dtype=str
        )

    features = table("features").reindex(columns=range(3))
    features.columns = ["id", "name", "type"]
    features["name"] = features["name"].fillna(features["id"])
    prefix = spec["gene_id_prefix"]
    if prefix:
        ours = features["id"].str.startswith(prefix).to_numpy()
        counts, features = counts[:, np.flatnonzero(ours)], features.loc[ours].copy()
        for column in ("id", "name"):
            features[column] = features[column].str.removeprefix(prefix)
    return _expression_and_tags(counts, table("barcodes")[0].tolist(), features)


def _read_tenx_h5(root: Path, spec: Mapping[str, Any]) -> Library:
    def text(values) -> list[str]:
        return [v.decode() if isinstance(v, bytes) else str(v) for v in values[:]]

    with h5py.File(root / spec["h5"], "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle[next(iter(handle))]
        shape = tuple(group["shape"][:])
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]), shape=shape
        )
        cells = text(group["barcodes"])
        if "features" in group:
            names = group["features"]
            features = pd.DataFrame(
                {
                    "id": text(names["id"]),
                    "name": text(names["name"]),
                    "type": text(names["feature_type"]),
                }
            )
        else:  # Cell Ranger 2: one genome group
            features = pd.DataFrame(
                {"id": text(group["genes"]), "name": text(group["gene_names"])}
            ).assign(type="Gene Expression")
    return _expression_and_tags(sparse.csr_matrix(matrix.T), cells, features)


def _read_tenx_bam(root: Path, spec: Mapping[str, Any]) -> Library:
    """Cell Ranger's UMI counts from its BAM: one count per read flagged as its
    molecule's representative (``xf`` bit 8) with a cell (``CB``), a UMI and one
    gene (``GX``, symbol ``GN``). Cells are named ``cell_prefix`` + ``CB``."""
    import pysam

    cells: dict[str, int] = {}
    genes: dict[str, int] = {}
    symbols: dict[str, str] = {}
    rows, columns = array("l"), array("l")
    with pysam.AlignmentFile(
        root / spec["bam"], "rb", check_sq=False, threads=8
    ) as handle:
        for read in handle.fetch(until_eof=True):
            try:  # a missing tag raises KeyError
                if not read.get_tag("xf") & 8:
                    continue
                cell, gene = read.get_tag("CB"), read.get_tag("GX")
                read.get_tag("UB")
            except KeyError:
                continue
            if ";" in gene:
                continue
            rows.append(cells.setdefault(cell, len(cells)))
            columns.append(genes.setdefault(gene, len(genes)))
            if gene not in symbols:
                symbols[gene] = read.get_tag("GN")
    counts = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int64), (np.asarray(rows), np.asarray(columns))),
        shape=(len(cells), len(genes)),
    )
    counts.sum_duplicates()
    return Library(
        counts=counts,
        cells=[spec["cell_prefix"] + cell for cell in cells],
        genes=list(genes),
        symbols=[symbols[gene] for gene in genes],
        tags=None,
        tag_names=[],
        obs=None,
    )


def _read_table(root: Path, spec: Mapping[str, Any]) -> Library:
    frame = pd.read_csv(
        root / spec["path"], sep=spec["sep"], skiprows=spec["skiprows"], index_col=0
    )
    if spec["genes_by_cells"]:
        frame = frame.T
    return Library(
        counts=sparse.csr_matrix(frame.to_numpy()),
        cells=[str(cell) for cell in frame.index],
        genes=[str(gene) for gene in frame.columns],
        symbols=None,
        tags=None,
        tag_names=[],
        obs=None,
    )


def _read_triplets(root: Path, spec: Mapping[str, Any]) -> Library:
    """A headerless (gene, cell, count) table, 1-based, with headerless cell and
    gene annotation tables (cell id first; gene id, then symbol), as sci-RNA-seq3
    pipelines write them."""

    def table(name: str) -> pd.DataFrame:
        return pd.read_csv(root / spec[name], sep="\t", header=None, dtype=str)

    cells, genes = table("cell_table")[0].tolist(), table("gene_table")
    entries = pd.read_csv(
        root / spec["matrix"], sep="\t", header=None, dtype=np.int64
    ).to_numpy()
    counts = sparse.csr_matrix(
        (entries[:, 2], (entries[:, 1] - 1, entries[:, 0] - 1)),
        shape=(len(cells), len(genes)),
    )
    return Library(
        counts=counts,
        cells=cells,
        genes=genes[0].tolist(),
        symbols=genes[1].tolist(),
        tags=None,
        tag_names=[],
        obs=None,
    )


def _read_mtx_names(root: Path, spec: Mapping[str, Any]) -> Library:
    """A genes x cells mtx whose names sit in R ``write.csv`` vectors: gene names
    are the ``gene_table``'s values, cell ids the ``cell_table``'s row names."""

    def vector(name: str) -> pd.DataFrame:
        return pd.read_csv(root / spec[name], dtype=str)

    return Library(
        counts=sparse.csr_matrix(mmread(root / spec["matrix"]).T),
        cells=vector("cell_table").iloc[:, 0].tolist(),
        genes=vector("gene_table").iloc[:, 1].tolist(),
        symbols=None,
        tags=None,
        tag_names=[],
        obs=None,
    )


def lognorm_counts(values: sparse.spmatrix, totals: np.ndarray) -> sparse.csr_matrix:
    """The integer counts behind ``log1p(count * s_cell)`` values, given each
    cell's UMI total: ``s_cell`` is the cell's ``expm1`` sum over its total.
    Raises unless every recovered value is within 0.01 of a positive integer and
    every cell's counts sum to its total (so a deposit normalised over other
    genes, or with corrected counts, fails)."""
    scaled = sparse.csr_matrix(values, dtype=np.float64)
    scaled.data = np.expm1(scaled.data)
    sums = np.asarray(scaled.sum(axis=1)).ravel()
    totals = np.asarray(totals, dtype=np.float64)
    if np.any(sums <= 0) or np.any(totals <= 0):
        raise ValueError("lognorm counts: a cell without values or total")
    counts = sparse.csr_matrix(sparse.diags(totals / sums) @ scaled)
    rounded = np.rint(counts.data)
    worst = float(np.max(np.abs(counts.data - rounded))) if counts.nnz else 0.0
    if worst > 0.01 or (rounded < 1).any():
        raise ValueError(f"lognorm counts: values are not integers (off by {worst})")
    counts.data = rounded
    counts = sparse.csr_matrix(counts, dtype=np.int64)
    if not np.array_equal(np.asarray(counts.sum(axis=1)).ravel(), totals):
        raise ValueError("lognorm counts: recovered counts do not sum to the totals")
    return counts


def _read_loom(path: Path) -> ad.AnnData:
    """A loom file (genes x cells HDF5, velocyto layout) as cells x genes; var is
    the row attributes indexed by ``Gene``, obs the column attributes indexed by
    ``CellID``, layers the loom layers."""

    def frame(group, index: str) -> pd.DataFrame:
        columns = {
            key: [v.decode() if isinstance(v, bytes) else v for v in group[key][:]]
            for key in group
            if group[key].ndim == 1
        }
        return pd.DataFrame(columns).set_index(index, drop=False)

    def matrix(dataset) -> sparse.csr_matrix:
        blocks = [
            sparse.csr_matrix(dataset[:, start : start + 4096].T)
            for start in range(0, dataset.shape[1], 4096)
        ]
        return sparse.vstack(blocks).tocsr()

    with h5py.File(path, "r") as handle:
        var, obs = (
            frame(handle["row_attrs"], "Gene"),
            frame(handle["col_attrs"], "CellID"),
        )
        adata = ad.AnnData(X=matrix(handle["matrix"]), obs=obs, var=var)
        for name in handle.get("layers", {}):
            adata.layers[name] = matrix(handle["layers"][name])
    adata.obs_names, adata.var_names = obs.index.astype(str), var.index.astype(str)
    return adata


def _read_anndata(root: Path, spec: Mapping[str, Any]) -> Library:
    """``counts``: ``X``, ``raw``, ``layer:<name>``, or ``lognorm:<one of those>:
    <obs column>`` for log-normalised values whose integer counts are recovered
    with each cell's UMI total from that column (``lognorm_counts``)."""
    path = root / spec["path"]
    adata = _read_loom(path) if path.suffix == ".loom" else ad.read_h5ad(path)
    where, total_column = spec["counts"], None
    if where.startswith("lognorm:"):
        where, total_column = where.removeprefix("lognorm:").rsplit(":", 1)
    if where == "X":
        counts, var = adata.X, adata.var
    elif where == "raw":
        counts, var = adata.raw.X, adata.raw.var
    elif where.startswith("layer:"):
        counts, var = adata.layers[where.split(":", 1)[1]], adata.var
    else:
        raise ValueError(f"{path}: counts must be X, raw or layer:<name>")
    if total_column is not None:
        counts = lognorm_counts(counts, adata.obs[total_column].to_numpy())

    def column(name):
        return var.index.astype(str) if name == "index" else var[name].astype(str)

    return Library(
        counts=sparse.csr_matrix(counts),
        cells=adata.obs_names.astype(str).tolist(),
        genes=column(spec["gene_column"]).tolist(),
        symbols=(
            None
            if spec["symbol_column"] is None
            else column(spec["symbol_column"]).tolist()
        ),
        tags=None,
        tag_names=[],
        obs=adata.obs.copy(),
    )


def _read_tahoe(root: Path, spec: Mapping[str, Any]) -> Library:
    """Tahoe-100M parquet shards (paths relative to the working directory): every
    cell of the listed Cellosaurus lines given ``drug``, labelled by
    ``cell_line_id`` in obs."""
    from src.data.basal import assemble_tahoe_matrix

    wanted = set(spec["cell_lines"])
    columns = ["genes", "expressions", "BARCODE_SUB_LIB_ID", "cell_line_id", "drug"]
    cells, lines, reservoir = [], [], []
    keep = [("cell_line_id", "in", sorted(wanted)), ("drug", "==", spec["drug"])]
    for shard in sorted(Path(spec["shards"]).glob("*.parquet")):
        frame = pd.read_parquet(shard, columns=columns, filters=keep)
        for number, row in frame.iterrows():
            cells.append(f"{shard.name}:{number}:{row['BARCODE_SUB_LIB_ID']}")
            lines.append(str(row["cell_line_id"]))
            reservoir.append((row["genes"], row["expressions"]))
    if not cells:
        raise ValueError(f"{spec['shards']}: no cells of {sorted(wanted)}")
    metadata = pd.read_parquet(spec["gene_metadata"]).set_index("token_id")
    counts, var = assemble_tahoe_matrix(reservoir, metadata)
    return Library(
        counts=sparse.csr_matrix(counts),
        cells=cells,
        genes=var.index.astype(str).tolist(),
        symbols=var["gene_symbol"].astype(str).tolist(),
        tags=None,
        tag_names=[],
        obs=pd.DataFrame({"cell_line_id": lines}, index=cells),
    )


_READ = {
    "tenx_mtx": _read_tenx_mtx,
    "tenx_h5": _read_tenx_h5,
    "tenx_bam": _read_tenx_bam,
    "table": _read_table,
    "triplets": _read_triplets,
    "mtx_names": _read_mtx_names,
    "anndata": _read_anndata,
    "tahoe": _read_tahoe,
}


def read_library(root: Path, spec: Mapping[str, Any]) -> Library:
    """Read one library with its reader; counts must be raw integers."""
    reader = spec["reader"]
    _require(spec, READERS[reader], f"{reader} library")
    library = _READ[reader](root, spec)
    assert_raw_integer(library.counts, f"{root} {reader} library")
    if library.counts.shape != (len(library.cells), len(library.genes)):
        raise ValueError(f"{root}: counts {library.counts.shape} do not match ids")
    if len(set(library.cells)) != len(library.cells):
        raise ValueError(f"{root}: duplicate cell ids")
    return library


def knee(totals: np.ndarray, min_umi: int) -> float:
    """UMI total at the inflection of the log-log barcode-rank curve: the left end
    of its steepest segment over distinct totals of at least ``min_umi`` (ignoring
    the first ten ranks). Barcodes at or above it are cells."""
    values = totals[totals >= min_umi]
    unique, counts = np.unique(values, return_counts=True)
    unique, counts = unique[::-1].astype(np.float64), counts[::-1]
    last_rank = np.cumsum(counts)
    allowed = last_rank[:-1] >= 10
    if not allowed.any():
        return float(min_umi)
    mid_rank = last_rank - (counts - 1) / 2
    slope = np.diff(np.log10(unique)) / np.diff(np.log10(mid_rank))
    step = np.flatnonzero(allowed)[np.argmin(slope[allowed])]
    return float(unique[step])


def call_hashtags(
    counts: np.ndarray,
    names: Sequence[str],
    hashtags: Mapping[str, str],
    *,
    min_count: int,
    min_ratio: float,
) -> tuple[list[str | None], dict[str, int]]:
    """Each cell's identifier from its hashtag counts: a singlet's top tag
    (at least ``min_count`` and ``min_ratio`` times the second); negatives and
    doublets get None."""
    columns = [list(names).index(tag) for tag in hashtags]
    values = np.asarray(counts)[:, columns].astype(np.float64)
    order = np.sort(values, axis=1)
    top = order[:, -1]
    second = order[:, -2] if values.shape[1] > 1 else np.zeros(len(values))
    best = np.argmax(values, axis=1)
    identifiers = list(hashtags.values())
    calls, tally = [], Counter()
    for index in range(len(values)):
        if top[index] < min_count:
            calls.append(None)
            tally["negative"] += 1
        elif top[index] >= min_ratio * second[index] and top[index] > second[index]:
            calls.append(identifiers[best[index]])
            tally["singlet"] += 1
        else:
            calls.append(None)
            tally["doublet"] += 1
    return calls, dict(tally)


def assign_pool(
    counts: sparse.csr_matrix,
    symbols: Sequence[str],
    members: Mapping[str, str],
    bulk: pd.DataFrame,
    *,
    clusters: int,
    min_margin: float,
    seed: int = 0,
) -> tuple[list[str | None], list[dict[str, Any]]]:
    """Each cell's identifier in a pool without labels: cluster the cells, then
    give each cluster the member (identifier to ModelID) whose DepMap bulk RNA
    deviation best correlates with the cluster's, when it beats the second best
    by ``min_margin``; other clusters' cells get None."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    missing = sorted(set(members.values()) - set(bulk.index))
    if missing:
        raise ValueError(f"pool members without DepMap bulk RNA: {missing}")
    genes = [g for g in dict.fromkeys(s.upper() for s in symbols) if g in bulk.columns]
    aligned, _ = align_columns(counts, [s.upper() for s in symbols], genes)
    library = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    logged = sparse.diags(1e4 / np.maximum(library, 1)) @ aligned
    logged.data = np.log1p(logged.data)
    variance = np.asarray(logged.power(2).mean(axis=0)).ravel() - np.square(
        np.asarray(logged.mean(axis=0)).ravel()
    )
    top = np.argsort(variance)[::-1][:2000]
    reduced = PCA(
        n_components=min(30, len(top), counts.shape[0] - 1), random_state=seed
    ).fit_transform(logged[:, top].toarray())
    labels = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit_predict(
        reduced
    )
    profile = bulk.loc[list(members.values()), genes]
    markers = profile.var(axis=0).sort_values(ascending=False).index[:1000]
    position = [genes.index(g) for g in markers]
    member_dev = profile[markers] - profile[markers].mean(axis=0)
    means = np.vstack(
        [
            np.asarray(logged[labels == k][:, position].mean(axis=0)).ravel()
            for k in range(clusters)
        ]
    )
    cluster_dev = means - means.mean(axis=0)
    identifiers = list(members)
    table, assigned = [], {}
    for k in range(clusters):
        r = [float(np.corrcoef(cluster_dev[k], row)[0, 1]) for row in member_dev.values]
        ranked = np.argsort(r)[::-1]
        margin = r[ranked[0]] - (r[ranked[1]] if len(r) > 1 else -1.0)
        chosen = identifiers[ranked[0]] if margin >= min_margin else None
        assigned[k] = chosen
        table.append(
            {
                "cluster": k,
                "cells": int((labels == k).sum()),
                "r": dict(zip(identifiers, r, strict=True)),
                "margin": margin,
                "assigned": chosen,
            }
        )
    return [assigned[k] for k in labels], table


def _identify(
    library: Library,
    identity: Mapping[str, Any],
    root: Path,
    members: Mapping[str, str],
    bulk_path: Path,
) -> tuple[list[str | None], list[str | None], dict[str, Any]]:
    """Per-cell identifier and arm (None where the identity rule names no line or
    no arm), and the rule's report."""
    kind = identity["kind"]
    _require(identity, IDENTITIES[kind], f"{kind} identity")
    n = len(library.cells)
    if kind == "line":
        return [identity["identifier"]] * n, [None] * n, {}
    if kind == "labels":
        if identity["path"] is None:
            table = library.obs.assign(_cell=library.obs.index.astype(str))
            cell_column = "_cell"
        else:
            table = pd.read_csv(
                root / identity["path"],
                sep=identity["sep"],
                skiprows=identity["skiprows"],
                dtype=str,
            )
            cell_column = identity["cell_column"]
            if cell_column is None:  # cell ids are the table's unnamed first column
                if isinstance(table.index, pd.RangeIndex):
                    table = table.rename(columns={table.columns[0]: "_cell"})
                else:  # an R table: header one field short, ids became the index
                    table = table.rename_axis("_cell").reset_index()
                cell_column = "_cell"
        for column, values in identity["keep"].items():
            table = table.loc[table[column].astype(str).isin(values)]
        cells = identity["cell_prefix"] + table[cell_column] + identity["cell_suffix"]
        table = table.set_index(cells.to_numpy())
        if not table.index.is_unique:
            raise ValueError(f"{identity['path']}: duplicate labelled cells")
        found = table.reindex(library.cells)
        arms = (
            [None] * n
            if identity["arm_column"] is None
            else found[identity["arm_column"]].tolist()
        )
        labels = found[identity["label_column"]].tolist()
        clean = [None if pd.isna(v) else str(v) for v in labels]
        return (
            clean,
            [None if pd.isna(a) else str(a) for a in arms],
            {"labelled_cells": int(sum(v is not None for v in clean))},
        )
    if kind == "barcode":
        # The line (group ``line``, else ``identifier``) and arm (group ``arm``)
        # carried by each cell id; ``rename`` turns a captured line into a name.
        pattern = re.compile(identity["pattern"])
        lines, arms = [], []
        for cell in library.cells:
            match = pattern.search(cell)
            groups = match.groupdict() if match else {}
            line = groups.get("line", identity["identifier"]) if match else None
            lines.append(None if line is None else identity["rename"].get(line, line))
            arms.append(groups.get("arm"))
        return lines, arms, {"matched_cells": int(sum(v is not None for v in lines))}
    if kind == "hashtag":
        if identity["counts"] is None:
            tags, names = library.tags, library.tag_names
            rows = np.arange(n)
        else:
            other = read_library(root, identity["counts"])
            if other.tags is not None:  # 10x features typed as antibody capture
                tags, names = other.tags, other.tag_names
            else:
                tags, names = other.counts, other.genes
            suffix = identity["cell_suffix"]
            position = {cell + suffix: i for i, cell in enumerate(other.cells)}
            rows = np.asarray([position.get(cell, -1) for cell in library.cells])
        if tags is None:
            raise ValueError("hashtag identity: the library has no tag features")
        dense = np.zeros((n, tags.shape[1]))
        present = rows >= 0
        dense[present] = tags[rows[present]].toarray()
        calls, tally = call_hashtags(
            dense,
            names,
            identity["hashtags"],
            min_count=identity["min_count"],
            min_ratio=identity["min_ratio"],
        )
        calls = [c if ok else None for c, ok in zip(calls, present, strict=True)]
        return calls, [None] * n, {"hashtags": tally}
    if kind == "pool":
        if library.symbols is None:
            raise ValueError("pool identity needs gene symbols")
        if len(identity["members"]) < 2:
            raise ValueError("a pool has at least two members")
        bulk = read_depmap_matrix(bulk_path)
        pool = {name: members[name] for name in identity["members"]}
        calls, table = assign_pool(
            library.counts,
            library.symbols,
            pool,
            bulk,
            clusters=identity["clusters"],
            min_margin=identity["min_margin"],
        )
        return calls, [None] * n, {"clusters": table}
    if kind == "rt_hash":
        return _rt_hash(library, identity, root)
    raise ValueError(f"unknown identity kind {kind!r}")


def _rt_hash(
    library: Library, identity: Mapping[str, Any], root: Path
) -> tuple[list[str | None], list[str | None], dict[str, Any]]:
    """sci-Plex: the line from the cell's RT barcode (the fifth ``_`` field of a
    ``P5_P7_RT_BC_<n>_Lig_BC_<m>`` id, named in ``rt_barcodes`` as ``RT_BC_<n>``)
    through the authors' RT-well map, and the arm from its hash: a singlet's top
    oligo (``call_hashtags``) up to its first ``_`` (``DMSO_0nM_10H01`` is DMSO)."""
    barcodes = pd.read_csv(
        root / identity["rt_barcodes"], sep=r"\s+", header=None, dtype=str
    )
    sequence = dict(zip(barcodes[0], barcodes[1], strict=True))
    wells = pd.read_csv(root / identity["rt_map"], dtype=str)
    line_of = dict(zip(wells["Barcode"], wells["Cell Line"], strict=True))
    lines = [
        line_of.get(sequence.get(f"RT_BC_{cell.split('_')[4]}"))
        for cell in library.cells
    ]
    hashes = pd.read_csv(
        root / identity["hash_table"],
        sep="\t",
        header=None,
        names=["sample", "cell", "oligo", "axis", "umis"],
        dtype={"cell": str, "oligo": str, "umis": np.int64},
    )
    oligos = sorted(set(hashes["oligo"]))
    position = {cell: i for i, cell in enumerate(library.cells)}
    found = hashes["cell"].map(position)
    hashes = hashes.loc[found.notna()]
    tags = sparse.csr_matrix(
        (
            hashes["umis"].to_numpy(),
            (
                found.dropna().astype(np.int64).to_numpy(),
                hashes["oligo"].map({o: i for i, o in enumerate(oligos)}).to_numpy(),
            ),
        ),
        shape=(len(library.cells), len(oligos)),
    )
    calls, tally = call_hashtags(
        tags.toarray(),
        oligos,
        {oligo: oligo for oligo in oligos},
        min_count=identity["min_count"],
        min_ratio=identity["min_ratio"],
    )
    arms = [None if call is None else call.split("_")[0] for call in calls]
    report = {
        "rt_lines": dict(Counter(line for line in lines if line is not None)),
        "hashtags": tally,
    }
    return lines, arms, report


@dataclass
class Context:
    """What every source of a run reads."""

    spec: Mapping[str, Any]
    candidates: pd.DataFrame
    admission: Admission
    split: FixedSplit
    models: pd.DataFrame
    gene_metadata: pd.DataFrame
    symbol_to_ensembl: dict[str, str]


def _mito(library: Library, gene_metadata: pd.DataFrame, id_kind: str) -> np.ndarray:
    if library.symbols is not None:
        symbols = library.symbols
    elif id_kind == "symbol":
        symbols = library.genes
    else:
        ids = [gene.split(".")[0] for gene in library.genes]
        symbols = gene_metadata["gene_symbol"].reindex(ids).fillna("").tolist()
    return np.asarray([str(s).upper().startswith("MT-") for s in symbols])


def build_source(
    name: str,
    source: Mapping[str, Any],
    context: Context,
    output: Path,
    wanted: Collection[str],
) -> dict[str, Any]:
    """Write the ``wanted`` lines' cells from this source; return its QC report."""
    _require(
        source, ("wave", "files", "unpack", "gene_id_kind", "arms", "libraries"), name
    )
    root = Path(context.spec["raw_root"]) / name
    for archive in source["unpack"]:
        _unpack(root / archive, root)
    accepted = context.candidates.loc[
        (context.candidates["source"] == name) & (context.candidates["review"] == "ok")
    ]
    ours = dict(zip(accepted["identifier"], accepted["model_id"], strict=True))
    flagged = set(
        context.candidates.loc[
            (context.candidates["source"] == name)
            & (context.candidates["review"] != "ok"),
            "identifier",
        ]
    )
    wanted = set(wanted)
    blocks: dict[str, list[tuple[Any, ...]]] = {m: [] for m in wanted}
    dropped, reports = Counter(), []
    floor = context.spec["min_genes"], context.spec["max_mito_fraction"]
    for index, spec in enumerate(source["libraries"]):
        _require(
            spec, ("reader", "cells", "identity", "arm"), f"{name} library {index}"
        )
        library = read_library(root, spec)
        report: dict[str, Any] = {"library": index, "cells_read": len(library.cells)}
        totals = np.asarray(library.counts.sum(axis=1)).ravel()
        if spec["cells"] == "knee":
            cut = knee(totals, int(spec["knee_min_umi"]))
            called = totals >= cut
            report["knee_umi"] = cut
        elif spec["cells"] == "deposited":
            called = np.ones(len(totals), dtype=bool)
        else:
            raise ValueError(f"{name}: cells must be deposited or knee")
        identifiers, arms, report["identity"] = _identify(
            library,
            spec["identity"],
            root,
            ours,
            Path(context.spec["bulk_expression_path"]),
        )
        arms = [arm if arm is not None else spec["arm"] for arm in arms]
        detected = np.asarray((library.counts > 0).sum(axis=1)).ravel()
        mito = _mito(library, context.gene_metadata, source["gene_id_kind"])
        mito_umi = np.asarray(
            library.counts[:, np.flatnonzero(mito)].sum(axis=1)
        ).ravel()
        report["mito_genes"] = int(mito.sum())
        keep = np.zeros(len(totals), dtype=bool)
        lines = np.empty(len(totals), dtype=object)
        for cell, identifier in enumerate(identifiers):
            if not called[cell]:
                dropped["not called"] += 1
            elif identifier is None:
                dropped["no line"] += 1
            elif identifier in flagged:
                dropped["identity not accepted"] += 1
            elif identifier not in ours:
                raise ValueError(f"{name}: label {identifier!r} absent from the map")
            elif ours[identifier] not in wanted:
                dropped[
                    "line of the 226"
                    if ours[identifier] in context.split.all_model_ids
                    else "line not taken from this source"
                ] += 1
            elif arms[cell] not in source["arms"]:
                dropped["arm"] += 1
            elif detected[cell] < floor[0] or mito_umi[cell] > floor[1] * totals[cell]:
                dropped["QC floor"] += 1
            else:
                keep[cell] = True
                lines[cell] = ours[identifier]
        report["cells_kept"] = int(keep.sum())
        reports.append(report)
        if not keep.any():
            continue
        rows = np.flatnonzero(keep)
        mapped, var, gene_stats = map_and_collapse_genes(
            library.counts[rows],
            [gene.split(".")[0] for gene in library.genes]
            if source["gene_id_kind"] == "ensembl"
            else library.genes,
            source["gene_id_kind"],
            context.gene_metadata,
            context.symbol_to_ensembl,
        )
        report["genes"] = gene_stats
        for model_id in sorted(set(lines[rows])):
            mine = lines[rows] == model_id
            blocks[model_id].append(
                (
                    mapped[np.flatnonzero(mine)],
                    var,
                    [f"{index}:{library.cells[c]}" for c in rows[mine]],
                    [arms[c] for c in rows[mine]],
                    [identifiers[c] for c in rows[mine]],
                    index,
                )
            )
    written = {}
    for model_id in sorted(blocks):
        if not blocks[model_id]:
            continue
        adata = _line_adata(model_id, name, blocks[model_id], context)
        if context.spec["run_tx1_loader_probe"]:
            probe_tx1_loader(adata, context.spec)
        _write_h5ad(adata, output / f"{model_id}.h5ad")
        written[model_id] = {
            "cells": int(adata.n_obs),
            "genes": int(adata.n_vars),
            "arms": dict(Counter(adata.obs["arm"])),
            "identifiers": sorted(set(adata.obs["identifier"])),
        }
    return {
        "source": name,
        "wave": source["wave"],
        "libraries": reports,
        "dropped": dict(dropped),
        "lines": written,
        "not_built": sorted(wanted - set(written)),
    }


def _line_adata(
    model_id: str, name: str, blocks: list[tuple[Any, ...]], context: Context
) -> ad.AnnData:
    """One line's cells over the genes every one of its libraries measures."""
    shared = set.intersection(*(set(block[1].index) for block in blocks))
    genes = [g for g in blocks[0][1].index if g in shared]
    matrix = sparse.vstack(
        [block[0][:, block[1].index.get_indexer(genes)] for block in blocks]
    ).tocsr()
    obs = pd.DataFrame(
        {
            "model_id": model_id,
            "cell_type": model_id,
            "patient_id": context.models.at[model_id, "patient_id"],
            "source": name,
            "identifier": [i for block in blocks for i in block[4]],
            "arm": [a for block in blocks for a in block[3]],
            "library": [block[5] for block in blocks for _ in block[2]],
        },
        index=[f"{model_id}:{cell}" for block in blocks for cell in block[2]],
    )
    var = blocks[0][1].loc[genes].drop(columns="source_gene_ids")
    adata = ad.AnnData(X=sparse.csr_matrix(matrix, dtype=np.int64), obs=obs, var=var)
    assert_tx1_input_contract(adata)
    return adata


def _write_h5ad(adata: ad.AnnData, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    adata.write_h5ad(temporary, compression="gzip")
    os.replace(temporary, path)


def _unpack(archive: Path, into: Path) -> None:
    marker = into / f".{archive.name}.unpacked"
    if marker.exists():
        return
    destination = into.resolve()
    if archive.suffix == ".gz" and not archive.name.endswith(".tar.gz"):
        with (
            gzip.open(archive, "rb") as source,
            archive.with_suffix("").open("wb") as out,
        ):
            shutil.copyfileobj(source, out)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            members = handle.namelist()
            bad = [
                m
                for m in members
                if not (destination / m).resolve().is_relative_to(destination)
            ]
            if bad:
                raise ValueError(f"{archive}: members escape {into}: {bad[:3]}")
            handle.extractall(into)
    else:
        with tarfile.open(archive) as handle:
            handle.extractall(into, filter="data")
    marker.write_text("")


def run(
    spec_path: Path,
    wave: int,
    output: Path,
    only: Sequence[str] | None = None,
    second_source: Path | None = None,
) -> None:
    """Build the wave's sources: each admitted line from its chosen source, or,
    with ``second_source`` (a plan of ``plan_second_sources``), each planned line
    from its second deposit, which may be a line of the 226."""
    spec = json.loads(spec_path.read_text())
    split = load_geneeffect_split(Path(spec["split_path"]))
    candidates = read_map(Path(spec["candidates_path"]))
    models = read_models(Path(spec["model_path"]))
    admission = admit(
        candidates,
        models,
        labelled_ids=read_model_ids(Path(spec["gene_effect_path"])),
        split=split,
    )
    gene_metadata, symbol_to_ensembl = load_gene_mapping(spec)
    context = Context(
        spec, candidates, admission, split, models, gene_metadata, symbol_to_ensembl
    )
    planned = (
        {} if second_source is None else json.loads(second_source.read_text())["lines"]
    )
    (output / "qc").mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(only or ()) - set(spec["sources"]))
    if unknown:
        raise ValueError(f"--sources not in the spec: {unknown}")
    for name, source in spec["sources"].items():
        if source["wave"] != wave or (only and name not in only):
            continue
        report_path = output / "qc" / f"{name}.json"
        if report_path.exists():
            print(f"{name}: built", flush=True)
            continue
        taken = (
            admission.labelled
            if second_source is None
            else {m: line["second"] for m, line in planned.items()}
        )
        wanted = [m for m, s in taken.items() if s == name]
        if second_source is not None and not wanted:
            continue
        partial = [m for m in wanted if (output / f"{m}.h5ad").exists()]
        if partial:
            raise FileExistsError(
                f"{name}: h5ad without a QC report (an interrupted run): {partial}"
            )
        print(f"{name}: building {len(wanted)} lines", flush=True)
        report = build_source(name, source, context, output, wanted)
        temporary = report_path.with_name(f".{report_path.name}.tmp")
        temporary.write_text(json.dumps(report, indent=2, default=float) + "\n")
        os.replace(temporary, report_path)
        print(f"{name}: wrote {len(report['lines'])} lines", flush=True)


def downloads(spec_path: Path, wave: int) -> list[tuple[str, Path]]:
    """``(url, path)`` of every file of the wave's sources not on disk, and of the
    re-count reference when a source of the wave is re-counted."""
    spec = json.loads(spec_path.read_text())
    files = []
    for name, source in spec["sources"].items():
        if source["wave"] != wave:
            continue
        root = Path(spec["raw_root"]) / name
        files += [(url, root / relative) for relative, url in source["files"].items()]
        if "recount" in source:
            reference = spec["recount"]
            files += [
                (url, Path(reference["index_dir"]) / relative)
                for relative, url in reference["files"].items()
            ]
    return [(url, path) for url, path in dict.fromkeys(files) if not path.exists()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-downloads", action="store_true")
    parser.add_argument("--sources", help="comma-separated source keys (default: all)")
    parser.add_argument(
        "--second-source",
        type=Path,
        help="a second-deposit plan: build its lines from their second deposits "
        "(into an output directory of their own)",
    )
    args = parser.parse_args(argv)
    if args.print_downloads:
        for url, path in downloads(args.spec, args.wave):
            print(f"{url}\t{path}")
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required to build")
    only = None if args.sources is None else args.sources.split(",")
    run(args.spec, args.wave, args.output_dir, only, args.second_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
