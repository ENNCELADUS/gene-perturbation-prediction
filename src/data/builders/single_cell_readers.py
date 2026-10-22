"""Readers for the extra single-cell deposits' counts: each reader kind's required keys
and its parser (10x matrix, h5 and BAM, tables, triplets, loom, AnnData, Tahoe),
returning one ``Library`` of raw counts and any hashtag counts."""

from __future__ import annotations

from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from src.data.builders.prepare_cell_line_atlas_raw_umi import (
    assert_raw_integer,
)

#: Keys each reader takes (all required).
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


def require(spec: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
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
    require(spec, READERS[reader], f"{reader} library")
    library = _READ[reader](root, spec)
    assert_raw_integer(library.counts, f"{root} {reader} library")
    if library.counts.shape != (len(library.cells), len(library.genes)):
        raise ValueError(f"{root}: counts {library.counts.shape} do not match ids")
    if len(set(library.cells)) != len(library.cells):
        raise ValueError(f"{root}: duplicate cell ids")
    return library
