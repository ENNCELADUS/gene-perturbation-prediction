"""Gene sources of the stage-one gene encoder (ladder design §4.4), aligned to the
GeneEffect gene axis."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = ("gene_text", "esmc_600m", "esm3_open", "string_space", "gene2vec")
PIE_SOURCES = ("gene_text", "string_space")
PROTEIN_SOURCES = ("esmc_600m", "esm3_open")


@dataclass(frozen=True)
class SourceTable:
    """One source as read: its keys and one vector per key (possibly memory-mapped)."""

    name: str
    keys: tuple[str, ...]
    vectors: np.ndarray


@dataclass(frozen=True)
class AlignedSource:
    """One source on the gene axis: NaN rows where ``present`` is false."""

    name: str
    axis: tuple[str, ...]
    vectors: np.ndarray
    present: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


def _checked_table(name: str, keys: Sequence[str], vectors: np.ndarray) -> SourceTable:
    if vectors.ndim != 2 or vectors.shape[0] != len(keys):
        raise ValueError(f"{name}: {vectors.shape} vectors for {len(keys)} keys")
    return SourceTable(name=name, keys=tuple(str(k) for k in keys), vectors=vectors)


def read_pie_dense(path: Path) -> SourceTable:
    """A PIE dense source dir, checked against its meta.json."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta["layout"] != "dense":
        raise ValueError(f"{path}: layout {meta['layout']!r}, expected dense")
    vectors = np.load(path / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if vectors.dtype != np.dtype(meta["dtype"]) or vectors.shape[1:] != (meta["dim"],):
        raise ValueError(
            f"{path}: embeddings {vectors.dtype} {vectors.shape} against meta "
            f"{meta['dtype']} dim {meta['dim']}"
        )
    return _checked_table(str(meta["name"]), meta["keys"], vectors)


def read_protein_npz(path: Path, name: str) -> SourceTable:
    """An ``embed_protein_sources`` output."""
    with np.load(path, allow_pickle=False) as payload:
        symbols = [str(s) for s in payload["symbols"]]
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
        if len(payload["accessions"]) != len(symbols):
            raise ValueError(f"{path}: {len(payload['accessions'])} accessions")
    return _checked_table(name, symbols, vectors)


def read_gene2vec(embeddings: Path, gene_names: Path) -> SourceTable:
    """Gene2vec keyed by symbol: every symbol of Geneformer's dictionary whose
    Ensembl ID has a Gene2vec vector."""
    import torch

    with open(gene_names, "rb") as handle:
        names = pickle.load(handle)
    table = torch.load(embeddings, weights_only=True)
    keys = [symbol for symbol, ensembl in names.items() if ensembl in table]
    if not keys:
        raise ValueError(f"{embeddings}: no Ensembl ID of {gene_names} has a vector")
    vectors = np.stack(
        [np.asarray(table[names[symbol]], dtype=np.float32) for symbol in keys]
    )
    return _checked_table("gene2vec", keys, vectors)


def align(table: SourceTable, axis: Sequence[str]) -> AlignedSource:
    """Rows of ``table`` in axis order, matched case-insensitively."""
    axis = tuple(axis)
    if any(gene != gene.upper() for gene in axis) or len(set(axis)) != len(axis):
        raise ValueError("the gene axis must be unique upper-case symbols")
    row_of: dict[str, int] = {}
    clashes: set[str] = set()
    for row, key in enumerate(table.keys):
        upper = key.upper()
        if upper in row_of:
            clashes.add(upper)
        row_of[upper] = row
    if clashes:
        raise ValueError(f"{table.name}: keys equal up to case: {sorted(clashes)[:10]}")
    rows = np.asarray([row_of.get(gene, -1) for gene in axis], dtype=np.int64)
    present = rows >= 0
    vectors = np.full((len(axis), table.vectors.shape[1]), np.nan, dtype=np.float32)
    vectors[present] = np.asarray(table.vectors[rows[present]], dtype=np.float32)
    if not np.isfinite(vectors[present]).all():
        raise ValueError(f"{table.name}: non-finite vectors among matched genes")
    return AlignedSource(name=table.name, axis=axis, vectors=vectors, present=present)


def coverage(source: AlignedSource, selective: Sequence[str]) -> dict[str, float | int]:
    """Genes covered: a count, and shares of the axis and of the selective genes."""
    covered = int(source.present.sum())
    record: dict[str, float | int] = {
        "dim": source.dim,
        "covered": covered,
        "share_of_axis": covered / len(source.axis),
    }
    on_axis = dict(zip(source.axis, source.present))
    missing = sorted(set(selective) - set(on_axis))
    if missing:
        raise ValueError(f"selective genes off the axis: {missing[:10]}")
    hit = sum(bool(on_axis[gene]) for gene in selective)
    record["selective_covered"] = hit
    record["share_of_selective"] = hit / len(selective)
    return record


@dataclass(frozen=True)
class SourcePaths:
    pie_sources: Path  # holds gene_text/ and string_space/
    protein: Path  # holds esmc_600m.npz and esm3_open.npz
    gene2vec: Path
    gene_names: Path

    def inputs(self) -> dict[str, list[Path]]:
        """The files each source is read from."""
        return {
            **{
                name: [
                    self.pie_sources / name / f for f in ("meta.json", "embeddings.npy")
                ]
                for name in PIE_SOURCES
            },
            **{name: [self.protein / f"{name}.npz"] for name in PROTEIN_SOURCES},
            "gene2vec": [self.gene2vec, self.gene_names],
        }


def read_source(name: str, paths: SourcePaths) -> SourceTable:
    if name in PIE_SOURCES:
        table = read_pie_dense(paths.pie_sources / name)
        if table.name != name:
            raise ValueError(f"{paths.pie_sources / name}: meta names {table.name!r}")
        return table
    if name in PROTEIN_SOURCES:
        return read_protein_npz(paths.protein / f"{name}.npz", name)
    if name == "gene2vec":
        return read_gene2vec(paths.gene2vec, paths.gene_names)
    raise ValueError(f"unknown source {name!r}; expected one of {SOURCES}")


def align_sources(
    axis: Sequence[str], paths: SourcePaths, names: Sequence[str] = SOURCES
) -> dict[str, AlignedSource]:
    """Every named source on the axis."""
    return {name: align(read_source(name, paths), axis) for name in names}


def write_sources(
    out: Path,
    aligned: Mapping[str, AlignedSource],
    paths: SourcePaths,
    selective: Sequence[str],
) -> dict:
    """``<out>/<source>.npz`` per source and ``<out>/coverage.json``."""
    out.mkdir(parents=True, exist_ok=True)
    inputs = paths.inputs()
    record: dict = {
        "axis": len(next(iter(aligned.values())).axis),
        "selective": len(selective),
        "sources": {},
    }
    for name, source in aligned.items():
        tmp = out / f".{name}.{os.getpid()}.tmp.npz"
        np.savez(
            tmp,
            axis=np.asarray(source.axis, dtype=str),
            vectors=source.vectors,
            present=source.present,
        )
        os.replace(tmp, out / f"{name}.npz")
        record["sources"][name] = {
            **coverage(source, selective),
            "inputs": [str(p) for p in inputs[name]],
        }
    tmp = out / f".coverage.{os.getpid()}.tmp.json"
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, out / "coverage.json")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--genes", type=Path, required=True, help="the ladder's genes.parquet"
    )
    parser.add_argument("--pie-sources", type=Path, required=True)
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--gene2vec", type=Path, required=True)
    parser.add_argument("--gene-names", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    genes = pd.read_parquet(args.genes)
    axis = [str(gene) for gene in genes.index]
    selective = [gene for gene, keep in zip(axis, genes["selective"]) if bool(keep)]
    paths = SourcePaths(args.pie_sources, args.protein, args.gene2vec, args.gene_names)
    record = write_sources(args.out, align_sources(axis, paths), paths, selective)
    for name, row in record["sources"].items():
        print(
            f"{name}: dim {row['dim']}, {row['covered']} of {record['axis']} genes "
            f"({row['share_of_axis']:.3f}), selective {row['share_of_selective']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
