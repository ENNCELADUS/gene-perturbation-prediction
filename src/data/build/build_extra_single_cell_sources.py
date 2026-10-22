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

uv run python -m src.data.build.build_extra_single_cell_sources \
    --spec configs/data/extra_single_cell_sources_26Q1.json --wave 1 --print-downloads
uv run python -m src.data.build.build_extra_single_cell_sources \
    --spec configs/data/extra_single_cell_sources_26Q1.json --wave 1 \
    --output-dir data/sl_dependency_v0/processed/basal_raw_umi
"""  # noqa: E501

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tarfile
import zipfile
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.basal import assert_tx1_input_contract
from src.data.build.build_extra_single_cell_lines import Admission, admit, read_map
from src.data.build.demultiplex import identify, knee
from src.data.build.prepare_cell_line_atlas_raw_umi import (
    load_gene_mapping,
    map_and_collapse_genes,
    probe_tx1_loader,
)
from src.data.build.single_cell_readers import Library, read_library, require
from src.data.depmap import read_model_ids, read_models
from src.data.splits import FixedSplit, load_geneeffect_split


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
    require(
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
        require(spec, ("reader", "cells", "identity", "arm"), f"{name} library {index}")
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
        identifiers, arms, report["identity"] = identify(
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
