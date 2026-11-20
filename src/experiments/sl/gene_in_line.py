"""The gene-in-line caches of the SL model's line term
(docs/data/gene-in-line-features.md): for each gene of the SL labels in each line,
``tx1_gene`` writes Tx1-3B's mean contextual embedding and the share of the line's cells
that hold the gene (GPU), and ``pie_change`` writes PIE's predicted change of the line's
profile after the gene's loss, in components fitted on the SL training lines (CPU, from
the PIE root of ``sl.pie_pass``).

Each cache is ``<out>/<step>/genes.json`` (the gene axis: the sorted genes of the label
table) and ``<out>/<step>/<ModelID>.npy`` (genes x width, float16, a NaN row where the
line has no value for the gene). ``tx1_gene`` skips a line whose file exists;
``pie_change`` writes every line again when one line's file is missing, so that all
lines are in one basis of components."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.data.io import write_json, write_npy
from src.experiments.config import read_yaml, require_keys
from src.experiments.run_dir import record_step, shard

log = logging.getLogger(__name__)

STEPS: Final[tuple[str, ...]] = ("tx1_gene", "pie_change")


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "joint", "labels", "split", "out", "tx1", "pie"},
        "config",
    )
    if config["kind"] != "sl_gene_in_line":
        raise ValueError(f"{path} is a {config['kind']} config")
    require_keys(
        config["tx1"],
        {"batch_size", "max_length", "max_cells", "cells_per_pass"},
        "tx1",
    )
    require_keys(config["pie"], {"root", "components", "sample"}, "pie")
    return config


def axis_and_lines(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """The gene axis (the sorted genes of the label table) and the split's lines
    with a labelled row."""
    from src.data.splits import load_geneeffect_split

    labels = pd.read_parquet(
        config["labels"], columns=["model_id", "gene_a", "gene_b", "label"]
    )
    genes = sorted(set(labels["gene_a"].dropna()) | set(labels["gene_b"].dropna()))
    labelled = set(labels.loc[labels["label"].notna(), "model_id"])
    split = load_geneeffect_split(Path(config["split"]))
    return genes, sorted(m for m in split.all_model_ids if m in labelled)


def open_cache(directory: Path, genes: Sequence[str]) -> None:
    """Write ``genes.json`` once; refuse a cache of another gene axis."""
    path = directory / "genes.json"
    if not path.is_file():
        write_json(path, list(genes))
    elif json.loads(path.read_text()) != list(genes):
        raise ValueError(f"{path}: another gene axis than the label table's")


def as_float16(values: np.ndarray, model_id: str) -> np.ndarray:
    """``values`` in float16; a finite value that float16 cannot hold raises."""
    with np.errstate(over="ignore"):
        stored = values.astype(np.float16)
    if (np.isfinite(values) & ~np.isfinite(stored)).any():
        raise ValueError(f"{model_id}: a value is outside the float16 range")
    return stored


def fit_components(
    vectors: np.ndarray, components: int
) -> tuple[np.ndarray, np.ndarray]:
    """The centre (panel,) and an orthonormal basis (panel, components) of the
    vectors' leading principal directions."""
    centre = vectors.mean(axis=0, dtype=np.float64).astype(np.float32)
    # float32 throughout: a float64 copy of the PIE sample would double its memory
    centred = np.subtract(vectors, centre, dtype=np.float32)
    covariance = (centred.T @ centred).astype(np.float64)
    _, vectors_of = np.linalg.eigh(covariance)
    return centre, vectors_of[:, ::-1][:, :components].astype(np.float32)


def var_rows(symbols: pd.Series, genes: Sequence[str]) -> np.ndarray:
    """The first ``var`` row of each axis gene's symbol (NaN where there is none)."""
    # the axis holds the upper-cased DepMap symbols (C10ORF105), var the source case
    upper = symbols.astype(str).str.upper().to_numpy()
    row = pd.Series(np.arange(len(upper)), index=upper)
    return row[~row.index.duplicated()].reindex(list(genes)).to_numpy()


def tx1_gene(
    config: Mapping[str, Any],
    genes: Sequence[str],
    lines: Sequence[str],
    shard_of: tuple[int, int],
) -> None:
    """Per line of this shard: the mean Tx1-3B embedding of each axis gene over at
    most ``tx1.max_cells`` cells, then the share of those cells that hold it
    (genes x 2,561). The axis gene takes the first ``var`` row of its upper-cased
    symbol."""
    import time

    from src.data.basal import symbol_column
    from src.data.tx1_cache import EMBEDDING_WIDTH, read_basal_source, tx1_cell_indices
    from src.experiments.joint.config import load_config as load_joint_config
    from src.model.tx1 import build_tx1_gene_encoder

    out = Path(config["out"]) / "tx1_gene"
    open_cache(out, genes)
    index, count = shard_of
    todo = [m for m in lines[index::count] if not (out / f"{m}.npy").is_file()]
    if not todo:
        return
    joint = load_joint_config(Path(config["joint"]))
    settings = config["tx1"]
    encoder = build_tx1_gene_encoder(
        Path(joint["paths"]["tx1_model_dir"]),
        int(settings["batch_size"]),
        int(settings["max_length"]),
        int(settings["cells_per_pass"]),
    )
    directory = Path(joint["paths"]["basal_raw_umi"])
    for model_id in todo:
        source = read_basal_source(
            directory / f"{model_id}.h5ad",
            model_id=model_id,
            var_ensembl_col=joint["preparation"]["var_ensembl_col"],
        )
        keep = tx1_cell_indices(
            list(source.obs_names),
            model_id,
            int(settings["max_cells"]),
            int(config["seed"]),
        )
        source = source[keep].copy()
        start = time.monotonic()
        mean, share = encoder(source)
        elapsed = time.monotonic() - start
        at = var_rows(source.var[symbol_column(source.var)], genes)
        values = np.full((len(genes), EMBEDDING_WIDTH + 1), np.nan, np.float32)
        on = ~np.isnan(at)
        take = at[on].astype(np.int64)
        values[on] = np.column_stack([mean[take], share[take]])
        values[values[:, -1] == 0] = np.nan
        write_npy(out / f"{model_id}.npy", as_float16(values, model_id))
        with_value = int(np.isfinite(values).all(axis=1).sum())
        record_step(
            out / "report.json",
            model_id,
            {"cells": int(source.n_obs), "genes_with_value": with_value},
        )
        log.info(
            "tx1_gene %s: %d cells in %.0f s, %d of %d genes with a value",
            model_id,
            source.n_obs,
            elapsed,
            with_value,
            len(genes),
        )


def pie_change(
    config: Mapping[str, Any], genes: Sequence[str], labelled: Sequence[str]
) -> None:
    """Per line of the PIE root: PIE's predicted change after the loss of each axis
    gene, in ``pie.components`` components fitted on a seeded sample of at most
    ``pie.sample`` (line, gene) vectors of the lines with role ``train``. The change
    files are read one at a time; every lost gene of ``genes.parquet`` must have
    one, so that an unfinished PIE pass does not read as genes without a value. When
    every line's file exists, nothing is done; when one is missing, every line is
    written again, because a new fit can give other components (a sign can change)."""
    root = Path(config["pie"]["root"])
    out = Path(config["out"]) / "pie_change"
    open_cache(out, genes)
    lines = pd.read_parquet(root / "lines.parquet")
    model_ids = [str(m) for m in lines.index]
    if all((out / f"{m}.npy").is_file() for m in model_ids):
        return
    lost = pd.read_parquet(root / "genes.parquet")
    lost = [str(g) for g in lost.index[lost["lost"]]]
    unfinished = [
        g for g in lost if not (root / "pie" / "change" / f"{g}.npy").is_file()
    ]
    if unfinished:
        raise FileNotFoundError(
            f"PIE's pass is incomplete: {len(unfinished)} lost genes have no change "
            f"file, e.g. {unfinished[:10]}"
        )
    files = {g: root / "pie" / "change" / f"{g.upper()}.npy" for g in genes}
    with_file = [g for g in genes if files[g].is_file()]
    train = np.flatnonzero(lines["role"].to_numpy() == "train")
    total = len(with_file) * len(train)
    rng = np.random.default_rng(int(config["seed"]))
    chosen = np.sort(
        rng.choice(total, size=min(int(config["pie"]["sample"]), total), replace=False)
    )
    gene_of, line_of = np.divmod(chosen, len(train))
    bounds = np.searchsorted(gene_of, np.arange(len(with_file) + 1))
    parts = []
    for k, gene in enumerate(with_file):
        if bounds[k] < bounds[k + 1]:
            change = np.load(files[gene], mmap_mode="r")
            parts.append(
                np.asarray(
                    change[train[line_of[bounds[k] : bounds[k + 1]]]], np.float32
                )
            )
    sample = np.concatenate(parts)
    del parts
    if not np.isfinite(sample).all():
        raise ValueError(f"{root}: a sampled PIE change is not finite")
    components = int(config["pie"]["components"])
    centre, basis = fit_components(sample, components)
    log.info("pie_change: %d components fitted on %d vectors", components, len(sample))
    del sample
    values = np.full((len(model_ids), len(genes), components), np.nan, np.float32)
    present = set(with_file)
    for k, gene in enumerate(genes):
        if gene in present:
            change = np.load(files[gene]).astype(np.float32)
            if change.shape[0] != len(model_ids):
                raise ValueError(
                    f"{files[gene]}: {change.shape[0]} rows, {len(model_ids)} PIE lines"
                )
            values[:, k] = (change - centre) @ basis
    for i in range(len(model_ids)):
        write_npy(out / f"{model_ids[i]}.npy", as_float16(values[i], model_ids[i]))
    write_json(
        out / "report.json",
        {
            "lines": len(model_ids),
            "train_lines": len(train),
            "genes": len(genes),
            "genes_with_change": len(with_file),
            "sample": len(chosen),
            "components": components,
            "labelled_lines_without_pie": sorted(set(labelled) - set(model_ids)),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", choices=STEPS, required=True)
    parser.add_argument("--shard", type=shard, default=(0, 1))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = load_config(args.config)
    genes, lines = axis_and_lines(config)
    if args.steps == "tx1_gene":
        tx1_gene(config, genes, lines, args.shard)
    else:
        pie_change(config, genes, lines)
    log.info("%s: done", args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
