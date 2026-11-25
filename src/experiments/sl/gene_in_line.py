"""The gene-in-line caches of the SL model's line term
(docs/data/gene-in-line-features.md): for each gene of the SL labels in each line,
``tx1_gene`` writes Tx1-3B's mean contextual embedding and the share of the line's cells
that hold the gene (GPU), and ``pie_change`` writes PIE's predicted change of the line's
profile after the gene's loss, in components fitted on the SL training lines (CPU, from
the PIE root of ``sl.pie_pass``). The CPU steps ``wide_expression``, ``cell_spread``
(the line's cell file in the ladder), ``paralog_buffer``, ``complex_buffer``,
``partner_low`` (the partner columns of ``knowledge_features`` on the wide expression),
``depmap_mutation`` and ``depmap_copy_number`` (DepMap's matrices) write a file of NaN
rows for a line without the source; a percentile is fitted on the SL training lines.
These CPU steps also write the bulk-only lines of the split file with a labelled row
(``bulk_only_lines``): such a line has no cell file, so its wide expression is its
DepMap bulk RNA, mapped gene by gene onto the pseudo-bulk (``wide_profiles``).

Each cache is ``<out>/<step>/genes.json`` (the gene axis: the sorted genes of the label
table) and ``<out>/<step>/<ModelID>.npy`` (genes x width, float16, a NaN row where the
line has no value for the gene). ``tx1_gene`` skips a line whose file exists;
``pie_change`` writes every line again when one line's file is missing, so that all
lines are in one basis of components; the other steps write every line on each call."""

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

STEPS: Final[tuple[str, ...]] = (
    "tx1_gene",
    "pie_change",
    "wide_expression",
    "cell_spread",
    "paralog_buffer",
    "complex_buffer",
    "partner_low",
    "depmap_mutation",
    "depmap_copy_number",
)
# the columns of ``knowledge_features`` that each buffer cache holds
BUFFER_COLUMNS: Final[dict[str, list[int]]] = {
    "paralog_buffer": [1, 2, 3],
    "complex_buffer": [4],
    "partner_low": [5],
}


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {
            "kind",
            "seed",
            "joint",
            "labels",
            "split",
            "out",
            "tx1",
            "pie",
            "ladder",
            "linear_model",
            "bulk_expression",
            "depmap",
        },
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
    require_keys(config["depmap"], {"damaging", "hotspot", "copy_number"}, "depmap")
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


def on_axis(
    values: np.ndarray, source: Sequence[str], genes: Sequence[str]
) -> np.ndarray:
    """Source genes x width on the axis: the first source row of each axis gene's
    upper-cased symbol, a NaN row where there is none."""
    at = var_rows(pd.Series(list(source)), genes)
    out = np.full((len(genes), values.shape[1]), np.nan, np.float32)
    on = ~np.isnan(at)
    out[on] = values[at[on].astype(np.int64)]
    return out


def low_threshold(pseudobulk: np.ndarray, train: np.ndarray) -> np.ndarray:
    """The 10th percentile of each source gene over the training rows."""
    return np.percentile(pseudobulk[train], 10, axis=0)


def read_cell_files(
    ladder: Path, lines: Sequence[str]
) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    """The ladder's source genes and, for each line with a cell file, its five
    arrays (each over the source genes)."""
    cells = Path(ladder) / "cells"
    source = json.loads((cells / "genes.json").read_text())["genes"]
    files = {m: cells / f"{m}.npz" for m in lines}
    return source, {
        m: dict(np.load(path)) for m, path in files.items() if path.is_file()
    }


def split_of(config: Mapping[str, Any]) -> Any:
    """The SL split of the config; its ``train`` side holds the SL training lines."""
    from src.data.splits import load_geneeffect_split

    return load_geneeffect_split(Path(config["split"]))


def bulk_only_lines(split_path: Path, split: Any, labelled: set[str]) -> list[str]:
    """The split file's bulk-only lines (``bulk_labelled``, ``bulk_unlabelled``) among
    ``labelled`` (the lines with a labelled row): the SL model's bulk-RNA training
    lines."""
    from src.data.extra_lines import load_bulk_lines

    extra = load_bulk_lines(Path(split_path), split)
    return sorted(m for m in (*extra.labelled, *extra.unlabelled) if m in labelled)


def fit_bulk_map(
    bulk: np.ndarray, pseudobulk: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each gene (column), the least-squares slope and intercept of the
    pseudo-bulk on the bulk value over the lines (rows) where both are finite; NaN
    for a gene without two such lines or with one bulk value in all of them."""
    both = np.isfinite(bulk) & np.isfinite(pseudobulk)
    with np.errstate(invalid="ignore", divide="ignore"):
        count = both.sum(axis=0)
        mean_x = np.where(both, bulk, 0.0).sum(axis=0) / count
        mean_y = np.where(both, pseudobulk, 0.0).sum(axis=0) / count
        x = np.where(both, bulk - mean_x, 0.0)
        slope = (x * np.where(both, pseudobulk - mean_y, 0.0)).sum(axis=0) / np.square(
            x
        ).sum(axis=0)
    return slope, mean_y - slope * mean_x


def wide_profiles(
    config: Mapping[str, Any], lines: Sequence[str]
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    """The wide expression of ``lines`` over the ladder's source genes, each gene's
    10th percentile over the SL training lines with a cell file among ``lines``, and
    the counts of the bulk map. A line with a cell file has its pseudo-bulk. A line
    without one that DepMap's bulk RNA holds has its bulk value mapped onto the
    pseudo-bulk (``fit_bulk_map`` on the SL split's training lines with a cell file
    and bulk RNA; a mapped value below 0 is 0). A line with neither has no entry."""
    from src.data.depmap import read_depmap_matrix

    split = split_of(config)
    source, cells = read_cell_files(Path(config["ladder"]), lines)
    profiles = {m: c["pseudobulk"] for m, c in cells.items()}
    low = low_threshold(
        np.stack(list(profiles.values())), np.isin(list(cells), split.train)
    )
    counts = {"lines_from_bulk": 0, "bulk_map_lines": 0}
    if len(cells) < len(lines):
        bulk = read_depmap_matrix(Path(config["bulk_expression"]))
        bulk = bulk.reindex(columns=[g.upper() for g in source])
        _, paired = read_cell_files(
            Path(config["ladder"]), [m for m in split.train if m in bulk.index]
        )
        slope, intercept = fit_bulk_map(
            bulk.loc[list(paired)].to_numpy(),
            np.stack([c["pseudobulk"] for c in paired.values()]).astype(np.float64),
        )
        mapped = {
            m: np.maximum(intercept + slope * bulk.loc[m].to_numpy(), 0.0)
            for m in lines
            if m not in cells and m in bulk.index
        }
        profiles |= {m: v.astype(np.float32) for m, v in mapped.items()}
        counts = {"lines_from_bulk": len(mapped), "bulk_map_lines": len(paired)}
    return source, profiles, low, counts


def write_cache(
    out: Path,
    genes: Sequence[str],
    lines: Sequence[str],
    values: Mapping[str, np.ndarray],
    width: int,
    counts: Mapping[str, int] | None = None,
) -> None:
    """One file for every line (axis genes x ``width``; a file of NaN rows for a
    line without the source; a row with a NaN value is a NaN row) and the report
    (with ``counts``, when given)."""
    open_cache(out, genes)
    with_value = np.zeros(len(genes), bool)
    for model_id in lines:
        rows = np.full((len(genes), width), np.nan, np.float32)
        if model_id in values:
            rows[:] = values[model_id]
            rows[~np.isfinite(rows).all(axis=1)] = np.nan
        with_value |= np.isfinite(rows).all(axis=1)
        write_npy(out / f"{model_id}.npy", as_float16(rows, model_id))
    write_json(
        out / "report.json",
        {
            "lines": len(lines),
            "lines_with_source": sum(m in values for m in lines),
            "genes_with_value": int(with_value.sum()),
        }
        | dict(counts or {}),
    )


def wide_expression(
    config: Mapping[str, Any], genes: Sequence[str], lines: Sequence[str]
) -> None:
    """Per line with a wide profile (``wide_profiles``: the pseudo-bulk
    ``log1p(CP10K)``, or the mapped bulk RNA of a line without a cell file): the value
    of each axis gene and a silenced flag (1.0 at or below the gene's 10th percentile
    over the SL training lines with a cell file)."""
    source, profiles, low, counts = wide_profiles(config, lines)
    values = {
        m: on_axis(np.column_stack([p, p <= low]), source, genes)
        for m, p in profiles.items()
    }
    write_cache(
        Path(config["out"]) / "wide_expression", genes, lines, values, 2, counts
    )


def cell_spread(
    config: Mapping[str, Any], genes: Sequence[str], lines: Sequence[str]
) -> None:
    """Per line with a cell file: the share of its cells that detect each axis gene
    and the gene's SD minus its depth-expected SD."""
    source, cells = read_cell_files(Path(config["ladder"]), lines)
    values = {
        m: on_axis(
            np.column_stack([c["detected"], c["sd"] - c["sd_expected"]]), source, genes
        )
        for m, c in cells.items()
    }
    write_cache(Path(config["out"]) / "cell_spread", genes, lines, values, 2)


def buffer_cache(
    config: Mapping[str, Any], genes: Sequence[str], lines: Sequence[str], step: str
) -> None:
    """Per line with a wide profile (``wide_profiles``): the ``BUFFER_COLUMNS[step]``
    columns of ``knowledge_features`` of each axis gene on the lines' wide expression
    (paralog minimum, sum and closest; complex mean; share of partners at or below
    their 10th percentile over the SL training lines), with the linear model's
    reference."""
    from src.linear_model.gene_features import knowledge_features, partner_index
    from src.linear_model.reference import load_reference

    split = split_of(config)
    source, profiles, low, counts = wide_profiles(config, lines)
    pseudobulk = np.stack(list(profiles.values()))
    reference = load_reference(
        Path(read_yaml(Path(config["linear_model"]))["paths"]["reference"]),
        blocked={*split.val, *split.test},
        drop=True,
    )
    index = partner_index(genes, [g.upper() for g in source], reference)
    features = knowledge_features(pseudobulk, index, low)[:, :, BUFFER_COLUMNS[step]]
    write_cache(
        Path(config["out"]) / step,
        genes,
        lines,
        dict(zip(profiles, features, strict=True)),
        len(BUFFER_COLUMNS[step]),
        counts,
    )


def by_symbol(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """ModelID x upper-cased symbol of ``SYMBOL (entrez)`` gene columns; the first
    column of a symbol is kept; a ModelID with two rows raises. A column without a
    symbol (the row number, a bare Ensembl ID) is left out; a file without any gene
    column raises."""
    from src.data.labels import parse_gene_symbol

    frame.index = frame.index.astype(str)
    if not frame.index.is_unique:
        raise ValueError(f"{path}: a ModelID has more than one row")
    symbols = {}
    for column in frame.columns:
        try:
            symbols[column] = parse_gene_symbol(column)
        except ValueError:
            continue
    if not symbols:
        raise ValueError(f"{path}: no SYMBOL (entrez) column")
    frame = frame[list(symbols)].set_axis(list(symbols.values()), axis=1)
    return frame.loc[:, ~frame.columns.duplicated()].astype(np.float32)


def read_mutation_matrix(path: Path) -> pd.DataFrame:
    """A DepMap mutation matrix (one row for each sequencing, five leading columns)
    as ModelID x symbol, from the rows with ``IsDefaultEntryForModel == "Yes"``."""
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.loc[frame["IsDefaultEntryForModel"] == "Yes"].set_index("ModelID")
    leading = [
        "SequencingID",
        "ModelConditionID",
        "IsDefaultEntryForModel",
        "IsDefaultEntryForMC",
    ]
    return by_symbol(frame.drop(columns=leading), path)


def read_copy_number(path: Path) -> pd.DataFrame:
    """DepMap's gene-level copy number (first column the ModelID) as ModelID x
    symbol."""
    return by_symbol(pd.read_csv(path, index_col=0), path)


def depmap_mutation(
    config: Mapping[str, Any], genes: Sequence[str], lines: Sequence[str]
) -> None:
    """Per line in both mutation matrices: the damaging and the hotspot call of
    each axis gene of the damaging matrix; a gene outside the hotspot matrix has
    hotspot 0."""
    damaging = read_mutation_matrix(Path(config["depmap"]["damaging"]))
    hotspot = read_mutation_matrix(Path(config["depmap"]["hotspot"]))
    hotspot = hotspot.reindex(columns=damaging.columns, fill_value=0.0)
    both = set(damaging.index) & set(hotspot.index)
    values = {
        m: on_axis(
            np.column_stack([damaging.loc[m], hotspot.loc[m]]), damaging.columns, genes
        )
        for m in lines
        if m in both
    }
    write_cache(Path(config["out"]) / "depmap_mutation", genes, lines, values, 2)


def depmap_copy_number(
    config: Mapping[str, Any], genes: Sequence[str], lines: Sequence[str]
) -> None:
    """Per line in the copy-number matrix: the stored ``log2(relative copy number +
    1)`` of each axis gene and a deep-loss flag (a log2 copy-number ratio below
    -0.8)."""
    copy_number = read_copy_number(Path(config["depmap"]["copy_number"]))
    values = {}
    for m in lines:
        if m in copy_number.index:
            value = copy_number.loc[m].to_numpy()
            deep_loss = 2.0**value - 1 < 2.0**-0.8
            values[m] = on_axis(
                np.column_stack([value, deep_loss]), copy_number.columns, genes
            )
    write_cache(Path(config["out"]) / "depmap_copy_number", genes, lines, values, 2)


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
    from src.experiments.joint_config import load_config as load_joint_config
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
    labels = pd.read_parquet(config["labels"], columns=["model_id", "label"])
    # the CPU caches also hold the bulk-only lines with a labelled row
    every = [
        *lines,
        *bulk_only_lines(
            Path(config["split"]),
            split_of(config),
            set(labels.loc[labels["label"].notna(), "model_id"]),
        ),
    ]
    steps = {
        "tx1_gene": lambda: tx1_gene(config, genes, lines, args.shard),
        "pie_change": lambda: pie_change(config, genes, lines),
        "wide_expression": lambda: wide_expression(config, genes, every),
        "cell_spread": lambda: cell_spread(config, genes, every),
        **{
            step: lambda step=step: buffer_cache(config, genes, every, step)
            for step in BUFFER_COLUMNS
        },
        "depmap_mutation": lambda: depmap_mutation(config, genes, every),
        "depmap_copy_number": lambda: depmap_copy_number(config, genes, every),
    }
    steps[args.steps]()
    log.info("%s: done", args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
