"""PIE's side of the PIE editor (SL protocol §13), in ``.venv-pie``: the PIE root's
tables for the screened SL lines, the frozen cross-dataset PIE's predicted profile
change after each lost gene in every screened line (DepMap source masked), the
measured Perturb-seq changes of the reference lines, and two measurements of PIE on
its own data (the masked re-score, and the held-out line's response against the mean
response of the other lines). The editor (``pie_editor``) reads the outputs without
the ``pie`` package.

Layout under ``pie_root``: ``lines.parquet`` (role train / val / test),
``genes.parquet`` (upper-case symbol; ``in_panel``: a decoded gene, the reader's
expression space with a PIE gene key; ``lost``: a lost gene with a key),
``mean/pie_input.parquet``, ``pie/change/<gene>.npy`` (lines x panel, float16, the
softmax expectation of PIE's delta-p bins: a change of the mean over cells of
log1p CP10K), ``measured/<dataset>__<context>.npz`` (perturbations x panel) and
``pie/own_task/``, ``pie/rescore/``."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.data.io import write_npy, write_npz, write_parquet
from src.experiments.config import read_yaml, require_keys
from src.experiments.run_dir import absolute, record_step, shard
from src.experiments.stage_one.ladder_inputs import read_cells

log = logging.getLogger(__name__)

STEPS: Final[tuple[str, ...]] = ("tables", "pass", "measured", "own_task", "rescore")
ROLES: Final[tuple[str, ...]] = ("train", "val", "test")
KEYS: Final[list[str]] = ["model_id", "design", "gene_a", "gene_b"]
CHANGE: Final[str] = "change"
MEASURED: Final[str] = "measured"
#: PIE's Replogle folds: one line held out each (``pie_reproduction.TARGETS``).
FOLD_SPLIT: Final[str] = "replogle_wdataset/unseen_ctx/{fold}/test.json"


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "output_root", "task", "pie_root", "ladder_root",
         "linear_model", "labels", "split", "simple_scores", "head_penalties",
         "top_k", "paths", "pie", "measured", "references"},
        "config",
    )  # fmt: skip
    if config["kind"] != "sl_pie_editor":
        raise ValueError(f"{path} is a {config['kind']} config")
    return config


def screened(simple_scores: Path) -> tuple[list[str], list[str]]:
    """The screened lines and lost genes of the rows the backbone scores (its
    ``conditional`` table): gene A of every row, gene B of a combinatorial one."""
    table = pd.read_parquet(Path(simple_scores) / "conditional.parquet", columns=KEYS)
    pair = table["design"] != "anchor"
    lost = set(table["gene_a"]) | set(table.loc[pair, "gene_b"])
    return sorted(table["model_id"].unique()), sorted(lost)


def cell_inputs(
    ladder_root: Path, lines: Sequence[str], genes: Sequence[str]
) -> pd.DataFrame:
    """PIE's input (mean over a line's cells of log1p CP10K) of ``genes`` from the
    ladder's cell files; a value the line's cells do not list takes the mean of the
    lines that list it."""
    union = json.loads((Path(ladder_root) / "cells" / "genes.json").read_text())[
        "genes"
    ]
    position = {g: i for i, g in enumerate(union)}
    columns = [position[g] for g in genes]
    values = np.empty((len(lines), len(genes)), dtype=np.float64)
    listed = np.empty((len(lines), len(genes)), dtype=bool)
    for i, model_id in enumerate(lines):
        cells = read_cells(ladder_root, model_id)
        if cells["pie_input"].shape != (len(union),):
            raise ValueError(
                f"{model_id}: {cells['pie_input'].shape[0]} PIE inputs, "
                f"{len(union)} genes in cells/genes.json"
            )
        values[i] = cells["pie_input"][columns]
        listed[i] = cells["listed"][columns]
    frame = pd.DataFrame(values, index=list(lines), columns=list(genes))
    fill = frame.where(listed).mean()
    if fill.isna().any():
        raise ValueError(f"genes no line lists: {list(fill.index[fill.isna()])[:10]}")
    return frame.where(listed, fill, axis=1)


def write_tables(config: Mapping[str, Any]) -> dict[str, Any]:
    """The three tables ``pie.read_tables`` reads, for the screened lines; the panel
    is the reader's expression space (the bulk genes every SL held-out line measures
    in pseudo-bulk) where PIE has a gene key."""
    from src.data.ladder_cells import symbol_key
    from src.data.pseudobulk import read_pseudobulk
    from src.data.splits import load_geneeffect_split
    from src.experiments.joint.config import load_config as load_joint_config
    from src.experiments.stage_one.ladder_inputs import bulk_gene_columns
    from src.linear_model.space import measured_genes

    root = absolute(config["pie_root"])
    split = load_geneeffect_split(absolute(config["split"]))
    lines, lost = screened(absolute(config["simple_scores"]))
    linear_model = read_yaml(absolute(config["linear_model"]))
    joint = load_joint_config(absolute(linear_model["joint_config"]))
    pseudo = read_pseudobulk(absolute(joint["prepared_root"]))
    bulk = bulk_gene_columns(absolute(linear_model["paths"]["bulk_expression"]))
    space = measured_genes(pseudo, bulk, [*split.val, *split.test])
    meta = absolute(config["paths"]["gene_text"]) / "meta.json"
    keys = symbol_key(
        sorted(set(space) | set(lost)), json.loads(meta.read_text())["keys"]
    )
    panel = [g for g in space if g in keys]
    genes = list(dict.fromkeys([*panel, *(g for g in lost if g in keys)]))
    ladder = pd.read_parquet(absolute(config["ladder_root"]) / "genes.parquet")
    selective = set(ladder.index[ladder["selective"]])
    table = pd.DataFrame(
        {
            "gene_text_key": [keys[g] for g in genes],
            "in_panel": [g in set(panel) for g in genes],
            "lost": [g in set(lost) for g in genes],
            "selective": [g in selective for g in genes],
        },
        index=pd.Index(genes, name="gene"),
    )
    role = {m: side for side in ROLES for m in getattr(split, side)}
    rows = pd.DataFrame(
        {"role": [role[m] for m in lines]}, index=pd.Index(lines, name="model_id")
    )
    pie_input = cell_inputs(absolute(config["ladder_root"]), lines, panel)
    (root / "mean").mkdir(parents=True, exist_ok=True)
    write_parquet(root / "genes.parquet", table)
    write_parquet(root / "lines.parquet", rows)
    write_parquet(root / "mean" / "pie_input.parquet", pie_input.astype(np.float32))
    return {
        "lines": len(lines),
        "roles": rows["role"].value_counts().to_dict(),
        "space": len(space),
        "panel": len(panel),
        "lost": len(lost),
        "lost_with_key": int(table["lost"].sum()),
    }


def pie_layout(config: Mapping[str, Any]) -> Any:
    from src.experiments.stage_one.pie import Layout

    return Layout(
        root=absolute(config["pie_root"]),
        checkpoint=absolute(config["paths"]["pie_checkpoint"]),
        context_text=absolute(config["paths"]["context_text"]),
        config=config,
    )


def lost_genes(tables: Any) -> list[str]:
    return list(tables.genes.index[tables.genes["lost"]])


def run_pass(
    config: Mapping[str, Any], shard_of: tuple[int, int], device: str
) -> dict[str, Any]:
    """PIE's predicted change of every screened line after each of this shard's
    lost genes lacking a file: ``forward_tensors``'s ``change`` over the panel."""
    import torch

    from src.experiments.stage_one.pie import (
        build_batch,
        exclusive,
        gene_evidence,
        load_pie,
        pie_environment,
        read_tables,
        shard_genes,
        write_dataset,
    )
    from src.experiments.stage_one.pie_decode import checked_agreement, forward_tensors

    out = pie_layout(config)
    tables = read_tables(out.root)
    with exclusive(out):
        write_dataset(out, tables)
    genes = shard_genes(lost_genes(tables), shard_of)
    directory = out.pie / CHANGE
    directory.mkdir(parents=True, exist_ok=True)
    todo = [g for g in genes if not (directory / f"{g}.npy").is_file()]
    if not todo:
        return {"genes": len(genes), "computed": 0}
    lines = list(tables.lines.index)
    ctrl = tables.pie_input.to_numpy(dtype=np.float32)
    size = int(config["pie"]["lines_per_batch"])
    seconds = []
    with pie_environment(out):
        pie = load_pie(out, tables, device)
        difference = checked_agreement(pie, tables, todo, ROLES)
        for number, gene in enumerate(todo, start=1):
            start = time.monotonic()
            key = tables.genes.at[gene, "gene_text_key"]
            evidence = gene_evidence(pie, key)
            parts = [
                forward_tensors(
                    pie,
                    build_batch(
                        pie, lines[i : i + size], key, ctrl[i : i + size], evidence
                    ),
                    None,
                    heads=True,
                )["change"]
                for i in range(0, len(lines), size)
            ]
            change = torch.cat(parts).float().cpu().numpy().astype(np.float16)
            write_npy(directory / f"{gene}.npy", change)
            seconds.append(time.monotonic() - start)
            if number % 100 == 0 or number == len(todo):
                log.info(
                    "pass: %d of %d genes, %.3f s per gene",
                    number,
                    len(todo),
                    float(np.mean(seconds)),
                )
    return {
        "genes": len(genes),
        "computed": len(todo),
        "lines": len(lines),
        "panel": len(tables.panel),
        "seconds_per_gene": float(np.mean(seconds)),
        "agreement": difference,
    }


def on_panel(genes: Sequence[str], panel: Sequence[str]) -> np.ndarray:
    """Each panel gene's column in ``genes`` (upper-cased; -1: absent, or spelled
    twice)."""
    upper = pd.Series([str(g).upper() for g in genes])
    column = pd.Series(np.arange(len(upper)), index=upper)
    column = column[~column.index.duplicated(keep=False)]
    return column.reindex(list(panel)).fillna(-1).to_numpy(dtype=np.int64)


def measured_change(
    delta_p: np.ndarray, genes: Sequence[str], panel: Sequence[str]
) -> np.ndarray:
    """Rows of a dataset's measured ``delta_p`` on the panel; NaN where the dataset
    lacks the gene."""
    take = on_panel(genes, panel)
    out = np.asarray(delta_p, dtype=np.float32)[:, np.maximum(take, 0)]
    out[:, take < 0] = np.nan
    return out


def export_measured(config: Mapping[str, Any]) -> dict[str, Any]:
    """The measured change (pseudo-bulk of the perturbed cells minus the control
    mean, PIE's ``delta_p``) of each reference context's perturbations, on the
    panel, as ``measured/<dataset>__<context>.npz``."""
    from pie.data.preprocessed import PreprocessedDir

    from src.experiments.stage_one.pie import read_tables

    root = absolute(config["pie_root"])
    panel = read_tables(root).panel
    wanted = sorted({c for pair in config["references"].values() for c in pair})
    (root / MEASURED).mkdir(exist_ok=True)
    record = {}
    for name in wanted:
        dataset, context = name.split("/")
        data = PreprocessedDir.open(absolute(config["measured"][dataset]))
        rows = np.flatnonzero(data.ctx_ids[:] == data.meta.context_to_id[context])
        perts = [data.perts[p].upper() for p in data.pert_ids[rows]]
        if len(set(perts)) != len(perts):
            raise ValueError(f"{name}: a perturbation spelled twice")
        change = measured_change(data.delta_p[rows], data.genes, panel)
        write_npz(
            root / MEASURED / f"{dataset}__{context}.npz",
            perturbations=np.asarray(perts),
            change=change.astype(np.float16),
        )
        record[name] = {
            "perturbations": len(perts),
            "panel_genes_measured": int(np.isfinite(change[0]).sum()),
        }
    return record


def response_scores(
    predicted: np.ndarray, truth: np.ndarray, others: np.ndarray
) -> dict[str, float]:
    """One (line, perturbation): Pearson and Spearman over genes of PIE's predicted
    response and of the other lines' mean response with the measured response, and
    of PIE's deviation from that mean with the measured deviation."""
    from scipy.stats import pearsonr, spearmanr

    keep = np.isfinite(predicted) & np.isfinite(truth) & np.isfinite(others)
    p, t, o = predicted[keep], truth[keep], others[keep]
    out = {}
    for name, (x, y) in {
        "pie": (p, t),
        "other_lines_mean": (o, t),
        "pie_deviation": (p - o, t - o),
    }.items():
        varies = np.ptp(x) > 0 and np.ptp(y) > 0
        out[f"{name}_pearson"] = float(pearsonr(x, y)[0]) if varies else np.nan
        out[f"{name}_spearman"] = float(spearmanr(x, y)[0]) if varies else np.nan
    return out


def own_task_table(
    predictions: pd.DataFrame, delta_p: np.ndarray, row_index: Mapping[tuple, int]
) -> pd.DataFrame:
    """Per predicted (context, perturbation) with a measured response in the
    context and in at least one other context: ``response_scores`` and the count of
    other contexts."""
    contexts = sorted({c for c, _ in row_index})
    records = []
    for context, perturbation, predicted in zip(
        predictions["context"], predictions["perturbation"], predictions["delta_p_pred"]
    ):
        others = [
            row_index[(c, perturbation)]
            for c in contexts
            if c != context and (c, perturbation) in row_index
        ]
        if not others or (context, perturbation) not in row_index:
            continue
        truth = np.asarray(delta_p[row_index[(context, perturbation)]], np.float64)
        mean = np.asarray(delta_p[others], np.float64).mean(axis=0)
        records.append(
            {"context": context, "perturbation": perturbation, "others": len(others)}
            | response_scores(np.asarray(predicted, np.float64), truth, mean)
        )
    return pd.DataFrame(records)


def _evaluate(runs: Path, experiment: str, checkpoint: Path, split: str, row_set: str,
              device: str, masked: bool, predictions: bool) -> Path:  # fmt: skip
    """PIE's ``run_eval`` of a released checkpoint linked under ``runs``; its
    output directory (reused when its tables exist)."""
    from pie.config import EvalConfig
    from pie.evaluate import run_eval

    from src.experiments.stage_one.pie import depmap_masked_prediction
    from src.experiments.stage_one.pie_reproduction import CHECKPOINT_FILES, CKPT

    directory = runs / experiment
    directory.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINT_FILES:
        link = directory / name
        if not link.is_symlink():
            link.symlink_to(checkpoint.parent / name)
    out = directory / "eval" / row_set
    if (out / f"granular_{CKPT}.csv").is_file():
        return out
    for name in ("metrics", "granular", "predictions"):
        (out / f"{name}_{CKPT}.{'parquet' if name == 'predictions' else 'csv'}").unlink(
            missing_ok=True
        )
    os.environ["PIE_RUNS_ROOT"] = str(runs)
    config = EvalConfig(
        experiment_name=experiment,
        run_dir=str(directory),
        ckpt=CKPT,
        split_path=split,
        row_set=row_set,
        preprocessed_dirs=None,
        save_predictions=predictions,
        overwrite=False,
        device=device,
        batch_size=16,
    )
    masking = depmap_masked_prediction() if masked else contextlib.nullcontext()
    with contextlib.chdir(directory), masking:
        run_eval(config)
    return out


def own_task(config: Mapping[str, Any], device: str) -> dict[str, Any]:
    """PIE's Replogle folds (one line held out each), DepMap unmasked and masked:
    the held-out line's predicted response against the mean measured response of
    the other lines (``own_task_table``), per fold and over the folds."""
    from pie.data.preprocessed import PreprocessedDir

    from src.experiments.stage_one.pie_reproduction import (
        CKPT,
        SPLITS_URI,
        WDATASET_FOLDS,
    )

    root = absolute(config["pie_root"]) / "pie" / "own_task"
    data = PreprocessedDir.open(absolute(config["measured"]["replogle"]))
    index = data.row_index()
    delta_p = data.delta_p
    released = absolute(config["paths"]["wdataset"])
    record: dict[str, Any] = {}
    tables = []
    for masking in ("unmasked", "masked"):
        for fold in WDATASET_FOLDS:
            out = _evaluate(
                root / masking / "runs",
                f"replogle_wdataset/{fold}",
                released / fold / f"{CKPT}.ckpt",
                f"{SPLITS_URI}/{FOLD_SPLIT.format(fold=fold)}",
                "unseen_ctx",
                device,
                masking == "masked",
                predictions=True,
            )
            predictions = pd.read_parquet(out / f"predictions_{CKPT}.parquet")
            if set(predictions["dataset"]) != {data.dataset}:
                raise ValueError(f"{out}: predictions of {set(predictions['dataset'])}")
            table = own_task_table(predictions, delta_p, index)
            tables.append(table.assign(masking=masking, fold=fold))
    joined = pd.concat(tables, ignore_index=True)
    joined.to_csv(root / "pairs.csv", index=False)
    scores = [c for c in joined.columns if c.endswith(("_pearson", "_spearman"))]
    for (masking, fold), group in joined.groupby(["masking", "fold"]):
        record.setdefault(masking, {})[fold] = {"pairs": len(group)} | {
            c: float(group[c].mean()) for c in scores
        }
    for masking, folds in record.items():
        folds["mean_over_folds"] = {
            c: float(np.mean([v[c] for v in folds.values()])) for c in scores
        }
    return record


def rescore(config: Mapping[str, Any], device: str) -> dict[str, Any]:
    """PIE's own metrics of the cross-dataset checkpoint on Replogle's seen and
    unseen perturbations with the DepMap source masked (``all`` row)."""
    from src.experiments.stage_one.pie import EXPERIMENT, ROW_SETS, pie_environment
    from src.experiments.stage_one.pie_reproduction import CKPT, SPLITS_URI, read_table

    out = pie_layout(config)
    record = {}
    with pie_environment(out):
        for row_set in ROW_SETS:
            directory = _evaluate(
                out.pie / "runs",
                EXPERIMENT,
                out.checkpoint,
                f"{SPLITS_URI}/replogle_xdataset/{row_set}.json",
                row_set,
                device,
                masked=True,
                predictions=False,
            )
            table = read_table(directory / f"metrics_{CKPT}.csv")
            overall = table[table["context"] == "all"].iloc[0]
            record[row_set] = {
                k: float(v) for k, v in overall.items() if k not in ("context",)
            }
    return record


def run(config: Mapping[str, Any], steps: Sequence[str], shard_of: tuple[int, int],
        device: str) -> None:  # fmt: skip
    root = absolute(config["pie_root"])
    (root / "pie").mkdir(parents=True, exist_ok=True)
    record = root / "pie" / "run.json"
    for step in steps:
        if step == "tables":
            record_step(record, "tables", write_tables(config))
        elif step == "pass":
            record_step(
                record,
                f"pass_shard_{shard_of[0]}_of_{shard_of[1]}",
                run_pass(config, shard_of, device),
            )
        elif step == "measured":
            record_step(record, "measured", export_measured(config))
        elif step == "own_task":
            record_step(record, "own_task", own_task(config, device))
        elif step == "rescore":
            record_step(record, "rescore", rescore(config, device))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--shard", type=shard, default=(0, 1))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    steps = [s for s in args.steps.split(",") if s]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {STEPS}")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(load_config(absolute(args.config)), steps, args.shard, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
