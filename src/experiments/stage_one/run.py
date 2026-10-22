"""Stage-one runs on the ladder inputs (PIE stage-one design §3.3, §4, Task 3;
protocol §16), in ``.venv-pie``.

``python -m src.experiments.stage_one run CONFIG [--run-id ID] [--fold K]
[--device cuda]`` (``hpc/run.sh stage-one``) trains one config (``kind: train``):
the target is ``GeneEffect - m_g`` over the frame's 981 lines in units of
``sigma_g`` (``--fold K`` drops fold K from training and refits ``m_g`` without
it); ``best.pt`` is chosen on validation per-line residual Spearman; then every line is
predicted once by ``best.pt``: ``predictions.parquet`` (every line x every gene, in
scoring units ``(m_g - mu_g) / sigma_g + z``; the 180 genes outside the decode
panel predict the base), ``V.npy`` (lines x panel genes x 32, fp16: the viability
embedding, ``V.json`` marks the lines trained on as in-sample) and ``row.json``
(validation; test once, in the main run only; the fold's single-cell training lines
out of fold, in a fold run; the training lines in-sample). A run directory
``<output_root>/<task>/<run id>[__fold<k>]`` belongs to one config; a run whose
``row.json`` exists is skipped, and ``last.pt`` resumes an interrupted one.

``queue --gpus 0,1,2,3 [--folds main,0,1,2,3,4] CONFIG...``
(``hpc/run.sh stage-one-queue``) runs every training config and fold, one process
per GPU, skipping configs of another kind and runs whose ``row.json`` exists.

``report CONFIG [--device cuda]`` (``hpc/run.sh stage-one-report``) scores the rows
a report config lists, trained and untrained (the base; the default prior refitted
on the frame, ``src.experiments.stage_one.prior_reference``; a per-gene ridge on the
line-centred decoder-feature delta, solved in its dual on the device), on
validation and test and out of fold over the single-cell training lines, with
the listed contrasts' 95% paired line-bootstrap intervals: ``results.md`` and
``results.json`` in ``<output_root>/<task>/<report config stem>/``.

Nothing is chosen in code: people read the report.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.context_prior.targets import Definitions
from src.data import ladder_reader as reader
from src.data.io import read_json, write_json
from src.experiments.run_dir import assert_bound, bind, locked
from src.experiments.stage_one.setup import Setup, load_setup
from src.experiments.stage_one.resident import Resident

log = logging.getLogger(__name__)

SPLITS = ("val", "test")
ROW = "row.json"
#: The metrics of the report's tables, in order.
REPORTED = (
    ("line_spearman", "Per-line residual Spearman"),
    ("line_spearman_scored", "Lines scored"),
    ("selective_spearman", "Selective Spearman"),
    ("selective_spearman_scored", "Genes scored"),
    ("within_gene_r2", "Within-gene R²"),
    ("selective_spearman_panel", "Selective Spearman, panel"),
    ("aupr_lift", "AUPR lift"),
    ("residual_pearson", "Residual Pearson"),
    ("absolute_pearson", "Per-line Pearson"),
    ("rmse", "RMSE"),
    ("calibration_slope", "Calibration slope"),
)


# --- setup -------------------------------------------------------------------------


def run_directory(config: Mapping[str, Any], run_id: str, fold: int | None) -> Path:
    name = run_id if fold is None else f"{run_id}__fold{fold}"
    return Path(config["output_root"]) / config["task"] / name


def predictions_frame(
    setup: Setup,
    m_g: pd.Series,
    z: np.ndarray,
    lines: Sequence[str],
    genes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Scoring units ``(m_g - mu_g) / sigma_g + z`` on every gene; ``z`` covers
    ``genes`` (default the decode panel; genes x lines), every other gene predicts
    the base."""
    frame = reader.base_prediction(setup.ladder, m_g, lines)
    frame.loc[:, list(setup.ladder.panel if genes is None else genes)] += z.T.astype(
        np.float64
    )
    return frame


def run_genes(setup: Setup, config: Mapping[str, Any]) -> tuple[list[str], dict]:
    """The genes a run trains on and predicts (``data.genes``) and their coverage:
    the decode panel, or the selective genes in it that have a full-panel profile
    delta (``pie/full/``), the same set for both matched profile rows."""
    ladder = setup.ladder
    if config["data"]["genes"] == "panel":
        return ladder.panel, {"genes": "panel", "count": len(ladder.panel)}
    selective = [g for g in ladder.panel if bool(ladder.genes.at[g, "selective"])]
    found = reader.full_panel_genes(ladder, selective)
    return found, {
        "genes": "selective_full",
        "count": len(found),
        "selective_in_panel": len(selective),
        "without_full_panel": [g for g in selective if g not in set(found)],
    }


def bind_genes(run_dir: Path, genes: Sequence[str]) -> None:
    """A run keeps the genes it started with: the full-panel set is found from the
    files present, so a resume after files were added or removed would train on
    another set; it is refused instead."""
    path = run_dir / "genes.json"
    with locked(run_dir):
        if path.is_file():
            if read_json(path) != list(genes):
                raise ValueError(
                    f"{run_dir} was started on another gene set ({path}); "
                    "use a new --run-id"
                )
            return
        write_json(path, list(genes))


# --- inputs on the device ------------------------------------------------------------


# --- one run -------------------------------------------------------------------------


def make_model(config: Mapping[str, Any], resident: Resident) -> torch.nn.Module:
    from src.model.stage_one import build_model, direct_readouts

    model = build_model(config, resident.dims())
    for direct in direct_readouts(model):
        direct.token_dropout = float(config["train"]["token_dropout"])
        direct.evidence_dropout = float(config["train"]["evidence_dropout"])
    return model.to(resident.device)


def warm_start_trunk(
    model: torch.nn.Module, path: str | Path, run_dir: Path
) -> list[str]:
    """The direct readout's shared modules from a released PIE checkpoint (at PIE's
    width), by ``trunk_name_map`` and the gene encoder's ``gene_text`` projection
    from PIE's gene query MLP; returns the tensors left fresh."""
    from pie.model.geneeffect import trunk_name_map, warm_start
    from pie.predict import load_checkpoint

    # The checkpoint's config saves its run_dir as ${PIE_RUNS_ROOT}/<experiment>,
    # which loading resolves (as in pie_reproduction); nothing is read there.
    os.environ["PIE_RUNS_ROOT"] = str(run_dir / "pie_runs")

    from src.model.stage_one import direct_readouts

    loaded = load_checkpoint(Path(path))
    layers = loaded.config.model.model_dump()
    names = {}
    for prefix, module in model.named_modules():
        if module in direct_readouts(model):
            head = f"{prefix}." if prefix else ""
            names |= trunk_name_map(module.trunk, layers, prefix=f"{head}trunk.")
            names[f"{head}gene_encoder.projections.gene_text"] = "gene_query_mlp"
    return warm_start(model, loaded.state_dict, names)


def run(config_path: Path, run_id: str | None, fold: int | None, device: str) -> Path:
    from src.experiments.stage_one.config import load_stage_one_config
    from src.training.stage_one_trainer import TrainingData, fit, load_strict, predict

    config = load_stage_one_config(config_path)
    if config["kind"] != "train":
        raise ValueError(f"{config_path} is a {config['kind']} config, not a run")
    run_dir = run_directory(config, run_id or Path(config_path).stem, fold)
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, {"run": config, "fold": fold})
    if (run_dir / ROW).is_file():
        log.info("%s: done", run_dir)
        return run_dir
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    setup = load_setup(config["data"]["ladder_config"])
    ladder = setup.ladder
    targets = setup.targets(fold)
    torch_device = torch.device(device)
    genes, coverage = run_genes(setup, config)
    bind_genes(run_dir, genes)
    resident = Resident(setup, config, fold, targets, torch_device, genes)
    model = make_model(config, resident)
    fresh = None
    trunk = config["model"]["trunk"]
    if trunk is not None and trunk["warm_start"] is not None:
        fresh = warm_start_trunk(model, trunk["warm_start"], run_dir)
    panel, lines = ladder.panel, ladder.line_ids
    position = {m: i for i, m in enumerate(lines)}
    table = targets.values.loc[:, genes].to_numpy(np.float32)
    labels = torch.full((len(lines), len(genes)), torch.nan, device=torch_device)
    trained = torch.tensor([position[m] for m in targets.lines], device=torch_device)
    labels[trained] = torch.from_numpy(table).to(torch_device)
    gene_table = ladder.genes.loc[list(genes)]

    def gene_tensor(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(values, dtype=dtype, device=torch_device)

    data = TrainingData(
        resident.batch,
        labels,
        trained,
        len(genes),
        m_g=gene_tensor(targets.m_g.loc[list(genes)].to_numpy(), torch.float32),
        sigma_g=gene_tensor(gene_table["sigma_g"].to_numpy(), torch.float32),
        selective=gene_tensor(gene_table["selective"].to_numpy(bool), torch.bool),
    )
    definitions = setup.definitions
    val = ladder.role_lines("val")
    val_positions = torch.tensor([position[m] for m in val], device=torch_device)
    val_truth = setup.truth(val)

    def validate(module: torch.nn.Module) -> dict[str, Any]:
        z = predict(module, resident.batch, val_positions, len(genes), config)
        scores = wide_scores(
            predictions_frame(setup, targets.m_g, z, val, genes),
            val_truth,
            definitions,
            panel,
        )
        return {f"val_{key}": value for key, value in scores.items()}

    result = fit(model, data, config, run_dir, setup.split, validate)
    best = make_model(config, resident)
    load_strict(result.best, best, config, setup.split)
    viability = np.lib.format.open_memmap(
        run_dir / ".V.npy.tmp",
        mode="w+",
        dtype=np.float16,
        shape=(len(lines), len(genes), 32),
    )
    z = predict(
        best,
        resident.batch,
        torch.arange(len(lines), device=torch_device),
        len(genes),
        config,
        viability=viability,
    )
    viability.flush()
    del viability
    os.replace(run_dir / ".V.npy.tmp", run_dir / "V.npy")
    in_sample = set(targets.lines)
    write_json(
        run_dir / "V.json",
        {
            "layout": "lines x the run's genes x 32, float16",
            "lines": lines,
            "genes": genes,
            "in_sample": [m in in_sample for m in lines],
        },
    )
    predictions = predictions_frame(setup, targets.m_g, z, lines, genes)
    temporary = run_dir / ".predictions.parquet.tmp"
    predictions.astype(np.float32).to_parquet(temporary)
    os.replace(temporary, run_dir / "predictions.parquet")
    scored = {"val": val}
    if fold is None:
        scored["test"] = ladder.role_lines("test")
    else:
        scored["oof"] = [
            m for m in ladder.role_lines("train_cells") if ladder.folds[m] == fold
        ]
    scored["train_in_sample"] = [
        m for m in ladder.role_lines("train_cells") if m in in_sample
    ]
    scores = {
        name: wide_scores(
            predictions.loc[group], setup.truth(group), definitions, panel
        )
        for name, group in scored.items()
    }
    write_json(
        run_dir / ROW,
        {
            "fold": fold,
            "best_epoch": result.state.best_epoch,
            "best_val_line_spearman": result.state.best_score,
            "epochs": result.state.next_epoch - 1,
            "updates": result.state.global_step,
            "scores": scores,
            "history": result.history,
            "lines_trained": len(targets.lines),
            "warm_start_fresh": fresh,
            "lineage_classes": resident.line_classes,
            "profile_variance_kept": resident.variance_kept,
            "genes": coverage,
            "full_panel_stream": resident.full_record,
        },
    )
    log.info(
        "%s: validation per-line residual Spearman %s",
        run_dir,
        scores["val"]["line_spearman"],
    )
    return run_dir


def wide_scores(
    prediction: pd.DataFrame,
    truth: pd.DataFrame,
    definitions: Definitions,
    panel: Sequence[str],
) -> dict[str, Any]:
    from src.eval.accuracy import wide_accuracy

    return wide_accuracy(prediction, truth, definitions, panel=panel)


# --- the queue ---------------------------------------------------------------------


def queue(configs: Sequence[Path], gpus: Sequence[str], folds: Sequence[str]) -> int:
    """Every training config x fold without a ``row.json``, one process per GPU;
    returns the number of runs that failed."""
    from src.experiments.stage_one.config import load_stage_one_config

    jobs = []
    for path in configs:
        config = load_stage_one_config(path)
        if config["kind"] != "train":
            print(f"skip {path}: a {config['kind']} config", flush=True)
            continue
        for value in folds:
            fold = None if value == "main" else int(value)
            run_dir = run_directory(config, Path(path).stem, fold)
            if (run_dir / "run_config.json").is_file():
                # Every existing run directory is checked before anything starts.
                assert_bound(run_dir, {"run": config, "fold": fold})
            if (run_dir / ROW).is_file():
                print(f"skip {run_dir}: done", flush=True)
                continue
            jobs.append((Path(path), fold, run_dir))
    free, running, failed = list(gpus), {}, []
    while jobs or running:
        while free and jobs:
            path, fold, run_dir = jobs.pop(0)
            gpu = free.pop(0)
            logs = run_dir.parent / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            argv = [sys.executable, "-m", "src.experiments.stage_one", "run", str(path)]
            if fold is not None:
                argv += ["--fold", str(fold)]
            handle = open(logs / f"{run_dir.name}.log", "a")
            process = subprocess.Popen(
                argv,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
            )
            running[process] = (gpu, run_dir, handle)
            print(f"start {run_dir.name} on GPU {gpu}", flush=True)
        for process in list(running):
            if process.poll() is None:
                continue
            gpu, run_dir, handle = running.pop(process)
            handle.close()
            free.append(gpu)
            status = (
                "done" if process.returncode == 0 else f"failed ({process.returncode})"
            )
            if process.returncode != 0:
                failed.append(run_dir.name)
            print(f"{status} {run_dir.name}", flush=True)
        time.sleep(5)
    print(f"queue finished; failed: {failed or 'none'}", flush=True)
    return len(failed)


# --- the per-gene ridge on the decoder-feature delta --------------------------------


# --- the report -----------------------------------------------------------------------


# --- entry point --------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("config", type=Path)
    one.add_argument("--run-id")
    one.add_argument("--fold", type=int)
    one.add_argument("--device", default="cuda")
    many = commands.add_parser("queue")
    many.add_argument("configs", type=Path, nargs="+")
    many.add_argument("--gpus", required=True)
    many.add_argument("--folds", default="main")
    summary = commands.add_parser("report")
    summary.add_argument("config", type=Path)
    summary.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "run":
        run(args.config, args.run_id, args.fold, args.device)
        return 0
    if args.command == "queue":
        folds = [f for f in args.folds.split(",") if f]
        bad = [f for f in folds if f != "main" and not f.isdigit()]
        if bad:
            parser.error(f"--folds takes main and fold numbers, got {bad}")
        gpus = [g for g in args.gpus.split(",") if g]
        return int(queue(args.configs, gpus, folds) > 0)
    from src.experiments.stage_one.report import report

    report(args.config, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
