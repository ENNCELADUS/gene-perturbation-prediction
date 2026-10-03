"""One revision run: training, validation evaluation and baselines, then summary.md.

``python -m src.experiments.revision CONFIG [--run-id ID] [--gpus 0,1,2,3]`` writes
``<output_root>/<run id>/{train/, evaluation/val/, baselines/val/, logs/,
revision.json, summary.md}``. Every step is skipped when its output exists, so
rerunning with the same run id resumes. There is no response comparison and no
readout, and the test split is never evaluated here.

Steps, in order: preparation (returns at once on an existing prepared root); joint
training (``accelerate launch`` on every chosen GPU, in this process on a CPU-only
machine); validation evaluation of ``train/best.pt`` and the validation baselines
(on the first chosen GPU); revision.json and summary.md, which hold the validation
table of the joint model against every baseline, the paired line bootstrap of
selective-gene Spearman against the Tx1 context-PCA ridge, and the training record
at the best epoch. The chosen GPUs are every visible one unless ``--gpus`` names
some. SIGINT or SIGTERM terminates the running subprocesses before the run exits.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
import json
import logging
import math
from pathlib import Path
import subprocess
from typing import Any

from src.experiments.all import (
    BASELINE_NAMES,
    Job,
    _check_resume_processes,
    _number,
    _read_json,
    choose_gpus,
    run_pool,
    sigterm_raises,
    start_process,
    training_argv,
    visible_gpus,
)
from src.experiments.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
JOINT_NAME = "Joint model"
TX1_RIDGE = "context_pca_ridge[tx1]"
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 0
# Table column name, revision.json key, metrics key of the joint model and of every
# baseline (both carry the ``val_`` prefix).
COLUMNS = (
    ("Selective Spearman", "selective_spearman", "val_selective_spearman"),
    ("Selective AUPR lift", "selective_aupr_lift", "val_selective_aupr_lift"),
    (
        "Residual Pearson (per variable gene)",
        "residual_pearson",
        "val_residual_pearson_macro_per_gene",
    ),
    ("Huber", "huber", "val_geneeffect_loss"),
    ("SD ratio (per gene)", "sd_ratio", "val_residual_sd_ratio_macro_per_gene"),
)
# The best epoch's record in train/metrics.jsonl: training-diagnostic and validation.
TRAIN_DIAGNOSTIC_KEY = "train_eval_selective_spearman"
VALIDATION_KEY = "val_selective_spearman"


def _finite(value: Any) -> float | None:
    """JSON-safe: undefined (None or NaN) stays None, never 0."""
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ----------------------------------------------------------------------------
# Training and validation
# ----------------------------------------------------------------------------


def train_joint(
    config_path: Path,
    config: dict,
    run: Path,
    gpus: tuple[str, ...],
    start: Callable = start_process,
) -> None:
    """Joint training on every chosen GPU, or in this process without GPUs."""
    from src.experiments.geneeffect import run_training

    train = run / "train"
    if (train / "done.json").is_file():
        return
    if gpus:
        _check_resume_processes(train, len(gpus))
        job = Job("train", training_argv(config_path, config, run, len(gpus)))
        run_pool([job], [gpus], run / "logs", start)
    else:
        run_training(config, train)


def _validation(config: dict, run: Path) -> None:
    from src.experiments.baselines import run_baselines
    from src.experiments.geneeffect import evaluate_checkpoint, export_evaluation

    evaluation = run / "evaluation" / "val"
    if not (evaluation / "metrics.json").is_file():
        print("validation evaluation of the joint model", flush=True)
        export_evaluation(
            evaluate_checkpoint(run / "train" / "best.pt", split="val"), evaluation
        )
    baselines = run / "baselines" / "val"
    if not (baselines / "metrics.json").is_file():
        print("validation baselines", flush=True)
        run_baselines(config, split="val", out_dir=baselines)


# ----------------------------------------------------------------------------
# revision.json and summary.md
# ----------------------------------------------------------------------------


def _row(metrics: dict) -> dict[str, float | None]:
    return {key: _finite(metrics[source]) for _, key, source in COLUMNS}


def _baseline_order(baselines: dict) -> list[str]:
    known = [method for method in BASELINE_NAMES if method in baselines]
    return known + [method for method in baselines if method not in BASELINE_NAMES]


def _bootstrap(run: Path, baselines: dict) -> dict[str, Any]:
    """Paired line bootstrap of selective Spearman, joint model minus Tx1 ridge."""
    import pandas as pd

    from src.eval import metrics
    from src.training.checkpoint import load_checkpoint

    if TX1_RIDGE not in baselines:
        raise ValueError(f"baselines/val/metrics.json has no {TX1_RIDGE} method")
    selective = frozenset(
        load_checkpoint(run / "train" / "best.pt")["preprocessing"]["selective_genes"]
    )
    joint = pd.read_parquet(run / "evaluation" / "val" / "predictions.parquet")
    ridge = pd.read_parquet(run / "baselines" / "val" / "predictions.parquet")
    ridge = ridge.loc[ridge["method"] == TX1_RIDGE]
    result = metrics.paired_line_bootstrap(
        joint,
        ridge,
        selective,
        repeats=BOOTSTRAP_REPEATS,
        seed=BOOTSTRAP_SEED,
    )
    low, high = result["interval"]
    return {
        "comparison": f"{JOINT_NAME} minus {BASELINE_NAMES[TX1_RIDGE]}",
        "repeats": BOOTSTRAP_REPEATS,
        "seed": BOOTSTRAP_SEED,
        "difference": _finite(result["difference"]),
        "interval": [_finite(low), _finite(high)],
    }


def _training_record(run: Path) -> dict[str, Any]:
    """Best epoch (1-based) and its diagnostic and validation selective Spearman."""
    done = _read_json(run / "train" / "done.json")
    best = done["best_epoch"]
    epoch_records = [
        json.loads(line)
        for line in (run / "train" / "metrics.jsonl").read_text().splitlines()
        if line and VALIDATION_KEY in json.loads(line)
    ]
    at_best = [record for record in epoch_records if record["epoch"] == best]
    if not at_best:
        raise ValueError(f"train/metrics.jsonl has no epoch record for epoch {best}")
    record = at_best[-1]
    return {
        "best_epoch": best + 1,
        "epochs_trained": done["next_epoch"],
        TRAIN_DIAGNOSTIC_KEY: _finite(record[TRAIN_DIAGNOSTIC_KEY]),
        VALIDATION_KEY: _finite(record[VALIDATION_KEY]),
    }


def build_record(
    config_path: Path, run: Path, run_id: str, git_revision: str
) -> dict[str, Any]:
    baselines = _read_json(run / "baselines" / "val" / "metrics.json")
    return {
        "run_id": run_id,
        "config": str(config_path),
        "git_revision": git_revision,
        "validation": {
            JOINT_NAME: _row(_read_json(run / "evaluation" / "val" / "metrics.json")),
            **{
                BASELINE_NAMES.get(method, method): _row(baselines[method])
                for method in _baseline_order(baselines)
            },
        },
        "bootstrap": _bootstrap(run, baselines),
        "training": _training_record(run),
    }


def _summary_lines(record: dict[str, Any]) -> list[str]:
    bootstrap, training = record["bootstrap"], record["training"]
    low, high = bootstrap["interval"]
    header = ["Model", *(name for name, _, _ in COLUMNS)]
    lines = [
        f"# Revision run {record['run_id']}",
        "",
        f"Config `{record['config']}`, git revision `{record['git_revision']}`. "
        "Validation only; the test split is not evaluated by this run. Nothing "
        "here is synthetic-lethality evidence.",
        "",
        "## Validation on the GeneEffect validation lines",
        "",
        "Selective Spearman and AUPR lift are macro means over the selective "
        "genes; residual Pearson and the SD ratio are macro means over the "
        "variable genes.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name, row in record["validation"].items():
        cells = [_number(row[key]) for _, key, _ in COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Undefined correlations come from predictors that are constant per gene "
        "across lines (gene mean, copy prior); they are not zero.",
        "",
        "## Selective Spearman against the Tx1 context-PCA ridge",
        "",
        f"{bootstrap['comparison']}: {_number(bootstrap['difference'])} "
        f"[{_number(low)}, {_number(high)}], paired bootstrap over validation "
        f"lines ({bootstrap['repeats']} resamples, seed {bootstrap['seed']}).",
        "",
        "## Training",
        "",
        f"`train/best.pt` is epoch {training['best_epoch']} of "
        f"{training['epochs_trained']} trained. At that epoch the selective "
        f"Spearman is {_number(training[TRAIN_DIAGNOSTIC_KEY])} on the training "
        f"diagnostic lines and {_number(training[VALIDATION_KEY])} on the "
        "validation lines.",
    ]
    return lines


def write_outputs(config_path: Path, run: Path, run_id: str) -> Path:
    """Write revision.json, then summary.md; return summary.md."""
    record = build_record(config_path, run, run_id, _git_revision())
    (run / "revision.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    path = run / "summary.md"
    path.write_text("\n".join(_summary_lines(record)) + "\n")
    return path


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------


def run_revision(
    config_path: Path, *, run_id: str | None, gpus: Sequence[str] | None = None
) -> Path:
    """Run (or resume) every step into ``<output_root>/<run id>``; return that dir.

    ``gpus`` names the GPUs to use (ids as ``CUDA_VISIBLE_DEVICES`` lists them);
    by default every visible GPU. It is not bound to the run: a resumed run may
    use other GPUs, except that unfinished training needs as many as it started on.
    """
    from src.experiments.prepare import prepare_inputs

    config_path = Path(config_path)
    config = load_config(config_path)
    visible = visible_gpus()
    chosen = choose_gpus(gpus, visible)
    if run_id is None:
        run_id = "revision_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"run id: {run_id}", flush=True)
    print(f"GPUs: {','.join(chosen) if chosen else 'none, CPU only'}", flush=True)
    run = Path(config["output_root"]) / run_id
    run.mkdir(parents=True, exist_ok=True)
    # A run directory belongs to one config: finished steps are skipped by
    # file existence, so resuming under another config would mix experiments.
    bound = run / "run_config.json"
    if bound.is_file():
        if json.loads(bound.read_text()) != config:
            raise ValueError(
                f"{run} was started with a different config ({bound}); "
                "use a new --run-id for this config"
            )
    else:
        bound.write_text(json.dumps(config, indent=2) + "\n")

    if chosen:
        import torch

        # In-process GPU work (Tx1 encoding of missing lines, checkpoint
        # evaluation) uses the current device; make it the first chosen GPU.
        torch.cuda.set_device(visible.index(chosen[0]))
    with sigterm_raises():
        prepare_inputs(config)
        train_joint(config_path, config, run, chosen)
        _validation(config, run)
        summary = write_outputs(config_path, run, run_id)
    print(f"summary: {summary}", flush=True)
    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--gpus",
        type=lambda text: tuple(gpu.strip() for gpu in text.split(",")),
        help="comma-separated GPU ids to use (default: every visible GPU)",
    )
    args = parser.parse_args(argv)
    # Preparation reports its progress through logging.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", force=True
    )
    run_revision(args.config, run_id=args.run_id, gpus=args.gpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
