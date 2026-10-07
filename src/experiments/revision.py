"""One revision run: training, validation and test with baselines, then summary.md.

``python -m src.experiments.revision CONFIG [--run-id ID] [--gpus 0,1,2,3]`` writes
``<output_root>/<run id>/{train/, evaluation/{val,test}/, baselines/{val,test}/,
logs/, revision.json, summary.md}``. Every step is skipped when its output exists, so
rerunning with the same run id resumes. There is no response comparison and no
readout. One config is one experiment at one seed: ``train/best.pt`` is chosen on
validation alone and then scored once on test.

Steps, in order: preparation (returns at once on an existing prepared root); joint
training (``accelerate launch`` on every chosen GPU, in this process on a CPU-only
machine); for validation, then test, the evaluation of ``train/best.pt`` and the
baselines (on the first chosen GPU); revision.json and summary.md, which hold per
split the table of the joint model against every baseline and the paired line
bootstrap of selective-gene Spearman against the Tx1 context-PCA ridge, and the
training record at the best epoch. The chosen GPUs are every visible one unless
``--gpus`` names some. SIGINT or SIGTERM terminates the running subprocesses before
the run exits.
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
# Evaluated splits, in order, with their summary section titles.
SPLITS = {"val": "Validation", "test": "Test"}
# Table column name, revision.json key, metrics key of the joint model and of every
# baseline (both prefix it with the split, ``val_`` or ``test_``).
COLUMNS = (
    ("Selective Spearman", "selective_spearman", "selective_spearman"),
    ("Selective AUPR lift", "selective_aupr_lift", "selective_aupr_lift"),
    (
        "Residual Pearson (per variable gene)",
        "residual_pearson",
        "residual_pearson_macro_per_gene",
    ),
    ("Huber", "huber", "geneeffect_loss"),
    ("SD ratio (per gene)", "sd_ratio", "residual_sd_ratio_macro_per_gene"),
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
# Training and evaluation
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


def _evaluate(config: dict, run: Path, split: str) -> None:
    """Score ``train/best.pt`` and fit and score the baselines on ``split``."""
    from src.experiments.baselines import run_baselines
    from src.experiments.geneeffect import evaluate_checkpoint, export_evaluation

    title = SPLITS[split].lower()
    evaluation = run / "evaluation" / split
    if not (evaluation / "metrics.json").is_file():
        print(f"{title} evaluation of the joint model", flush=True)
        export_evaluation(
            evaluate_checkpoint(run / "train" / "best.pt", split=split), evaluation
        )
    baselines = run / "baselines" / split
    if not (baselines / "metrics.json").is_file():
        print(f"{title} baselines", flush=True)
        run_baselines(config, split=split, out_dir=baselines)


# ----------------------------------------------------------------------------
# revision.json and summary.md
# ----------------------------------------------------------------------------


def _row(metrics: dict, split: str) -> dict[str, float | None]:
    return {key: _finite(metrics[f"{split}_{source}"]) for _, key, source in COLUMNS}


def _baseline_order(baselines: dict) -> list[str]:
    known = [method for method in BASELINE_NAMES if method in baselines]
    return known + [method for method in baselines if method not in BASELINE_NAMES]


def _bootstrap(
    run: Path, split: str, baselines: dict, selective: frozenset[str]
) -> dict[str, Any]:
    """Paired line bootstrap of selective Spearman, joint model minus Tx1 ridge."""
    import pandas as pd

    from src.eval import metrics

    if TX1_RIDGE not in baselines:
        raise ValueError(f"baselines/{split}/metrics.json has no {TX1_RIDGE} method")
    joint = pd.read_parquet(run / "evaluation" / split / "predictions.parquet")
    ridge = pd.read_parquet(run / "baselines" / split / "predictions.parquet")
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
    """Best epoch (1-based; 0 is the pre-update validation of a stack) and its
    diagnostic and validation selective Spearman."""
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


def _split_record(run: Path, split: str, selective: frozenset[str]) -> dict[str, Any]:
    """The split's table (joint model, then baselines) and its bootstrap."""
    baselines = _read_json(run / "baselines" / split / "metrics.json")
    joint = _read_json(run / "evaluation" / split / "metrics.json")
    return {
        "models": {
            JOINT_NAME: _row(joint, split),
            **{
                BASELINE_NAMES.get(method, method): _row(baselines[method], split)
                for method in _baseline_order(baselines)
            },
        },
        "bootstrap": _bootstrap(run, split, baselines, selective),
    }


def build_record(
    config_path: Path, run: Path, run_id: str, git_revision: str
) -> dict[str, Any]:
    from src.training.checkpoint import load_checkpoint

    selective = frozenset(
        load_checkpoint(run / "train" / "best.pt")["preprocessing"]["selective_genes"]
    )
    return {
        "run_id": run_id,
        "config": str(config_path),
        "git_revision": git_revision,
        **{split: _split_record(run, split, selective) for split in SPLITS},
        "training": _training_record(run),
    }


def _split_lines(title: str, split: dict[str, Any]) -> list[str]:
    bootstrap = split["bootstrap"]
    low, high = bootstrap["interval"]
    header = ["Model", *(name for name, _, _ in COLUMNS)]
    lines = [
        f"## {title} lines",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name, row in split["models"].items():
        cells = [_number(row[key]) for _, key, _ in COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines + [
        "",
        f"Selective Spearman, {bootstrap['comparison']}: "
        f"{_number(bootstrap['difference'])} [{_number(low)}, {_number(high)}], "
        f"paired bootstrap over {title.lower()} lines ({bootstrap['repeats']} "
        f"resamples, seed {bootstrap['seed']}).",
        "",
    ]


def _summary_lines(record: dict[str, Any]) -> list[str]:
    training = record["training"]
    lines = [
        f"# Revision run {record['run_id']}",
        "",
        f"Config `{record['config']}`, git revision `{record['git_revision']}`. "
        "`train/best.pt` is chosen on validation alone, then scored once on test. "
        "Nothing here is synthetic-lethality evidence.",
        "",
        "Selective Spearman and AUPR lift are macro means over the selective "
        "genes; residual Pearson and the SD ratio are macro means over the "
        "variable genes. Undefined correlations come from predictors that are "
        "constant per gene across lines (gene mean, copy prior); they are not zero.",
        "",
    ]
    for split, title in SPLITS.items():
        lines += _split_lines(title, record[split])
    when = (
        "the model before its first update (the prior alone)"
        if training["best_epoch"] == 0
        else f"epoch {training['best_epoch']}"
    )
    return lines + [
        "## Training",
        "",
        f"`train/best.pt` is {when}; {training['epochs_trained']} epochs trained. "
        f"At that point the selective Spearman is "
        f"{_number(training[TRAIN_DIAGNOSTIC_KEY])} on the training diagnostic "
        f"lines and {_number(training[VALIDATION_KEY])} on the validation lines.",
    ]


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
        for split in SPLITS:
            _evaluate(config, run, split)
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
