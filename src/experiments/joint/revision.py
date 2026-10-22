"""One revision run: training, validation and test with baselines, then summary.md.

Every step is skipped when its output exists, so rerunning with the same run id
resumes; ``train/best.pt`` is chosen on validation alone and then scored once on test.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data.io import read_json
from src.eval.accuracy import BOOTSTRAP_REPEATS, BOOTSTRAP_SEED
from src.experiments.joint.config import load_config
from src.experiments.joint.pipeline import (
    BASELINE_NAMES,
    PRIOR_CONTROLS,
    Job,
    check_resume_processes,
    choose_gpus,
    run_pool,
    sigterm_raises,
    start_process,
    training_argv,
    visible_gpus,
)
from src.experiments.run_dir import git_revision, number

JOINT_NAME = "Joint model"
TX1_RIDGE = "context_pca_ridge[tx1]"
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
# Controls the joint model is bootstrapped against, in order, when present.
COMPARED = (TX1_RIDGE, *PRIOR_CONTROLS)
# The best epoch's record in train/metrics.jsonl: training-diagnostic and validation.
TRAIN_DIAGNOSTIC_KEY = "train_eval_selective_spearman"
VALIDATION_KEY = "val_selective_spearman"


def _finite(value: Any) -> float | None:
    """JSON-safe: undefined (None or NaN) stays None, never 0."""
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def train_joint(
    config_path: Path,
    config: dict,
    run: Path,
    gpus: tuple[str, ...],
    start: Callable = start_process,
) -> None:
    """Joint training on every chosen GPU, or in this process without GPUs."""
    from src.experiments.joint.train import run_training

    train = run / "train"
    if (train / "done.json").is_file():
        return
    if gpus:
        check_resume_processes(train, len(gpus))
        job = Job("train", training_argv(config_path, run, len(gpus)))
        run_pool([job], [gpus], run / "logs", start)
    else:
        run_training(config, train)


def evaluate(config: dict, run: Path, split: str) -> None:
    """Score ``train/best.pt`` and fit and score the baselines on ``split``."""
    from src.experiments.joint.baselines import run_baselines
    from src.experiments.joint.evaluate import evaluate_checkpoint, export_evaluation

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


def _row(metrics: dict, split: str) -> dict[str, float | None]:
    return {key: _finite(metrics[f"{split}_{source}"]) for _, key, source in COLUMNS}


def _baseline_order(baselines: dict) -> list[str]:
    known = [method for method in BASELINE_NAMES if method in baselines]
    return known + [method for method in baselines if method not in BASELINE_NAMES]


def _bootstrap(
    run: Path, split: str, method: str, selective: frozenset[str]
) -> dict[str, Any]:
    """Paired line bootstrap of selective Spearman, joint model minus ``method``."""
    import pandas as pd

    from src.eval import metrics

    joint = pd.read_parquet(run / "evaluation" / split / "predictions.parquet")
    control = pd.read_parquet(run / "baselines" / split / "predictions.parquet")
    control = control.loc[control["method"] == method]
    result = metrics.paired_line_bootstrap(
        joint, control, selective, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED
    )
    low, high = result["interval"]
    return {
        "comparison": f"{JOINT_NAME} minus {BASELINE_NAMES[method]}",
        "repeats": BOOTSTRAP_REPEATS,
        "seed": BOOTSTRAP_SEED,
        "difference": _finite(result["difference"]),
        "interval": [_finite(low), _finite(high)],
    }


def _lineages(
    run: Path, split: str, selective: frozenset[str], split_table: Path
) -> list[dict[str, Any]]:
    """Per lineage: lines, and the mean over its lines of each model's per-line
    residual Spearman across the selective genes (descriptive; 1-5 lines each)."""
    import pandas as pd
    from scipy.stats import spearmanr

    lineage = pd.read_csv(split_table).set_index("model_id")["lineage"]
    frames = {
        JOINT_NAME: pd.read_parquet(run / "evaluation" / split / "predictions.parquet")
    }
    controls = pd.read_parquet(run / "baselines" / split / "predictions.parquet")
    for method in COMPARED:
        if method in set(controls["method"]):
            frames[BASELINE_NAMES[method]] = controls.loc[controls["method"] == method]
    per_line = {}
    for name, frame in frames.items():
        rows = frame.loc[frame["gene_symbol"].isin(selective)]
        per_line[name] = rows.groupby("model_id")[
            ["residual", "residual_prediction"]
        ].apply(lambda g: spearmanr(g["residual"], g["residual_prediction"]).statistic)
    table = pd.DataFrame(per_line)
    table["lineage"] = table.index.map(lineage)
    return [
        {
            "lineage": name,
            "lines": len(group),
            **{model: _finite(group[model].mean()) for model in per_line},
        }
        for name, group in table.groupby("lineage", sort=True)
    ]


def training_record(run: Path) -> dict[str, Any]:
    """Best epoch (1-based; 0 is the pre-update validation of a stack) and its
    diagnostic and validation selective Spearman."""
    done = read_json(run / "train" / "done.json")
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


def _split_record(
    run: Path, split: str, selective: frozenset[str], split_table: Path
) -> dict[str, Any]:
    """The split's table (joint model, then baselines), its bootstraps and lineages."""
    baselines = read_json(run / "baselines" / split / "metrics.json")
    joint = read_json(run / "evaluation" / split / "metrics.json")
    if TX1_RIDGE not in baselines:
        raise ValueError(f"baselines/{split}/metrics.json has no {TX1_RIDGE} method")
    return {
        "models": {
            JOINT_NAME: _row(joint, split),
            **{
                BASELINE_NAMES.get(method, method): _row(baselines[method], split)
                for method in _baseline_order(baselines)
            },
        },
        "bootstraps": [
            _bootstrap(run, split, method, selective)
            for method in COMPARED
            if method in baselines
        ],
        "lineages": _lineages(run, split, selective, split_table),
    }


def build_record(
    config_path: Path, run: Path, run_id: str, git_revision: str, split_table: Path
) -> dict[str, Any]:
    from src.training.checkpoint import load_checkpoint

    selective = frozenset(
        load_checkpoint(run / "train" / "best.pt")["preprocessing"]["selective_genes"]
    )
    return {
        "run_id": run_id,
        "config": str(config_path),
        "git_revision": git_revision,
        **{
            split: _split_record(run, split, selective, split_table) for split in SPLITS
        },
        "training": training_record(run),
    }


def _split_lines(title: str, split: dict[str, Any]) -> list[str]:
    header = ["Model", *(name for name, _, _ in COLUMNS)]
    lines = [
        f"## {title} lines",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name, row in split["models"].items():
        cells = [number(row[key]) for _, key, _ in COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    for bootstrap in split["bootstraps"]:
        low, high = bootstrap["interval"]
        lines.append(
            f"Selective Spearman, {bootstrap['comparison']}: "
            f"{number(bootstrap['difference'])} [{number(low)}, {number(high)}], "
            f"paired bootstrap over {title.lower()} lines ({bootstrap['repeats']} "
            f"resamples, seed {bootstrap['seed']})."
        )
    models = [key for key in split["lineages"][0] if key not in {"lineage", "lines"}]
    lines += [
        "",
        f"Per lineage, descriptive: mean over the lineage's {title.lower()} lines of "
        "the per-line residual Spearman across the selective genes.",
        "",
        "| Lineage | Lines | " + " | ".join(models) + " |",
        "|---|---|" + "---|" * len(models),
    ]
    for row in split["lineages"]:
        cells = [number(row[model]) for model in models]
        lines.append(
            f"| {row['lineage']} | {row['lines']} | " + " | ".join(cells) + " |"
        )
    return lines + [""]


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
        f"{number(training[TRAIN_DIAGNOSTIC_KEY])} on the training diagnostic "
        f"lines and {number(training[VALIDATION_KEY])} on the validation lines.",
    ]


def write_outputs(config_path: Path, run: Path, run_id: str, split_table: Path) -> Path:
    """Write revision.json, then summary.md; return summary.md. ``split_table`` is
    the split's CSV (``model_id``, ``lineage``) for the per-lineage table."""
    record = build_record(config_path, run, run_id, git_revision(), split_table)
    (run / "revision.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    path = run / "summary.md"
    path.write_text("\n".join(_summary_lines(record)) + "\n")
    return path


def run_revision(
    config_path: Path, *, run_id: str | None, gpus: Sequence[str] | None = None
) -> Path:
    """Run (or resume) every step into ``<output_root>/<run id>``; return that dir.

    ``gpus`` (default: every visible GPU) is not bound to the run: a resumed run
    may use other GPUs, except that unfinished training needs as many as it
    started on.
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
            evaluate(config, run, split)
        split_table = Path(config["paths"]["split"]).with_suffix(".csv")
        if not split_table.is_file():
            raise FileNotFoundError(
                f"{split_table}: the split table next to the split file"
            )
        summary = write_outputs(config_path, run, run_id, split_table)
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
