"""One automatic run: preparation through validation evaluation, then summary.md.

``python -m src.experiments.all CONFIG [--run-id ID]`` writes
``<output_root>/<run id>/{comparison/, train/, evaluation/val/, baselines/val/,
readout/, logs/, summary.md}``. Every step is skipped when its output exists, so
rerunning with the same run id resumes. The test split is never evaluated here.

GPU use: with two or more visible GPUs the response-model comparison runs on the
last one while joint training runs on the others, concurrently; with one GPU they
run one after the other; without CUDA both run in this process on the CPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from src.experiments.config import load_config

LINE_NAMES = {
    "ACH-000551": "K562",
    "ACH-000739": "HepG2",
    "ACH-000971": "HCT116",
    "ACH-000995": "Jurkat",
}
ARM_NAMES = {
    "no_change": "No change",
    "global_mean_effect": "Global mean effect",
    "released_state": "Released STATE checkpoint",
    "state_joint": "STATE as in the joint model",
    "mlp_hvg": "MLP on HVG",
    "mlp_tx1": "MLP on Tx1",
}
BASELINE_NAMES = {
    "gene_mean": "Gene mean",
    "copy_prior": "K562 copy prior",
    "nearest_line[tx1]": "Nearest line (Tx1)",
    "nearest_line[hvg]": "Nearest line (HVG)",
    "context_pca_ridge[tx1]": "Context-PCA ridge (Tx1)",
    "context_pca_ridge[hvg]": "Context-PCA ridge (HVG)",
}
VALIDATION_COLUMNS = {
    "Huber": "val_geneeffect_loss",
    "Absolute Pearson (per line)": "val_geneeffect_pearson_macro_per_line",
    "Residual Pearson (per gene)": "val_residual_pearson_macro_per_gene",
    "Residual Spearman (per gene)": "val_residual_spearman_macro_per_gene",
    "SD ratio (per gene)": "val_residual_sd_ratio_macro_per_gene",
}
POLL_SECONDS = 10.0


def _name(model_id: str) -> str:
    return (
        f"{LINE_NAMES[model_id]} ({model_id})" if model_id in LINE_NAMES else model_id
    )


def _number(value: Any, digits: int = 4) -> str:
    """Undefined (None or NaN) is written as such, never as 0."""
    if value is None or not math.isfinite(float(value)):
        return "undefined"
    return f"{float(value):.{digits}f}"


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------
# Response-model comparison and joint training, scheduled over visible GPUs
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One subprocess step; its log is ``<run>/logs/<step>.log``."""

    step: str
    argv: tuple[str, ...]
    gpus: tuple[str, ...]


def visible_gpus() -> tuple[str, ...]:
    """CUDA device ids as the child processes must name them."""
    import torch

    count = torch.cuda.device_count()
    listed = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = listed.split(",") if listed else [str(i) for i in range(count)]
    return tuple(i.strip() for i in ids[:count])


def gpu_schedule(
    config_path: Path, config: dict, run: Path, gpus: tuple[str, ...]
) -> list[list[Job]]:
    """Batches of jobs; the jobs of one batch run concurrently.

    Two or more GPUs: the comparison on the last GPU beside training on the
    rest. One GPU: comparison, then training. Finished steps are left out.
    Training keeps the same GPUs when the comparison is already finished,
    because resuming from ``last.pt`` requires the same number of processes.
    """
    comparison = Job(
        "comparison",
        (
            sys.executable,
            "-m",
            "src.experiments.response_comparison",
            "--config",
            str(config_path),
            "--out-dir",
            str(run / "comparison"),
            "--device",
            "cuda",
        ),
        gpus[-1:],
    )
    train_gpus = gpus[:-1] if len(gpus) > 1 else gpus
    launch = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(len(train_gpus)),
        "--num_machines",
        "1",
        "--mixed_precision",
        str(config["precision"]),
    ]
    if len(train_gpus) > 1:
        launch.append("--multi_gpu")
    train = Job(
        "train",
        (
            *launch,
            "--module",
            "src.train",
            "--config",
            str(config_path),
            "--run-dir",
            str(run / "train"),
        ),
        train_gpus,
    )
    pending = [
        job
        for job, done in (
            (comparison, run / "comparison" / "verdicts.json"),
            (train, run / "train" / "done.json"),
        )
        if not done.is_file()
    ]
    return [pending] if len(gpus) > 1 else [[job] for job in pending]


def run_jobs(jobs: list[Job], run: Path, on_poll=lambda: None) -> None:
    """Start the jobs, wait for all of them, raise naming every failed step."""
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = []
    for job in jobs:
        log = logs / f"{job.step}.log"
        handle = log.open("a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(job.gpus))
        env["PYTHONUNBUFFERED"] = "1"
        print(f"{job.step}: started on GPU {','.join(job.gpus)}, log {log}", flush=True)
        process = subprocess.Popen(
            job.argv, stdout=handle, stderr=subprocess.STDOUT, env=env
        )
        running.append((job, process, handle, log))
    failures = []
    while running:
        on_poll()
        for entry in list(running):
            job, process, handle, log = entry
            code = process.poll()
            if code is None:
                continue
            handle.close()
            running.remove(entry)
            if code:
                failures.append(f"{job.step} failed with exit code {code}; see {log}")
                print(failures[-1], flush=True)
            else:
                print(f"{job.step}: finished", flush=True)
        if running:
            time.sleep(POLL_SECONDS)
    on_poll()
    if failures:
        raise RuntimeError("; ".join(failures))


def _comparison_and_training(config_path: Path, config: dict, run: Path) -> None:
    comparison, train = run / "comparison", run / "train"
    announced = []

    def announce_sanity() -> None:
        if not announced and (comparison / "sanity.json").is_file():
            announced.append(True)
            print(sanity_line(comparison), flush=True)

    gpus = visible_gpus()
    if gpus:
        for batch in gpu_schedule(config_path, config, run, gpus):
            run_jobs(batch, run, announce_sanity)
    else:
        from src.experiments.geneeffect import run_training
        from src.experiments.response_comparison import run_comparison

        if not (comparison / "verdicts.json").is_file():
            run_comparison(config, comparison, device="cpu")
        announce_sanity()
        if not (train / "done.json").is_file():
            run_training(config, train)
    announce_sanity()


# ----------------------------------------------------------------------------
# Validation evaluation
# ----------------------------------------------------------------------------


def _validation(config: dict, run: Path) -> None:
    import torch

    from src.experiments.baselines import run_baselines
    from src.experiments.geneeffect import evaluate_checkpoint, export_evaluation
    from src.experiments.readout import run_readout

    best = run / "train" / "best.pt"
    evaluation = run / "evaluation" / "val"
    if not (evaluation / "metrics.json").is_file():
        print("validation evaluation of the joint model", flush=True)
        export_evaluation(evaluate_checkpoint(best, split="val"), evaluation)
    baselines = run / "baselines" / "val"
    if not (baselines / "metrics.json").is_file():
        print("validation baselines", flush=True)
        run_baselines(config, split="val", out_dir=baselines)
    if not (run / "readout" / "metrics.json").is_file():
        print("readout head with the explicit gene-specific context slope", flush=True)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        run_readout(best, run / "readout", device=device)


# ----------------------------------------------------------------------------
# summary.md
# ----------------------------------------------------------------------------


def target_sum_line(manifest: dict) -> str:
    space = manifest["expression_space"]
    sources = " and ".join(_name(m) for m in space["target_sum_sources"])
    return (
        f"Target total T = {space['target_sum']:.1f}: the median whole-library UMI "
        f"count of the non-targeting cells in {sources}. Every expression quantity "
        f"except Tx1's input is log1p(x * T / library size), library size over all "
        f"genes."
    )


def sanity_line(comparison: Path) -> str:
    sanity = _read_json(comparison / "sanity.json")
    parts = [
        f"{_name(anchor)} {_number(row['ratio_to_no_change'], 3)} "
        f"({row['covered_genes']} of {row['total_genes']} genes in its vocabulary)"
        for anchor, row in sanity["anchors"].items()
    ]
    return (
        "STATE sanity line, released checkpoint untrained, held-out loss / "
        "no-change loss: " + "; ".join(parts)
    )


def _interval(pooled: dict) -> str:
    low, high = pooled["interval"]
    return f"{_number(pooled['ratio'])} [{_number(low)}, {_number(high)}]"


def _comparison_section(comparison: Path) -> list[str]:
    import pandas as pd

    from src.experiments.response_comparison import VERDICTS

    table = pd.read_csv(comparison / "comparison.csv")
    verdicts = _read_json(comparison / "verdicts.json")
    folds = list(dict.fromkeys(table.fold))
    by_fold = table.set_index(["arm", "fold"])
    header = [
        "Arm",
        *(f"{_name(f)} held out" for f in folds),
        "Pooled, all folds [95% interval]",
        "Pooled, without HCT116 [95% interval]",
        "Identity share (mean over folds)",
    ]
    lines = [
        "## Response-model comparison",
        "",
        "Held-out loss divided by the no-change loss on the same conditions "
        "(below 1 beats no-change); each fold trains on three anchors and scores "
        f"the fourth. Intervals: {verdicts['bootstrap']}-resample gene bootstrap "
        "of the pooled ratio.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for arm, name in ARM_NAMES.items():
        identity = by_fold.loc[arm].identity_share.dropna()
        cells = [
            name,
            *(_number(by_fold.loc[(arm, f), "held_out_ratio"]) for f in folds),
            _interval(verdicts["pooled"]["all_folds"][arm]),
            _interval(verdicts["pooled"]["without_hct116"][arm]),
            _number(identity.mean() if len(identity) else None),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Verdicts (difference = first arm's pooled ratio minus the second's; "
        "below 0 favours the first):",
        "",
    ]
    for key, first, second in VERDICTS:
        words = []
        for variant, label in (
            ("all_folds", "all folds"),
            ("without_hct116", "without HCT116"),
        ):
            verdict = verdicts["verdicts"][variant][key]
            low, high = verdict["interval"]
            difference = verdict["difference"]
            if verdict["label"] == "separated":
                better = ARM_NAMES[first] if difference < 0 else ARM_NAMES[second]
                outcome = f"{better} is better, the interval excludes 0"
            else:
                outcome = "no separation, the interval includes 0"
            words.append(
                f"{label}: difference {_number(difference)} "
                f"[{_number(low)}, {_number(high)}], {outcome}"
            )
        lines.append(
            f"- {ARM_NAMES[first]} vs {ARM_NAMES[second]}: " + "; ".join(words) + "."
        )
    return lines


def _validation_section(config: dict, run: Path) -> list[str]:
    done = _read_json(run / "train" / "done.json")
    train = config["train"]
    if done["bad_epochs"] >= train["patience"]:
        stop = (
            f"training stopped early after {done['bad_epochs']} epochs without a "
            "lower validation GeneEffect loss"
        )
    else:
        stop = f"training ran to the maximum of {train['max_epochs']} epochs"
    baselines = _read_json(run / "baselines" / "val" / "metrics.json")
    rows = {
        "Joint model": _read_json(run / "evaluation" / "val" / "metrics.json"),
        "Readout head with the explicit gene-specific context slope": _read_json(
            run / "readout" / "metrics.json"
        ),
        **{name: baselines.get(method) for method, name in BASELINE_NAMES.items()},
    }
    lines = [
        "## Validation on the GeneEffect validation lines",
        "",
        f"Joint model: `train/best.pt` is epoch {done['best_epoch'] + 1} of "
        f"{done['next_epoch']} trained (validation GeneEffect loss "
        f"{_number(done['best_loss'])}); {stop}.",
        "",
        "| Model | " + " | ".join(VALIDATION_COLUMNS) + " |",
        "|" + "---|" * (len(VALIDATION_COLUMNS) + 1),
    ]
    for name, metrics in rows.items():
        cells = (
            ["not produced"] * len(VALIDATION_COLUMNS)
            if metrics is None
            else [_number(metrics[key]) for key in VALIDATION_COLUMNS.values()]
        )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Undefined correlations come from predictors that are constant per gene "
        "across lines (gene mean, copy prior); they are not zero.",
    ]
    return lines


def write_summary(config: dict, run: Path, run_id: str) -> Path:
    from src.data.prepared import PREPARED_METADATA_FILENAME

    manifest = _read_json(Path(config["prepared_root"]) / PREPARED_METADATA_FILENAME)
    lines = [
        f"# Run {run_id}",
        "",
        "Validation only; the test split is not evaluated by this run. Nothing "
        "here is synthetic-lethality evidence.",
        "",
        "## Expression space",
        "",
        target_sum_line(manifest),
        "",
        "## STATE sanity line",
        "",
        sanity_line(run / "comparison"),
        "",
        *_comparison_section(run / "comparison"),
        "",
        *_validation_section(config, run),
    ]
    path = run / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------


def run_all(config_path: Path, *, run_id: str | None) -> Path:
    """Run (or resume) every step into ``<output_root>/<run id>``; return that dir."""
    from src.experiments.prepare import prepare_inputs

    config_path = Path(config_path)
    config = load_config(config_path)
    if run_id is None:
        run_id = "all_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"run id: {run_id}", flush=True)
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

    manifest = _read_json(prepare_inputs(config))
    print(target_sum_line(manifest), flush=True)
    _comparison_and_training(config_path, config, run)
    _validation(config, run)
    summary = write_summary(config, run, run_id)
    print(f"summary: {summary}", flush=True)
    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    run_all(args.config, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
