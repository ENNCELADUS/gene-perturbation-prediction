"""One automatic run: preparation through validation evaluation, then summary.md.

``python -m src.experiments.all CONFIG [--run-id ID] [--gpus 0,1,2,3]`` writes
``<output_root>/<run id>/{comparison/, train/, evaluation/val/, baselines/val/,
readout/, logs/, summary.md}``. Every step is skipped when its output exists, so
rerunning with the same run id resumes. The test split is never evaluated here.

Steps, in order: preparation; the untrained response-comparison arms (one
subprocess on the first chosen GPU); joint training (``accelerate launch`` on
every chosen GPU); the trained response-comparison jobs (one subprocess per
chosen GPU at a time, the next starting as soon as a GPU frees) and their
summary; validation evaluation, baselines and the readout head (on the first
chosen GPU); summary.md. The chosen GPUs are every visible one unless ``--gpus``
names some. Without CUDA every step runs in this process on the CPU. SIGINT or
SIGTERM terminates the running subprocesses before the run exits.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import math
import os
from pathlib import Path
import signal
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
    "Selective Spearman (per selective gene)": "val_selective_spearman",
    "Selective AUPR lift": "val_selective_aupr_lift",
    "Huber": "val_geneeffect_loss",
    "Absolute Pearson (per line)": "val_geneeffect_pearson_macro_per_line",
    "Residual Pearson (per gene)": "val_residual_pearson_macro_per_gene",
    "Residual Spearman (per gene)": "val_residual_spearman_macro_per_gene",
    "SD ratio (per gene)": "val_residual_sd_ratio_macro_per_gene",
}
POLL_SECONDS = 10.0
STOP_SECONDS = 30.0


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
# GPU subprocesses: one pool of GPU slots per step
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One subprocess step; its log is ``<run>/logs/<step>.log``."""

    step: str
    argv: tuple[str, ...]


def visible_gpus() -> tuple[str, ...]:
    """CUDA device ids as the child processes must name them (empty without CUDA)."""
    import torch

    count = torch.cuda.device_count()
    listed = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = listed.split(",") if listed else [str(i) for i in range(count)]
    return tuple(i.strip() for i in ids[:count])


def choose_gpus(
    requested: Sequence[str] | None, visible: tuple[str, ...]
) -> tuple[str, ...]:
    """Every visible GPU, or the requested ones after checking they are visible.

    GPU ids are the ones ``CUDA_VISIBLE_DEVICES`` lists, or 0..n-1 without it.
    """
    if requested is None:
        return visible
    chosen = tuple(requested)
    if not visible:
        raise ValueError("--gpus was given but no CUDA GPU is visible")
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(visible):
        raise ValueError(
            f"--gpus {','.join(chosen)}: choose distinct GPUs among the visible "
            f"ones {','.join(visible)}"
        )
    return chosen


def start_process(argv: Sequence[str], gpus: Sequence[str], log: Path):
    """Start ``argv`` on ``gpus`` in its own process group, output to ``log``."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus), PYTHONUNBUFFERED="1")
    with log.open("a") as handle:
        return subprocess.Popen(
            argv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )


def stop_processes(processes: Sequence[subprocess.Popen]) -> None:
    """Terminate each process group, wait, then kill whatever is left of it."""
    for process in processes:
        if process.poll() is None:
            with contextlib.suppress(OSError):
                os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_SECONDS
    for process in processes:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        # Workers of a launcher can outlive it; the group id still names them.
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_pool(
    jobs: Sequence[Job],
    slots: Sequence[tuple[str, ...]],
    logs: Path,
    start: Callable = start_process,
) -> None:
    """Run ``jobs`` with at most one job per slot (a tuple of GPU ids) at a time.

    The next job starts as soon as a slot frees. After a failure no new job
    starts; the running ones finish, then this raises naming every failed step,
    its exit code and log. On any exception (SIGINT, SIGTERM) the running
    process groups are terminated.
    """
    logs.mkdir(parents=True, exist_ok=True)
    waiting, free = list(jobs), list(slots)
    running = []  # (job, slot, process, log)
    failures = []
    try:
        while running or (waiting and not failures):
            while waiting and free and not failures:
                job, slot = waiting.pop(0), free.pop(0)
                log = logs / f"{job.step}.log"
                print(
                    f"{job.step}: started on GPU {','.join(slot)}, log {log}",
                    flush=True,
                )
                running.append((job, slot, start(job.argv, slot, log), log))
            time.sleep(POLL_SECONDS)
            for entry in list(running):
                job, slot, process, log = entry
                code = process.poll()
                if code is None:
                    continue
                running.remove(entry)
                free.append(slot)
                if code:
                    failures.append(
                        f"{job.step} failed with exit code {code}; see {log}"
                    )
                    print(failures[-1], flush=True)
                else:
                    print(f"{job.step}: finished", flush=True)
    finally:
        stop_processes([process for _, _, process, _ in running])
    if failures:
        raise RuntimeError("; ".join(failures))


@contextlib.contextmanager
def sigterm_raises():
    """Turn SIGTERM into SystemExit, as SIGINT is KeyboardInterrupt, so that
    :func:`run_pool` terminates its children on either."""

    def handler(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def comparison_argv(config_path: Path, run: Path, *mode: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "src.experiments.response_comparison",
        "--config",
        str(config_path),
        "--out-dir",
        str(run / "comparison"),
        "--device",
        "cuda",
        *mode,
    )


def training_argv(
    config_path: Path, config: dict, run: Path, processes: int
) -> tuple[str, ...]:
    launch = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        str(config["precision"]),
    ]
    if processes > 1:
        launch.append("--multi_gpu")
    return (
        *launch,
        "--module",
        "src.train",
        "--config",
        str(config_path),
        "--run-dir",
        str(run / "train"),
    )


def _check_resume_processes(train: Path, processes: int) -> None:
    """Resuming ``last.pt`` needs the process count it was started with."""
    if not (train / "last.pt").is_file():
        return
    started = _read_json(train / "run.json")["world_size"]
    if started != processes:
        raise ValueError(
            f"training in {train} was started on {started} GPU(s) and resumes "
            f"only on {started} (the effective batch depends on it); pass --gpus "
            f"with {started} GPU(s), not {processes}, or use a new --run-id"
        )


# ----------------------------------------------------------------------------
# Response-model comparison and joint training
# ----------------------------------------------------------------------------


def comparison_and_training(
    config_path: Path,
    config: dict,
    run: Path,
    gpus: tuple[str, ...],
    start: Callable = start_process,
) -> None:
    """Untrained comparison arms, joint training, trained comparison jobs, summary.

    With GPUs every step is a subprocess: the untrained arms on the first GPU,
    training on all of them, then one trained comparison job per GPU at a time.
    Without GPUs every step runs in this process on the CPU, in the same order.
    """
    from src.experiments import response_comparison
    from src.experiments.geneeffect import run_training

    comparison, train, logs = run / "comparison", run / "train", run / "logs"
    compared = (comparison / "verdicts.json").is_file()
    if not compared:
        if gpus:
            job = Job(
                "comparison_untrained", comparison_argv(config_path, run, "--untrained")
            )
            run_pool([job], [gpus[:1]], logs, start)
        else:
            response_comparison.run_untrained(config, comparison, device="cpu")
        if (comparison / "sanity.json").is_file():
            print(sanity_line(comparison), flush=True)
    if not (train / "done.json").is_file():
        if gpus:
            _check_resume_processes(train, len(gpus))
            job = Job("train", training_argv(config_path, config, run, len(gpus)))
            run_pool([job], [gpus], logs, start)
        else:
            run_training(config, train)
    if not compared:
        pending = response_comparison.pending_trained_jobs(config, comparison)
        if gpus:
            jobs = [
                Job(
                    f"comparison_{arm}__{anchor}",
                    comparison_argv(config_path, run, "--job", arm, anchor),
                )
                for arm, anchor in pending
            ]
            run_pool(jobs, [(gpu,) for gpu in gpus], logs, start)
        else:
            for arm, anchor in pending:
                response_comparison.run_trained_job(
                    config, comparison, arm, anchor, device="cpu"
                )
        response_comparison.summarise(config, comparison)


# ----------------------------------------------------------------------------
# Validation evaluation
# ----------------------------------------------------------------------------


def _validation(config: dict, run: Path, device: str) -> None:
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
            "higher validation selective-gene Spearman"
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
        f"{done['next_epoch']} trained (validation selective-gene Spearman "
        f"{_number(done['best_score'])}); {stop}.",
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


def run_all(
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
        run_id = "all_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
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
        manifest = _read_json(prepare_inputs(config))
        print(target_sum_line(manifest), flush=True)
        comparison_and_training(config_path, config, run, chosen)
        device = f"cuda:{visible.index(chosen[0])}" if chosen else "cpu"
        _validation(config, run, device)
        summary = write_summary(config, run, run_id)
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
    # Preparation reports its progress (T, anchors, lines) through logging.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", force=True
    )
    run_all(args.config, run_id=args.run_id, gpus=args.gpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
