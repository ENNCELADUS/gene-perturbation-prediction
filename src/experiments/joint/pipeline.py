"""What the joint model's runs (``all`` and ``revision``) share: the GPU pool their
processes run on, the training launch under ``accelerate``, the resume check, and the
names of the baselines and prior controls their summaries report."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.data.io import read_json
from src.experiments.joint.config import PRECISION

BASELINE_NAMES = {
    "gene_mean": "Gene mean",
    "copy_prior": "K562 copy prior",
    "nearest_line[tx1]": "Nearest line (Tx1)",
    "nearest_line[hvg]": "Nearest line (HVG)",
    "context_pca_ridge[tx1]": "Context-PCA ridge (Tx1)",
    "context_pca_ridge[hvg]": "Context-PCA ridge (HVG)",
    "context_prior": "Linear context prior",
    "context_prior+context_pca_ridge[tx1]": (
        "Linear context prior + Tx1 context-PCA ridge"
    ),
}
# Controls that exist only when the config names a prior export (paths.prior).
PRIOR_CONTROLS = ("context_prior", "context_prior+context_pca_ridge[tx1]")
POLL_SECONDS = 10.0
STOP_SECONDS = 30.0


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
    """Every visible GPU, or the requested ones after checking they are visible."""
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


def training_argv(config_path: Path, run: Path, processes: int) -> tuple[str, ...]:
    launch = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        PRECISION,
    ]
    if processes > 1:
        launch.append("--multi_gpu")
    return (
        *launch,
        "--module",
        "src.experiments.joint.train",
        "--config",
        str(config_path),
        "--run-dir",
        str(run / "train"),
    )


def check_resume_processes(train: Path, processes: int) -> None:
    """Resuming ``last.pt`` needs the process count it was started with."""
    if not (train / "last.pt").is_file():
        return
    started = read_json(train / "run.json")["world_size"]
    if started != processes:
        raise ValueError(
            f"training in {train} was started on {started} GPU(s) and resumes "
            f"only on {started} (the effective batch depends on it); pass --gpus "
            f"with {started} GPU(s), not {processes}, or use a new --run-id"
        )
