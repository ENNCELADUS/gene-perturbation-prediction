"""Filesystem coordination primitives for the exp08 fold work-queue.

Replaces the single end-of-run NCCL all-gather collective. Every rank
walks the same job list and uses these primitives to claim, run, and record
``(split_type, fold_id)`` jobs through the filesystem only — no
``torch.distributed`` collective is involved (Guard G1).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from sl_dl_model.config import SLDLConfig


def run_token() -> str:
    """Return a per-run token used to scope intra-run claim markers.

    Claims are *per-run* (design decision C): a hard-crashed worker leaves an
    orphan claim dir, and the next run must ignore it. Keying the ``.claims``
    subdir by this token guarantees a fresh run never collides with a prior
    run's claims, so the only cross-run state is the ``.result.json`` files.

    Resolution order:

    1. ``SLURM_JOB_ID`` — distinct per cluster job (the production case).
    2. ``SL_DL_RUN_ID`` — explicit override (set by a launcher to share one
       token across ranks of the same local/CLI run).
    3. ``local-<parent-pid>`` — fallback for a bare local run; the parent pid
       is stable across the ranks an ``accelerate launch`` spawns.

    Returns:
        A non-empty token string.
    """
    slurm = os.environ.get("SLURM_JOB_ID")
    if slurm:
        return slurm
    explicit = os.environ.get("SL_DL_RUN_ID")
    if explicit:
        return explicit
    return f"local-{os.getppid()}"


def fold_results_dir(config: SLDLConfig) -> Path:
    """Return the per-run fold-results directory under ``output_dir``."""
    return Path(config.output_dir) / config.fold_results_subdir


def result_path(results_dir: Path, split: str, fold: int) -> Path:
    """Return the success-result JSON path for one job."""
    return results_dir / f"{split}_fold{fold}.result.json"


def failed_path(results_dir: Path, split: str, fold: int) -> Path:
    """Return the quarantine-marker path for one job."""
    return results_dir / f"{split}_fold{fold}.failed"


def claim_path(
    results_dir: Path,
    split: str,
    fold: int,
    run_token: str | None = None,
    *,
    _default_token=run_token,
) -> Path:
    """Return the atomic-claim directory path for one job, scoped by run token.

    Args:
        results_dir: The fold-results directory.
        split: CV split type.
        fold: Fold id.
        run_token: Per-run token; defaults to :func:`run_token`. Claims live
            under ``.claims/<run_token>/`` so a prior run's orphan claims never
            block a resume.
    """
    token = run_token if run_token is not None else _default_token()
    return results_dir / ".claims" / token / f"{split}_fold{fold}"


def atomic_write_json(path: Path, obj: object) -> None:
    """Write ``obj`` as JSON atomically (temp file in the same dir + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def read_json(path: Path) -> object:
    """Read and parse a JSON file written by :func:`atomic_write_json`."""
    return json.loads(Path(path).read_text())


def try_claim(
    results_dir: Path, split: str, fold: int, run_token: str | None = None
) -> bool:
    """Atomically claim one job. Return ``True`` if this caller won the claim.

    Uses ``os.mkdir`` (POSIX/Lustre-atomic). A returned ``False`` means another
    rank in the *same run* already owns the job. Claims are scoped by
    ``run_token`` (see :func:`run_token`), so a prior run's orphan claim never
    blocks a resume.
    """
    claim = claim_path(results_dir, split, fold, run_token)
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(claim)
        return True
    except FileExistsError:
        return False


def is_done(results_dir: Path, split: str, fold: int) -> bool:
    """Return ``True`` if this job already has a success-result file."""
    return result_path(results_dir, split, fold).exists()
