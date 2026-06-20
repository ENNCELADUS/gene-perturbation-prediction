"""Filesystem coordination primitives for the exp08 fold work-queue.

Replaces the single end-of-run NCCL ``gather_object`` collective. Every rank
walks the same job list and uses these primitives to claim, run, and record
``(split_type, fold_id)`` jobs through the filesystem only — no
``torch.distributed`` collective is involved (Guard G1).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from sl_dl_model.config import SLDLConfig


def fold_results_dir(config: SLDLConfig) -> Path:
    """Return the per-run fold-results directory under ``output_dir``."""
    return Path(config.output_dir) / config.fold_results_subdir


def result_path(results_dir: Path, split: str, fold: int) -> Path:
    """Return the success-result JSON path for one job."""
    return results_dir / f"{split}_fold{fold}.result.json"


def failed_path(results_dir: Path, split: str, fold: int) -> Path:
    """Return the quarantine-marker path for one job."""
    return results_dir / f"{split}_fold{fold}.failed"


def claim_path(results_dir: Path, split: str, fold: int) -> Path:
    """Return the atomic-claim directory path for one job."""
    return results_dir / ".claims" / f"{split}_fold{fold}"


def atomic_write_json(path: Path, obj: object) -> None:
    """Write ``obj`` as JSON atomically (temp file in the same dir + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def read_json(path: Path) -> object:
    """Read and parse a JSON file written by :func:`atomic_write_json`."""
    return json.loads(Path(path).read_text())


def try_claim(results_dir: Path, split: str, fold: int) -> bool:
    """Atomically claim one job. Return ``True`` if this caller won the claim.

    Uses ``os.mkdir`` (POSIX/Lustre-atomic). A returned ``False`` means another
    rank already owns the job in this run.
    """
    claim = claim_path(results_dir, split, fold)
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(claim)
        return True
    except FileExistsError:
        return False


def is_done(results_dir: Path, split: str, fold: int) -> bool:
    """Return ``True`` if this job already has a success-result file."""
    return result_path(results_dir, split, fold).exists()
