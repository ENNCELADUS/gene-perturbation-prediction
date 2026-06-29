"""Shared queue utilities for exp08b pass runners."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sl_dl_model import fold_queue as fq
from sl_dl_model.exp08b_config import Exp08bConfig


def jobs(frame: pd.DataFrame, config: Exp08bConfig) -> list[tuple[str, int]]:
    """Return ordered split/fold jobs present in the benchmark frame."""
    requested = config.split_types or ("CV1", "CV2", "CV3")
    available = set(frame["split_type"].unique())
    split_types = [split for split in requested if split in available]
    return [(split, fold) for split in split_types for fold in config.folds]


def raise_if_step_incomplete(
    results_dir: Path,
    job_list: list[tuple[str, int]],
    fingerprint: str,
    step: str,
) -> None:
    """Raise when any pass job lacks a matching success marker."""
    failed: list[tuple[str, int, Path, str | None]] = []
    missing: list[tuple[str, int]] = []
    for split_type, fold_id in job_list:
        if fq.is_done(results_dir, split_type, fold_id, fingerprint=fingerprint):
            continue
        if fq.is_failed(results_dir, split_type, fold_id, fingerprint=fingerprint):
            path = fq.failed_path(results_dir, split_type, fold_id)
            payload = fq.read_json(path)
            trace = payload.get("traceback") if isinstance(payload, dict) else None
            failed.append((split_type, fold_id, path, trace))
        else:
            missing.append((split_type, fold_id))

    if not failed and not missing:
        return

    lines = [f"{step} incomplete."]
    if failed:
        lines.append(f"failed jobs: {[(s, f) for s, f, _p, _t in failed]}")
        for split_type, fold_id, path, trace in failed:
            lines.append(f"- {split_type}/fold{fold_id}: {path}")
            if trace:
                lines.append(str(trace))
    if missing:
        lines.append(f"missing jobs: {missing}")
    raise RuntimeError("\n".join(lines))
