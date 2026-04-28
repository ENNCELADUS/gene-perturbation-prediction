"""Distributed-rank helpers for process-local output control."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path


def is_primary_rank() -> bool:
    """Return True for single-process runs or distributed rank zero."""
    rank = _rank_from_env(("RANK", "ACCELERATE_PROCESS_INDEX"))
    if rank is not None:
        return rank == 0
    local_rank = _rank_from_env(
        ("LOCAL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK")
    )
    return local_rank in (None, 0)


def configure_process_output(suppress_stdout: bool = True) -> None:
    """Suppress non-primary rank informational output in distributed runs."""
    suppress_torchtext_deprecation_warning()
    if is_primary_rank():
        return

    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    if suppress_stdout:
        sys.stdout = Path(os.devnull).open("w")


def log_primary_info(logger: logging.Logger, message: str, *args: object) -> None:
    """Log an info message only from primary rank."""
    if is_primary_rank():
        logger.info(message, *args)


def disable_tqdm(config: dict) -> bool:
    """Return whether progress bars should be disabled for this process."""
    return (
        bool(config["run_config"].get("disable_tqdm", False))
        or not is_primary_rank()
    )


def suppress_torchtext_deprecation_warning() -> None:
    """Suppress torchtext's import-time deprecation warning."""
    warnings.filterwarnings(
        "ignore",
        message=r"Torchtext is deprecated.*",
        category=UserWarning,
    )


def _rank_from_env(names: Sequence[str]) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None
