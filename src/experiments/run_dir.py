"""Run directories bound to one config, shared by every experiment command (the
stage-one runs in ``.venv-pie`` included)."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.data.io import jsonable, read_json, write_json


@contextmanager
def locked(run_dir: Path) -> Iterator[None]:
    """Serialises the run directory's shared files across processes: the config
    check and its creation, and a results scan with its replacement (so an older
    scan never overwrites a newer one)."""
    with open(run_dir / ".lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def assert_bound(run_dir: Path, config: Mapping[str, Any]) -> None:
    """Read-only: raise unless ``run_dir`` was bound to ``config`` (before its
    results are skipped over or read)."""
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} has no run_config.json; it is not a run")
    if read_json(path) != jsonable({"config": config}):
        raise ValueError(
            f"{run_dir} was written under another config ({path}); move it aside"
        )


def bind(run_dir: Path, config: Mapping[str, Any]) -> None:
    """A run directory belongs to one config: finished steps are skipped by file
    existence, so resuming under another would mix experiments."""
    record = jsonable({"config": config})
    path = run_dir / "run_config.json"
    with locked(run_dir):
        if path.is_file():
            if read_json(path) != record:
                raise ValueError(
                    f"{run_dir} was started with a different config ({path}); "
                    "use a new --run-id"
                )
            return
        write_json(path, record)
