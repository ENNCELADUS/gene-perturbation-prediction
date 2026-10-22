"""What every experiment command does with its run directory: bind it to one config,
lock its shared files, record finished steps and status, and format report numbers
(the stage-one commands in ``.venv-pie`` included)."""

from __future__ import annotations

import argparse
import fcntl
import math
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.io import jsonable, read_json, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]


def absolute(path: str | Path) -> Path:
    """A config path: relative paths resolve from the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def shard(value: str) -> tuple[int, int]:
    """``--shard I/N``: every N-th item from the I-th."""
    index, count = (int(part) for part in value.split("/"))
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError(f"shard {value}: need 0 <= I < N")
    return index, count


def number(value: Any, digits: int = 4) -> str:
    """Undefined (None or NaN) is written as such, never as 0."""
    if value is None or not math.isfinite(float(value)):
        return "undefined"
    return f"{float(value):.{digits}f}"


def interval(entry: Mapping[str, Any] | None) -> str:
    """A difference with its bootstrap interval, ``x [low, high]``."""
    if entry is None:
        return ""
    low, high = entry["interval"]
    return f"{number(entry['difference'])} [{number(low)}, {number(high)}]"


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


def set_status(run_dir: Path, status_key: str, status: str, **details) -> None:
    """One step's status in ``run.json``, beside the others."""
    path = run_dir / "run.json"
    record = read_json(path) if path.exists() else {}
    record[status_key] = {"status": status, **details}
    write_json(path, record)


def open_manifest(
    path: Path,
    settings: Mapping[str, Any],
    *,
    fresh: Mapping[str, Any] | None = None,
    keep_nan: bool = False,
) -> dict[str, Any]:
    """A root's manifest, written now (with ``fresh`` fields) if new; refuses a root
    written under other settings."""
    if path.is_file():
        manifest = read_json(path)
        if manifest["settings"] != settings:
            raise ValueError(
                f"{path.parent} was written under other settings; choose a new root "
                "or move it aside"
            )
        return manifest
    manifest = {"settings": dict(settings), **(fresh or {}), "steps": {}}
    write_json(path, manifest, keep_nan=keep_nan)
    return manifest


def record_step(
    path: Path, step: str, payload: Mapping[str, Any], *, keep_nan: bool = False
) -> None:
    """Add a finished step to a manifest, under the root's lock so concurrent shards
    keep each other's records."""
    with locked(path.parent):
        manifest = read_json(path) if path.is_file() else {"steps": {}}
        manifest["steps"][step] = {**payload, "finished": now()}
        write_json(path, manifest, keep_nan=keep_nan)
