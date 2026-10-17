"""Run directories bound to one config, and the JSON and text writes they share.

Lifted out of ``src.experiments.context_prior`` so the stage-one runs in
``.venv-pie`` bind their run directories the same way; ``context_prior``
re-imports these under their old names.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    """Plain JSON values: numpy scalars become Python numbers, NaN becomes None
    (undefined, never 0) and an infinite float the string ``"inf"``."""
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def write_text(path: Path, text: str) -> None:
    """Atomic, and safe when several processes write the same path."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


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
