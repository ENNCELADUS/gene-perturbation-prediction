"""Atomic file writes every step shares, and plain-JSON conversion.

A write goes to a hidden temporary beside the target, then replaces it, so a reader
never sees a partial file and concurrent writers of one path leave one whole copy.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def replace_with(path: Path, write: Callable[[Path], object]) -> None:
    """``write(temporary)``, then move the temporary onto ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    replace_with(path, lambda temporary: temporary.write_text(text))


def write_json(
    path: Path, payload: Any, *, sort_keys: bool = False, keep_nan: bool = False
) -> None:
    """Indented JSON; NaN becomes null unless ``keep_nan`` writes it as ``NaN``."""
    if keep_nan:
        text = json.dumps(payload, indent=2, sort_keys=sort_keys)
    else:
        text = json.dumps(
            jsonable(payload), indent=2, allow_nan=False, sort_keys=sort_keys
        )
    write_text(path, text + "\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_parquet(path: Path, frame: Any) -> None:
    replace_with(path, frame.to_parquet)


def write_npy(path: Path, array: np.ndarray) -> None:
    def write(temporary: Path) -> None:
        # Through a handle: np.save would append ".npy" to the temporary's name.
        with temporary.open("wb") as handle:
            np.save(handle, array)

    replace_with(path, write)


def write_npz(path: Path, **arrays: Any) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)

    replace_with(path, write)
