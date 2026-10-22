"""What every track's strict config loader shares: every key explicit, unknown keys
rejected. Each track's loader lives in its package (``joint``, ``prior``,
``stage_one``, ``states``)."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def require_keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    missing, unknown = expected - value.keys(), value.keys() - expected
    if missing or unknown:
        raise ValueError(
            f"{name}: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def read_yaml(path: Path) -> Any:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)
