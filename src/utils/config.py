"""Configuration helpers for config-driven model pipelines."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import yaml

ALLOWED_MODELS = {"pca_knn", "random_forest", "scgpt"}
ALLOWED_STAGES = {"prepare", "train", "evaluate"}
REQUIRED_SECTIONS = {
    "run_config",
    "device_config",
    "data_config",
    "model_config",
}


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file."""
    with Path(config_path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def validate_config(config: Mapping[str, object]) -> None:
    """Validate the shared config contract."""
    missing = sorted(REQUIRED_SECTIONS - set(config))
    if missing:
        raise ValueError(f"Missing required config section(s): {', '.join(missing)}")

    model_config = _require_mapping(config, "model_config")
    model_name = model_config.get("model")
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            "model_config.model must be one of "
            f"{sorted(ALLOWED_MODELS)}, got {model_name!r}"
        )

    run_config = _require_mapping(config, "run_config")
    stages = run_config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("run_config.stages must be a non-empty list")
    invalid_stages = [stage for stage in stages if stage not in ALLOWED_STAGES]
    if invalid_stages:
        raise ValueError(
            "run_config.stages contains unsupported stage(s): "
            f"{', '.join(invalid_stages)}"
        )


def set_seed(seed: int | None) -> None:
    """Set Python, NumPy, and Torch seeds when provided."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_value: str | None) -> Path | None:
    """Return a Path for non-empty config path values."""
    if path_value is None or path_value == "":
        return None
    return Path(path_value)


def _require_mapping(
    config: Mapping[str, object], section: str
) -> Mapping[str, object]:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
    return value
