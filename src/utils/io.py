"""
I/O utilities for saving/loading configs, checkpoints, and results.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        path: Config file path

    Returns:
        Configuration dictionary
    """
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(
    obj: Any,
    path: Union[str, Path],
    format: str = "pickle",
) -> None:
    """
    Save checkpoint/model to file.

    Args:
        obj: Object to save
        path: Output path
        format: 'pickle' or 'json'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "pickle":
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    elif format == "json":
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_checkpoint(
    path: Union[str, Path],
    format: str = "pickle",
) -> Any:
    """
    Load checkpoint from file.

    Args:
        path: Checkpoint path
        format: 'pickle' or 'json'

    Returns:
        Loaded object
    """
    if format == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif format == "json":
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")
