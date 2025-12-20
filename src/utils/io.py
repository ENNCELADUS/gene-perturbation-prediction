"""I/O utilities for saving/loading configs."""

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
