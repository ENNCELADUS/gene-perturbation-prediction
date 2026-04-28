"""Prepare Norman condition split artifacts for scGPT."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.utils.data import (
    get_condition_splits,
    infer_condition_splits,
    load_adata,
    save_condition_split,
)


def run(config: dict) -> dict:
    """Generate a condition-level split artifact when config does not define one."""
    data_config = config["data_config"]
    split_path = _split_path(data_config)
    inline_split = data_config.get("condition_split", {})
    if isinstance(inline_split, Mapping) and any(inline_split.values()):
        split = get_condition_splits(config)
    else:
        adata = load_adata(data_config["h5ad_path"])
        split_config = data_config.get("split_config", {})
        if not isinstance(split_config, Mapping):
            raise ValueError("data_config.split_config must be a mapping")
        strategy = split_config.get("strategy", "random_condition")
        if strategy != "random_condition":
            raise ValueError("Only random_condition split strategy is supported")
        split = infer_condition_splits(
            adata=adata,
            condition_key=str(data_config.get("condition_key", "condition")),
            seed=int(config["run_config"].get("seed", 42)),
            train_fraction=float(split_config.get("train_fraction", 0.8)),
            validation_fraction=float(split_config.get("validation_fraction", 0.1)),
            test_fraction=float(split_config.get("test_fraction", 0.1)),
        )
    save_condition_split(split, split_path)
    return {
        "split_path": str(split_path),
        "n_train": len(split["train"]),
        "n_validation": len(split["validation"]),
        "n_test": len(split["test"]),
    }


def _split_path(data_config: Mapping[str, object]) -> Path:
    path = data_config.get("condition_split_path")
    if path:
        return Path(str(path))
    return Path("results/scgpt/norman_condition_split.yaml")
