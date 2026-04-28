"""Prepare Norman condition split artifacts for scGPT."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.utils.data import (
    infer_gene_heldout_condition_split,
    get_condition_splits,
    load_adata,
    save_condition_split,
)


def run(config: dict) -> dict:
    """Generate a condition-level split artifact when config does not define one."""
    data_config = config["data_config"]
    split_path = _split_path(config)
    inline_split = data_config.get("condition_split", {})
    if isinstance(inline_split, Mapping) and any(inline_split.values()):
        split = get_condition_splits(config)
    else:
        adata = load_adata(data_config["h5ad_path"])
        split_config = data_config.get("split_config", {})
        if not isinstance(split_config, Mapping):
            raise ValueError("data_config.split_config must be a mapping")
        strategy = split_config.get("strategy", "gene_heldout")
        if strategy != "gene_heldout":
            raise ValueError("Only gene_heldout split strategy is supported")
        split = infer_gene_heldout_condition_split(
            adata=adata,
            condition_key=str(data_config.get("condition_key", "condition")),
            train_gene_fraction=float(split_config.get("train_gene_fraction", 0.7)),
            validation_gene_fraction=float(
                split_config.get("validation_gene_fraction", 0.1)
            ),
            test_gene_fraction=float(split_config.get("test_gene_fraction", 0.2)),
            min_cells_per_condition=int(split_config.get("min_cells_per_condition", 1)),
        )
    save_condition_split(split, split_path)
    conditions = split["conditions"] if "conditions" in split else split
    raw_stats = split.get("stats", {})
    stats = raw_stats if isinstance(raw_stats, Mapping) else {}
    return {
        "split_path": str(split_path),
        "n_train": len(conditions["train"]),
        "n_validation": len(conditions["validation"]),
        "n_test": len(conditions["test"]),
        **stats,
    }


def _split_path(config: Mapping[str, object]) -> Path:
    data_config = config.get("data_config", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("data_config must be a mapping")
    path = data_config.get("condition_split_path")
    if path:
        return Path(str(path))
    run_config = config.get("run_config", {})
    study_name = "norman"
    if isinstance(run_config, Mapping) and run_config.get("study_name"):
        study_name = str(run_config["study_name"])
    h5ad_path = data_config.get("h5ad_path")
    if h5ad_path:
        return Path(str(h5ad_path)).with_name(f"{study_name}_condition_split.yaml")
    return Path("data") / study_name / f"{study_name}_condition_split.yaml"
