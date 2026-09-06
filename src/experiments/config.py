"""Strict configuration for the joint GeneEffect protocol."""

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

import yaml


_GROUPS = {
    "seeds": "train collator projection",
    "train": (
        "max_epochs patience dependency_batch_size response_batch_size "
        "response_interval response_weight state_learning_rate "
        "adapter_learning_rate head_learning_rate weight_decay"
    ),
    "selection": "metric mode",
    "features": (
        "cells_per_context hvg_dim esm2_dim "
        "variable_gene_min_observations variable_gene_percentile"
    ),
    "model": "cell_sentence_len esm2_adapter_hidden head_hidden head_layers",
    "preparation": (
        "response_max_cells_per_gene response_total_cells_per_line "
        "response_sampling_seed response_holdout_fraction "
        "response_holdout_seed tx1_batch_size tx1_max_length "
        "var_ensembl_col hvg_gene_symbol_col"
    ),
    "paths": (
        "split gene_effect source_registry tx1_registration "
        "cell_line_manifest tx1_model_dir tx1_cache q_sc_cache "
        "esm2_embeddings common_gene_panel state_checkpoint "
        "state_model_dir perturbseq_sources response_cache"
    ),
}


def _keys(value, required, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    missing, unknown = set(required) - value.keys(), value.keys() - set(required)
    if missing or unknown:
        raise ValueError(
            f"{name}: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate explicit settings without opening input files or importing Torch."""
    _keys(config, {*_GROUPS, "precision", "output_root", "prepared_root"}, "config")
    for name, fields in _GROUPS.items():
        _keys(config[name], fields.split(), name)
    for name in ("output_root", "prepared_root"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty path")
    for name, value in config["paths"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{name} must be a nonempty path")
    if config["precision"] not in {"bf16", "fp16", "no"}:
        raise ValueError("precision must be bf16, fp16 or no")
    if config["selection"] != {"metric": "val_geneeffect_loss", "mode": "min"}:
        raise ValueError("selection must minimize val_geneeffect_loss")
    for name, value in config["seeds"].items():
        if type(value) is not int or value != 0:
            raise ValueError(f"seeds.{name} must be 0")
    continuous = {
        "response_weight",
        "state_learning_rate",
        "adapter_learning_rate",
        "head_learning_rate",
        "weight_decay",
        "variable_gene_percentile",
        "response_holdout_fraction",
    }
    for group in ("train", "features", "model", "preparation"):
        for name, value in config[group].items():
            if name in {"var_ensembl_col", "hvg_gene_symbol_col"}:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{group}.{name} must be a nonempty string")
                continue
            if name == "response_total_cells_per_line" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{group}.{name} must be a finite number")
            if name not in continuous and type(value) is not int:
                raise ValueError(f"{group}.{name} must be an integer")
            minimum_zero = name in {
                "response_weight",
                "weight_decay",
                "variable_gene_percentile",
                "response_sampling_seed",
                "response_holdout_seed",
            }
            if value < 0 or (value == 0 and not minimum_zero):
                raise ValueError(f"{group}.{name} is outside its numeric domain")
    if config["train"]["response_batch_size"] % 4:
        raise ValueError("train.response_batch_size must be divisible by four")
    if config["train"]["response_interval"] != 4:
        raise ValueError("train.response_interval must be 4")
    if config["features"]["variable_gene_percentile"] > 100:
        raise ValueError("features.variable_gene_percentile must be in [0, 100]")
    if config["features"]["hvg_dim"] != 2000:
        raise ValueError("features.hvg_dim must be 2000")
    for name, expected in {
        "response_sampling_seed": 42,
        "response_holdout_seed": 13,
        "response_holdout_fraction": 0.1,
    }.items():
        if config["preparation"][name] != expected:
            raise ValueError(f"preparation.{name} must be {expected}")
    return dict(config)


def load_config(path: Path) -> dict[str, Any]:
    """Load an explicit YAML config; relative data paths use repository cwd."""
    with Path(path).open() as handle:
        return validate_config(yaml.safe_load(handle))
