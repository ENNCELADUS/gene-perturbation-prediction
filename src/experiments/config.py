"""Joint GeneEffect configuration: every key explicit, unknown keys rejected."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_GROUPS = {
    "seeds": "train collator projection",
    "train": (
        "max_epochs patience dependency_batch_size response_batch_size "
        "response_interval response_weight state_learning_rate "
        "adapter_learning_rate head_learning_rate weight_decay "
        "objective state_mode warmup_epochs genes_per_block"
    ),
    "comparison": (
        "epochs hidden learning_rate state_learning_rate batch_size shuffles bootstrap"
    ),
    "features": (
        "cells_per_context hvg_dim esm2_dim "
        "variable_gene_min_observations variable_gene_percentile "
        "selective_min_lines selective_max_fraction residual_sd_floor_percentile"
    ),
    "model": (
        "cell_sentence_len esm2_adapter_hidden head_blocks factor_rank "
        "context_components dropout"
    ),
    "preparation": (
        "response_max_cells_per_gene response_total_cells_per_line "
        "response_sampling_seed tx1_batch_size tx1_max_length "
        "var_ensembl_col hvg_gene_symbol_col"
    ),
    "paths": (
        "split gene_effect source_registry tx1_registration cell_line_manifest "
        "tx1_model_dir tx1_cache esm2_embeddings state_checkpoint state_model_dir "
        "perturbseq_sources"
    ),
}
_TOP_LEVEL = "precision output_root prepared_root"
_HEAD_BLOCKS = "use_delta_proj use_s use_q_sc use_e_g use_z_c"
OBJECTIVES = ("huber", "standardized_mse", "pearson_blocks")
STATE_MODES = ("frozen", "trainable")


def _require_keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    missing, unknown = expected - value.keys(), value.keys() - expected
    if missing or unknown:
        raise ValueError(
            f"{name}: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject missing and unknown keys and unknown choices; values as written."""
    _require_keys(config, {*_GROUPS, *_TOP_LEVEL.split()}, "config")
    for name, keys in _GROUPS.items():
        _require_keys(config[name], set(keys.split()), name)
    _require_keys(
        config["model"]["head_blocks"], set(_HEAD_BLOCKS.split()), "model.head_blocks"
    )
    for key, choices in (("objective", OBJECTIVES), ("state_mode", STATE_MODES)):
        if config["train"][key] not in choices:
            raise ValueError(f"train.{key} must be one of {choices}")
    return dict(config)


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML config; relative paths resolve from the repo root."""
    with Path(path).open() as handle:
        return validate_config(yaml.safe_load(handle))


_PRIOR_GROUPS = {
    "paths": "extra_lines reference model bulk_expression",
    "training_side": "exclude_lineages",
    "curve": "sizes subsets selected",
    "selection": "penalties shrinkages selected rank view_weights",
    "prior": "components folds bootstrap_repeats",
}
_PRIOR_TOP_LEVEL = "seed joint_config output_root"


def validate_prior_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject missing and unknown keys of a linear-context-prior config."""
    _require_keys(config, {*_PRIOR_GROUPS, *_PRIOR_TOP_LEVEL.split()}, "config")
    for name, keys in _PRIOR_GROUPS.items():
        _require_keys(config[name], set(keys.split()), name)
    return dict(config)


def load_prior_config(path: Path) -> dict[str, Any]:
    """Load and validate a prior YAML config; paths resolve from the repo root."""
    with Path(path).open() as handle:
        return validate_prior_config(yaml.safe_load(handle))
