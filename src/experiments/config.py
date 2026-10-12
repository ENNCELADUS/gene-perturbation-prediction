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
        "objective state_mode warmup_epochs genes_per_block single_cell_extras"
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
        "split gene_effect basal_raw_umi extra_single_cell_lines tx1_registration "
        "cell_line_manifest "
        "tx1_model_dir tx1_cache esm2_embeddings state_checkpoint state_model_dir "
        "perturbseq_sources prior"
    ),
}
_TOP_LEVEL = "precision output_root prepared_root"
_HEAD_BLOCKS = "use_delta_proj use_s use_q_sc use_e_g use_z_c"
OBJECTIVES = (
    "huber",
    "standardized_mse",
    "pearson_blocks",
    "line_ranking",
    "dependency_classification",
)
STATE_MODES = ("frozen", "trainable")


def _require_bool(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")


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
    _require_bool(config["train"]["single_cell_extras"], "train.single_cell_extras")
    return dict(config)


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML config; relative paths resolve from the repo root."""
    with Path(path).open() as handle:
        return validate_config(yaml.safe_load(handle))


_PRIOR_GROUPS = {
    "paths": "extra_lines reference model bulk_expression",
    "training_side": "exclude_lineages single_cell_extras",
    "prior": "components folds bootstrap_repeats selected_genes",
    "penalties": "components gene",
}
_PRIOR_TOP_LEVEL = "seed joint_config output_root reference block_sets experiments"
_EXPERIMENT_KEYS = "kind settings block_sets"
#: The pinned row every gain is measured against: the experiment's first setting.
_REFERENCE_KEYS = "experiment block_set components_penalty gene_penalty"
GENE_BLOCKS = ("own_expression", "partners", "data_selected")


def validate_prior_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keys of a linear-context-prior experiment config; experiments and block
    sets are named by the config, their shape is checked."""
    _require_keys(config, {*_PRIOR_GROUPS, *_PRIOR_TOP_LEVEL.split()}, "config")
    for name, keys in _PRIOR_GROUPS.items():
        _require_keys(config[name], set(keys.split()), name)
    side = config["training_side"]
    _require_bool(side["single_cell_extras"], "training_side.single_cell_extras")
    for name, blocks in config["block_sets"].items():
        if (
            not blocks
            or blocks[0] != "expression_components"
            or any(b not in GENE_BLOCKS for b in blocks[1:])
        ):
            raise ValueError(
                f"block set {name}: expression_components first, then gene-level blocks"
            )
    for name, experiment in config["experiments"].items():
        _require_keys(experiment, set(_EXPERIMENT_KEYS.split()), f"experiments.{name}")
        unknown = set(experiment["block_sets"]) - set(config["block_sets"])
        if unknown:
            raise ValueError(
                f"experiments.{name}: unknown block sets {sorted(unknown)}"
            )
    reference = config["reference"]
    _require_keys(reference, set(_REFERENCE_KEYS.split()), "reference")
    if reference["experiment"] not in config["experiments"]:
        raise ValueError("reference must name an experiment")
    if reference["block_set"] not in config["block_sets"]:
        raise ValueError("reference must name a block set")
    components_only = config["block_sets"][reference["block_set"]] == [
        "expression_components"
    ]
    if components_only != (reference["gene_penalty"] is None):
        raise ValueError(
            "reference gene_penalty: null for expression components alone, "
            "a number otherwise"
        )
    return dict(config)


def load_prior_config(path: Path) -> dict[str, Any]:
    """Load and validate a prior YAML config; paths resolve from the repo root."""
    with Path(path).open() as handle:
        return validate_prior_config(yaml.safe_load(handle))


_STACK_TOP_LEVEL = (
    "seed output_root prior_run bootstrap_repeats selected_genes penalties arms"
)
_STACK_PENALTIES = "own selected"
_STACK_STATISTICS = ("mean", "detected", "variance")


def validate_stack_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keys of a single-cell statistics stack config: the prior run whose export
    it stacks on, the penalty grids, and arms naming the statistics they read."""
    _require_keys(config, set(_STACK_TOP_LEVEL.split()), "config")
    _require_keys(config["penalties"], set(_STACK_PENALTIES.split()), "penalties")
    if not all(p > 0 for p in config["penalties"]["own"]):
        raise ValueError("penalties.own must be positive (.inf: pooled weights only)")
    if not all(0 < p < float("inf") for p in config["penalties"]["selected"]):
        raise ValueError("penalties.selected must be positive and finite")
    if not config["arms"]:
        raise ValueError("arms must name at least one arm")
    for name, chosen in config["arms"].items():
        if (
            not chosen
            or set(chosen) - set(_STACK_STATISTICS)
            or len(set(chosen)) != len(chosen)
        ):
            raise ValueError(
                f"arms.{name}: distinct statistics from {_STACK_STATISTICS}, "
                f"got {chosen}"
            )
    return dict(config)


def load_stack_config(path: Path) -> dict[str, Any]:
    """Load and validate a single-cell statistics stack YAML config."""
    with Path(path).open() as handle:
        return validate_stack_config(yaml.safe_load(handle))
