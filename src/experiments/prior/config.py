"""The linear context prior's configs: the prior itself, the single-cell statistics
stack and the noise diagnosis. Every key explicit, unknown keys rejected."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.experiments.config import read_yaml, require_keys

_PRIOR_GROUPS = {
    "paths": "reference model bulk_expression",
    "training_side": "exclude_lineages",
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
    require_keys(config, {*_PRIOR_GROUPS, *_PRIOR_TOP_LEVEL.split()}, "config")
    for name, keys in _PRIOR_GROUPS.items():
        require_keys(config[name], set(keys.split()), name)
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
        require_keys(experiment, set(_EXPERIMENT_KEYS.split()), f"experiments.{name}")
        unknown = set(experiment["block_sets"]) - set(config["block_sets"])
        if unknown:
            raise ValueError(
                f"experiments.{name}: unknown block sets {sorted(unknown)}"
            )
    reference = config["reference"]
    require_keys(reference, set(_REFERENCE_KEYS.split()), "reference")
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
    return validate_prior_config(read_yaml(path))


_STACK_TOP_LEVEL = (
    "seed output_root prior_run bootstrap_repeats selected_genes penalties arms"
)
_STACK_PENALTIES = "own selected"
_STACK_STATISTICS = ("mean", "detected", "variance")


def validate_stack_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keys of a single-cell statistics stack config: the prior run whose export
    it stacks on, the penalty grids, and arms naming the statistics they read."""
    require_keys(config, set(_STACK_TOP_LEVEL.split()), "config")
    require_keys(config["penalties"], set(_STACK_PENALTIES.split()), "penalties")
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
    return validate_stack_config(read_yaml(path))


_NOISE_GROUPS = {
    "cell_sampling": "cells",
    "learning_curve": "fractions cilantro_predictions",
    "deposit": "min_lines permutations second_source_root",
    "cell_states": (
        "programs threshold_percentile cluster_components cluster_counts penalties"
    ),
}
_NOISE_TOP_LEVEL = "seed output_root prior_run bootstrap_repeats"


def validate_noise_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keys of a noise-diagnosis config: the prior run it reads, and one group per
    measurement; the shapes of the grids are checked."""
    require_keys(config, {*_NOISE_GROUPS, *_NOISE_TOP_LEVEL.split()}, "config")
    for name, keys in _NOISE_GROUPS.items():
        require_keys(config[name], set(keys.split()), name)
    if not all(isinstance(n, int) and n > 0 for n in config["cell_sampling"]["cells"]):
        raise ValueError("cell_sampling.cells must be positive integers")
    if not all(0 < f < 1 for f in config["learning_curve"]["fractions"]):
        raise ValueError("learning_curve.fractions must lie in (0, 1)")
    cilantro = config["learning_curve"]["cilantro_predictions"]
    if not (isinstance(cilantro, str) and cilantro):
        raise ValueError("learning_curve.cilantro_predictions must be a path")
    states = config["cell_states"]
    if not states["programs"] or len(set(states["programs"])) != len(
        states["programs"]
    ):
        raise ValueError("cell_states.programs must name distinct programs")
    if not 0 < states["threshold_percentile"] < 100:
        raise ValueError("cell_states.threshold_percentile must lie in (0, 100)")
    if not all(isinstance(k, int) and k > 1 for k in states["cluster_counts"]):
        raise ValueError("cell_states.cluster_counts must be integers above 1")
    if not all(0 < p < float("inf") for p in states["penalties"]):
        raise ValueError("cell_states.penalties must be positive and finite")
    return dict(config)


def load_noise_config(path: Path) -> dict[str, Any]:
    """Load and validate a noise-diagnosis YAML config."""
    return validate_noise_config(read_yaml(path))
