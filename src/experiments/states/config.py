"""The cell-state telemetry's config (protocol §17). Every key explicit, unknown keys
rejected."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.experiments.config import read_yaml, require_keys

_STATES_GROUPS = {
    "fit": "cells_per_line seeds cycle_states mixture_counts mixture_inits components",
    "programs": (
        "ks iterations density_threshold highly_variable top_genes robust_overlap "
        "redundant_overlap cycle_overlap cluster_overlap min_lines"
    ),
    "scvi": "latent hidden layers max_epochs batch_size",
    "project": "roles",
    "inputs": "merge_effective_count permutation_seeds cycling_floor",
    "references": "kinker gavish hallmarks tirosh chemistry",
    "pie": "genes contexts_per_batch",
    "report": "permutations recurrence_mass clr_floor stability_cells",
}
_STATES_TOP_LEVEL = "seed ladder_config output_root"
#: Roles a state vocabulary may be projected onto; fits read train_cells only.
STATE_ROLES = ("train_cells", "val")


def _positive_integers(config: Mapping[str, Any], group: str, keys: str) -> None:
    for key in keys.split():
        value = config[group][key]
        if not (isinstance(value, int) and value > 0):
            raise ValueError(f"{group}.{key} must be a positive integer")


def validate_states_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keys of the state-telemetry config (PIE stage-one design §3.5, Task 8): the
    ladder config whose inputs it reads, the vocabulary fits, the projection
    roles, the state inputs and their controls, the reference tables, the frozen
    PIE's state pass and the report."""
    require_keys(config, {*_STATES_GROUPS, *_STATES_TOP_LEVEL.split()}, "config")
    for name, keys in _STATES_GROUPS.items():
        require_keys(config[name], set(keys.split()), name)
    _positive_integers(
        config, "fit", "cells_per_line cycle_states mixture_inits components"
    )
    _positive_integers(
        config,
        "programs",
        "iterations highly_variable top_genes robust_overlap redundant_overlap "
        "cycle_overlap cluster_overlap min_lines",
    )
    _positive_integers(config, "scvi", "latent hidden layers max_epochs batch_size")
    _positive_integers(config, "pie", "genes contexts_per_batch")
    _positive_integers(config, "report", "permutations stability_cells")
    for key, values in (
        ("fit.seeds", config["fit"]["seeds"]),
        ("fit.mixture_counts", config["fit"]["mixture_counts"]),
        ("programs.ks", config["programs"]["ks"]),
        ("inputs.permutation_seeds", config["inputs"]["permutation_seeds"]),
    ):
        if (
            not values
            or len(set(values)) != len(values)
            or not all(isinstance(v, int) and v >= 0 for v in values)
        ):
            raise ValueError(f"{key} must be distinct non-negative integers")
    if config["fit"]["seeds"][0] != config["seed"]:
        raise ValueError("fit.seeds must start with seed, the fit the PIE pass reads")
    if not all(
        k > 1 for k in [*config["fit"]["mixture_counts"], *config["programs"]["ks"]]
    ):
        raise ValueError("fit.mixture_counts and programs.ks must be integers above 1")
    roles = config["project"]["roles"]
    if (
        not roles
        or roles[0] != "train_cells"
        or set(roles) - set(STATE_ROLES)
        or len(set(roles)) != len(roles)
    ):
        raise ValueError(
            f"project.roles: train_cells first, then others of {STATE_ROLES}"
        )
    for key in ("merge_effective_count", "cycling_floor"):
        if not config["inputs"][key] > 0:
            raise ValueError(f"inputs.{key} must be positive")
    for key in ("recurrence_mass", "clr_floor"):
        if not 0 < config["report"][key] < 1:
            raise ValueError(f"report.{key} must lie in (0, 1)")
    if not 0 < config["programs"]["density_threshold"]:
        raise ValueError("programs.density_threshold must be positive")
    return dict(config)


def load_states_config(path: Path) -> dict[str, Any]:
    """Load and validate a state-telemetry YAML config."""
    return validate_states_config(read_yaml(path))
