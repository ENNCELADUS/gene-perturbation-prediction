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
        "split gene_effect basal_raw_umi tx1_registration "
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
    _require_keys(config, {*_PRIOR_GROUPS, *_PRIOR_TOP_LEVEL.split()}, "config")
    for name, keys in _PRIOR_GROUPS.items():
        _require_keys(config[name], set(keys.split()), name)
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
    _require_keys(config, {*_NOISE_GROUPS, *_NOISE_TOP_LEVEL.split()}, "config")
    for name, keys in _NOISE_GROUPS.items():
        _require_keys(config[name], set(keys.split()), name)
    if not all(isinstance(n, int) and n > 0 for n in config["cell_sampling"]["cells"]):
        raise ValueError("cell_sampling.cells must be positive integers")
    if not all(0 < f < 1 for f in config["learning_curve"]["fractions"]):
        raise ValueError("learning_curve.fractions must lie in (0, 1)")
    cilantro = config["learning_curve"]["cilantro_predictions"]
    if cilantro is not None and not (isinstance(cilantro, str) and cilantro):
        raise ValueError("learning_curve.cilantro_predictions must be a path or null")
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
    with Path(path).open() as handle:
        return validate_noise_config(yaml.safe_load(handle))


_LADDER_GROUPS = {
    "paths": "gene_text pie_checkpoint context_text tx1_cache",
    "frame": "folds",
    "pie": "pilot_genes components lines_per_batch",
    "tx1": "batch_size max_length",
}
_LADDER_TOP_LEVEL = "seed prior_config noise_config ladder_root"


def validate_ladder_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keys of the stage-one inputs config: the default prior's config (split,
    labels, bulk RNA, lineage drop), the noise diagnosis's (programs, clusters),
    the ladder root, and the frozen PIE and Tx1 settings."""
    _require_keys(config, {*_LADDER_GROUPS, *_LADDER_TOP_LEVEL.split()}, "config")
    for name, keys in _LADDER_GROUPS.items():
        _require_keys(config[name], set(keys.split()), name)
    if not (isinstance(config["frame"]["folds"], int) and config["frame"]["folds"] > 1):
        raise ValueError("frame.folds must be an integer above 1")
    for key in ("pilot_genes", "components", "lines_per_batch"):
        value = config["pie"][key]
        if not (isinstance(value, int) and value > 0):
            raise ValueError(f"pie.{key} must be a positive integer")
    return dict(config)


def load_ladder_config(path: Path) -> dict[str, Any]:
    """Load and validate a stage-one inputs YAML config."""
    with Path(path).open() as handle:
        return validate_ladder_config(yaml.safe_load(handle))


# --- the stage-one model (PIE stage-one design §3.3, §4; protocol §16) -------------

STAGE_ONE_KINDS = ("train", "prior_reference", "report")
STAGE_ONE_ARMS = ("film", "multi_encoder", "direct_readout", "hybrid")
#: Arms with the direct readout's trunk.
STAGE_ONE_TRUNK_ARMS = ("direct_readout", "hybrid")
#: The pair input before the FiLM layer: a frozen-PIE delta, a learned constant
#: (``none``), a line-level reference input, or the prior's prediction alone
#: (``prior``).
#: ``dE_full``: the profile delta on the full decode panel (``pie/full/``), kept
#: for the selective genes only.
STAGE_ONE_DELTAS = ("dH", "dE", "dL", "dE_full")
STAGE_ONE_PROFILES = ("dE", "dE_full")
#: The genes a run trains on and predicts: the decode panel, or the selective
#: genes in it with a full-panel profile delta (the matched profile rows).
STAGE_ONE_GENE_SETS = ("panel", "selective_full")
STAGE_ONE_INPUTS = (
    *STAGE_ONE_DELTAS,
    "none",
    "components",
    "lineage",
    "text",
    "own_x",
    "prior",
)
STAGE_ONE_OWN_SLOTS = ("own_h", "own_dp", "own_x")
STAGE_ONE_GENE_SOURCES = (
    "gene_text",
    "esmc_600m",
    "esm3_open",
    "string_space",
    "gene2vec",
)
_STAGE_ONE_TOP_LEVEL = {
    "train": "kind task seed output_root data model train eval",
    "prior_reference": "kind task seed output_root ladder_config",
    "report": "kind task output_root ladder_config bootstrap_repeats rows contrasts",
}
_STAGE_ONE_GROUPS = {
    "data": (
        "ladder_config gene_sources centre_gene_sources input own_slots permuted "
        "prior_run genes"
    ),
    "model": "arm gene_width hidden state_width context trunk",
    "train": (
        "max_epochs patience warmup_epochs learning_rate weight_decay ema_decay "
        "dependency_weight lines_per_step genes_per_step token_dropout evidence_dropout"
    ),
    "eval": "lines_per_chunk genes_per_chunk",
}
_STAGE_ONE_CONTEXT = "width queries"
_STAGE_ONE_TRUNK = (
    "d_model n_latents n_encoder_layers n_processor_layers n_decoder_layers "
    "num_heads ff_mult dropout attention_chunk warm_start"
)
_STAGE_ONE_ROWS = {
    "base": "name kind",
    "prior": "name kind run",
    "dh_ridge": "name kind penalty",
    "run": "name kind config",
}


def _positive_int(value: Any, name: str) -> None:
    if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
        raise ValueError(f"{name} must be a positive integer")


def _share(value: Any, name: str) -> None:
    if not (isinstance(value, (int, float)) and 0 <= value < 1):
        raise ValueError(f"{name} must lie in [0, 1)")


def _validate_stage_one_data(data: Mapping[str, Any]) -> None:
    sources = data["gene_sources"]
    if (
        not sources
        or set(sources) - set(STAGE_ONE_GENE_SOURCES)
        or len(set(sources)) != len(sources)
    ):
        raise ValueError(
            f"data.gene_sources: distinct names from {STAGE_ONE_GENE_SOURCES}"
        )
    if data["centre_gene_sources"] not in (True, False):
        raise ValueError("data.centre_gene_sources must be a boolean")
    if data["input"] not in STAGE_ONE_INPUTS:
        raise ValueError(f"data.input must be one of {STAGE_ONE_INPUTS}")
    slots = data["own_slots"]
    if set(slots) - set(STAGE_ONE_OWN_SLOTS) or len(set(slots)) != len(slots):
        raise ValueError(f"data.own_slots: distinct names from {STAGE_ONE_OWN_SLOTS}")
    delta = data["input"] in STAGE_ONE_DELTAS
    if slots and not delta:
        raise ValueError("data.own_slots go beside a frozen-PIE delta only")
    if data["permuted"] not in (True, False) or (data["permuted"] and not delta):
        raise ValueError("data.permuted: a boolean, true only with a frozen-PIE delta")
    if data["genes"] not in STAGE_ONE_GENE_SETS:
        raise ValueError(f"data.genes must be one of {STAGE_ONE_GENE_SETS}")
    if data["input"] == "dE_full" and data["genes"] != "selective_full":
        raise ValueError("data.input dE_full exists for data.genes selective_full only")
    # prior_run: the components input reads its component scores; every other
    # input reads its prediction as a pair feature (``prior`` alone, or beside a
    # delta).
    if data["input"] in ("components", "prior") and not isinstance(
        data["prior_run"], str
    ):
        raise ValueError(f"data.prior_run: required for input {data['input']}")
    if data["prior_run"] is not None and data["input"] not in (
        "components",
        "prior",
        *STAGE_ONE_DELTAS,
    ):
        raise ValueError(
            "data.prior_run: for components, prior or a frozen-PIE delta, else null"
        )


def _validate_stage_one_model(
    model: Mapping[str, Any], data: Mapping[str, Any]
) -> None:
    arm = model["arm"]
    if arm not in STAGE_ONE_ARMS:
        raise ValueError(f"model.arm must be one of {STAGE_ONE_ARMS}")
    if (data["input"] in STAGE_ONE_PROFILES) != (model["state_width"] is not None):
        raise ValueError(
            "model.state_width: the profile delta's state encoder width, else null"
        )
    if model["state_width"] is not None:
        _positive_int(model["state_width"], "model.state_width")
    _positive_int(model["gene_width"], "model.gene_width")
    _positive_int(model["hidden"], "model.hidden")
    if arm == "film" and data["input"] == "none":
        raise ValueError(
            "a film arm needs a pair input; the learned constant is the multi-encoder's"
        )
    if arm == "multi_encoder" and data["input"] not in (*STAGE_ONE_DELTAS, "none"):
        raise ValueError(
            "a multi-encoder reads a frozen-PIE delta or the learned constant"
        )
    if arm == "direct_readout" and (
        data["input"] != "none" or data["own_slots"] or data["permuted"]
    ):
        raise ValueError(
            "a direct readout reads the line's tokens: input none, no own slots"
        )
    if (arm == "multi_encoder") != (model["context"] is not None):
        raise ValueError(
            "model.context: the multi-encoder's context encoder, else null"
        )
    if model["context"] is not None:
        _require_keys(
            model["context"], set(_STAGE_ONE_CONTEXT.split()), "model.context"
        )
        for key in _STAGE_ONE_CONTEXT.split():
            _positive_int(model["context"][key], f"model.context.{key}")
    if arm == "hybrid" and data["input"] not in STAGE_ONE_DELTAS:
        raise ValueError("a hybrid joins the direct readout with a frozen-PIE delta")
    if (arm in STAGE_ONE_TRUNK_ARMS) != (model["trunk"] is not None):
        raise ValueError("model.trunk: the direct readout's trunk, else null")
    if model["trunk"] is not None:
        trunk = model["trunk"]
        _require_keys(trunk, set(_STAGE_ONE_TRUNK.split()), "model.trunk")
        for key in _STAGE_ONE_TRUNK.split():
            if key not in ("dropout", "warm_start", "n_processor_layers"):
                _positive_int(trunk[key], f"model.trunk.{key}")
        processor = trunk["n_processor_layers"]
        if not (isinstance(processor, int) and processor >= 0):
            raise ValueError(
                "model.trunk.n_processor_layers must be a non-negative integer"
            )
        _share(trunk["dropout"], "model.trunk.dropout")
        if model["gene_width"] != trunk["d_model"]:
            raise ValueError("model.gene_width must equal model.trunk.d_model")
        if trunk["warm_start"] is not None and not isinstance(trunk["warm_start"], str):
            raise ValueError("model.trunk.warm_start: a PIE checkpoint path or null")


def _validate_stage_one_train(config: Mapping[str, Any]) -> None:
    for name, keys in _STAGE_ONE_GROUPS.items():
        _require_keys(config[name], set(keys.split()), name)
    _validate_stage_one_data(config["data"])
    _validate_stage_one_model(config["model"], config["data"])
    train = config["train"]
    for key in ("max_epochs", "patience", "lines_per_step", "genes_per_step"):
        _positive_int(train[key], f"train.{key}")
    warmup = train["warmup_epochs"]
    if not (isinstance(warmup, int) and 0 <= warmup < train["max_epochs"]):
        raise ValueError("train.warmup_epochs must be an integer in [0, max_epochs)")
    if not (train["learning_rate"] > 0 and train["weight_decay"] >= 0):
        raise ValueError(
            "train.learning_rate must be positive, weight_decay non-negative"
        )
    if not train["dependency_weight"] >= 0:
        raise ValueError("train.dependency_weight must be non-negative")
    if train["ema_decay"] is not None and not 0 < train["ema_decay"] < 1:
        raise ValueError("train.ema_decay must lie in (0, 1), or be null for no EMA")
    for key in ("token_dropout", "evidence_dropout"):
        _share(train[key], f"train.{key}")
        if train[key] > 0 and config["model"]["arm"] not in STAGE_ONE_TRUNK_ARMS:
            raise ValueError(f"train.{key} applies to the direct readout only")
    for key in ("lines_per_chunk", "genes_per_chunk"):
        _positive_int(config["eval"][key], f"eval.{key}")


def _validate_stage_one_report(config: Mapping[str, Any]) -> None:
    _positive_int(config["bootstrap_repeats"], "bootstrap_repeats")
    names = []
    for index, row in enumerate(config["rows"]):
        if not isinstance(row, Mapping) or row.get("kind") not in _STAGE_ONE_ROWS:
            raise ValueError(
                f"rows[{index}].kind must be one of {tuple(_STAGE_ONE_ROWS)}"
            )
        _require_keys(row, set(_STAGE_ONE_ROWS[row["kind"]].split()), f"rows[{index}]")
        if row["kind"] == "dh_ridge" and not row["penalty"] > 0:
            raise ValueError(f"rows[{index}].penalty must be positive")
        names.append(row["name"])
    if len(set(names)) != len(names):
        raise ValueError("rows must have distinct names")
    for pair in config["contrasts"]:
        if len(pair) != 2 or set(pair) - set(names) or pair[0] == pair[1]:
            raise ValueError(f"contrast {pair}: two distinct row names")


def validate_stage_one_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keys of a stage-one config: a training run (``kind: train``), the default
    prior refitted on the frame (``prior_reference``) or a report over runs
    (``report``); each kind's groups and choices are checked."""
    if not isinstance(config, Mapping) or config.get("kind") not in STAGE_ONE_KINDS:
        raise ValueError(f"kind must be one of {STAGE_ONE_KINDS}")
    _require_keys(config, set(_STAGE_ONE_TOP_LEVEL[config["kind"]].split()), "config")
    if not (isinstance(config["task"], str) and config["task"]):
        raise ValueError("task must name the stage (the run directory's parent)")
    if config["kind"] != "report":
        seed = config["seed"]
        if not (isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0):
            raise ValueError("seed must be a non-negative integer")
    if config["kind"] == "train":
        _validate_stage_one_train(config)
    elif config["kind"] == "report":
        _validate_stage_one_report(config)
    return dict(config)


def load_stage_one_config(path: Path) -> dict[str, Any]:
    """Load and validate a stage-one YAML config."""
    with Path(path).open() as handle:
        return validate_stage_one_config(yaml.safe_load(handle))


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
STATE_ROLES = ("train_cells", "val", "test")


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
    _require_keys(config, {*_STATES_GROUPS, *_STATES_TOP_LEVEL.split()}, "config")
    for name, keys in _STATES_GROUPS.items():
        _require_keys(config[name], set(keys.split()), name)
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
    with Path(path).open() as handle:
        return validate_states_config(yaml.safe_load(handle))
