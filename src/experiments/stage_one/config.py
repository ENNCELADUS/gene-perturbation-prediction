"""The stage-one configs: the ladder's inputs and every stage-one run, prior reference
and report (PIE stage-one design; protocol §15, §16). Every key explicit, unknown keys
rejected."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.experiments.config import read_yaml, require_keys


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
    require_keys(config, {*_LADDER_GROUPS, *_LADDER_TOP_LEVEL.split()}, "config")
    for name, keys in _LADDER_GROUPS.items():
        require_keys(config[name], set(keys.split()), name)
    if not (isinstance(config["frame"]["folds"], int) and config["frame"]["folds"] > 1):
        raise ValueError("frame.folds must be an integer above 1")
    for key in ("pilot_genes", "components", "lines_per_batch"):
        value = config["pie"][key]
        if not (isinstance(value, int) and value > 0):
            raise ValueError(f"pie.{key} must be a positive integer")
    return dict(config)


def load_ladder_config(path: Path) -> dict[str, Any]:
    """Load and validate a stage-one inputs YAML config."""
    return validate_ladder_config(read_yaml(path))


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
        require_keys(model["context"], set(_STAGE_ONE_CONTEXT.split()), "model.context")
        for key in _STAGE_ONE_CONTEXT.split():
            _positive_int(model["context"][key], f"model.context.{key}")
    if arm == "hybrid" and data["input"] not in STAGE_ONE_DELTAS:
        raise ValueError("a hybrid joins the direct readout with a frozen-PIE delta")
    if (arm in STAGE_ONE_TRUNK_ARMS) != (model["trunk"] is not None):
        raise ValueError("model.trunk: the direct readout's trunk, else null")
    if model["trunk"] is not None:
        trunk = model["trunk"]
        require_keys(trunk, set(_STAGE_ONE_TRUNK.split()), "model.trunk")
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
        require_keys(config[name], set(keys.split()), name)
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
        require_keys(row, set(_STAGE_ONE_ROWS[row["kind"]].split()), f"rows[{index}]")
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
    require_keys(config, set(_STAGE_ONE_TOP_LEVEL[config["kind"]].split()), "config")
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
    return validate_stage_one_config(read_yaml(path))
