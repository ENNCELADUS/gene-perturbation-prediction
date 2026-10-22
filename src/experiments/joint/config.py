"""The joint GeneEffect model's config (``all``, ``revision``, ``test``; the prior and
the ladder read it too): every key explicit, unknown keys rejected."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.experiments.config import read_yaml, require_keys

_GROUPS = {
    "seeds": "train collator projection",
    "train": (
        "max_epochs patience dependency_batch_size state_learning_rate "
        "adapter_learning_rate head_learning_rate weight_decay "
        "objective state_mode warmup_epochs genes_per_block"
    ),
    "comparison": (
        "epochs hidden learning_rate state_learning_rate batch_size shuffles bootstrap"
    ),
    "features": (
        "cells_per_context variable_gene_min_observations variable_gene_percentile "
        "selective_min_lines selective_max_fraction residual_sd_floor_percentile"
    ),
    "model": (
        "cell_sentence_len esm2_adapter_hidden head_blocks factor_rank "
        "context_components dropout"
    ),
    "preparation": (
        "response_max_cells_per_gene response_sampling_seed tx1_batch_size "
        "tx1_max_length var_ensembl_col"
    ),
    "paths": (
        "split gene_effect basal_raw_umi tx1_model_dir tx1_cache esm2_embeddings "
        "state_checkpoint state_model_dir perturbseq_sources prior"
    ),
}
_TOP_LEVEL = "output_root prepared_root"
_HEAD_BLOCKS = "use_delta_proj use_s"
#: The joint model trains, validates and extracts features in bf16 on CUDA.
PRECISION = "bf16"
OBJECTIVES = (
    "huber",
    "standardized_mse",
    "pearson_blocks",
    "line_ranking",
    "dependency_classification",
)
STATE_MODES = ("frozen", "trainable")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject missing and unknown keys and unknown choices; values as written."""
    require_keys(config, {*_GROUPS, *_TOP_LEVEL.split()}, "config")
    for name, keys in _GROUPS.items():
        require_keys(config[name], set(keys.split()), name)
    require_keys(
        config["model"]["head_blocks"], set(_HEAD_BLOCKS.split()), "model.head_blocks"
    )
    for key, choices in (("objective", OBJECTIVES), ("state_mode", STATE_MODES)):
        if config["train"][key] not in choices:
            raise ValueError(f"train.{key} must be one of {choices}")
    return dict(config)


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML config; relative paths resolve from the repo root."""
    return validate_config(read_yaml(path))


def mixed_precision() -> str:
    """bf16 on CUDA, where every joint run goes; fp32 on the CPU (the tests)."""
    import torch

    return PRECISION if torch.cuda.is_available() else "no"
