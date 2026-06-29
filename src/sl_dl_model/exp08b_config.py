"""Configuration for exp08b two-step STATE-adapter experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from sl_dl_model.config import SLDLConfig


@dataclass(frozen=True)
class Exp08bConfig(SLDLConfig):
    """Exp08b config with Step 1 and Step 2 specific fields."""

    output_dir: Path = Path(
        "results/experiments/08b_k562_sl_pair_two_step_state_adapter/run"
    )
    embedding_method: str = "exp08b_state_adapter_meanstd"

    # Step 1 generator.
    generator_kind: str = "state_adapter"
    generator_val_fraction: float = 0.2
    generator_val_seed: int = 17
    direct_mlp_hidden: int = 512
    bag_scale_mode: str = "fixed_warmup"
    bag_scale_min: float = 1e-3
    bag_scale_ema_decay: float = 0.95

    # Exp08b keeps the distill anchor at full weight for the full Step 1 run.
    lambda_sl: float = 0.0
    lambda_distill: float = 1.0
    lambda_distill_after_warmup: float = 1.0
    lambda_bag: float = 1.0
    warmup_epochs: int = 1

    # Step-scoped artifacts.
    step1_artifacts_subdir: str = "step1_generator"
    step2_results_subdir: str = "step2_sl_head"
    generator_embedding_filename: str = "predicted_embeddings.npz"
    generator_manifest_filename: str = "generator_manifest.json"
    generator_weights_filename: str = "generator_weights.pt"
    generator_monitor_filename: str = "generator_monitor.csv"


# generator_kind → side-by-side ladder label (spec §5.2). Step 2 metric rows
# must carry distinct model names so exp08b, the direct-ESM2-MLP control, and
# the NN-copy rung land as separate rows in the official-metric summary.
_GENERATOR_KIND_TO_MODEL_NAME = {
    "state_adapter": "exp08b",
    "direct_mlp": "direct_esm2_mlp",
    "nn_copy": "nn_copy",
}


def metric_model_name_for(generator_kind: str) -> str:
    """Map a ``generator_kind`` to its §5.2 ladder model label.

    Raises:
        ValueError: If ``generator_kind`` is not a known exp08b rung.
    """
    try:
        return _GENERATOR_KIND_TO_MODEL_NAME[generator_kind]
    except KeyError as exc:
        raise ValueError(f"unknown generator_kind: {generator_kind!r}") from exc


@dataclass(frozen=True)
class SlHeadConfig:
    """Slim Step-2 config: pair-head, scoring, and optimization fields only.

    Spec §7.1 forbids Step 2 from holding a STATE checkpoint. This dataclass
    carries exactly the fields ``CachedEmbeddingPairHeadProducer`` reads — no
    ``state_checkpoint``, ``esm2_npz``, ``gwps_h5ad``, or generator field — so
    the Step-2 producer module never even has a checkpoint path to leak.
    """

    pair_hidden: tuple[int, ...] = (256, 64)
    include_coverage_flag: bool = True
    seed: int = 17
    lr: float = 1e-3
    max_epochs: int = 20
    batch_pairs: int = 1024
    max_grad_norm: float = 1.0

    @classmethod
    def from_exp08b(cls, config: "Exp08bConfig") -> "SlHeadConfig":
        """Project the relevant Step-2 fields out of a full exp08b config."""
        return cls(
            pair_hidden=tuple(config.pair_hidden),
            include_coverage_flag=bool(config.include_coverage_flag),
            seed=int(config.seed),
            lr=float(config.lr),
            max_epochs=int(config.max_epochs),
            batch_pairs=int(config.batch_pairs),
            max_grad_norm=float(config.max_grad_norm),
        )


_PATH_FIELDS = {
    "input_csv",
    "output_dir",
    "esm2_npz",
    "state_checkpoint",
    "gwps_h5ad",
    "gwps_overlap_csv",
    "bags_npz",
}
_TUPLE_FIELDS = {"split_types", "folds", "ranking_k", "pair_hidden"}


def load_exp08b_config(path: Path) -> Exp08bConfig:
    """Load an :class:`Exp08bConfig` from YAML."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    valid = {f.name for f in fields(Exp08bConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    kwargs: dict[str, object] = {}
    for key, value in raw.items():
        if key in _PATH_FIELDS and value is not None:
            kwargs[key] = Path(value)
        elif key in _TUPLE_FIELDS and value is not None:
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return Exp08bConfig(**kwargs)
