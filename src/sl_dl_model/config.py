"""Configuration for the exp08 STATE-adapter DL SL-pair model."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SLDLConfig:
    """Defaults and hyperparameters for the exp08 DL run."""

    input_csv: Path = Path(
        "data/SL_benchmark/derived/k562_depmap_rand_1to1/"
        "all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv"
    )
    output_dir: Path = Path("results/experiments/08_k562_sl_pair_state_dl/run")
    split_types: tuple[str, ...] | None = None
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    ranking_k: tuple[int, ...] = (10, 20, 50)
    seed: int = 17

    # Gene embedding (ESM2)
    esm2_model: str = "facebook/esm2_t33_650M_UR50D"
    esm2_npz: Path | None = None
    fallback_strategy: str = "zero"
    include_coverage_flag: bool = True

    # STATE encoder
    state_checkpoint: Path = Path(
        "model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt"
    )
    state_backend: str = "state_checkpoint"
    gwps_h5ad: Path = Path(
        "data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad"
    )
    gwps_overlap_csv: Path = Path(
        "data/sl_dependency_v0/interim/k562_replogle_depmap_overlap.csv"
    )
    bags_npz: Path | None = None
    control_template_size: int = 256
    cells_per_bag: int = 256

    # Adapter / pooling / pair head
    # pert_dim: for state_checkpoint this is a hint only — the adapter output
    # width is inferred from the loaded checkpoint's pert_dim. It is
    # authoritative only for the linear_mock backend.
    pert_dim: int = 328
    adapter_hidden: int = 512
    pooling: str = "mean_std"
    pair_hidden: tuple[int, ...] = (256, 64)

    # Loss weights + schedule
    lambda_sl: float = 1.0
    lambda_distill: float = 0.5
    lambda_distill_after_warmup: float = 0.1
    lambda_bag: float = 1.0
    lambda_rank: float = 0.0
    warmup_epochs: int = 3
    max_epochs: int = 20
    batch_pairs: int = 1024
    lr: float = 1e-3

    embedding_method: str = "state_adapter_esm2_meanstd"

    @property
    def augmented(self) -> bool:
        """exp08 always runs the augmented (transcript) scoring path."""
        return True


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


def load_config(path: Path) -> SLDLConfig:
    """Load an :class:`SLDLConfig` from YAML, coercing paths and tuples."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    valid = {f.name for f in fields(SLDLConfig)}
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
    return SLDLConfig(**kwargs)
