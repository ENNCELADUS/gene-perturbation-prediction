"""Configuration for the exp10 DDGCN reproduction run."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DdgcnConfig:
    """Defaults and hyperparameters for the DDGCN reproduction.

    Defaults match the official Dual-Dropout GCN repo (Cai et al. 2020) and the
    vendored port in ``data/SL_benchmark/src/``.

    Attributes:
        input_csv: Canonical all-CV balanced benchmark CSV.
        output_dir: Run directory for metrics and manifest.
        split_types: CV splits to evaluate; ``None`` auto-discovers the input.
        folds: CV fold ids to evaluate.
        ranking_k: Cutoffs for NDCG/Recall/Precision@k.
        seed: Per-fold seed for torch/numpy.
        dropout: Dropout probability (input, hidden, decoder).
        lr: Adam learning rate.
        hidden1: First GCN layer output dim.
        hidden2: Second GCN layer (node embedding) dim.
        init_type: Weight init scheme (``"Kaiming"``/``"Xavier"``/other).
        use_bias: Whether GCN layers use a bias term.
        rho: Geometric-mean / second-stream loss weight.
        normal_dim: Adjacency normalization mode (``"Row&Column"``/``"Row"``).
        max_epochs: Maximum training epochs per fold.
        tolerance_epoch: Minimum epochs before early-stop is considered.
        stop_threshold: Relative loss-change early-stop threshold.
        eval_interval: Epoch cadence for the early-stop check.
    """

    input_csv: Path = Path(
        "data/SL_benchmark/derived/k562_depmap_rand_1to1/"
        "all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv"
    )
    output_dir: Path = Path("results/experiments/10_k562_sl_pair_ddgcn/run")
    split_types: tuple[str, ...] | None = None
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    ranking_k: tuple[int, ...] = (10, 20, 50)
    seed: int = 456

    dropout: float = 0.5
    lr: float = 0.01
    hidden1: int = 512
    hidden2: int = 256
    init_type: str = "Kaiming"
    use_bias: bool = False
    rho: float = 1.0
    normal_dim: str = "Row&Column"
    max_epochs: int = 2000
    tolerance_epoch: int = 1000
    stop_threshold: float = 1e-5
    eval_interval: int = 50


_PATH_FIELDS = {"input_csv", "output_dir"}
_TUPLE_FIELDS = {"split_types", "folds", "ranking_k"}


def load_config(path: Path) -> DdgcnConfig:
    """Load a :class:`DdgcnConfig` from YAML, coercing paths and tuples.

    Args:
        path: Path to a YAML file with a subset of ``DdgcnConfig`` fields.

    Returns:
        The constructed :class:`DdgcnConfig`.

    Raises:
        ValueError: If the YAML contains keys not present on ``DdgcnConfig``.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    valid = {f.name for f in fields(DdgcnConfig)}
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
    return DdgcnConfig(**kwargs)
