"""Configuration for the Feng DepMap-profile baseline."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProfileBaselineConfig:
    """Frozen inputs and model settings for a profile-only benchmark run."""

    sl_root: Path = Path("data/SL_benchmark")
    gene_effect_csv: Path = Path(
        "data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv"
    )
    output_dir: Path = Path(
        "results/experiments/11_feng_depmap_geneeffect_profiles/run"
    )
    config_source: Path | None = None
    split_types: tuple[str, ...] = ("CV1", "CV2", "CV3")
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    ranking_k: tuple[int, ...] = (10, 20, 50)
    models: tuple[str, ...] = (
        "pearson_abs",
        "spearman_abs",
        "cell_line_residual_pearson_abs",
        "missingness_only",
        "summary_logreg",
        "summary_xgb",
        "raw_sgd",
    )
    ranking_models: tuple[str, ...] = (
        "pearson_abs",
        "spearman_abs",
        "cell_line_residual_pearson_abs",
    )
    min_finite_lines: int = 20
    dependency_threshold: float = -0.5
    pan_essential_fraction: float = 0.5
    seed: int = 17
    logreg_c: float = 1.0
    logreg_max_iter: int = 1000
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.1
    raw_alpha: float = 1.0e-4
    raw_l1_ratio: float = 0.15
    raw_max_iter: int = 200
    pair_feature_chunk_size: int = 2048
    score_matrix_chunk_rows: int = 4

    @property
    def entities_csv(self) -> Path:
        """Canonical Feng entity table."""
        return self.sl_root / "data" / "fin_entities.csv"

    def split_path(self, split_type: str) -> Path:
        """Return the official Rand 1:1 split cache for a CV type."""
        return self.sl_root / "data" / "data_split" / f"{split_type}_1.npy"


_PATH_FIELDS = {"sl_root", "gene_effect_csv", "output_dir", "config_source"}
_TUPLE_FIELDS = {"split_types", "folds", "ranking_k", "models", "ranking_models"}


def load_config(path: Path) -> ProfileBaselineConfig:
    """Load a validated YAML config."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    valid = {field.name for field in fields(ProfileBaselineConfig)}
    unknown = set(raw).difference(valid)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    kwargs: dict[str, object] = {}
    for key, value in raw.items():
        if key in _PATH_FIELDS:
            kwargs[key] = Path(value)
        elif key in _TUPLE_FIELDS:
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return replace(ProfileBaselineConfig(**kwargs), config_source=Path(path))
