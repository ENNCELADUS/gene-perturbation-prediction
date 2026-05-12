"""Configuration loading for dependency baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    h5ad_path: Path
    overlap_csv: Path
    output_dir: Path
    obs_perturbation_col: str = "gene"
    control_label: str = "non-targeting"
    var_gene_symbol_col: str = "gene_name"
    depmap_label_col: str = "depmap_gene_effect"
    matched_label_col: str = "has_depmap_label"
    n_cells_col: str = "n_cells_or_pseudobulk"
    external_overlap_csvs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class FeatureConfig:
    chunk_size: int = 4096
    top_abs_delta_sizes: tuple[int, ...] = (50, 100, 500)


@dataclass(frozen=True)
class CvConfig:
    n_splits: int = 5
    n_repeats: int = 5
    random_state: int = 42
    stratify_bins: int = 10
    pca_components: tuple[int, ...] = (20, 50, 100)
    model_set: str = "full"
    essential_thresholds: tuple[float, ...] = (-0.5, -1.0)


@dataclass(frozen=True)
class BaselineConfig:
    data: DataConfig
    features: FeatureConfig
    cv: CvConfig


def _path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _tuple_int(values: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    return tuple(int(value) for value in values)


def _tuple_float(values: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if values is None:
        return default
    return tuple(float(value) for value in values)


def _tuple_path(values: Any) -> tuple[Path, ...]:
    if values is None:
        return ()
    return tuple(_path(value) for value in values)


def load_config(path: str | Path) -> BaselineConfig:
    """Load a YAML config file."""
    config_path = _path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data = raw.get("data", {})
    features = raw.get("features", {})
    cv = raw.get("cv", {})

    return BaselineConfig(
        data=DataConfig(
            h5ad_path=_path(data["h5ad_path"]),
            overlap_csv=_path(data["overlap_csv"]),
            output_dir=_path(data["output_dir"]),
            obs_perturbation_col=data.get("obs_perturbation_col", "gene"),
            control_label=data.get("control_label", "non-targeting"),
            var_gene_symbol_col=data.get("var_gene_symbol_col", "gene_name"),
            depmap_label_col=data.get("depmap_label_col", "depmap_gene_effect"),
            matched_label_col=data.get("matched_label_col", "has_depmap_label"),
            n_cells_col=data.get("n_cells_col", "n_cells_or_pseudobulk"),
            external_overlap_csvs=_tuple_path(data.get("external_overlap_csvs")),
        ),
        features=FeatureConfig(
            chunk_size=int(features.get("chunk_size", 4096)),
            top_abs_delta_sizes=_tuple_int(
                features.get("top_abs_delta_sizes"),
                (50, 100, 500),
            ),
        ),
        cv=CvConfig(
            n_splits=int(cv.get("n_splits", 5)),
            n_repeats=int(cv.get("n_repeats", 5)),
            random_state=int(cv.get("random_state", 42)),
            stratify_bins=int(cv.get("stratify_bins", 10)),
            pca_components=_tuple_int(cv.get("pca_components"), (20, 50, 100)),
            model_set=cv.get("model_set", "full"),
            essential_thresholds=_tuple_float(
                cv.get("essential_thresholds"),
                (-0.5, -1.0),
            ),
        ),
    )
