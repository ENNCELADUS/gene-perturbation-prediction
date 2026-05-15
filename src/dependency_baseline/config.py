"""Configuration loading for dependency baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dependency_baseline.program_scores import DEFAULT_PROGRAM_SCORE_SETS


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
    external_evaluations: tuple[ExternalEvaluationConfig, ...] = ()


@dataclass(frozen=True)
class ExternalEvaluationConfig:
    name: str
    features_npz: Path


@dataclass(frozen=True)
class FeatureConfig:
    chunk_size: int = 4096
    top_abs_delta_sizes: tuple[int, ...] = (50, 100, 500)
    program_score_sets: tuple[str, ...] = DEFAULT_PROGRAM_SCORE_SETS


@dataclass(frozen=True)
class ViabilityAxisArtifactConfig:
    name: str
    url: str
    sha256: str


@dataclass(frozen=True)
class ViabilityAxisConfig:
    enabled: bool = False
    cache_dir: Path | None = None
    artifacts: tuple[ViabilityAxisArtifactConfig, ...] = ()


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
class ExperimentConfig:
    name: str = "replogle_k562_b_to_c_baseline"
    seed: int = 42
    run_id: str | None = None
    human_result_tables: tuple[str, ...] = ("summary_metrics",)
    machine_result_format: str = "parquet"
    checkpoint_policy: str = "all_cv_and_final"
    save_predictions: bool = True
    save_rankings: bool = True
    save_splits: bool = True
    topk_candidates: tuple[int, ...] = (20, 50, 100)


@dataclass(frozen=True)
class SelectionConfig:
    scopes: tuple[str, ...] | None = None
    features: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    folds: tuple[int, ...] | None = None
    weightings: tuple[str, ...] | None = None


@dataclass(frozen=True)
class BaselineConfig:
    data: DataConfig
    features: FeatureConfig
    cv: CvConfig
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    viability_axis: ViabilityAxisConfig = field(default_factory=ViabilityAxisConfig)
    models: dict[str, dict[str, Any]] | None = None


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


def _tuple_str_or_none(values: Any) -> tuple[str, ...] | None:
    if values is None:
        return None
    return tuple(str(value) for value in values)


def _tuple_int_or_none(values: Any) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(int(value) for value in values)


def _tuple_path(values: Any) -> tuple[Path, ...]:
    if values is None:
        return ()
    return tuple(_path(value) for value in values)


def _external_evaluations(values: Any) -> tuple[ExternalEvaluationConfig, ...]:
    if values is None:
        return ()
    return tuple(
        ExternalEvaluationConfig(
            name=str(value["name"]),
            features_npz=_path(value["features_npz"]),
        )
        for value in values
    )


def _viability_axis_config(values: Any) -> ViabilityAxisConfig:
    if values is None:
        return ViabilityAxisConfig()
    artifacts = tuple(
        ViabilityAxisArtifactConfig(
            name=str(value["name"]),
            url=str(value["url"]),
            sha256=str(value["sha256"]),
        )
        for value in values.get("artifacts", ())
    )
    cache_dir = values.get("cache_dir")
    return ViabilityAxisConfig(
        enabled=bool(values.get("enabled", False)),
        cache_dir=_path(cache_dir) if cache_dir else None,
        artifacts=artifacts,
    )


def _model_config(values: Any, cv: CvConfig) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {
        "mean_label": {"enabled": True},
        "ridge": {"enabled": True, "alpha": 10.0},
        "elastic_net": {
            "enabled": True,
            "alpha": 0.02,
            "l1_ratio": 0.1,
            "max_iter": 20000,
            "tol": 1e-3,
            "selection": "random",
        },
        "pca_ridge": {
            "enabled": True,
            "components": list(cv.pca_components),
            "alpha": 10.0,
        },
        "pca_random_forest": {
            "enabled": cv.model_set != "quick",
            "components": list(cv.pca_components),
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "n_jobs": -1,
        },
        "random_forest": {
            "enabled": cv.model_set != "quick",
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "n_jobs": -1,
        },
        "xgboost": {
            "enabled": cv.model_set != "quick",
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "n_jobs": 4,
        },
        "lasso": {
            "enabled": False,
            "alpha": 0.01,
            "max_iter": 20000,
            "tol": 1e-3,
            "selection": "random",
        },
        "nar_viability_axis": {
            "enabled": None,
            "alpha": 10.0,
            "n_score_columns": 2,
            "score_ridge": True,
            "score_plus_burden_ridge": True,
            "resid_pca50_ridge": True,
            "resid_pca50_random_forest": True,
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "n_jobs": -1,
        },
        "signal_decomposition": {
            "enabled": False,
            "alpha": 10.0,
            "lasso_alpha": 0.01,
            "n_score_columns": None,
            "pca_components": 50,
            "score_ridge": True,
            "resid_pca_ridge": True,
            "resid_pca_random_forest": True,
            "resid_pca_plus_scores_ridge": True,
            "resid_pca_plus_scores_random_forest": True,
            "resid_lasso": True,
            "program_score_ridge": True,
            "program_score_elastic_net": True,
            "program_score_random_forest": True,
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "n_jobs": -1,
        },
    }
    if cv.model_set == "quick":
        models["elastic_net"]["enabled"] = False
        models["pca_ridge"]["enabled"] = False
        models["pca_random_forest"]["enabled"] = False

    if values is None:
        return models
    for name, overrides in values.items():
        current = dict(models.get(str(name), {}))
        current.update(overrides or {})
        models[str(name)] = current
    return models


def load_config(path: str | Path) -> BaselineConfig:
    """Load a YAML config file."""
    config_path = _path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data = raw.get("data", {})
    features = raw.get("features", {})
    cv = raw.get("cv", {})
    experiment = raw.get("experiment", {})
    selection = raw.get("selection", {})
    cv_config = CvConfig(
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
    )

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
            external_evaluations=_external_evaluations(
                data.get("external_evaluations")
            ),
        ),
        features=FeatureConfig(
            chunk_size=int(features.get("chunk_size", 4096)),
            top_abs_delta_sizes=_tuple_int(
                features.get("top_abs_delta_sizes"),
                (50, 100, 500),
            ),
            program_score_sets=_tuple_str_or_none(features.get("program_score_sets"))
            or DEFAULT_PROGRAM_SCORE_SETS,
        ),
        cv=cv_config,
        experiment=ExperimentConfig(
            name=experiment.get("name", "replogle_k562_b_to_c_baseline"),
            seed=int(experiment.get("seed", 42)),
            run_id=experiment.get("run_id"),
            human_result_tables=_tuple_str_or_none(
                experiment.get("human_result_tables")
            )
            or ("summary_metrics",),
            machine_result_format=str(
                experiment.get("machine_result_format", "parquet")
            ),
            checkpoint_policy=experiment.get(
                "checkpoint_policy",
                "all_cv_and_final",
            ),
            save_predictions=bool(experiment.get("save_predictions", True)),
            save_rankings=bool(experiment.get("save_rankings", True)),
            save_splits=bool(experiment.get("save_splits", True)),
            topk_candidates=_tuple_int(
                experiment.get("topk_candidates"),
                (20, 50, 100),
            ),
        ),
        selection=SelectionConfig(
            scopes=_tuple_str_or_none(selection.get("scopes")),
            features=_tuple_str_or_none(selection.get("features")),
            models=_tuple_str_or_none(selection.get("models")),
            folds=_tuple_int_or_none(selection.get("folds")),
            weightings=_tuple_str_or_none(selection.get("weightings")),
        ),
        viability_axis=_viability_axis_config(raw.get("viability_axis")),
        models=_model_config(raw.get("models"), cv_config),
    )
