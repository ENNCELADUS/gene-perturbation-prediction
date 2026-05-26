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
    external_feature_sources: tuple[ExternalFeatureSourceConfig, ...] = ()


@dataclass(frozen=True)
class ExternalEvaluationConfig:
    name: str
    features_npz: Path


@dataclass(frozen=True)
class ExternalFeatureSourceConfig:
    name: str
    h5ad_path: Path
    obs_perturbation_col: str = "perturbation"
    control_label: str | None = None
    var_gene_symbol_col: str = "gene_name"


@dataclass(frozen=True)
class FeatureConfig:
    chunk_size: int = 4096
    top_abs_delta_sizes: tuple[int, ...] = (50, 100, 500)
    program_score_sets: tuple[str, ...] = DEFAULT_PROGRAM_SCORE_SETS


@dataclass(frozen=True)
class SingleCellConfig:
    n_hvg: int = 2000
    n_pcs: int = 128
    feature_sets: tuple[str, ...] = ("single_cell_pc_delta",)
    scvi_latent_dim: int = 128
    scvi_hidden_units: int = 128
    scvi_layers: int = 1
    scvi_max_epochs: int = 400
    scvi_batch_size: int = 128
    max_cells_per_bag: int = 256
    hidden_units: tuple[int, ...] = (128, 64)
    bag_hidden_units: tuple[int, ...] = (64,)
    attention_heads: int = 4
    attention_orthogonality_lambda: float = 0.01
    attention_orthogonality: str = "cosine_squared_offdiag"
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 500
    patience: int = 40
    validation_fraction: float = 0.15
    batch_size: int = 32
    device: str = "auto"


@dataclass(frozen=True)
class DistributionConfig:
    component_counts: tuple[int, ...] = (32,)
    sensitivity_component_counts: tuple[int, ...] = (16, 64)
    covariance_type: str = "diag"
    prototype_fit_scope: str = "train_genes_plus_controls"
    feature_blocks: str = "occupancy_first"
    views: tuple[str, ...] = ("centered", "deltap")
    weightings: tuple[str, ...] = ("unweighted",)
    ridge_alphas: tuple[float, ...] = (1.0, 10.0, 100.0)
    random_forest_n_estimators: int = 300
    random_forest_min_samples_leaf: int = 5
    random_forest_n_jobs: int = -1
    mlp_hidden_units: tuple[int, ...] = (32,)
    mlp_max_epochs: int = 300
    mlp_patience: int = 30
    cloudpred_hidden_units: tuple[int, ...] = (32,)
    cloudpred_learning_rate: float = 1e-3
    cloudpred_weight_decay: float = 1e-3
    cloudpred_max_epochs: int = 300
    cloudpred_patience: int = 30
    cloudpred_validation_fraction: float = 0.15
    cloudpred_batch_size: int = 32
    max_gmm_fit_cells: int | None = None
    max_cells_per_bag: int = 512
    device: str = "auto"


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
    features: FeatureConfig = field(default_factory=FeatureConfig)
    cv: CvConfig = field(default_factory=CvConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    single_cell: SingleCellConfig = field(default_factory=SingleCellConfig)
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
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


def _tuple_int_or_default(values: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    return tuple(int(value) for value in values)


def _tuple_int_or_none(values: Any) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(int(value) for value in values)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


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


def _external_feature_sources(values: Any) -> tuple[ExternalFeatureSourceConfig, ...]:
    if values is None:
        return ()
    return tuple(
        ExternalFeatureSourceConfig(
            name=str(value["name"]),
            h5ad_path=_path(value["h5ad_path"]),
            obs_perturbation_col=str(value.get("obs_perturbation_col", "perturbation")),
            control_label=(
                str(value["control_label"]) if value.get("control_label") else None
            ),
            var_gene_symbol_col=str(value.get("var_gene_symbol_col", "gene_name")),
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
        "mlp": {
            "enabled": False,
            "hidden_units": [64, 16],
            "dropout": 0.2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "max_epochs": 1000,
            "patience": 75,
            "min_delta": 0.0001,
            "validation_fraction": 0.15,
            "validation_bins": cv.stratify_bins,
            "device": "auto",
            "variants": [
                {"name": "raw", "pca_components": None},
                {"name": "pca50", "pca_components": 50},
                {"name": "pca100", "pca_components": 100},
            ],
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
    single_cell = raw.get("single_cell", {})
    distribution = raw.get("distribution", {})
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
            external_feature_sources=_external_feature_sources(
                data.get("external_feature_sources")
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
        single_cell=SingleCellConfig(
            n_hvg=int(single_cell.get("n_hvg", 2000)),
            n_pcs=int(single_cell.get("n_pcs", 128)),
            feature_sets=_tuple_str_or_none(single_cell.get("feature_sets"))
            or ("single_cell_pc_delta",),
            scvi_latent_dim=int(single_cell.get("scvi_latent_dim", 128)),
            scvi_hidden_units=int(single_cell.get("scvi_hidden_units", 128)),
            scvi_layers=int(single_cell.get("scvi_layers", 1)),
            scvi_max_epochs=int(single_cell.get("scvi_max_epochs", 400)),
            scvi_batch_size=int(single_cell.get("scvi_batch_size", 128)),
            max_cells_per_bag=int(single_cell.get("max_cells_per_bag", 256)),
            hidden_units=_tuple_int_or_default(
                single_cell.get("hidden_units"),
                (128, 64),
            ),
            bag_hidden_units=_tuple_int_or_default(
                single_cell.get("bag_hidden_units"),
                (64,),
            ),
            attention_heads=int(single_cell.get("attention_heads", 4)),
            attention_orthogonality_lambda=float(
                single_cell.get("attention_orthogonality_lambda", 0.01),
            ),
            attention_orthogonality=str(
                single_cell.get(
                    "attention_orthogonality",
                    "cosine_squared_offdiag",
                )
            ),
            dropout=float(single_cell.get("dropout", 0.1)),
            learning_rate=float(single_cell.get("learning_rate", 1e-3)),
            weight_decay=float(single_cell.get("weight_decay", 1e-3)),
            max_epochs=int(single_cell.get("max_epochs", 500)),
            patience=int(single_cell.get("patience", 40)),
            validation_fraction=float(single_cell.get("validation_fraction", 0.15)),
            batch_size=int(single_cell.get("batch_size", 32)),
            device=str(single_cell.get("device", "auto")),
        ),
        distribution=DistributionConfig(
            component_counts=_tuple_int(
                distribution.get("component_counts"),
                (32,),
            ),
            sensitivity_component_counts=_tuple_int(
                distribution.get("sensitivity_component_counts"),
                (16, 64),
            ),
            covariance_type=str(distribution.get("covariance_type", "diag")),
            prototype_fit_scope=str(
                distribution.get("prototype_fit_scope", "train_genes_plus_controls")
            ),
            feature_blocks=str(distribution.get("feature_blocks", "occupancy_first")),
            views=_tuple_str_or_none(distribution.get("views"))
            or ("centered", "deltap"),
            weightings=_tuple_str_or_none(distribution.get("weightings"))
            or ("unweighted",),
            ridge_alphas=_tuple_float(
                distribution.get("ridge_alphas"),
                (1.0, 10.0, 100.0),
            ),
            random_forest_n_estimators=int(
                distribution.get("random_forest_n_estimators", 300)
            ),
            random_forest_min_samples_leaf=int(
                distribution.get("random_forest_min_samples_leaf", 5)
            ),
            random_forest_n_jobs=int(distribution.get("random_forest_n_jobs", -1)),
            mlp_hidden_units=_tuple_int_or_default(
                distribution.get("mlp_hidden_units"),
                (32,),
            ),
            mlp_max_epochs=int(distribution.get("mlp_max_epochs", 300)),
            mlp_patience=int(distribution.get("mlp_patience", 30)),
            cloudpred_hidden_units=_tuple_int_or_default(
                distribution.get("cloudpred_hidden_units"),
                (32,),
            ),
            cloudpred_learning_rate=float(
                distribution.get("cloudpred_learning_rate", 1e-3)
            ),
            cloudpred_weight_decay=float(
                distribution.get("cloudpred_weight_decay", 1e-3)
            ),
            cloudpred_max_epochs=int(distribution.get("cloudpred_max_epochs", 300)),
            cloudpred_patience=int(distribution.get("cloudpred_patience", 30)),
            cloudpred_validation_fraction=float(
                distribution.get("cloudpred_validation_fraction", 0.15)
            ),
            cloudpred_batch_size=int(distribution.get("cloudpred_batch_size", 32)),
            max_gmm_fit_cells=_int_or_none(distribution.get("max_gmm_fit_cells")),
            max_cells_per_bag=int(distribution.get("max_cells_per_bag", 512)),
            device=str(distribution.get("device", "auto")),
        ),
        viability_axis=_viability_axis_config(raw.get("viability_axis")),
        models=_model_config(raw.get("models"), cv_config),
    )
