"""Model factories and estimator fitting helpers."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dependency_baseline.config import BaselineConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    supports_weight: bool


def build_model_specs(config: BaselineConfig) -> list[ModelSpec]:
    """Build configured sklearn baseline models."""
    specs: list[ModelSpec] = []
    models = config.models or {}
    _add_mean_label(specs, models)
    _add_ridge(specs, models)
    _add_elastic_net(specs, models, config)
    _add_lasso(specs, models, config)
    _add_pca_ridge(specs, models, config)
    _add_random_forest(specs, models, config)
    _add_pca_random_forest(specs, models, config)
    _add_nar_viability_axis_models(specs, models, config)
    _add_signal_decomposition_models(specs, models, config)
    _add_xgboost(specs, models, config)
    return specs


def fit_estimator(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    weighting: str,
) -> tuple[object, float]:
    """Clone and fit one estimator, returning fit wall time."""
    fit_params = {}
    if weighting == "sqrt_n_cells":
        fit_params = fit_params_for_sample_weight(spec.estimator, sample_weight)
    fitted = clone(spec.estimator)
    fit_started = time.perf_counter()
    fitted.fit(x_train, y_train, **fit_params)
    return fitted, time.perf_counter() - fit_started


def compatible_model_feature(
    model_name: str,
    feature_name: str,
    x_train: np.ndarray,
) -> bool:
    """Return whether a model can consume a concrete feature matrix."""
    return compatible_model_feature_shape(model_name, feature_name, x_train.shape)


def compatible_model_feature_shape(
    model_name: str,
    feature_name: str,
    x_train_shape: tuple[int, int],
) -> bool:
    """Return whether a model can consume a feature shape."""
    if model_name == "nar_score_ridge":
        return feature_name == "nar_viability_scores"
    if model_name == "nar_score_plus_burden_ridge":
        return feature_name == "nar_viability_scores_plus_burden"
    if model_name in {"nar_resid_pca50_ridge", "nar_resid_pca50_random_forest"}:
        if feature_name not in {"nar_resid_delta_all", "nar_resid_delta_mask_target"}:
            return False
        return 50 <= min(x_train_shape[0], x_train_shape[1] - 2)
    signal_decomposition_features = {
        "nuisance_resid_delta_all",
        "nuisance_resid_delta_mask_target",
    }
    if model_name == "nuisance_resid_lasso":
        if feature_name not in signal_decomposition_features:
            return False
        return x_train_shape[1] > 1
    if model_name.startswith("nuisance_resid_pca"):
        if feature_name not in signal_decomposition_features:
            return False
        n_components = _pca_component_count(model_name.replace("nuisance_resid_", ""))
        return n_components <= min(x_train_shape[0], x_train_shape[1] - 2)
    if model_name == "nuisance_score_ridge":
        return feature_name == "nuisance_scores"
    if model_name in {
        "program_score_ridge",
        "program_score_elastic_net",
        "program_score_random_forest",
    }:
        return feature_name in {"program_scores", "program_scores_plus_burden"}
    if not model_name.startswith("pca"):
        return True
    if feature_name not in {"delta_all", "delta_mask_target"}:
        return False
    n_components = _pca_component_count(model_name)
    return n_components <= min(x_train_shape[0], x_train_shape[1])


def fit_params_for_sample_weight(
    model: object,
    sample_weight: np.ndarray,
) -> dict[str, np.ndarray]:
    """Route sample weights to sklearn pipelines or estimators."""
    if hasattr(model, "steps"):
        final_name = model.steps[-1][0]
        return {f"{final_name}__sample_weight": sample_weight}
    return {"sample_weight": sample_weight}


def sample_weights(n_cells: np.ndarray) -> np.ndarray:
    """Compute sqrt(n_cells) weights normalized to mean one."""
    weights = np.sqrt(np.maximum(n_cells, 1.0))
    return weights / np.mean(weights)


class NuisanceResidualizer(BaseEstimator, TransformerMixin):
    """Regress feature columns on appended nuisance score columns."""

    def __init__(self, n_score_columns: int = 2) -> None:
        self.n_score_columns = n_score_columns

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "NuisanceResidualizer":
        del y
        matrix = np.asarray(x, dtype=np.float64)
        n_score_columns = int(self.n_score_columns)
        if n_score_columns <= 0 or n_score_columns >= matrix.shape[1]:
            msg = "n_score_columns must be positive and smaller than feature count"
            raise ValueError(msg)
        delta = matrix[:, :-n_score_columns]
        scores = matrix[:, -n_score_columns:]
        design = np.column_stack([np.ones(scores.shape[0]), scores])
        self.coefficients_ = np.linalg.lstsq(design, delta, rcond=None)[0]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        n_score_columns = int(self.n_score_columns)
        delta = matrix[:, :-n_score_columns]
        scores = matrix[:, -n_score_columns:]
        design = np.column_stack([np.ones(scores.shape[0]), scores])
        residual = delta - design @ self.coefficients_
        return residual.astype(np.float32)


class ViabilityAxisResidualizer(NuisanceResidualizer):
    """Backward-compatible name for NAR viability-axis residualization."""


class ResidualizedPCAWithScores(BaseEstimator, TransformerMixin):
    """Return residualized-delta PCs concatenated with nuisance scores."""

    def __init__(
        self,
        n_score_columns: int,
        n_components: int,
        random_state: int | None = None,
    ) -> None:
        self.n_score_columns = n_score_columns
        self.n_components = n_components
        self.random_state = random_state

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "ResidualizedPCAWithScores":
        del y
        self.residualizer_ = NuisanceResidualizer(self.n_score_columns).fit(x)
        residual = self.residualizer_.transform(x)
        self.scaler_ = StandardScaler().fit(residual)
        scaled = self.scaler_.transform(residual)
        self.pca_ = PCA(
            n_components=int(self.n_components),
            random_state=self.random_state,
        ).fit(scaled)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        scores = matrix[:, -int(self.n_score_columns) :]
        residual = self.residualizer_.transform(matrix)
        pcs = self.pca_.transform(self.scaler_.transform(residual))
        return np.hstack([pcs, scores]).astype(np.float32)


def filter_models(
    model_specs: list[ModelSpec],
    selected: tuple[str, ...] | None,
) -> list[ModelSpec]:
    """Apply optional model-name selection."""
    if selected is None:
        return model_specs
    selected_set = set(selected)
    return [spec for spec in model_specs if spec.name in selected_set]


def _add_mean_label(
    specs: list[ModelSpec],
    models: dict[str, dict[str, object]],
) -> None:
    mean_config = models.get("mean_label", {})
    if mean_config.get("enabled", True):
        specs.append(ModelSpec("mean_label", DummyRegressor(strategy="mean"), False))


def _add_ridge(
    specs: list[ModelSpec],
    models: dict[str, dict[str, object]],
) -> None:
    ridge_config = models.get("ridge", {})
    if not ridge_config.get("enabled", True):
        return
    variants = ridge_config.get("variants")
    if variants:
        for variant in variants:
            alpha = float(variant.get("alpha", ridge_config.get("alpha", 10.0)))
            specs.append(_ridge_spec(f"ridge_alpha{_param_token(alpha)}", alpha))
        return
    specs.append(_ridge_spec("ridge", float(ridge_config.get("alpha", 10.0))))


def _ridge_spec(name: str, alpha: float) -> ModelSpec:
    return ModelSpec(
        name,
        make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=alpha),
        ),
        True,
    )


def _param_token(value: float | int) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "neg").replace(".", "p")


def _variant_values(
    family_config: dict[str, object],
    default_components: tuple[int, ...],
) -> list[dict[str, object]]:
    variants = family_config.get("variants")
    if variants:
        return [dict(variant) for variant in variants]
    return [
        {"components": component}
        for component in family_config.get(
            "components",
            default_components,
        )
    ]


def _variant_component(variant: dict[str, object]) -> int:
    value = variant.get("components", variant.get("n_components"))
    if value is None:
        msg = "PCA model variant must define components or n_components"
        raise ValueError(msg)
    return int(value)


def _add_elastic_net(
    specs: list[ModelSpec],
    models: dict[str, dict[str, object]],
    config: BaselineConfig,
) -> None:
    elastic_config = models.get("elastic_net", {})
    if not elastic_config.get("enabled", config.cv.model_set != "quick"):
        return
    specs.append(
        ModelSpec(
            "elastic_net",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                ElasticNet(
                    alpha=float(elastic_config.get("alpha", 0.02)),
                    l1_ratio=float(elastic_config.get("l1_ratio", 0.1)),
                    max_iter=int(elastic_config.get("max_iter", 20000)),
                    tol=float(elastic_config.get("tol", 1e-3)),
                    selection=str(elastic_config.get("selection", "random")),
                    random_state=config.cv.random_state,
                ),
            ),
            True,
        )
    )


def _add_lasso(
    specs: list[ModelSpec],
    models: dict[str, dict[str, object]],
    config: BaselineConfig,
) -> None:
    lasso_config = models.get("lasso", {})
    if not lasso_config.get("enabled", False):
        return
    specs.append(
        ModelSpec(
            "lasso",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                Lasso(
                    alpha=float(lasso_config.get("alpha", 0.01)),
                    max_iter=int(lasso_config.get("max_iter", 20000)),
                    tol=float(lasso_config.get("tol", 1e-3)),
                    selection=str(lasso_config.get("selection", "random")),
                    random_state=config.cv.random_state,
                ),
            ),
            True,
        )
    )


def _add_pca_ridge(
    specs: list[ModelSpec],
    models: dict[str, dict[str, object]],
    config: BaselineConfig,
) -> None:
    pca_config = models.get("pca_ridge", {})
    if not pca_config.get("enabled", config.cv.model_set != "quick"):
        return
    has_variants = bool(pca_config.get("variants"))
    for variant in _variant_values(pca_config, config.cv.pca_components):
        n_components = _variant_component(variant)
        alpha = float(variant.get("alpha", pca_config.get("alpha", 10.0)))
        name = f"pca{n_components}_ridge"
        if has_variants:
            name = f"{name}_alpha{_param_token(alpha)}"
        specs.append(
            ModelSpec(
                name,
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    PCA(
                        n_components=n_components,
                        random_state=config.cv.random_state,
                    ),
                    Ridge(alpha=alpha),
                ),
                True,
            )
        )


def _add_random_forest(
    specs: list[ModelSpec],
    models: dict[str, dict[str, object]],
    config: BaselineConfig,
) -> None:
    forest_config = models.get("random_forest", {})
    if not forest_config.get("enabled", config.cv.model_set != "quick"):
        return
    specs.append(
        ModelSpec(
            "random_forest",
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(
                    n_estimators=int(forest_config.get("n_estimators", 300)),
                    min_samples_leaf=int(forest_config.get("min_samples_leaf", 5)),
                    random_state=config.cv.random_state,
                    n_jobs=int(forest_config.get("n_jobs", -1)),
                ),
            ),
            True,
        )
    )


def _add_pca_random_forest(
    specs: list[ModelSpec],
    models: dict[str, dict[str, object]],
    config: BaselineConfig,
) -> None:
    pca_forest_config = models.get("pca_random_forest", {})
    if not pca_forest_config.get("enabled", config.cv.model_set != "quick"):
        return
    has_variants = bool(pca_forest_config.get("variants"))
    for variant in _variant_values(pca_forest_config, config.cv.pca_components):
        n_components = _variant_component(variant)
        min_samples_leaf = int(
            variant.get(
                "min_samples_leaf",
                pca_forest_config.get("min_samples_leaf", 5),
            )
        )
        name = f"pca{n_components}_random_forest"
        if has_variants:
            name = f"{name}_leaf{min_samples_leaf}"
        specs.append(
            ModelSpec(
                name,
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    PCA(
                        n_components=n_components,
                        random_state=config.cv.random_state,
                    ),
                    RandomForestRegressor(
                        n_estimators=int(pca_forest_config.get("n_estimators", 300)),
                        min_samples_leaf=min_samples_leaf,
                        random_state=config.cv.random_state,
                        n_jobs=int(pca_forest_config.get("n_jobs", -1)),
                    ),
                ),
                True,
            )
        )


def _add_nar_viability_axis_models(
    specs: list[ModelSpec],
    models: dict[str, dict[str, object]],
    config: BaselineConfig,
) -> None:
    nar_config = models.get("nar_viability_axis", {})
    enabled = nar_config.get("enabled")
    if enabled is None:
        enabled = config.viability_axis.enabled
    if not enabled:
        return
    alpha = float(nar_config.get("alpha", 10.0))
    n_score_columns = int(nar_config.get("n_score_columns", 2))
    if nar_config.get("score_ridge", True):
        specs.append(
            ModelSpec(
                "nar_score_ridge",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    Ridge(alpha=alpha),
                ),
                True,
            )
        )
    if nar_config.get("score_plus_burden_ridge", True):
        specs.append(
            ModelSpec(
                "nar_score_plus_burden_ridge",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    Ridge(alpha=alpha),
                ),
                True,
            )
        )
    if nar_config.get("resid_pca50_ridge", True):
        specs.append(
            ModelSpec(
                "nar_resid_pca50_ridge",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    NuisanceResidualizer(n_score_columns=n_score_columns),
                    StandardScaler(),
                    PCA(n_components=50, random_state=config.cv.random_state),
                    Ridge(alpha=alpha),
                ),
                True,
            )
        )
    if nar_config.get("resid_pca50_random_forest", True):
        specs.append(
            ModelSpec(
                "nar_resid_pca50_random_forest",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    NuisanceResidualizer(n_score_columns=n_score_columns),
                    StandardScaler(),
                    PCA(n_components=50, random_state=config.cv.random_state),
                    RandomForestRegressor(
                        n_estimators=int(nar_config.get("n_estimators", 300)),
                        min_samples_leaf=int(nar_config.get("min_samples_leaf", 5)),
                        random_state=config.cv.random_state,
                        n_jobs=int(nar_config.get("n_jobs", -1)),
                    ),
                ),
                True,
            )
        )


def _add_signal_decomposition_models(
    specs: list[ModelSpec],
    models: dict[str, dict[str, object]],
    config: BaselineConfig,
) -> None:
    signal_config = models.get("signal_decomposition", {})
    if not signal_config.get("enabled", False):
        return

    alpha = float(signal_config.get("alpha", 10.0))
    lasso_alpha = float(signal_config.get("lasso_alpha", 0.01))
    n_score_columns = signal_config.get("n_score_columns")
    n_score_columns = 11 if n_score_columns is None else int(n_score_columns)
    n_components = int(signal_config.get("pca_components", 50))
    n_estimators = int(signal_config.get("n_estimators", 300))
    min_samples_leaf = int(signal_config.get("min_samples_leaf", 5))
    n_jobs = int(signal_config.get("n_jobs", -1))

    if signal_config.get("score_ridge", True):
        specs.append(
            ModelSpec(
                "nuisance_score_ridge",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    Ridge(alpha=alpha),
                ),
                True,
            )
        )
    if signal_config.get("resid_pca_ridge", True):
        specs.append(
            ModelSpec(
                f"nuisance_resid_pca{n_components}_ridge",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    NuisanceResidualizer(n_score_columns=n_score_columns),
                    StandardScaler(),
                    PCA(n_components=n_components, random_state=config.cv.random_state),
                    Ridge(alpha=alpha),
                ),
                True,
            )
        )
    if signal_config.get("resid_pca_random_forest", True):
        specs.append(
            ModelSpec(
                f"nuisance_resid_pca{n_components}_random_forest",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    NuisanceResidualizer(n_score_columns=n_score_columns),
                    StandardScaler(),
                    PCA(n_components=n_components, random_state=config.cv.random_state),
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        min_samples_leaf=min_samples_leaf,
                        random_state=config.cv.random_state,
                        n_jobs=n_jobs,
                    ),
                ),
                True,
            )
        )
    if signal_config.get("resid_pca_plus_scores_ridge", True):
        specs.append(
            ModelSpec(
                f"nuisance_resid_pca{n_components}_plus_scores_ridge",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    ResidualizedPCAWithScores(
                        n_score_columns=n_score_columns,
                        n_components=n_components,
                        random_state=config.cv.random_state,
                    ),
                    StandardScaler(),
                    Ridge(alpha=alpha),
                ),
                True,
            )
        )
    if signal_config.get("resid_pca_plus_scores_random_forest", True):
        specs.append(
            ModelSpec(
                f"nuisance_resid_pca{n_components}_plus_scores_random_forest",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    ResidualizedPCAWithScores(
                        n_score_columns=n_score_columns,
                        n_components=n_components,
                        random_state=config.cv.random_state,
                    ),
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        min_samples_leaf=min_samples_leaf,
                        random_state=config.cv.random_state,
                        n_jobs=n_jobs,
                    ),
                ),
                True,
            )
        )
    if signal_config.get("resid_lasso", True):
        specs.append(
            ModelSpec(
                "nuisance_resid_lasso",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    NuisanceResidualizer(n_score_columns=n_score_columns),
                    StandardScaler(),
                    Lasso(
                        alpha=lasso_alpha,
                        max_iter=20000,
                        tol=1e-3,
                        selection="random",
                        random_state=config.cv.random_state,
                    ),
                ),
                True,
            )
        )
    if signal_config.get("program_score_ridge", True):
        specs.append(
            ModelSpec(
                "program_score_ridge",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    Ridge(alpha=alpha),
                ),
                True,
            )
        )
    if signal_config.get("program_score_elastic_net", True):
        specs.append(
            ModelSpec(
                "program_score_elastic_net",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    ElasticNet(
                        alpha=float(signal_config.get("elastic_alpha", 0.02)),
                        l1_ratio=float(signal_config.get("elastic_l1_ratio", 0.5)),
                        max_iter=20000,
                        tol=1e-3,
                        selection="random",
                        random_state=config.cv.random_state,
                    ),
                ),
                True,
            )
        )
    if signal_config.get("program_score_random_forest", True):
        specs.append(
            ModelSpec(
                "program_score_random_forest",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        min_samples_leaf=min_samples_leaf,
                        random_state=config.cv.random_state,
                        n_jobs=n_jobs,
                    ),
                ),
                True,
            )
        )


def _add_xgboost(
    specs: list[ModelSpec],
    models: dict[str, dict[str, object]],
    config: BaselineConfig,
) -> None:
    xgb_config = models.get("xgboost", {})
    if not xgb_config.get("enabled", config.cv.model_set != "quick"):
        return
    try:
        from xgboost import XGBRegressor

        variants = xgb_config.get("variants") or [{}]
        has_variants = bool(xgb_config.get("variants"))
        for variant in variants:
            max_depth = int(variant.get("max_depth", xgb_config.get("max_depth", 3)))
            learning_rate = float(
                variant.get("learning_rate", xgb_config.get("learning_rate", 0.03))
            )
            name = "xgboost"
            if has_variants:
                name = f"xgboost_depth{max_depth}_lr{_param_token(learning_rate)}"
            specs.append(
                ModelSpec(
                    name,
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        XGBRegressor(
                            n_estimators=int(xgb_config.get("n_estimators", 300)),
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=float(xgb_config.get("subsample", 0.8)),
                            colsample_bytree=float(
                                xgb_config.get("colsample_bytree", 0.8)
                            ),
                            objective=str(
                                xgb_config.get("objective", "reg:squarederror")
                            ),
                            random_state=config.cv.random_state,
                            n_jobs=int(xgb_config.get("n_jobs", 4)),
                        ),
                    ),
                    True,
                )
            )
    except ImportError:
        LOGGER.info("xgboost is enabled but not installed; skipping")


def _pca_component_count(model_name: str) -> int:
    match = re.fullmatch(r"pca(?P<n_components>\d+)_.+", model_name)
    if match is None:
        raise ValueError(f"Invalid PCA model name: {model_name}")
    return int(match.group("n_components"))
