"""Model factories and estimator fitting helpers."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
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
    _add_pca_ridge(specs, models, config)
    _add_random_forest(specs, models, config)
    _add_pca_random_forest(specs, models, config)
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
    specs.append(
        ModelSpec(
            "ridge",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                Ridge(alpha=float(ridge_config.get("alpha", 10.0))),
            ),
            True,
        )
    )


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


def _add_pca_ridge(
    specs: list[ModelSpec],
    models: dict[str, dict[str, object]],
    config: BaselineConfig,
) -> None:
    pca_config = models.get("pca_ridge", {})
    if not pca_config.get("enabled", config.cv.model_set != "quick"):
        return
    for n_components in pca_config.get("components", config.cv.pca_components):
        n_components = int(n_components)
        specs.append(
            ModelSpec(
                f"pca{n_components}_ridge",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    PCA(
                        n_components=n_components,
                        random_state=config.cv.random_state,
                    ),
                    Ridge(alpha=float(pca_config.get("alpha", 10.0))),
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
    for n_components in pca_forest_config.get("components", config.cv.pca_components):
        n_components = int(n_components)
        specs.append(
            ModelSpec(
                f"pca{n_components}_random_forest",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    PCA(
                        n_components=n_components,
                        random_state=config.cv.random_state,
                    ),
                    RandomForestRegressor(
                        n_estimators=int(
                            pca_forest_config.get("n_estimators", 300)
                        ),
                        min_samples_leaf=int(
                            pca_forest_config.get("min_samples_leaf", 5)
                        ),
                        random_state=config.cv.random_state,
                        n_jobs=int(pca_forest_config.get("n_jobs", -1)),
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

        specs.append(
            ModelSpec(
                "xgboost",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    XGBRegressor(
                        n_estimators=int(xgb_config.get("n_estimators", 300)),
                        max_depth=int(xgb_config.get("max_depth", 3)),
                        learning_rate=float(xgb_config.get("learning_rate", 0.03)),
                        subsample=float(xgb_config.get("subsample", 0.8)),
                        colsample_bytree=float(
                            xgb_config.get("colsample_bytree", 0.8)
                        ),
                        objective=str(xgb_config.get("objective", "reg:squarederror")),
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
