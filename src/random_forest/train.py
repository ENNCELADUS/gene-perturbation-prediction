"""Train random forest baseline."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm

from src.utils.data import build_pseudobulk_matrices, build_single_cell_matrices
from src.utils.distributed import disable_tqdm
from src.utils.metrics import (
    build_label_matrix,
    compute_gene_metrics,
    target_indices_for_conditions,
)


def run(config: dict) -> dict:
    """Fit a multi-output random forest gene scorer."""
    with tqdm(
        total=4,
        desc="random_forest train",
        unit="step",
        dynamic_ncols=True,
        disable=disable_tqdm(config),
    ) as progress:
        data = build_pseudobulk_matrices(config)
        progress.update()
        X_train = data["matrices"]["train"]
        train_conditions = data["conditions"]["train"]
        if X_train.shape[0] == 0:
            raise ValueError(
                "Random forest training requires at least one train condition"
            )

        validation_data = build_single_cell_matrices(
            config,
            split_names=("validation",),
        )
        progress.update()

        fit_result = _fit_best_random_forest(
            X_train=X_train,
            train_conditions=train_conditions,
            X_validation=validation_data["matrices"].get(
                "validation",
                np.empty((0, X_train.shape[1]), dtype=np.float32),
            ),
            validation_conditions=validation_data["conditions"].get("validation", []),
            gene_name_to_idx=data["gene_name_to_idx"],
            n_genes=len(data["gene_names"]),
            model_config=_search_model_config(config),
            top_k_values=config.get("evaluation_config", {}).get(
                "top_k_values",
                [1, 5, 10],
            ),
            selection_metric=_selection_metric(config),
            seed=config["run_config"].get("seed"),
            disable_tqdm=disable_tqdm(config),
        )
        progress.update()
        checkpoint_path = _checkpoint_path(config)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": fit_result["model"],
                "gene_names": data["gene_names"],
                "model_selection": fit_result["metadata"],
            },
            checkpoint_path,
        )
        progress.update()
    return {
        "checkpoint_path": str(checkpoint_path),
        "n_train": X_train.shape[0],
        "model_selection": fit_result["metadata"],
    }


def _fit_best_random_forest(
    X_train: np.ndarray,
    train_conditions: Sequence[str],
    X_validation: np.ndarray,
    validation_conditions: Sequence[str],
    gene_name_to_idx: dict[str, int],
    n_genes: int,
    model_config: dict,
    top_k_values: Sequence[int],
    selection_metric: str,
    seed: int | None,
    disable_tqdm: bool,
) -> dict[str, object]:
    labels = build_label_matrix(train_conditions, gene_name_to_idx, n_genes)
    validation_targets = target_indices_for_conditions(
        validation_conditions,
        gene_name_to_idx,
    )
    use_validation = X_validation.shape[0] > 0 and any(validation_targets)
    best_result: dict[str, object] | None = None
    candidates: list[dict[str, object]] = []

    grid_config = model_config if use_validation else _without_grid_values(model_config)
    for params in _random_forest_candidates(grid_config):
        candidate_config = {**model_config, **params}
        model = _fit_random_forest_with_progress(
            X_train=X_train,
            labels=labels,
            model_config=candidate_config,
            seed=seed,
            disable_tqdm=disable_tqdm,
        )
        metrics: dict[str, float | int] = {}
        score = 0.0
        if use_validation:
            raw_scores = model.predict(X_validation)
            scores = [raw_scores[row_idx] for row_idx in range(raw_scores.shape[0])]
            metrics = compute_gene_metrics(scores, validation_targets, top_k_values)
            score = _metric_score(metrics, selection_metric)

        candidate = {
            "params": params,
            "validation_metrics": metrics,
            "score": score,
        }
        candidates.append(candidate)
        if best_result is None or score > float(best_result["score"]):
            best_result = {"model": model, **candidate}

    if best_result is None:
        raise ValueError("Random forest hyperparameter search produced no candidates")

    metadata = {
        "enabled": use_validation,
        "selection_metric": selection_metric,
        "best_params": best_result["params"],
        "best_score": best_result["score"],
        "validation_metrics": best_result["validation_metrics"],
        "candidates": candidates,
    }
    if not use_validation:
        metadata["skip_reason"] = "validation split is empty or has no target genes"
    return {"model": best_result["model"], "metadata": metadata}


def _fit_random_forest_with_progress(
    X_train: np.ndarray,
    labels: np.ndarray,
    model_config: dict,
    seed: int | None,
    disable_tqdm: bool,
) -> RandomForestRegressor:
    n_estimators = int(model_config.get("n_estimators", 300))
    if n_estimators < 1:
        raise ValueError("Random forest n_estimators must be at least 1")
    estimator_chunk_size = max(1, int(model_config.get("estimator_chunk_size", 1)))
    model = RandomForestRegressor(
        n_estimators=0,
        max_depth=model_config.get("max_depth"),
        min_samples_leaf=int(model_config.get("min_samples_leaf", 1)),
        max_features=model_config.get("max_features", 1.0),
        random_state=seed,
        n_jobs=int(model_config.get("n_jobs", 1)),
        warm_start=True,
    )
    with tqdm(
        total=n_estimators,
        desc="random_forest trees",
        unit="tree",
        dynamic_ncols=True,
        disable=disable_tqdm,
    ) as progress:
        fitted_estimators = 0
        while fitted_estimators < n_estimators:
            next_estimators = min(
                fitted_estimators + estimator_chunk_size,
                n_estimators,
            )
            model.set_params(n_estimators=next_estimators)
            model.fit(X_train, labels)
            progress.update(next_estimators - fitted_estimators)
            fitted_estimators = next_estimators
    return model


def _random_forest_candidates(model_config: dict) -> list[dict[str, object]]:
    candidates = []
    seen = set()
    for n_estimators, max_depth, min_samples_leaf, max_features in product(
        _grid_values(
            model_config,
            "n_estimators_grid",
            model_config.get("n_estimators", 300),
        ),
        _grid_values(
            model_config,
            "max_depth_grid",
            model_config.get("max_depth"),
        ),
        _grid_values(
            model_config,
            "min_samples_leaf_grid",
            model_config.get("min_samples_leaf", 1),
        ),
        _grid_values(
            model_config,
            "max_features_grid",
            model_config.get("max_features", 1.0),
        ),
    ):
        params = {
            "n_estimators": int(n_estimators),
            "max_depth": None if max_depth is None else int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": max_features,
        }
        key = tuple(params.items())
        if key in seen:
            continue
        candidates.append(params)
        seen.add(key)
    return candidates


def _search_model_config(config: dict) -> dict:
    model_config = dict(config["model_config"])
    search_config = (
        config.get("training_config", {}).get("hyperparameter_search", {})
        if isinstance(config.get("training_config", {}), dict)
        else {}
    )
    if not isinstance(search_config, dict) or not search_config.get("enabled", False):
        return model_config
    return {**model_config, **search_config}


def _without_grid_values(model_config: dict) -> dict:
    return {
        key: value
        for key, value in model_config.items()
        if not str(key).endswith("_grid")
    }


def _selection_metric(config: dict) -> str:
    training_config = config.get("training_config", {})
    if not isinstance(training_config, dict):
        return "mrr"
    search_config = training_config.get("hyperparameter_search", {})
    if not isinstance(search_config, dict):
        return "mrr"
    return str(search_config.get("selection_metric", "mrr"))


def _metric_score(metrics: dict[str, float | int], selection_metric: str) -> float:
    if selection_metric not in metrics:
        raise ValueError(
            f"selection metric {selection_metric!r} is not available in validation "
            f"metrics: {sorted(metrics)}"
        )
    return float(metrics[selection_metric])


def _grid_values(config: dict, grid_key: str, fallback: object) -> list[object]:
    values = config.get(grid_key)
    if values is None:
        return [fallback]
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise ValueError(f"model_config.{grid_key} must be a list")
    if not values:
        raise ValueError(f"model_config.{grid_key} must not be empty")
    return list(values)


def _checkpoint_path(config: dict) -> Path:
    path = config["run_config"].get("save_checkpoint_path")
    if path:
        return Path(path)
    study_name = config["run_config"]["study_name"]
    return Path("model") / "random_forest" / study_name / "model.joblib"
