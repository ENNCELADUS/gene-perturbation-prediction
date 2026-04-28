"""Train PCA+kNN baseline artifacts."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.pca_knn.evaluate import _knn_scores
from src.utils.data import build_pseudobulk_matrices, build_single_cell_matrices
from src.utils.distributed import disable_tqdm
from src.utils.metrics import (
    build_label_matrix,
    compute_gene_metrics,
    target_indices_for_conditions,
)


def run(config: dict) -> dict:
    """Fit PCA on train condition profiles and persist nearest-neighbor labels."""
    with tqdm(
        total=4,
        desc="pca_knn train",
        unit="step",
        dynamic_ncols=True,
        disable=disable_tqdm(config),
    ) as progress:
        data = build_pseudobulk_matrices(config)
        progress.update()
        X_train = data["matrices"]["train"]
        train_conditions = data["conditions"]["train"]
        if X_train.shape[0] == 0:
            raise ValueError("PCA+kNN training requires at least one train condition")

        validation_data = build_single_cell_matrices(
            config,
            split_names=("validation",),
        )
        progress.update()

        search_config = _search_model_config(config)
        fit_result = _fit_best_pca_knn(
            X_train=X_train,
            train_conditions=train_conditions,
            X_validation=validation_data["matrices"].get(
                "validation",
                np.empty((0, X_train.shape[1]), dtype=np.float32),
            ),
            validation_conditions=validation_data["conditions"].get("validation", []),
            gene_name_to_idx=data["gene_name_to_idx"],
            n_genes=len(data["gene_names"]),
            model_config=search_config,
            top_k_values=config.get("evaluation_config", {}).get(
                "top_k_values",
                [1, 5, 10],
            ),
            selection_metric=_selection_metric(config),
            seed=config["run_config"].get("seed"),
            disable_tqdm=disable_tqdm(config),
        )
        progress.update()
        artifact = fit_result["artifact"]
        checkpoint_path = _checkpoint_path(config)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pca": artifact["pca"],
                "scaler": artifact["scaler"],
                "train_embeddings": artifact["train_embeddings"],
                "train_labels": artifact["train_labels"],
                "train_conditions": train_conditions,
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


def _fit_best_pca_knn(
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
    for params in _pca_knn_candidates(grid_config, X_train):
        artifact = _fit_pca_knn_artifact(
            X_train=X_train,
            labels=labels,
            params=params,
            seed=seed,
        )
        metrics: dict[str, float | int] = {}
        score = 0.0
        if use_validation:
            X_validation_proc = (
                artifact["scaler"].transform(X_validation)
                if artifact["scaler"] is not None
                else X_validation
            )
            validation_embeddings = artifact["pca"].transform(X_validation_proc)
            scores, _, _ = _knn_scores(
                train_embeddings=artifact["train_embeddings"],
                train_labels=artifact["train_labels"],
                query_embeddings=validation_embeddings,
                k=int(params["n_neighbors"]),
                disable_tqdm=disable_tqdm,
            )
            metrics = compute_gene_metrics(scores, validation_targets, top_k_values)
            score = _metric_score(metrics, selection_metric)

        candidate = {
            "params": params,
            "validation_metrics": metrics,
            "score": score,
        }
        candidates.append(candidate)
        if best_result is None or score > float(best_result["score"]):
            best_result = {"artifact": artifact, **candidate}

    if best_result is None:
        raise ValueError("PCA+kNN hyperparameter search produced no candidates")

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
    return {"artifact": best_result["artifact"], "metadata": metadata}


def _fit_pca_knn_artifact(
    X_train: np.ndarray,
    labels: np.ndarray,
    params: dict[str, object],
    seed: int | None,
) -> dict[str, object]:
    scaler = StandardScaler() if bool(params["standardize"]) else None
    X_proc = scaler.fit_transform(X_train) if scaler is not None else X_train
    pca = PCA(n_components=int(params["n_components"]), random_state=seed)
    train_embeddings = pca.fit_transform(X_proc)
    return {
        "pca": pca,
        "scaler": scaler,
        "train_embeddings": train_embeddings,
        "train_labels": labels,
    }


def _pca_knn_candidates(
    model_config: dict,
    X_train: np.ndarray,
) -> list[dict[str, object]]:
    max_components = min(X_train.shape[0], X_train.shape[1])
    if max_components < 1:
        raise ValueError("PCA+kNN requires at least one train row and feature")
    requested_components = _grid_values(
        model_config,
        "n_components_grid",
        model_config.get("n_components", 16),
    )
    requested_neighbors = _grid_values(
        model_config,
        "n_neighbors_grid",
        model_config.get("n_neighbors", 5),
    )
    standardize_values = _grid_values(
        model_config,
        "standardize_grid",
        model_config.get("standardize", False),
    )

    candidates = []
    seen = set()
    for requested_n_components, requested_n_neighbors, standardize in product(
        requested_components,
        requested_neighbors,
        standardize_values,
    ):
        params = {
            "n_components": min(int(requested_n_components), max_components),
            "n_neighbors": max(1, min(int(requested_n_neighbors), X_train.shape[0])),
            "standardize": bool(standardize),
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
    return Path("model") / "pca_knn" / study_name / "model.joblib"
