from __future__ import annotations

import numpy as np

from src.random_forest.train import (
    _fit_best_random_forest,
    _fit_random_forest_with_progress,
)


def test_random_forest_train_passes_regularization_config() -> None:
    X_train = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    model = _fit_random_forest_with_progress(
        X_train=X_train,
        labels=labels,
        model_config={
            "n_estimators": 3,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "estimator_chunk_size": 2,
        },
        seed=13,
        disable_tqdm=True,
    )

    assert model.n_estimators == 3
    assert model.max_depth == 8
    assert model.min_samples_leaf == 2
    assert model.n_jobs == -1


def test_random_forest_selects_params_by_validation_metric() -> None:
    X_train = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    X_validation = np.asarray([[0.1], [2.9]], dtype=np.float32)
    gene_name_to_idx = {"A": 0, "B": 1}

    result = _fit_best_random_forest(
        X_train=X_train,
        train_conditions=["A", "A", "B", "B"],
        X_validation=X_validation,
        validation_conditions=["A", "B"],
        gene_name_to_idx=gene_name_to_idx,
        n_genes=2,
        model_config={
            "n_estimators_grid": [20],
            "max_depth_grid": [1],
            "min_samples_leaf_grid": [3, 1],
            "max_features_grid": [1.0],
            "n_jobs": 1,
            "estimator_chunk_size": 20,
        },
        top_k_values=[1],
        selection_metric="relevant_hit@1",
        seed=13,
        disable_tqdm=True,
    )

    assert result["metadata"]["best_params"]["min_samples_leaf"] == 1
    assert result["metadata"]["best_score"] == 1.0
