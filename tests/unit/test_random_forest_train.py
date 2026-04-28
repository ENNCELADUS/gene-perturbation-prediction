from __future__ import annotations

import numpy as np

from src.random_forest.train import _fit_random_forest_with_progress


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
