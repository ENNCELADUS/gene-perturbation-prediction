from __future__ import annotations

import numpy as np

from src.pca_knn.train import _fit_best_pca_knn


def test_pca_knn_selects_neighbors_by_validation_metric() -> None:
    X_train = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ],
        dtype=np.float32,
    )
    X_validation = np.asarray([[9.0, 0.0]], dtype=np.float32)
    gene_name_to_idx = {"A": 0, "B": 1, "C": 2}

    result = _fit_best_pca_knn(
        X_train=X_train,
        train_conditions=["A", "B", "C"],
        X_validation=X_validation,
        validation_conditions=["B"],
        gene_name_to_idx=gene_name_to_idx,
        n_genes=3,
        model_config={
            "n_components_grid": [1],
            "n_neighbors_grid": [2, 1],
            "standardize_grid": [False],
        },
        top_k_values=[1],
        selection_metric="relevant_hit@1",
        seed=7,
        disable_tqdm=True,
    )

    assert result["metadata"]["best_params"]["n_neighbors"] == 1
    assert result["metadata"]["best_score"] == 1.0
