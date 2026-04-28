"""Evaluate PCA+kNN baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.distance import cdist

from src.utils.data import build_pseudobulk_matrices
from src.utils.metrics import compute_gene_metrics, target_indices_for_conditions


def run(config: dict) -> dict:
    """Rank genes by nearest-neighbor label aggregation."""
    data = build_pseudobulk_matrices(config)
    artifact = joblib.load(_checkpoint_path(config))
    X_test = data["matrices"]["test"]
    test_conditions = data["conditions"]["test"]
    if X_test.shape[0] == 0:
        raise ValueError("PCA+kNN evaluation requires at least one test condition")

    scaler = artifact["scaler"]
    pca = artifact["pca"]
    X_proc = scaler.transform(X_test) if scaler is not None else X_test
    query_embeddings = pca.transform(X_proc)
    scores = _knn_scores(
        train_embeddings=artifact["train_embeddings"],
        train_labels=artifact["train_labels"],
        query_embeddings=query_embeddings,
        k=int(config["model_config"].get("n_neighbors", 5)),
    )
    targets = target_indices_for_conditions(test_conditions, data["gene_name_to_idx"])
    top_k_values = config.get("evaluation_config", {}).get("top_k_values", [1, 5, 10])
    metrics = compute_gene_metrics(scores, targets, top_k_values)
    _write_json(config["run_config"].get("eval_log_path"), {"metrics": metrics})
    return {"metrics": metrics}


def _knn_scores(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
) -> list[np.ndarray]:
    k = max(1, min(k, train_embeddings.shape[0]))
    distances = cdist(query_embeddings, train_embeddings, metric="euclidean")
    neighbors = np.argsort(distances, axis=1)[:, :k]
    return [train_labels[row].mean(axis=0) for row in neighbors]


def _checkpoint_path(config: dict) -> Path:
    path = config["run_config"].get("load_checkpoint_path")
    if path:
        return Path(path)
    path = config["run_config"].get("save_checkpoint_path")
    if path:
        return Path(path)
    return Path("results/pca_knn/model.joblib")


def _write_json(path: str | None, payload: dict) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
