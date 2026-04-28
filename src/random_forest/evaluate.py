"""Evaluate random forest baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.utils.data import build_pseudobulk_matrices
from src.utils.metrics import compute_gene_metrics, target_indices_for_conditions


def run(config: dict) -> dict:
    """Predict gene scores and compute retrieval metrics."""
    data = build_pseudobulk_matrices(config)
    artifact = joblib.load(_checkpoint_path(config))
    X_test = data["matrices"]["test"]
    test_conditions = data["conditions"]["test"]
    if X_test.shape[0] == 0:
        raise ValueError(
            "Random forest evaluation requires at least one test condition"
        )

    raw_scores = artifact["model"].predict(X_test)
    scores = [raw_scores[row_idx] for row_idx in range(raw_scores.shape[0])]
    targets = target_indices_for_conditions(test_conditions, data["gene_name_to_idx"])
    top_k_values = config.get("evaluation_config", {}).get("top_k_values", [1, 5, 10])
    metrics = compute_gene_metrics(scores, targets, top_k_values)
    _write_json(config["run_config"].get("eval_log_path"), {"metrics": metrics})
    return {"metrics": metrics}


def _checkpoint_path(config: dict) -> Path:
    path = config["run_config"].get("load_checkpoint_path")
    if path:
        return Path(path)
    path = config["run_config"].get("save_checkpoint_path")
    if path:
        return Path(path)
    study_name = config["run_config"]["study_name"]
    return Path("model") / "random_forest" / study_name / "model.joblib"


def _write_json(path: str | None, payload: dict) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
