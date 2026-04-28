"""Evaluate random forest baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from tqdm.auto import tqdm

from src.utils.data import build_pseudobulk_matrices, get_gene_splits
from src.utils.metrics import (
    build_gene_ranking_diagnostics,
    compute_gene_metrics,
    target_indices_for_conditions,
)


def run(config: dict) -> dict:
    """Predict gene scores and compute retrieval metrics."""
    with tqdm(
        total=5,
        desc="random_forest evaluate",
        unit="step",
        dynamic_ncols=True,
        disable=_disable_tqdm(config),
    ) as progress:
        data = build_pseudobulk_matrices(config)
        progress.update()
        artifact = joblib.load(_checkpoint_path(config))
        progress.update()
        X_test = data["matrices"]["test"]
        test_conditions = data["conditions"]["test"]
        if X_test.shape[0] == 0:
            raise ValueError(
                "Random forest evaluation requires at least one test condition"
            )

        raw_scores = artifact["model"].predict(X_test)
        progress.update()
        scores = [
            raw_scores[row_idx]
            for row_idx in tqdm(
                range(raw_scores.shape[0]),
                desc="random_forest query",
                unit="query",
                dynamic_ncols=True,
                disable=_disable_tqdm(config),
            )
        ]
        targets = target_indices_for_conditions(
            test_conditions,
            data["gene_name_to_idx"],
        )
        top_k_values = config.get("evaluation_config", {}).get(
            "top_k_values",
            [1, 5, 10],
        )
        metrics = compute_gene_metrics(scores, targets, top_k_values)
        payload: dict[str, object] = {"metrics": metrics}
        if _diagnostics_enabled(config):
            payload["diagnostics"] = build_gene_ranking_diagnostics(
                scores=scores,
                targets=targets,
                gene_names=data["gene_names"],
                conditions=test_conditions,
                top_k_values=top_k_values,
                top_n_predictions=_top_n_predictions(config),
                split_genes=get_gene_splits(config),
            )
        progress.update()
        _write_json(config["run_config"].get("eval_log_path"), payload)
        progress.update()
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


def _diagnostics_enabled(config: dict) -> bool:
    diagnostics = config.get("evaluation_config", {}).get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        return False
    return bool(diagnostics.get("enabled", False))


def _top_n_predictions(config: dict) -> int:
    diagnostics = config.get("evaluation_config", {}).get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        return 10
    return int(diagnostics.get("top_n_predictions", 10))


def _disable_tqdm(config: dict) -> bool:
    return bool(config["run_config"].get("disable_tqdm", False))
