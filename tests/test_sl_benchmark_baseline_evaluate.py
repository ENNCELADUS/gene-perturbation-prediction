from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_run_cv_writes_outputs_and_aggregates(
    synthetic_benchmark_csv: Path, tmp_path: Path
) -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "run"
    config = SLBaselineConfig(
        input_csv=synthetic_benchmark_csv,
        output_dir=output_dir,
        folds=(0, 1),
        ranking_k=(2, 5),
    )
    summary = run_cv(config)

    assert (output_dir / "fold_metrics.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "manifest.json").exists()

    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    assert set(fold_metrics["model"].unique()) == {"A", "B", "C"}
    assert set(fold_metrics["fold_id"].unique()) == {0, 1}
    assert {"model", "fold_id", "metric", "value"}.issubset(fold_metrics.columns)
    metric_names = set(fold_metrics["metric"].unique())
    assert {"auroc", "aupr", "f1@0.5"}.issubset(metric_names)
    assert {"ndcg@2", "recall@2", "precision@2"}.issubset(metric_names)

    assert {"model", "metric", "mean", "std"}.issubset(summary.columns)
    assert len(summary) == len(fold_metrics["metric"].unique()) * 3

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert "input_csv_sha256" in manifest
    assert "leakage_notes" in manifest
    assert "ranking_semantics" in manifest
    assert manifest["seed"] == config.seed
