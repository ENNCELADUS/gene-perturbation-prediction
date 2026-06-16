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
    assert set(fold_metrics["split_type"].unique()) == {"CV1"}
    assert set(fold_metrics["fold_id"].unique()) == {0, 1}
    assert {"split_type", "model", "fold_id", "metric", "value"}.issubset(
        fold_metrics.columns
    )
    metric_names = set(fold_metrics["metric"].unique())
    assert {"auroc", "aupr", "f1@0.5"}.issubset(metric_names)
    assert {"ndcg@2", "recall@2", "precision@2"}.issubset(metric_names)

    assert {"split_type", "model", "metric", "mean", "std"}.issubset(summary.columns)
    assert len(summary) == len(fold_metrics["metric"].unique()) * 3

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert "input_csv_sha256" in manifest
    assert "leakage_notes" in manifest
    assert "ranking_semantics" in manifest
    assert "model_c_f1_note" in manifest
    assert manifest["split_types"] == ["CV1"]
    assert manifest["seed"] == config.seed


def test_run_cv_preserves_all_cv_split_boundaries(
    synthetic_all_cv_benchmark_csv: Path, tmp_path: Path
) -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "all_cv_run"
    config = SLBaselineConfig(
        input_csv=synthetic_all_cv_benchmark_csv,
        output_dir=output_dir,
        split_types=("CV1", "CV2"),
        folds=(0,),
        ranking_k=(2, 4),
    )
    summary = run_cv(config)

    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    assert set(fold_metrics["split_type"].unique()) == {"CV1", "CV2"}
    assert (fold_metrics.groupby(["split_type", "fold_id"]).size() > 0).all()
    assert {"CV1", "CV2"} == set(summary["split_type"].unique())
    assert not summary.duplicated(["split_type", "model", "metric"]).any()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["split_types"] == ["CV1", "CV2"]


def test_run_cv_rejects_missing_requested_split_type(
    synthetic_all_cv_benchmark_csv: Path, tmp_path: Path
) -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    config = SLBaselineConfig(
        input_csv=synthetic_all_cv_benchmark_csv,
        output_dir=tmp_path / "missing_split_run",
        split_types=("CV3",),
        folds=(0,),
    )

    try:
        run_cv(config)
    except ValueError as error:
        assert "split_types" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing split_type")
