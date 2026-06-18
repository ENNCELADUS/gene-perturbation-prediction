# tests/test_ddgcn_evaluate.py
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd

from ddgcn.config import DdgcnConfig


def _toy_csv(path: Path) -> None:
    rows = [
        ("CV1", 0, "train", 1, 0, 1, "G0", "G1", -0.5, -0.4),
        ("CV1", 0, "train", 1, 2, 3, "G2", "G3", -0.6, -0.3),
        ("CV1", 0, "train", 0, 0, 4, "G0", "G4", -0.5, 0.1),
        ("CV1", 0, "train", 0, 1, 5, "G1", "G5", -0.4, 0.2),
        ("CV1", 0, "test", 1, 0, 2, "G0", "G2", -0.5, -0.6),
        ("CV1", 0, "test", 0, 3, 5, "G3", "G5", -0.3, 0.2),
    ]
    cols = [
        "split_type", "fold_id", "split_role", "sl_label",
        "gene_a_unified_id", "gene_b_unified_id",
        "gene_a_symbol", "gene_b_symbol",
        "gene_a_k562_gene_effect", "gene_b_k562_gene_effect",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_run_cv_writes_artifacts(tmp_path: Path) -> None:
    from ddgcn.evaluate import run_cv

    csv = tmp_path / "bench.csv"
    _toy_csv(csv)
    out = tmp_path / "run"
    cfg = dataclasses.replace(
        DdgcnConfig(),
        input_csv=csv,
        output_dir=out,
        split_types=("CV1",),
        folds=(0,),
        hidden1=8,
        hidden2=4,
        max_epochs=15,
        tolerance_epoch=2,
        eval_interval=5,
    )
    summary = run_cv(cfg)

    assert (out / "fold_metrics.csv").exists()
    assert (out / "summary.csv").exists()
    assert (out / "manifest.json").exists()
    assert (out / "official_metrics_summary.csv").exists()
    assert (out / "CV1" / "fold_metrics.csv").exists()
    assert (out / "CV1" / "summary.csv").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["candidate_gene_count"] == 6
    assert manifest["split_types"] == ["CV1"]
    assert manifest["dropout"] == 0.5
    assert manifest["lr"] == 0.01
    assert "input_csv_sha256" in manifest
    assert "train_edge_counts" in manifest

    assert set(summary.columns) == {
        "split_type", "model", "slice", "metric", "mean", "std"
    }
    assert (summary["model"] == "ddgcn").all()
