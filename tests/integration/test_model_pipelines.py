from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import pytest

from src import main


def write_tiny_adata(path: Path) -> None:
    gene_names = ["A", "B", "C", "D"]
    rows = [
        ("ctrl", 1, [1.0, 1.0, 1.0, 1.0]),
        ("ctrl", 1, [1.1, 0.9, 1.0, 1.0]),
        ("A+ctrl", 0, [5.0, 1.0, 1.0, 1.0]),
        ("A+ctrl", 0, [4.8, 1.2, 1.0, 1.0]),
        ("B+ctrl", 0, [1.0, 5.0, 1.0, 1.0]),
        ("B+ctrl", 0, [1.1, 4.7, 1.0, 1.0]),
        ("A+B", 0, [4.5, 4.4, 1.0, 1.0]),
        ("A+B", 0, [4.6, 4.3, 1.1, 1.0]),
    ]
    obs = pd.DataFrame(
        {
            "condition": [row[0] for row in rows],
            "control": [row[1] for row in rows],
        }
    )
    var = pd.DataFrame({"gene_name": gene_names}, index=gene_names)
    adata = ad.AnnData(
        X=np.asarray([row[2] for row in rows], dtype=np.float32),
        obs=obs,
        var=var,
    )
    adata.write_h5ad(path)


def baseline_config(tmp_path: Path, model: str) -> dict:
    h5ad_path = tmp_path / "tiny.h5ad"
    artifact_dir = tmp_path / model
    return {
        "run_config": {
            "stages": ["prepare", "train", "evaluate"],
            "seed": 7,
            "study_name": "test",
            "save_checkpoint_path": str(artifact_dir / "model.joblib"),
            "eval_log_path": str(artifact_dir / "eval.json"),
        },
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {
            "h5ad_path": str(h5ad_path),
            "condition_key": "condition",
            "control_key": "control",
            "condition_split": {
                "train": ["A"],
                "validation": ["B"],
                "test": ["A+B"],
            },
        },
        "model_config": {
            "model": model,
            "n_components": 2,
            "n_neighbors": 1,
            "standardize": False,
            "n_estimators": 4,
            "max_depth": 3,
        },
        "evaluation_config": {
            "top_k_values": [1, 2],
            "diagnostics": {"enabled": True, "top_n_predictions": 2},
        },
    }


def test_pca_knn_full_pipeline_runs_from_config(tmp_path: Path) -> None:
    write_tiny_adata(tmp_path / "tiny.h5ad")

    results = main.run_from_config(baseline_config(tmp_path, "pca_knn"))

    assert [stage["stage"] for stage in results["stages"]] == [
        "prepare",
        "train",
        "evaluate",
    ]
    assert results["stages"][-1]["metrics"]["n_queries"] == 2
    log_payload = json.loads(
        Path(baseline_config(tmp_path, "pca_knn")["run_config"]["eval_log_path"])
        .read_text()
    )
    assert log_payload["diagnostics"]["summary"]["n_queries"] == 2
    per_query = log_payload["diagnostics"]["per_query"]
    assert [query["condition"] for query in per_query] == ["A+B", "A+B"]
    assert [query["cell_index"] for query in per_query] == [6, 7]
    assert "nearest_neighbors" in log_payload["diagnostics"]["per_query"][0]


def test_random_forest_full_pipeline_runs_from_config(tmp_path: Path) -> None:
    write_tiny_adata(tmp_path / "tiny.h5ad")

    results = main.run_from_config(baseline_config(tmp_path, "random_forest"))

    assert [stage["stage"] for stage in results["stages"]] == [
        "prepare",
        "train",
        "evaluate",
    ]
    assert results["stages"][-1]["metrics"]["n_queries"] == 2
    log_payload = json.loads(
        Path(baseline_config(tmp_path, "random_forest")["run_config"]["eval_log_path"])
        .read_text()
    )
    assert log_payload["diagnostics"]["summary"]["n_queries"] == 2
    per_query = log_payload["diagnostics"]["per_query"]
    assert [query["condition"] for query in per_query] == ["A+B", "A+B"]
    assert [query["cell_index"] for query in per_query] == [6, 7]
    assert "nearest_neighbors" not in log_payload["diagnostics"]["per_query"][0]


@pytest.mark.parametrize("model", ["pca_knn", "random_forest"])
def test_baseline_evaluate_rejects_checkpoint_gene_order_mismatch(
    tmp_path: Path,
    model: str,
) -> None:
    write_tiny_adata(tmp_path / "tiny.h5ad")
    config = baseline_config(tmp_path, model)
    config["run_config"]["stages"] = ["train"]
    main.run_from_config(config)

    checkpoint_path = Path(config["run_config"]["save_checkpoint_path"])
    artifact = joblib.load(checkpoint_path)
    artifact["gene_names"] = list(reversed(artifact["gene_names"]))
    joblib.dump(artifact, checkpoint_path)

    config["run_config"]["stages"] = ["evaluate"]
    with pytest.raises(ValueError, match="gene order"):
        main.run_from_config(config)
