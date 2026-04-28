from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from src.scgpt import prepare as scgpt_prepare
from src.utils.data import get_condition_splits


def test_scgpt_prepare_generates_gene_heldout_condition_split_artifact(
    tmp_path: Path,
) -> None:
    h5ad_path = tmp_path / "norman.h5ad"
    split_path = tmp_path / "norman_condition_split.yaml"
    rows = [
        ("ctrl", 1),
        ("A+ctrl", 0),
        ("B+ctrl", 0),
        ("C+ctrl", 0),
        ("D+ctrl", 0),
        ("E+ctrl", 0),
        ("F+ctrl", 0),
        ("G+ctrl", 0),
        ("H+ctrl", 0),
        ("I+ctrl", 0),
        ("J+ctrl", 0),
        ("A+B", 0),
        ("B+C", 0),
        ("C+D", 0),
        ("D+E", 0),
        ("E+F", 0),
        ("F+G", 0),
        ("G+H", 0),
        ("H+I", 0),
        ("I+J", 0),
        ("A+J", 0),
    ]
    adata = ad.AnnData(
        X=np.ones((len(rows), 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "condition": [condition for condition, _ in rows],
                "control": [control for _, control in rows],
            }
        ),
        var=pd.DataFrame({"gene_name": ["A", "B"]}),
    )
    adata.write_h5ad(h5ad_path)
    config = {
        "run_config": {"stages": ["prepare"], "seed": 11, "study_name": "test"},
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {
            "h5ad_path": str(h5ad_path),
            "condition_key": "condition",
            "condition_split_path": str(split_path),
            "condition_split": {"train": [], "validation": [], "test": []},
            "split_config": {
                "strategy": "gene_heldout",
                "train_gene_fraction": 0.7,
                "validation_gene_fraction": 0.1,
                "test_gene_fraction": 0.2,
            },
        },
        "model_config": {"model": "scgpt"},
    }

    result = scgpt_prepare.run(config)
    split = get_condition_splits(config)

    assert result["split_path"] == str(split_path)
    assert split_path.exists()
    assert result["n_train_genes"] == 7
    assert result["n_validation_genes"] == 1
    assert result["n_test_genes"] == 2
    assert "ctrl" not in set().union(*[set(values) for values in split.values()])
    assert sum(len(values) for values in split.values()) == 20
    assert len(set().union(*[set(values) for values in split.values()])) == 20

    payload = yaml.safe_load(split_path.read_text())
    assert payload["strategy"] == "gene_heldout"
    train_genes = set(payload["genes"]["train"])
    validation_genes = set(payload["genes"]["validation"])
    test_genes = set(payload["genes"]["test"])
    assert len(train_genes) == 7
    assert len(validation_genes) == 1
    assert len(test_genes) == 2
    assert not train_genes & validation_genes
    assert not train_genes & test_genes
    assert not validation_genes & test_genes

    for condition in split["train"]:
        assert set(condition.split("+")) <= train_genes
    for condition in split["validation"]:
        genes = set(condition.split("+"))
        assert genes & validation_genes
        assert not genes & test_genes
    for condition in split["test"]:
        assert set(condition.split("+")) & test_genes

    fractions = payload["stats"]["condition_fractions"]
    deltas = payload["stats"]["condition_fraction_deltas"]
    assert fractions["train"] == pytest.approx(0.65)
    assert fractions["validation"] == pytest.approx(0.10)
    assert fractions["test"] == pytest.approx(0.25)
    assert max(abs(value) for value in deltas.values()) <= 0.05


def test_scgpt_prepare_rejects_random_condition_strategy(tmp_path: Path) -> None:
    h5ad_path = tmp_path / "norman.h5ad"
    adata = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "condition": ["ctrl", "A+ctrl"],
                "control": [1, 0],
            }
        ),
        var=pd.DataFrame({"gene_name": ["A"]}),
    )
    adata.write_h5ad(h5ad_path)
    config = {
        "run_config": {"stages": ["prepare"], "seed": 11, "study_name": "test"},
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {
            "h5ad_path": str(h5ad_path),
            "condition_key": "condition",
            "condition_split_path": str(tmp_path / "split.yaml"),
            "condition_split": {"train": [], "validation": [], "test": []},
            "split_config": {"strategy": "random_condition"},
        },
        "model_config": {"model": "scgpt"},
    }

    with pytest.raises(ValueError, match="gene_heldout"):
        scgpt_prepare.run(config)
