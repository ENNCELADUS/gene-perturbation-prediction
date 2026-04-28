from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from src.scgpt import prepare as scgpt_prepare
from src.utils.data import get_condition_splits


def test_scgpt_prepare_generates_norman_condition_split_artifact(
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
        "run_config": {"stages": ["prepare"], "seed": 11},
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
                "strategy": "random_condition",
                "train_fraction": 0.6,
                "validation_fraction": 0.2,
                "test_fraction": 0.2,
            },
        },
        "model_config": {"model": "scgpt"},
    }

    result = scgpt_prepare.run(config)
    split = get_condition_splits(config)

    assert result["split_path"] == str(split_path)
    assert split_path.exists()
    assert len(split["train"]) == 3
    assert len(split["validation"]) == 1
    assert len(split["test"]) == 1
    assert "ctrl" not in set().union(*[set(values) for values in split.values()])
    assert sum(len(values) for values in split.values()) == 5
    assert len(set().union(*[set(values) for values in split.values()])) == 5
