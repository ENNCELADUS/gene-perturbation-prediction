"""The prior export: reference row on validation and test, out of fold on training."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from src.context_prior.folds import training_side_folds
from src.experiments import context_prior as run
from src.experiments import prior_export as export
from src.experiments.config import validate_prior_config
from tests.test_context_prior_run import synthetic_base, tiny_config


def selected_config(tmp_path):
    config = tiny_config(tmp_path)
    config["reference"] = {
        "experiment": "affine",
        "block_set": "selected",
        "components_penalty": 1.0,
        "gene_penalty": 1.0,
    }
    return validate_prior_config(config)


def test_training_side_folds_follow_patients():
    folds = training_side_folds(
        {"S0": 0, "S1": 1},
        ["S0", "S1", "E0", "E1", "E2"],
        {"S0": "P0", "S1": "P1", "E0": "P1", "E1": "P9", "E2": float("nan")},
    )
    assert folds == {"S0": 0, "S1": 1, "E0": 1, "E1": -1, "E2": -1}


def test_validation_and_test_rows_are_the_reference_row(tmp_path):
    base = synthetic_base(selected_config(tmp_path))
    predictions = export.export_predictions(base)
    reference = run.reference_predictions(base)
    for split in ("val", "test"):
        np.testing.assert_allclose(predictions[split], reference[split])
    assert list(predictions["train"].index) == list(base.bridge.single_cell_train)
    assert list(predictions["train"].columns) == list(base.definitions.genes)


def test_out_of_fold_rows_ignore_their_own_labels(tmp_path):
    base = synthetic_base(selected_config(tmp_path))
    before = export.export_predictions(base)
    held = [m for m, fold in base.bridge.folds.items() if fold == 0]
    base.gene_effect.loc[held] = base.gene_effect.loc[held] + 3.0
    after = export.export_predictions(base)
    np.testing.assert_allclose(after["train"].loc[held], before["train"].loc[held])
    others = [m for m in base.bridge.single_cell_train if m not in held]
    assert not np.allclose(after["train"].loc[others], before["train"].loc[others])
    assert not np.allclose(after["val"], before["val"])


def test_export_refuses_a_bridge_it_cannot_refit(tmp_path):
    config = selected_config(tmp_path)
    config["experiments"]["affine"]["kind"] = "gating"
    with pytest.raises(ValueError, match="affine bridge only"):
        export.export_predictions(synthetic_base(config))


def test_main_writes_the_export_once(tmp_path, monkeypatch):
    config = selected_config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(export, "load_base", synthetic_base)
    assert export.main([str(path), "--run-id", "r"]) == 0
    out = tmp_path / "out" / "r" / "export"
    record = json.loads((out / "prior.json").read_text())
    assert record["run_id"] == "r"
    assert record["reference"]["block_set"] == "selected"
    assert record["units"] == "residual SD"
    with np.load(out / "prior.npz") as payload:
        lines = [str(m) for m in payload["lines"]]
        assert payload["values"].shape == (len(lines), len(payload["genes"]))
        assert payload["values"].dtype == np.float32
    assert lines == [
        *record["lines"]["train"],
        *record["lines"]["val"],
        *record["lines"]["test"],
    ]
    assert set(record["scores"]) == {"train", "val", "test"}
    with pytest.raises(FileExistsError, match="new --run-id"):
        export.main([str(path), "--run-id", "r"])
