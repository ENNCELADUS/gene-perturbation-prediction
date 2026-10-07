"""The data-selected analysis on the runner's synthetic lines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.experiments import context_prior as run
from src.experiments import data_selected_analysis as analysis
from src.experiments.config import validate_prior_config
from tests.test_context_prior_run import synthetic_base, tiny_config


def gene_level_config(tmp_path, block_set="all"):
    config = tiny_config(tmp_path)
    config["reference"] = {
        "experiment": "affine",
        "block_set": block_set,
        "components_penalty": 1.0,
        "gene_penalty": 1.0,
    }
    return validate_prior_config(config)


def test_masked_stage_keeps_or_strips_weights():
    rng = np.random.default_rng(0)
    from src.context_prior.ridge import SelectedFit

    model = SelectedFit(
        selection=np.array([[0, 2], [1, 3]]),
        coef=rng.normal(size=(2, 2)),
        intercept=np.array([0.5, -0.5]),
    )
    x = rng.normal(size=(5, 4))
    everything = analysis.masked(model, np.ones(4, dtype=bool))
    assert np.allclose(everything.predict(x), model.predict(x))
    nothing = analysis.masked(model, np.zeros(4, dtype=bool))
    assert np.allclose(nothing.predict(x), np.broadcast_to(model.intercept, (5, 2)))
    first = analysis.masked(model, np.array([True, False, False, False]))
    assert np.allclose(first.predict(x)[:, 0], model.coef[0, 0] * x[:, 0] + 0.5)


def test_variance_explained_by_groups_and_design():
    values = np.array([[1.0, 0.0], [1.0, 1.0], [3.0, 0.0], [3.0, 1.0]])
    groups = np.array(["a", "a", "b", "b"])
    assert np.allclose(analysis.variance_explained(values, groups=groups), [1.0, 0.0])
    design = np.array([[0.0], [1.0], [0.0], [1.0]])
    assert np.allclose(analysis.variance_explained(values, design=design), [0.0, 1.0])
    constant = analysis.variance_explained(np.ones((4, 1)), groups=groups)
    assert np.isnan(constant).all()


def test_analysis_matches_the_reference_row(tmp_path):
    config = gene_level_config(tmp_path)
    base = synthetic_base(config)
    result = analysis.analyse(base)
    reference = run.reference_predictions(base)
    scores = result["ablation"]["reference"]
    for split in ("val", "test"):
        expected = run.score(
            reference[split], base.truth(list(reference[split].index)), base.definitions
        )
        assert scores[split]["selective_spearman"] == pytest.approx(
            expected["selective_spearman"]
        )
    features = result["features"]
    count = config["prior"]["selected_genes"]
    selective = len(base.definitions.selective)
    assert features["times_selected"].sum() == count * selective
    assert features["summed_abs_weight"].is_monotonic_decreasing
    targets = result["targets"]
    assert list(targets["gene"]) == list(base.definitions.selective)
    for split in ("val", "test"):
        assert np.allclose(
            targets[f"{split}_gain"],
            targets[f"{split}_spearman"] - targets[f"{split}_without_stage"],
            equal_nan=True,
        )
    variants = result["ablation"]["variants"]
    assert "without the stage" in variants
    # Ten features kept of twelve; the larger counts exceed the space.
    assert set(variants) == {"without the stage", "top 10 only", "without top 10"}
    spread = analysis.concentration(targets)
    assert len(spread["test_gain_by_val_decile"]) == 10


def test_analysis_needs_data_selected_genes_last(tmp_path):
    config = gene_level_config(tmp_path)
    config["block_sets"]["own_and_partners"] = [
        "expression_components",
        "own_expression",
        "partners",
    ]
    config["reference"]["block_set"] = "own_and_partners"
    with pytest.raises(ValueError, match="data-selected"):
        analysis.analyse(synthetic_base(validate_prior_config(config)))


def test_main_writes_the_tables(tmp_path, monkeypatch):
    config = gene_level_config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(analysis, "load_base", synthetic_base)
    assert analysis.main([str(path), "--run-id", "r"]) == 0
    out = tmp_path / "out" / "r" / "data_selected"
    assert {p.name for p in out.iterdir()} == {
        "features.csv",
        "targets.csv",
        "ablation.json",
        "summary.md",
    }
    assert len(pd.read_csv(out / "targets.csv")) == 6
    assert "Top 30 features" in (out / "summary.md").read_text()
