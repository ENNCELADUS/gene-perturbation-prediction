"""Single-cell statistics stacked on the prior: config, normalisation fitted on
training lines, the two-stage stack, and the arm runner on synthetic lines."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.context_prior.gene_features import select_genes
from src.context_prior.single_cell import (
    STATISTICS,
    normalize_statistics,
    stack_predictions,
)
from src.experiments.prior import single_cell_stack as stack
from src.experiments.prior.config import validate_stack_config
from tests.fixtures.prior import TARGETS, synthetic_base, tiny_config

CONFIG = yaml.safe_load(
    Path("configs/context_prior/single_cell_stack.yaml").read_text()
)


def test_stack_config_is_strict():
    validate_stack_config(CONFIG)
    unknown = copy.deepcopy(CONFIG)
    unknown["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        validate_stack_config(unknown)
    statistic = copy.deepcopy(CONFIG)
    statistic["arms"]["bad"] = ["mean", "kurtosis"]
    with pytest.raises(ValueError, match="arms.bad"):
        validate_stack_config(statistic)
    infinite = copy.deepcopy(CONFIG)
    infinite["penalties"]["selected"] = [float("inf")]
    with pytest.raises(ValueError, match="selected"):
        validate_stack_config(infinite)
    assert CONFIG["penalties"]["own"][-1] == float("inf")  # pooled weights only


def test_selection_leaves_out_every_own_column():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(40, 6))
    residual = pool[:, [0]] + pool[:, [3]]  # the gene's own two columns explain it
    selection = select_genes(pool, residual, 2, own=np.array([[0, 3]]))
    assert not {0, 3} & set(selection[0])


def _statistics(lines, genes, seed=0):
    rng = np.random.default_rng(seed)
    return {
        name: pd.DataFrame(
            rng.normal(size=(len(lines), len(genes))), index=lines, columns=genes
        )
        for name in STATISTICS
    }


def test_normalisation_is_fitted_on_training_lines_only():
    train, val = [f"S{i}" for i in range(8)], ["V0", "V1"]
    genes = [f"G{i}" for i in range(5)]
    raw = _statistics([*train, *val], genes)
    raw["detected"].loc["S0", "G1"] = np.nan  # a training gap: the training mean
    raw["mean"].loc["S2", "G4"] = np.nan
    raw["mean"].loc["V0", "G4"] = np.nan  # a validation gap: G4 leaves the space
    before = normalize_statistics(raw, train, measured_in=val)
    assert list(before["mean"].columns) == genes[:4]
    assert np.isfinite(before["detected"].to_numpy()).all()
    moved = {name: frame.copy() for name, frame in raw.items()}
    moved["variance"].loc["V1"] = moved["variance"].loc["V1"] * 10 + 3
    after = normalize_statistics(moved, train, measured_in=val)
    for name in STATISTICS:
        pd.testing.assert_frame_equal(after[name].loc[train], before[name].loc[train])
    # Quantile normalisation keeps only V1's ranks, which an affine change keeps.
    pd.testing.assert_frame_equal(
        after["variance"].loc[["V1"]], before["variance"].loc[["V1"]]
    )


def _planted(seed=0):
    """The target follows another gene's detected fraction; the mean is noise."""
    rng = np.random.default_rng(seed)
    train = [f"S{i}" for i in range(120)]
    queries = [f"Q{i}" for i in range(60)]
    genes = [f"G{i}" for i in range(10)]
    statistics = _statistics([*train, *queries], genes, seed)
    target = pd.DataFrame(
        0.1 * rng.normal(size=(len(train), 3)), index=train, columns=genes[:3]
    )
    signal = statistics["detected"]["G7"]
    target["G0"] += signal.loc[train].to_numpy()
    truth = signal.loc[queries].to_numpy()
    return statistics, target, queries, truth


def _fit(statistics, target, queries, chosen):
    return stack_predictions(
        statistics,
        target,
        queries,
        chosen=chosen,
        own_shrinkages=[1.0, float("inf")],
        selected_penalties=[0.01, 1.0],
        count=2,
    )


def test_the_stack_finds_a_detected_fraction_signal_the_mean_arm_cannot():
    statistics, target, queries, truth = _planted()
    single_cell = _fit(statistics, target, queries, ["mean", "detected", "variance"])
    mean = _fit(statistics, target, queries, ["mean"])
    assert set(single_cell) == {(1.0, 0.01), (1.0, 1.0), (np.inf, 0.01), (np.inf, 1.0)}
    found = single_cell[(1.0, 0.01)]
    assert list(found.index) == queries and list(found.columns) == ["G0", "G1", "G2"]
    assert np.corrcoef(found["G0"], truth)[0, 1] > 0.9
    assert abs(np.corrcoef(mean[(1.0, 0.01)]["G0"], truth)[0, 1]) < 0.4


def test_a_query_prediction_reads_only_its_own_line():
    statistics, target, queries, _ = _planted()
    before = _fit(statistics, target, queries, ["mean", "detected"])
    moved = {name: frame.copy() for name, frame in statistics.items()}
    moved["detected"].loc["Q0"] = moved["detected"].loc["Q0"] + 5.0
    after = _fit(moved, target, queries, ["mean", "detected"])
    for key in before:
        pd.testing.assert_frame_equal(
            after[key].loc[queries[1:]], before[key].loc[queries[1:]]
        )
        assert not np.allclose(after[key].loc["Q0"], before[key].loc["Q0"])


def test_run_arms_reports_every_row_and_the_control_gain(tmp_path):
    base = synthetic_base(tiny_config(tmp_path))
    train = list(base.bridge.single_cell_train)
    every = [*train, *base.bridge.val, *base.bridge.test]
    genes = [f"G{i}" for i in range(12)]
    statistics = normalize_statistics(
        _statistics(every, genes, seed=3), train, [*base.bridge.val, *base.bridge.test]
    )
    prior = pd.DataFrame(0.0, index=every, columns=TARGETS)
    config = validate_stack_config(
        {
            **CONFIG,
            "bootstrap_repeats": 20,
            "selected_genes": 2,
            "penalties": {"own": [10.0, float("inf")], "selected": [1.0]},
        }
    )
    result = stack.run_arms(config, base, prior, statistics)
    assert set(result["arms"]) == set(CONFIG["arms"])
    for arm, record in result["arms"].items():
        assert len(record["rows"]) == 2
        for row in record["rows"]:
            for split in ("val", "test"):
                entry = row[split]
                assert len(entry["gain_over_prior"]["interval"]) == 2
                assert ("gain_over_mean" in entry) == (arm != "mean")
    text = stack.results_markdown("run", config, result)
    assert "pooled only" in text and "Arm `distribution`" in text
