"""Cilantro-SL's rank-value encoding and top-k truncation as a bridge remedy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.remedies import rank_value
from tests.test_context_prior_bridging import base


def test_medians_skip_zeros_and_send_silent_genes_last():
    bulk = pd.DataFrame(
        np.log1p([[0.0, 4.0, 0.0], [2.0, 8.0, 0.0], [6.0, 0.0, 0.0]]),
        columns=["A", "B", "C"],
    )
    medians = rank_value.nonzero_medians(bulk)
    assert np.allclose(medians[["A", "B"]], [4.0, 6.0])
    assert medians["C"] == np.inf


def test_rank_value_ranks_relative_expression_within_each_line():
    medians = pd.Series({"A": 1.0, "B": 10.0, "C": 1.0, "D": 1.0})
    # Linear values 5, 20, 4, 0 -> relative 5, 2, 4, 0: B is the most expressed but
    # ranks third once divided by its median.
    frame = pd.DataFrame(np.log1p([[5.0, 20.0, 4.0, 0.0]]), columns=list("ABCD"))
    full = rank_value.encode(frame, medians, "rank_value", None)
    assert np.allclose(full.iloc[0], [1.0, 0.5, 0.75, 0.25])
    top = rank_value.encode(frame, medians, "rank_value", 2)
    assert np.allclose(top.iloc[0], [1.0, 0.0, 0.5, 0.0])
    # Truncation alone keeps the same top genes' expression and zeroes the rest.
    kept = rank_value.encode(frame, medians, "expression", 2)
    assert np.allclose(kept.iloc[0], [np.log1p(5.0), 0.0, np.log1p(4.0), 0.0])


def test_settings_are_checked():
    b = base()
    for setting in (
        {"encoding": "rank_value"},
        {"encoding": "ranks", "top_k": None},
        {"encoding": "rank_value", "top_k": 0},
        {"encoding": "expression", "top_k": None},
    ):
        with pytest.raises(ValueError):
            rank_value.build(b, setting)


def test_build_encodes_every_source_before_the_bridge():
    b = base()
    inputs = rank_value.build(b, {"encoding": "rank_value", "top_k": None})
    medians = rank_value.nonzero_medians(b.bulk)
    expected = rank_value.encode(b.bulk, medians, "rank_value", None)
    pd.testing.assert_frame_equal(inputs.expression, expected)
    assert set(inputs.queries) == {"val", "test", "oracle"}
    assert list(inputs.queries["val"].index) == list(b.val)
    assert list(inputs.oof_paired.index) == list(b.paired)
    assert list(inputs.oof_bulk.index) == list(b.paired)
    assert inputs.expression.to_numpy().min() >= 0.0
