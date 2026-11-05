"""Controls built on the linear model: the linear model alone, and the linear model
plus the Tx1 ridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines.linear_model_controls import (
    LINEAR_MODEL,
    LINEAR_MODEL_PLUS_TX1_RIDGE,
    linear_model_control_rows,
)
from src.baselines.residual import run_r1_ladder
from src.data.splits import FixedSplit
from tests.fixtures.joint import TRAIN, VAL, make_inputs
from tests.fixtures.joint import with_linear_model

LINEAR_MODEL_VALUES = {m: 0.1 * i for i, m in enumerate((*TRAIN, *VAL))}


def test_linear_model_rows_are_the_offsets_on_the_splits_labels():
    inputs = with_linear_model(make_inputs(), LINEAR_MODEL_VALUES)
    rows = linear_model_control_rows(inputs, "val")
    alone = rows.loc[rows.method == LINEAR_MODEL].set_index(["model_id", "gene_symbol"])
    labels = inputs.labels.loc[inputs.labels.model_id.isin(VAL)]
    assert len(alone) == len(labels)
    for _, row in labels.iterrows():
        expected = (
            LINEAR_MODEL_VALUES[row.model_id] * inputs.residual_scale[row.gene_symbol]
        )
        assert alone.loc[(row.model_id, row.gene_symbol), "residual_prediction"] == (
            pytest.approx(expected, rel=1e-6)
        )
    assert set(rows.slice) == {"val"}
    assert list(rows.columns) == [
        "slice",
        "model_id",
        "gene_symbol",
        "method",
        "gene_effect",
        "residual",
        "residual_prediction",
    ]


def test_linear_model_controls_need_a_linear_model():
    with pytest.raises(ValueError, match="no linear_model"):
        linear_model_control_rows(make_inputs(), "val")


def test_ridge_on_what_the_linear_model_leaves_reproduces_a_linear_model_that_explains_nothing():  # noqa: E501
    plain = make_inputs()
    zero = with_linear_model(plain, {m: 0.0 for m in (*TRAIN, *VAL)})
    rows = linear_model_control_rows(zero, "val")
    stacked = rows.loc[rows.method == LINEAR_MODEL_PLUS_TX1_RIDGE]
    shifted = with_linear_model(plain, {m: 0.0 for m in TRAIN} | {m: 1.0 for m in VAL})
    moved = linear_model_control_rows(shifted, "val")
    moved = moved.loc[moved.method == LINEAR_MODEL_PLUS_TX1_RIDGE]
    assert len(stacked) == len(plain.labels.loc[plain.labels.model_id.isin(VAL)])
    scale = stacked.gene_symbol.map(plain.residual_scale).to_numpy()
    np.testing.assert_allclose(
        moved.residual_prediction.to_numpy(),
        stacked.residual_prediction.to_numpy() + scale,
        rtol=1e-6,
    )


def test_ridge_on_a_zero_linear_model_is_the_ladders_tx1_ridge():
    plain = make_inputs()
    rows = linear_model_control_rows(
        with_linear_model(plain, dict.fromkeys((*TRAIN, *VAL), 0.0)), "val"
    )
    stacked = rows.loc[rows.method == LINEAR_MODEL_PLUS_TX1_RIDGE]
    lines = (*TRAIN, *VAL)
    view = pd.DataFrame(
        np.stack(
            [
                np.concatenate(
                    (
                        plain.lines[m].controls_tx1.mean(0),
                        plain.lines[m].controls_tx1.var(0),
                    )
                )
                for m in lines
            ]
        ),
        index=pd.Index(lines, name="model_id"),
    )
    labels = plain.labels.loc[plain.labels.model_id.isin(lines)]
    ladder = run_r1_ladder(
        labels[["model_id", "gene_symbol", "gene_effect"]],
        {"tx1": view},
        None,
        split=FixedSplit(train=TRAIN, val=VAL, test=()),
    )
    ridge = ladder.loc[ladder.method == "context_pca_ridge[tx1]"]
    key = ["model_id", "gene_symbol"]
    merged = stacked.merge(ridge, on=key, suffixes=("", "_ladder"))
    assert len(merged) == len(stacked) == len(ridge)
    np.testing.assert_allclose(
        merged.residual_prediction, merged.residual_prediction_ladder, atol=1e-8
    )
