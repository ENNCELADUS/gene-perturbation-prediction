"""Controls built on the prior: the prior alone and the prior plus the Tx1 ridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines.prior_controls import (
    PRIOR,
    PRIOR_PLUS_TX1_RIDGE,
    prior_control_rows,
)
from src.baselines.residual import run_r1_ladder
from src.data.splits import FixedSplit
from tests.test_joint import TRAIN, VAL, make_inputs
from tests.test_prior_offsets import with_prior

PRIOR_VALUES = {m: 0.1 * i for i, m in enumerate((*TRAIN, *VAL))}


def test_prior_rows_are_the_offsets_on_the_splits_labels():
    inputs = with_prior(make_inputs(), PRIOR_VALUES)
    rows = prior_control_rows(inputs, "val")
    alone = rows.loc[rows.method == PRIOR].set_index(["model_id", "gene_symbol"])
    labels = inputs.labels.loc[inputs.labels.model_id.isin(VAL)]
    assert len(alone) == len(labels)
    for _, row in labels.iterrows():
        expected = PRIOR_VALUES[row.model_id] * inputs.residual_scale[row.gene_symbol]
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


def test_prior_controls_need_a_prior():
    with pytest.raises(ValueError, match="no prior"):
        prior_control_rows(make_inputs(), "val")


def test_ridge_on_what_the_prior_leaves_reproduces_a_prior_that_explains_nothing():
    plain = make_inputs()
    zero = with_prior(plain, {m: 0.0 for m in (*TRAIN, *VAL)})
    rows = prior_control_rows(zero, "val")
    stacked = rows.loc[rows.method == PRIOR_PLUS_TX1_RIDGE]
    shifted = with_prior(plain, {m: 0.0 for m in TRAIN} | {m: 1.0 for m in VAL})
    moved = prior_control_rows(shifted, "val")
    moved = moved.loc[moved.method == PRIOR_PLUS_TX1_RIDGE]
    assert len(stacked) == len(plain.labels.loc[plain.labels.model_id.isin(VAL)])
    scale = stacked.gene_symbol.map(plain.residual_scale).to_numpy()
    np.testing.assert_allclose(
        moved.residual_prediction.to_numpy(),
        stacked.residual_prediction.to_numpy() + scale,
        rtol=1e-6,
    )


def test_ridge_on_a_zero_prior_is_the_ladders_tx1_ridge():
    plain = make_inputs()
    rows = prior_control_rows(
        with_prior(plain, dict.fromkeys((*TRAIN, *VAL), 0.0)), "val"
    )
    stacked = rows.loc[rows.method == PRIOR_PLUS_TX1_RIDGE]
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
        seed=0,
        outer="fixed",
        split=FixedSplit(train=TRAIN, val=VAL, test=()),
    ).predictions
    ridge = ladder.loc[ladder.method == "context_pca_ridge[tx1]"]
    key = ["model_id", "gene_symbol"]
    merged = stacked.merge(ridge, on=key, suffixes=("", "_ladder"))
    assert len(merged) == len(stacked) == len(ridge)
    np.testing.assert_allclose(
        merged.residual_prediction, merged.residual_prediction_ladder, atol=1e-8
    )
