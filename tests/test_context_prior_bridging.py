"""Out-of-fold bridging, diagnostics and the affine remedy."""

from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.context_prior.bridge import fit_bridge
from src.context_prior.bridging import BridgeBase, bridge_diagnostics, oof_bridged
from src.context_prior.remedies import affine

GENES = [f"G{i}" for i in range(12)]


def base(seed=0):
    rng = np.random.default_rng(seed)
    sc = [f"S{i}" for i in range(20)]  # S18, S19 have no bulk row
    paired = sc[:18]
    extras = [f"E{i}" for i in range(10)]
    signal = rng.normal(size=(len(sc) + 12, 12))
    pseudo = pd.DataFrame(
        signal + 0.3 * rng.normal(size=signal.shape),
        index=[*sc, *[f"V{i}" for i in range(6)], *[f"T{i}" for i in range(6)]],
        columns=GENES,
    )
    bulk = pd.DataFrame(
        np.vstack([2 * signal[:18] + 1, rng.normal(size=(10, 12))]),
        index=[*paired, *extras],
        columns=GENES,
    )
    oracle = pd.DataFrame(
        rng.normal(size=(6, 12)), index=[f"V{i}" for i in range(6)], columns=GENES
    )
    return BridgeBase(
        bulk=bulk,
        oracle=oracle,
        pseudobulk=pseudo,
        paired=tuple(paired),
        single_cell_train=tuple(sc),
        val=tuple(f"V{i}" for i in range(6)),
        test=tuple(f"T{i}" for i in range(6)),
        folds={m: i % 4 for i, m in enumerate(sc)},
    )


def test_oof_rows_come_from_bridges_fitted_without_their_fold():
    b = base()
    rows = oof_bridged(b.pseudobulk, b.bulk, b.paired, b.single_cell_train, b.folds)
    assert list(rows.index) == list(b.single_cell_train)  # S18, S19 included
    fold = b.folds["S3"]
    outside = [m for m in b.paired if b.folds[m] != fold]
    manual = fit_bridge(b.pseudobulk.loc[outside], b.bulk.loc[outside]).apply(
        b.pseudobulk.loc[["S3"]]
    )
    assert np.allclose(rows.loc[["S3"]], manual)


def test_affine_remedy_and_diagnostics():
    b = base()
    inputs = affine.build(b, {})
    assert inputs.expression is b.bulk
    assert set(inputs.queries) == {"val", "test", "oracle"}
    assert list(inputs.queries["test"].index) == list(b.test)
    assert list(inputs.oof_paired.index) == list(b.paired)
    paralogs = pd.DataFrame({"gene": ["G0"], "paralog": ["G1"], "identity": [50.0]})
    report = bridge_diagnostics(
        inputs.oof_paired, b.bulk.loc[list(b.paired)], ["G0", "G2"], paralogs
    )
    assert report["selective"]["total"] == 2
    assert report["selective_paralogs"]["total"] == 1
    assert 0.5 < report["median"] <= 1.0


@pytest.mark.parametrize(
    ("kind", "setting"),
    [
        ("affine", {}),
        ("contrastive", {"pseudo_components": 2, "bulk_components": 1}),
        ("gating", {"threshold": 0.3}),
        ("noise_matched", {}),
        ("denoise", {"rank": 4}),
    ],
)
def test_oof_rows_never_read_their_own_folds_bulk(kind, setting):
    """Out-of-fold rows stand in for unseen queries, so nothing a remedy learns
    for them may read the held fold's bulk RNA."""
    build = importlib.import_module(f"src.context_prior.remedies.{kind}").build
    b = base()
    held = [m for m in b.paired if b.folds[m] == 0]
    bulk = b.bulk.copy()
    bulk.loc[held] = np.random.default_rng(9).normal(size=(len(held), len(GENES)))
    before = build(b, setting)
    after = build(replace(b, bulk=bulk), setting)
    pd.testing.assert_frame_equal(
        after.oof_paired.loc[held], before.oof_paired.loc[held]
    )
    assert list(before.oof_bulk.index) == list(b.paired)
    assert list(before.oof_bulk.columns) == list(before.oof_paired.columns)
