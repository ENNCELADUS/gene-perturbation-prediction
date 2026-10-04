"""Contrastive directions between paired pseudo-bulk and bulk, their removal, and
the contrastive-PCA remedy."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.context_prior.alignment import contrastive_directions, fit_alignment
from src.context_prior.remedies import affine, contrastive
from tests.test_context_prior_bridging import base

GENES = [f"G{i}" for i in range(30)]
RNG = np.random.default_rng(7)
#: The pseudo-bulk-only direction: spread over every gene, not one gene's axis.
NOISE_AXIS = RNG.normal(size=30) / np.linalg.norm(RNG.normal(size=30))


def paired(seed=0, lines=60):
    rng = np.random.default_rng(seed)
    axis = NOISE_AXIS / np.linalg.norm(NOISE_AXIS)
    shared = rng.normal(size=(lines, 3)) @ rng.normal(size=(3, 30))
    pseudo = (
        shared
        + 5.0 * rng.normal(size=(lines, 1)) * axis
        + 0.1 * rng.normal(size=(lines, 30))
    )
    bulk = shared + 0.1 * rng.normal(size=(lines, 30))
    index = [f"L{i}" for i in range(lines)]
    return (
        pd.DataFrame(pseudo, index=index, columns=GENES),
        pd.DataFrame(bulk, index=index, columns=GENES),
    )


def mean_gene_correlation(left, right):
    return np.mean([np.corrcoef(left[g], right[g])[0, 1] for g in GENES])


def test_directions_find_the_pseudo_bulk_only_axis():
    pseudo, bulk = paired()
    directions = contrastive_directions(pseudo.to_numpy(), bulk.to_numpy(), 1)
    axis = NOISE_AXIS / np.linalg.norm(NOISE_AXIS)
    assert directions.shape == (30, 1) and abs(directions[:, 0] @ axis) > 0.95


def test_only_positive_excess_variance_counts():
    x = np.random.default_rng(1).normal(size=(40, 10))
    assert contrastive_directions(x, 2 * x, 3).shape == (10, 0)
    # Identical sources: every excess variance is numerically zero.
    assert contrastive_directions(x, x, 3).shape == (10, 0)


def test_zero_counts_are_the_identity():
    pseudo, bulk = paired()
    alignment = fit_alignment(pseudo, bulk, pseudo_components=0, bulk_components=0)
    assert alignment.count == 0 and alignment.apply(pseudo, source="pseudo").equals(
        pseudo
    )


def test_removal_makes_the_sources_agree_and_is_affine_for_other_rows():
    pseudo, bulk = paired()
    alignment = fit_alignment(pseudo, bulk, pseudo_components=1, bulk_components=0)
    before = mean_gene_correlation(pseudo, bulk)
    after = mean_gene_correlation(
        alignment.apply(pseudo, source="pseudo"), alignment.apply(bulk, source="bulk")
    )
    assert after > before + 0.05 and after > 0.95
    other = bulk.iloc[:5] + 1.0
    shifted = alignment.apply(other, source="bulk") - alignment.apply(
        bulk.iloc[:5], source="bulk"
    )
    keep = np.eye(30) - alignment.directions @ alignment.directions.T
    assert np.allclose(shifted.to_numpy(), (other - bulk.iloc[:5]).to_numpy() @ keep)


def test_overlapping_direction_sets_are_removed_once():
    pseudo, bulk = paired()
    alignment = fit_alignment(pseudo, bulk, pseudo_components=3, bulk_components=3)
    assert np.allclose(
        alignment.directions.T @ alignment.directions, np.eye(alignment.count)
    )
    assert alignment.count <= 6


def test_misaligned_frames_and_unknown_source_raise():
    pseudo, bulk = paired()
    with pytest.raises(ValueError, match="paired"):
        fit_alignment(pseudo, bulk.iloc[::-1], pseudo_components=1, bulk_components=0)
    alignment = fit_alignment(pseudo, bulk, pseudo_components=1, bulk_components=0)
    with pytest.raises(ValueError, match="source"):
        alignment.apply(pseudo, source="tumour")


def assert_same_inputs(left, right):
    pd.testing.assert_frame_equal(left.expression, right.expression)
    assert set(left.queries) == set(right.queries)
    for name in left.queries:
        pd.testing.assert_frame_equal(left.queries[name], right.queries[name])
    pd.testing.assert_frame_equal(left.oof_paired, right.oof_paired)
    assert left.gene_space is None and left.gene_rows is None


def test_zero_components_remedy_is_the_affine_remedy():
    b = base()
    inputs = contrastive.build(b, {"pseudo_components": 0, "bulk_components": 0})
    assert_same_inputs(inputs, affine.build(b, {}))


def test_remedy_fits_on_paired_lines_and_bridges_aligned_sources():
    b = base()
    setting = {"pseudo_components": 2, "bulk_components": 1}
    inputs = contrastive.build(b, setting)
    assert list(inputs.expression.index) == list(b.bulk.index)
    assert list(inputs.queries["val"].index) == list(b.val)
    assert list(inputs.queries["test"].index) == list(b.test)
    assert list(inputs.queries["oracle"].index) == list(b.oracle.index)
    assert list(inputs.oof_paired.index) == list(b.paired)
    assert not np.allclose(inputs.expression, b.bulk)
    # Rows outside the paired lines are transformed, never fitted on.
    shuffled = b.pseudobulk.copy()
    outside = [*b.val, *b.test, "S18", "S19"]
    shuffled.loc[outside] = shuffled.loc[outside].to_numpy()[::-1] * 3.0
    extras = b.bulk.copy()
    extras.loc[~extras.index.isin(b.paired)] *= -2.0
    moved = contrastive.build(replace(b, pseudobulk=shuffled, bulk=extras), setting)
    paired = list(b.paired)
    pd.testing.assert_frame_equal(
        moved.expression.loc[paired], inputs.expression.loc[paired]
    )
    pd.testing.assert_frame_equal(moved.oof_paired, inputs.oof_paired)


@pytest.mark.parametrize(
    "setting",
    [
        {"pseudo_components": 2},
        {"pseudo_components": 2, "bulk_components": 0, "rank": 4},
        {"pseudo_components": 2.0, "bulk_components": 0},
    ],
)
def test_remedy_setting_is_strict(setting):
    with pytest.raises(ValueError):
        contrastive.build(base(), setting)
