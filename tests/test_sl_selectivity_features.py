"""Tests for cross-cell-line selectivity pair features."""

from __future__ import annotations

import numpy as np

from sl_benchmark_baseline.features import (
    SELECTIVITY_FEATURE_NAMES,
    build_selectivity_pair_features,
)


def test_build_selectivity_pair_features_shape_and_values():
    sel_ab = np.array([1.0, 2.0])
    sel_ba = np.array([3.0, -2.0])
    pan_a = np.array([-0.5, -1.0])
    pan_b = np.array([-0.2, -3.0])
    out = build_selectivity_pair_features(sel_ab, sel_ba, pan_a, pan_b)
    assert out.shape == (2, 3)
    assert len(SELECTIVITY_FEATURE_NAMES) == 3
    # row 0: mean=(1+3)/2=2, absdiff=|1-3|=2, pan_min=min(-0.5,-0.2)=-0.5
    np.testing.assert_allclose(out[0], [2.0, 2.0, -0.5])
    # row 1: mean=0, absdiff=4, pan_min=-3.0
    np.testing.assert_allclose(out[1], [0.0, 4.0, -3.0])


def test_build_selectivity_pair_features_is_swap_invariant():
    sel_ab = np.array([1.0])
    sel_ba = np.array([3.0])
    pan_a = np.array([-0.5])
    pan_b = np.array([-0.2])
    forward = build_selectivity_pair_features(sel_ab, sel_ba, pan_a, pan_b)
    swapped = build_selectivity_pair_features(sel_ba, sel_ab, pan_b, pan_a)
    np.testing.assert_allclose(forward, swapped)
