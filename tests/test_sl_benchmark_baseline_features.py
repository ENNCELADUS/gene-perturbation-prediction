from __future__ import annotations

import numpy as np
from sl_benchmark_baseline.features import (
    build_pair_features,
    build_transcript_pair_features,
    build_augmented_pair_features,
    transcript_feature_names,
)


def test_transcript_pair_features_are_swap_invariant():
    emb_a = np.array([[1.0, 2.0], [0.5, -1.0]])
    emb_b = np.array([[3.0, -1.0], [2.0, 4.0]])
    flag_a = np.array([1.0, 1.0])
    flag_b = np.array([0.0, 1.0])
    forward = build_transcript_pair_features(
        emb_a, emb_b, flag_a, flag_b, include_coverage_flag=True
    )
    swapped = build_transcript_pair_features(
        emb_b, emb_a, flag_b, flag_a, include_coverage_flag=True
    )
    np.testing.assert_allclose(forward, swapped)


def test_transcript_pair_features_shape_and_names():
    emb_a = np.zeros((4, 8))
    emb_b = np.zeros((4, 8))
    flag = np.ones(4)
    feats = build_transcript_pair_features(
        emb_a, emb_b, flag, flag, include_coverage_flag=True
    )
    assert feats.shape == (4, 3 * 8 + 2)
    assert len(transcript_feature_names(8, include_coverage_flag=True)) == 3 * 8 + 2
    assert len(transcript_feature_names(8, include_coverage_flag=False)) == 3 * 8


def test_transcript_pair_features_omit_coverage_flag():
    emb_a = np.zeros((2, 5))
    emb_b = np.zeros((2, 5))
    flag = np.ones(2)
    feats = build_transcript_pair_features(
        emb_a, emb_b, flag, flag, include_coverage_flag=False
    )
    assert feats.shape == (2, 15)


def test_augmented_pair_features_concatenate_geneeffect_then_transcript():
    ea = np.array([-1.0, 0.2])
    eb = np.array([-0.8, 0.3])
    emb_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    emb_b = np.array([[0.0, 1.0], [2.0, 2.0]])
    flag = np.ones(2)
    feats = build_augmented_pair_features(
        ea, eb, emb_a, emb_b, flag, flag, include_coverage_flag=True
    )
    # 5 gene-effect + (3*2 transcript) + 2 coverage = 13
    assert feats.shape == (2, 13)
    np.testing.assert_allclose(feats[:, :5], build_pair_features(ea, eb))


def test_augmented_pair_features_are_swap_invariant():
    ea = np.array([-1.0])
    eb = np.array([0.4])
    emb_a = np.array([[1.0, -2.0]])
    emb_b = np.array([[0.5, 3.0]])
    flag_a = np.array([1.0])
    flag_b = np.array([0.0])
    forward = build_augmented_pair_features(
        ea, eb, emb_a, emb_b, flag_a, flag_b, include_coverage_flag=True
    )
    swapped = build_augmented_pair_features(
        eb, ea, emb_b, emb_a, flag_b, flag_a, include_coverage_flag=True
    )
    np.testing.assert_allclose(forward, swapped)


def test_pair_features_are_swap_invariant() -> None:
    from sl_benchmark_baseline.features import FEATURE_NAMES, build_pair_features

    ea = np.array([-1.0, 0.5])
    eb = np.array([0.2, -0.3])
    forward = build_pair_features(ea, eb)
    swapped = build_pair_features(eb, ea)
    assert forward.shape == (2, len(FEATURE_NAMES))
    np.testing.assert_allclose(forward, swapped)
    np.testing.assert_allclose(forward[0], [-1.0, 0.2, -0.8, -0.2, 1.2], rtol=1e-6)


def test_standardizer_fits_on_train_only() -> None:
    from sl_benchmark_baseline.features import Standardizer

    train = np.array([[0.0, 2.0], [2.0, 4.0]])
    standardizer = Standardizer.fit(train)
    transformed = standardizer.transform(train)
    np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(transformed.std(axis=0), [1.0, 1.0], atol=1e-9)
    const = np.array([[5.0], [5.0]])
    const_std = Standardizer.fit(const).transform(const)
    np.testing.assert_allclose(const_std, [[0.0], [0.0]])
