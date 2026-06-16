from __future__ import annotations

import numpy as np


def test_pair_features_are_swap_invariant() -> None:
    from sl_benchmark_baseline.features import FEATURE_NAMES, build_pair_features

    ea = np.array([-1.0, 0.5])
    eb = np.array([0.2, -0.3])
    forward = build_pair_features(ea, eb)
    swapped = build_pair_features(eb, ea)
    assert forward.shape == (2, len(FEATURE_NAMES))
    np.testing.assert_allclose(forward, swapped)
    np.testing.assert_allclose(
        forward[0], [-1.0, 0.2, -0.8, -0.2, 1.2], rtol=1e-6
    )


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
