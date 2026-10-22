"""Closed-form ridge stages against scikit-learn and limiting cases."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src.context_prior.ridge import (
    column_scale,
    gene_ridge,
    selected_ridge,
    shared_ridge,
)


def test_shared_ridge_matches_sklearn_on_standardised_features():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3)) * [1.0, 5.0, 0.1]
    x = np.column_stack([x, np.full(40, 7.0)])  # constant column: no effect
    y = x[:, :2] @ [[1.0, -1.0], [0.5, 0.2]] + rng.normal(size=(40, 2))
    fit = shared_ridge(x, y, [0.1])[0]
    z = (x[:, :3] - x[:, :3].mean(0)) / x[:, :3].std(0)
    reference = Ridge(alpha=0.1 * 40).fit(z, y)
    assert np.allclose(fit.predict(x), reference.predict(z))


def test_constant_column_with_an_inexact_float_mean_contributes_nothing():
    # 50 copies of 0.1 average to 0.1 - 2.8e-17, so the column's float SD is not
    # zero; it must still count as constant, or a query off the constant is divided
    # by that rounding error.
    rng = np.random.default_rng(4)
    x = np.column_stack([rng.normal(size=(50, 2)), np.full(50, 0.1)])
    assert column_scale(x)[1][2] == 1.0
    y = x[:, :2] @ [[1.0], [-0.5]] + rng.normal(size=(50, 1))
    fit = shared_ridge(x, y, [0.1])[0]
    query = x[:5].copy()
    query[:, 2] = 5.0
    assert np.allclose(fit.predict(query), fit.predict(x[:5]))


def test_reduced_rank_projects_targets_on_the_basis():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 2))
    y = rng.normal(size=(30, 4))
    basis = np.linalg.qr(rng.normal(size=(4, 2)))[0]
    fit = shared_ridge(x, y, [1.0], basis=basis)[0]
    assert np.linalg.matrix_rank(fit.coef) <= 2
    full = shared_ridge(x, y, [1.0])[0]  # ridge is linear in the targets
    assert np.allclose(fit.coef, full.coef @ basis @ basis.T)


def test_pooled_gene_ridge_limits():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(60, 5, 1))
    features = (features - features.mean(axis=0)).astype(np.float32)  # standardised
    targets = 2.0 * features[:, :, 0] + rng.normal(size=(60, 5)) * 0.01
    pooled_only, per_gene = gene_ridge(features, targets, [np.inf, 1e-6], pooled=True)
    assert np.allclose(pooled_only.deviation, 0.0)
    assert np.isclose(pooled_only.pooled[0], 2.0, atol=0.01)
    assert np.allclose(per_gene.predict(features), targets, atol=0.05)


def test_selected_ridge_matches_a_per_gene_fit():
    rng = np.random.default_rng(3)
    expression = rng.normal(size=(50, 6))
    expression -= expression.mean(axis=0)  # the prior passes standardised columns
    selection = np.array([[1, 2], [0, 4]])
    targets = np.column_stack([expression[:, 1], expression[:, 4] - expression[:, 0]])
    fit = selected_ridge(expression, selection, targets, [1e-6])[0]
    assert np.allclose(fit.predict(expression), targets, atol=1e-3)
