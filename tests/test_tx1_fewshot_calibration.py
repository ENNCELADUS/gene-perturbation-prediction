"""Unit tests for the Phase-E few-shot RE-RANKING calibrator, on synthetic data.

The central claim under test: the per-line metric (Spearman rank
correlation) is invariant to any monotone transform, so a scalar-affine
k-shot calibration (``tx1_geneeffect_eval.affine_kshot_calibrate``) cannot
re-rank genes and therefore cannot move the Spearman score. A per-gene
FEATURE-based ridge calibration (:mod:`aivc_model.tx1_fewshot_calibration`)
can, because different genes sharing the same scalar ``base_pred`` can
have different feature vectors and hence different adapted scores. Every
synthetic fixture here is built in-process; nothing touches real data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from aivc_model.tx1_geneeffect_eval import affine_kshot_calibrate
from aivc_model.tx1_fewshot_calibration import (
    DEFAULT_MAX_COMPONENTS,
    MIN_FINITE_LABELS,
    InsufficientFiniteLabelsError,
    calibrate_line,
    fit_feature_reducer,
    fit_ridge_calibration,
    make_predictions_long,
)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def _label_mask(n_genes: int, k: int) -> np.ndarray:
    """First-``k``-genes-True label mask, matching the Phase-F convention."""
    mask = np.zeros(n_genes, dtype=bool)
    mask[:k] = True
    return mask


# ---------------------------------------------------------------------------
# Central test: re-ranking beats both base_pred and the affine calibrator.
# ---------------------------------------------------------------------------


def _mis_ranking_fixture(
    n_genes: int = 300, n_features: int = 4, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build features/base_pred/y_true where base_pred mis-ranks vs. y_true.

    ``y_true`` is (up to small noise) a linear function of ``features[:, 0]``
    only. ``base_pred`` is dominated by ``features[:, 1]`` and
    ``features[:, 2]`` (irrelevant to ``y_true``), with only a weak,
    same-signed dependence on ``features[:, 0]`` — so ``base_pred`` ranks
    genes mostly by noise directions and is a poor (but not perfectly
    anti-correlated) proxy for ``y_true``.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_genes, n_features))
    y_true = 2.0 * features[:, 0] + 0.1 * rng.normal(size=n_genes)
    base_pred = 0.3 * features[:, 0] + 5.0 * features[:, 1] - 3.0 * features[:, 2]
    return features, base_pred, y_true


def test_reranking_beats_base_and_affine_calibration() -> None:
    n_genes = 300
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=n_genes)
    k = 50
    label_mask = _label_mask(n_genes, k)
    scored_mask = ~label_mask

    base_rho = _spearman(base_pred[scored_mask], y_true[scored_mask])
    affine_pred = affine_kshot_calibrate(base_pred, y_true, label_mask)
    affine_rho = _spearman(affine_pred[scored_mask], y_true[scored_mask])
    adapted = calibrate_line(features, base_pred, y_true, label_mask, k=k)
    adapted_rho = _spearman(adapted[scored_mask], y_true[scored_mask])

    # base_pred is a genuinely weak/mis-ranked proxy for y_true.
    assert base_rho < 0.4
    # Affine calibration is a monotone map of base_pred: it cannot re-rank,
    # so its Spearman must stay essentially where base_pred's was.
    assert abs(affine_rho - base_rho) < 0.05
    # The feature-based ridge adapter re-ranks and should recover most of
    # the true (linear-in-features) signal on the disjoint scored genes.
    assert adapted_rho > 0.85
    # Clear margin over both base_pred and the (provably inert) affine map.
    assert adapted_rho > base_rho + 0.3
    assert adapted_rho > affine_rho + 0.3


# ---------------------------------------------------------------------------
# k=0 identity.
# ---------------------------------------------------------------------------


def test_k_zero_returns_base_pred_unchanged() -> None:
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=50, seed=1)
    label_mask = np.zeros(50, dtype=bool)
    result = calibrate_line(features, base_pred, y_true, label_mask, k=0)
    assert np.array_equal(result, base_pred)
    assert result is not base_pred  # returns a copy, not an alias


# ---------------------------------------------------------------------------
# Determinism (closed-form, no RNG inside the adapter).
# ---------------------------------------------------------------------------


def test_calibrate_line_is_deterministic() -> None:
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=120, seed=2)
    label_mask = _label_mask(120, 20)
    first = calibrate_line(features, base_pred, y_true, label_mask, k=20)
    second = calibrate_line(features, base_pred, y_true, label_mask, k=20)
    assert np.array_equal(first, second)


def test_fit_ridge_calibration_is_deterministic() -> None:
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=40, seed=3)
    calibrator_a = fit_ridge_calibration(features[:20], y_true[:20], alpha=1.0)
    calibrator_b = fit_ridge_calibration(features[:20], y_true[:20], alpha=1.0)
    assert np.array_equal(calibrator_a.weights, calibrator_b.weights)
    assert calibrator_a.intercept == calibrator_b.intercept


# ---------------------------------------------------------------------------
# Regularization sanity.
# ---------------------------------------------------------------------------


def test_larger_alpha_shrinks_weights_toward_zero() -> None:
    features, _, y_true = _mis_ranking_fixture(n_genes=30, n_features=5, seed=4)
    low = fit_ridge_calibration(features[:15], y_true[:15], alpha=0.1)
    high = fit_ridge_calibration(features[:15], y_true[:15], alpha=100.0)
    assert np.linalg.norm(high.weights) < np.linalg.norm(low.weights)


def test_high_dimensional_features_have_no_high_dimensional_fit_path() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(80, 4000))
    y_true = rng.normal(size=80)
    k = 10
    reducer = fit_feature_reducer(features)
    calibrator = fit_ridge_calibration(
        features[:k],
        y_true[:k],
        alpha=1.0,
        feature_reducer=reducer,
    )

    assert reducer.input_dim == 4000
    assert reducer.output_dim == DEFAULT_MAX_COMPONENTS
    assert calibrator.weights.shape == (k // 2,)
    assert np.all(np.isfinite(calibrator.weights))
    assert np.linalg.norm(calibrator.weights) < 1e3
    assert calibrator.apply(features).shape == (80,)


# ---------------------------------------------------------------------------
# Disjointness: the fit only ever reads the k selected label genes.
# ---------------------------------------------------------------------------


def test_scored_genes_are_excluded_from_the_fit() -> None:
    n_genes = 60
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=n_genes, seed=6)
    k = 10
    label_mask = _label_mask(n_genes, k)
    fit_indices = np.flatnonzero(label_mask)[:k]
    scored_mask = np.ones(n_genes, dtype=bool)
    scored_mask[fit_indices] = False

    # Sanity: the fit and scored gene sets never overlap.
    assert not np.any(scored_mask & np.isin(np.arange(n_genes), fit_indices))

    baseline = calibrate_line(features, base_pred, y_true, label_mask, k=k)

    # Perturbing y_true only at scored genes must not change the fit (the
    # adapter is closed-form on features + the k label targets only), so
    # the resulting adapted predictions are bit-identical.
    perturbed_y = y_true.copy()
    perturbed_y[scored_mask] += 1000.0
    perturbed = calibrate_line(features, base_pred, perturbed_y, label_mask, k=k)
    assert np.array_equal(baseline, perturbed)

    # Perturbing y_true at a fit (label) gene DOES change the output,
    # confirming the fit genuinely reads the label targets it is supposed
    # to (this is not a vacuously-passing no-op test).
    perturbed_label_y = y_true.copy()
    perturbed_label_y[fit_indices[0]] += 1000.0
    perturbed_label = calibrate_line(
        features, base_pred, perturbed_label_y, label_mask, k=k
    )
    assert not np.array_equal(baseline, perturbed_label)


def test_label_pool_larger_than_k_uses_only_first_k() -> None:
    """A label_mask pool bigger than k must only consume its first k Trues."""
    n_genes = 50
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=n_genes, seed=7)
    pool_mask = _label_mask(n_genes, k=20)  # 20 candidate label genes
    result_k5 = calibrate_line(features, base_pred, y_true, pool_mask, k=5)

    # Genes 5..19 are in the label pool but unused at k=5: corrupting their
    # y_true must not affect the output, since only genes 0..4 are fit on.
    corrupted_y = y_true.copy()
    corrupted_y[5:20] += 1000.0
    result_corrupted = calibrate_line(features, base_pred, corrupted_y, pool_mask, k=5)
    assert np.array_equal(result_k5, result_corrupted)


# ---------------------------------------------------------------------------
# Increasing k improves the held-out curve (unlike the flat affine curve).
# ---------------------------------------------------------------------------


def test_increasing_k_improves_scored_spearman_curve() -> None:
    n_genes = 400
    n_features = 10
    rng = np.random.default_rng(11)
    features = rng.normal(size=(n_genes, n_features))
    # y_true depends on the first 3 (of 10) feature dims; the other 7 are
    # pure noise directions relative to y_true.
    true_w = np.array([1.5, -1.0, 0.8] + [0.0] * (n_features - 3))
    y_true = features @ true_w + 0.2 * rng.normal(size=n_genes)
    # base_pred is dominated by the irrelevant noise directions.
    base_pred = features[:, 3:].sum(axis=1) + 0.1 * features[:, 0]

    rhos = {}
    for k in (5, 10, 25):
        label_mask = _label_mask(n_genes, k)
        scored_mask = ~label_mask
        adapted = calibrate_line(features, base_pred, y_true, label_mask, k=k)
        rhos[k] = _spearman(adapted[scored_mask], y_true[scored_mask])

    # k=5 < n_features=10: ridge is under-determined but regularized, so it
    # should still be finite and not catastrophically negative.
    assert rhos[5] > -0.5
    # The curve is not flat (unlike the provably inert affine curve): more
    # labels give a materially better held-out rank correlation, with a
    # tolerance since k=10 (near F) can be noisy.
    assert rhos[25] > rhos[10] - 0.1
    assert rhos[10] > rhos[5] - 0.1
    assert rhos[25] > rhos[5] + 0.2


# ---------------------------------------------------------------------------
# apply() and RidgeCalibrator basics.
# ---------------------------------------------------------------------------


def test_apply_residual_mode_requires_base_pred() -> None:
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=20, seed=8)
    calibrator = fit_ridge_calibration(
        features[:10], y_true[:10], alpha=1.0, base_pred_labeled=base_pred[:10]
    )
    assert calibrator.residual is True
    with pytest.raises(ValueError, match="requires base_pred_all"):
        calibrator.apply(features)


def test_apply_non_residual_mode_ignores_base_pred() -> None:
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=20, seed=9)
    calibrator = fit_ridge_calibration(features[:10], y_true[:10], alpha=1.0)
    assert calibrator.residual is False
    out_without = calibrator.apply(features)
    out_with = calibrator.apply(features, base_pred_all=base_pred)
    assert np.array_equal(out_without, out_with)


def test_apply_rejects_wrong_feature_width() -> None:
    features, _, y_true = _mis_ranking_fixture(n_genes=20, n_features=4, seed=10)
    calibrator = fit_ridge_calibration(features[:10], y_true[:10], alpha=1.0)
    wrong_width = np.zeros((5, 3))
    with pytest.raises(ValueError, match="2-D with"):
        calibrator.apply(wrong_width)


def test_fit_ridge_calibration_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one labeled gene"):
        fit_ridge_calibration(np.zeros((0, 3)), np.zeros(0), alpha=1.0)


def test_fit_ridge_calibration_rejects_negative_alpha() -> None:
    features, _, y_true = _mis_ranking_fixture(n_genes=10, seed=12)
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        fit_ridge_calibration(features, y_true, alpha=-1.0)


# ---------------------------------------------------------------------------
# alpha=0 with a rank-deficient design (F > k) must not raise LinAlgError.
# ---------------------------------------------------------------------------


def test_fit_ridge_calibration_alpha_zero_rank_deficient_is_finite() -> None:
    """alpha=0 with F > k makes the normal-equations Gram exactly singular.

    ``np.linalg.solve`` raises ``LinAlgError: Singular matrix`` in this
    case; the fix (``np.linalg.lstsq``) must instead return a finite,
    well-defined min-norm solution.
    """
    features, _, y_true = _mis_ranking_fixture(n_genes=20, n_features=8, seed=20)
    k = 3  # k < n_features=8: Gram is rank-deficient at alpha=0.

    calibrator = fit_ridge_calibration(features[:k], y_true[:k], alpha=0.0)

    assert np.all(np.isfinite(calibrator.weights))
    assert np.isfinite(calibrator.intercept)


def test_calibrate_line_alpha_zero_rank_deficient_returns_finite_predictions() -> None:
    """The same alpha=0/F>k case, exercised through the calibrate_line seam."""
    n_genes = 20
    features, base_pred, y_true = _mis_ranking_fixture(
        n_genes=n_genes, n_features=8, seed=21
    )
    label_mask = _label_mask(n_genes, k=3)

    result = calibrate_line(
        features, base_pred, y_true, label_mask, k=3, alpha=0.0, residual=False
    )

    assert result.shape == (n_genes,)
    assert np.all(np.isfinite(result))


def test_fit_ridge_calibration_matches_reduced_manual_solve() -> None:
    """The closed form is solved in the bounded, label-free PCA space."""
    features, _, y_true = _mis_ranking_fixture(n_genes=40, n_features=3, seed=22)
    features_labeled = features[:20]
    y_labeled = y_true[:20]
    reducer = fit_feature_reducer(features)

    for alpha in (0.0, 1.0):
        calibrator = fit_ridge_calibration(
            features_labeled,
            y_labeled,
            alpha=alpha,
            feature_reducer=reducer,
        )

        reduced = reducer.transform(features_labeled)[:, : calibrator.weights.size]
        reduced = reduced - reduced.mean(axis=0)
        target_centered = y_labeled - y_labeled.mean()
        gram = reduced.T @ reduced + alpha * np.eye(calibrator.weights.size)
        expected_weights = np.linalg.solve(gram, reduced.T @ target_centered)

        assert np.allclose(calibrator.weights, expected_weights, atol=1e-6), alpha


def test_feature_reducer_is_label_free_and_deterministic() -> None:
    features, _, y_true = _mis_ranking_fixture(n_genes=100, n_features=20, seed=35)
    first = fit_feature_reducer(features)
    second = fit_feature_reducer(features)

    corrupted_y = y_true.copy()
    corrupted_y[50:] += 10_000.0
    del corrupted_y  # labels are deliberately never accepted by the reducer API

    assert np.array_equal(first.feature_mean, second.feature_mean)
    assert np.array_equal(first.feature_std, second.feature_std)
    assert np.array_equal(first.standardized_mean, second.standardized_mean)
    assert np.array_equal(first.components, second.components)
    assert np.array_equal(first.transform(features), second.transform(features))


def test_calibrate_line_rejects_k_exceeding_available_labels() -> None:
    n_genes = 20
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=n_genes, seed=13)
    label_mask = _label_mask(n_genes, k=5)
    with pytest.raises(ValueError, match="exceeds available labeled genes"):
        calibrate_line(features, base_pred, y_true, label_mask, k=6)


def test_calibrate_line_rejects_negative_k() -> None:
    n_genes = 10
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=n_genes, seed=14)
    label_mask = _label_mask(n_genes, k=5)
    with pytest.raises(ValueError, match="k must be >= 0"):
        calibrate_line(features, base_pred, y_true, label_mask, k=-1)


# ---------------------------------------------------------------------------
# T2 code review regression: non-finite held-out labels must not poison the
# ridge fit. Held-out lines are NOT completeness-guaranteed (the
# differential slice was frozen on TRAINING lines), so a k-shot panel can
# land on a gene with a missing GeneEffect label.
# ---------------------------------------------------------------------------


def test_fit_ridge_calibration_raises_on_insufficient_finite_labels() -> None:
    """Below MIN_FINITE_LABELS finite targets, fit_ridge_calibration refuses."""
    features, _, y_true = _mis_ranking_fixture(n_genes=10, seed=32)
    all_nan_y = np.full_like(y_true[:5], np.nan)
    with pytest.raises(InsufficientFiniteLabelsError):
        fit_ridge_calibration(features[:5], all_nan_y, alpha=1.0)


def test_fit_ridge_calibration_fits_on_the_finite_subset() -> None:
    """Above MIN_FINITE_LABELS finite targets, the fit drops only the NaNs."""
    n_genes = 40
    features, _, y_true = _mis_ranking_fixture(n_genes=n_genes, seed=33)
    corrupted_y = y_true[:20].copy()
    corrupted_y[0] = np.nan  # 1 of 20 labels is non-finite; 19 >= minimum

    calibrator = fit_ridge_calibration(features[:20], corrupted_y, alpha=1.0)
    finite_mask = np.isfinite(corrupted_y)
    expected = fit_ridge_calibration(
        features[:20][finite_mask], corrupted_y[finite_mask], alpha=1.0
    )

    assert np.allclose(calibrator.weights, expected.weights)
    assert calibrator.intercept == pytest.approx(expected.intercept)


def test_calibrate_line_drops_nonfinite_target_and_returns_finite_predictions() -> None:
    """A NaN held-out label among the k labels must not poison the panel.

    Regression for the bug where ``target.mean()`` becomes NaN (and hence
    every fitted weight) as soon as one of the k labeled genes has a
    non-finite GeneEffect. With enough OTHER finite labels remaining, the
    fit proceeds on the finite subset and returns finite predictions for
    every gene, not just the scored ones.
    """
    n_genes = 60
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=n_genes, seed=30)
    k = 10
    label_mask = _label_mask(n_genes, k)
    fit_indices = np.flatnonzero(label_mask)[:k]

    corrupted_y = y_true.copy()
    corrupted_y[fit_indices[0]] = np.nan  # one of the k labels is NaN

    result = calibrate_line(features, base_pred, corrupted_y, label_mask, k=k)

    assert np.all(np.isfinite(result))


def test_calibrate_line_falls_back_to_identity_with_too_few_finite_labels(
    caplog,
) -> None:
    """Below MIN_FINITE_LABELS finite labels, calibrate_line returns base_pred.

    Rather than raising or returning NaN across the panel, too few finite
    labeled genes falls back to identity (unchanged ``base_pred``), with a
    warning logged so the fallback is observable.
    """
    n_genes = 30
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=n_genes, seed=31)
    k = MIN_FINITE_LABELS  # exactly at the minimum; corrupt all of it below
    label_mask = _label_mask(n_genes, k)
    fit_indices = np.flatnonzero(label_mask)[:k]

    corrupted_y = y_true.copy()
    corrupted_y[fit_indices] = np.nan  # every k=MIN_FINITE_LABELS label is NaN

    with caplog.at_level("WARNING"):
        result = calibrate_line(features, base_pred, corrupted_y, label_mask, k=k)

    assert np.array_equal(result, base_pred)
    assert result is not base_pred  # a copy, not an alias, matching k=0 identity
    assert any("falling back" in r.message.lower() for r in caplog.records)


def test_calibrate_line_k_zero_identity_is_unaffected_by_finite_label_fix() -> None:
    """The k=0 identity path is untouched by the finite-label fallback logic."""
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=20, seed=34)
    label_mask = np.zeros(20, dtype=bool)
    result = calibrate_line(features, base_pred, y_true, label_mask, k=0)
    assert np.array_equal(result, base_pred)
    assert result is not base_pred


# ---------------------------------------------------------------------------
# make_predictions_long integration helper.
# ---------------------------------------------------------------------------


def test_make_predictions_long_shape_and_disjoint_scoring() -> None:
    import pandas as pd

    n_genes = 40
    features, base_pred, y_true = _mis_ranking_fixture(n_genes=n_genes, seed=15)
    genes = np.array([f"GENE{i}" for i in range(n_genes)])
    k_schedule = [0, 5, 10]

    rng = np.random.default_rng(16)
    panel_rows = []
    for panel in range(3):
        order = rng.permutation(n_genes)[:10]
        for label_order, gene_idx in enumerate(order, start=1):
            panel_rows.append(
                {
                    "panel": panel,
                    "label_order": label_order,
                    "depmap_column": genes[gene_idx],
                }
            )
    panels_for_line = pd.DataFrame(panel_rows)

    long_df = make_predictions_long(
        model_id="LINE_0",
        genes=genes,
        features=features,
        base_pred=base_pred,
        y_true=y_true,
        panels_for_line=panels_for_line,
        k_schedule=k_schedule,
        method="tx1_3b_st",
    )
    assert set(long_df["k"].unique()) == set(k_schedule)
    assert len(long_df) == 3 * len(k_schedule) * n_genes

    # k=0 rows must reproduce base_pred unchanged (identity), per gene.
    zero_k = long_df[(long_df["k"] == 0) & (long_df["panel"] == 0)]
    zero_k = zero_k.set_index("depmap_column").loc[genes]
    assert np.allclose(zero_k["base_pred"].to_numpy(), base_pred)
