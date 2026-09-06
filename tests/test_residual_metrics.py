"""Tests for :mod:`src.eval.metrics`.

All frames are synthetic and built in-test; nothing here depends on
gitignored data. The tests are organized around the two mathematical facts
the module exists to encode (see the module docstring of
``residual_metrics.py``): per-gene across-line Spearman is invariant to a
per-gene constant shift, and a context-blind predictor is undefined (NaN,
never 0.0) on that axis while scoring well on the historical per-line axis.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import (
    ResidualScore,
    ShuffleControl,
    bootstrap_delta,
    per_gene_spearman,
    per_line_spearman,
    score_predictions,
    shuffled_context_control,
)


def _assert_series_close(series: pd.Series, expected: float) -> None:
    """Assert every entry of ``series`` is close to the scalar ``expected``.

    ``pandas.Series.__eq__`` does not broadcast correctly against a
    ``pytest.approx`` object (it compares the whole object per-element
    instead of unwrapping it), so plain ``(series == pytest.approx(x)).all()``
    silently evaluates to ``False`` even when every entry matches -- this
    helper uses ``numpy.allclose`` instead.
    """
    assert np.allclose(series.to_numpy(dtype=float), expected)


def _long_frame(genes: list[str], lines: list[str], truth_fn, pred_fn) -> pd.DataFrame:
    """Build a long-form (model_id, gene_symbol, truth, pred) frame."""
    rows = []
    for line in lines:
        for gene in genes:
            rows.append(
                {
                    "model_id": line,
                    "gene_symbol": gene,
                    "truth": truth_fn(gene, line),
                    "pred": pred_fn(gene, line),
                }
            )
    return pd.DataFrame(rows)


# A perfectly-predicted fixture: mu_g dominates but a small per-line-varying
# per-gene term keeps every gene's across-line vector non-constant, and every
# line's across-gene vector non-constant, with no rank ties on either axis.
_GENES = ["G0", "G1", "G2", "G3", "G4", "G5"]
_LINES = ["L0", "L1", "L2", "L3", "L4"]
_MU = {gene: -3.0 + i * 1.0 for i, gene in enumerate(_GENES)}


def _true_value(gene: str, line: str) -> float:
    gene_idx = _GENES.index(gene)
    line_idx = _LINES.index(line)
    return _MU[gene] + 0.01 * (line_idx - 2) * (gene_idx + 1)


def _perfect_frame() -> pd.DataFrame:
    """pred == truth exactly: Spearman is exactly 1.0 on both axes."""
    return _long_frame(_GENES, _LINES, _true_value, _true_value)


def _context_blind_frame() -> pd.DataFrame:
    """pred(g, c) = mu_g only: constant per gene, varying per line."""
    return _long_frame(_GENES, _LINES, _true_value, lambda gene, _line: _MU[gene])


# --------------------------------------------------------------------------
# Fact 1: per-gene axis is invariant to a per-gene constant shift.
# --------------------------------------------------------------------------


def test_per_gene_spearman_invariant_to_per_gene_constant_shift() -> None:
    rng = np.random.default_rng(7)
    genes = [f"G{i}" for i in range(6)]
    lines = [f"L{i}" for i in range(5)]
    truth_vals = {
        (g, ln): float(v)
        for g in genes
        for ln, v in zip(lines, rng.normal(size=len(lines)), strict=True)
    }
    pred_vals = {
        (g, ln): float(v)
        for g in genes
        for ln, v in zip(lines, rng.normal(size=len(lines)), strict=True)
    }
    frame = _long_frame(
        genes,
        lines,
        lambda g, ln: truth_vals[(g, ln)],
        lambda g, ln: pred_vals[(g, ln)],
    )

    raw = per_gene_spearman(frame, truth_col="truth", pred_col="pred")

    # Different arbitrary per-gene constants applied independently to truth
    # and pred -- invariance must hold even when the two shifts differ.
    truth_shift = {g: 2.5 * i - 4.0 for i, g in enumerate(genes)}
    pred_shift = {g: -1.7 * i + 9.0 for i, g in enumerate(genes)}
    gene_ln_pairs = list(zip(frame["gene_symbol"], frame["model_id"], strict=True))
    shifted = frame.copy()
    shifted["truth"] = [truth_vals[(g, ln)] - truth_shift[g] for g, ln in gene_ln_pairs]
    shifted["pred"] = [pred_vals[(g, ln)] - pred_shift[g] for g, ln in gene_ln_pairs]

    residual = per_gene_spearman(shifted, truth_col="truth", pred_col="pred")

    pd.testing.assert_series_equal(raw.sort_index(), residual.sort_index())


# --------------------------------------------------------------------------
# Fact 2 (the central behaviour): context-blind predictor is undefined on
# the per-gene axis, and does NOT crash, and scores well on per-line.
# --------------------------------------------------------------------------


def test_context_blind_predictor_undefined_on_per_gene_axis() -> None:
    frame = _context_blind_frame()
    score = score_predictions(frame, truth_col="truth", pred_col="pred")

    assert isinstance(score, ResidualScore)
    assert score.per_gene.isna().all()
    assert math.isnan(score.macro_per_gene)
    assert score.n_gene_undefined == len(_GENES)
    assert score.n_genes == 0


def test_context_blind_predictor_scores_well_on_per_line_axis() -> None:
    # Same fixture as above: the two axes must disagree, on purpose.
    frame = _context_blind_frame()
    score = score_predictions(frame, truth_col="truth", pred_col="pred")

    assert score.n_line_undefined == 0
    assert score.n_lines == len(_LINES)
    assert score.macro_per_line == pytest.approx(1.0)
    _assert_series_close(score.per_line, 1.0)


# --------------------------------------------------------------------------
# Known-answer case: exact monotone relationship gives rho == 1.0 on both
# axes, for every unit and for the macro aggregates.
# --------------------------------------------------------------------------


def test_perfect_prediction_scores_exactly_one_on_both_axes() -> None:
    frame = _perfect_frame()
    score = score_predictions(frame, truth_col="truth", pred_col="pred")

    assert score.n_line_undefined == 0
    assert score.n_gene_undefined == 0
    assert score.macro_per_line == pytest.approx(1.0)
    assert score.macro_per_gene == pytest.approx(1.0)
    _assert_series_close(score.per_line, 1.0)
    _assert_series_close(score.per_gene, 1.0)


# --------------------------------------------------------------------------
# Sign sensitivity: negating the prediction flips both axes to -1.0.
# --------------------------------------------------------------------------


def test_negating_prediction_flips_sign_on_both_axes() -> None:
    frame = _perfect_frame()
    negated = frame.copy()
    negated["pred"] = -negated["pred"]

    score = score_predictions(negated, truth_col="truth", pred_col="pred")

    assert score.macro_per_line == pytest.approx(-1.0)
    assert score.macro_per_gene == pytest.approx(-1.0)
    _assert_series_close(score.per_line, -1.0)
    _assert_series_close(score.per_gene, -1.0)


# --------------------------------------------------------------------------
# Undefined-unit causes: too few observations, constant truth, constant
# prediction -- each must land in the undefined counts without crashing.
# --------------------------------------------------------------------------


def test_undefined_units_cover_all_three_causes() -> None:
    rows = [
        # GA: only 2 lines -> fewer than MIN_OBSERVATIONS (3).
        {"model_id": "L0", "gene_symbol": "GA", "truth": 1.0, "pred": 1.0},
        {"model_id": "L1", "gene_symbol": "GA", "truth": 2.0, "pred": 2.0},
        # GB: constant truth across 4 lines, varying prediction.
        {"model_id": "L0", "gene_symbol": "GB", "truth": 5.0, "pred": 1.0},
        {"model_id": "L1", "gene_symbol": "GB", "truth": 5.0, "pred": 2.0},
        {"model_id": "L2", "gene_symbol": "GB", "truth": 5.0, "pred": 3.0},
        {"model_id": "L3", "gene_symbol": "GB", "truth": 5.0, "pred": 4.0},
        # GC: varying truth, constant prediction across 4 lines.
        {"model_id": "L0", "gene_symbol": "GC", "truth": 1.0, "pred": 5.0},
        {"model_id": "L1", "gene_symbol": "GC", "truth": 2.0, "pred": 5.0},
        {"model_id": "L2", "gene_symbol": "GC", "truth": 3.0, "pred": 5.0},
        {"model_id": "L3", "gene_symbol": "GC", "truth": 4.0, "pred": 5.0},
        # GD: control -- well-posed, monotone, must NOT be undefined.
        {"model_id": "L0", "gene_symbol": "GD", "truth": 1.0, "pred": 1.0},
        {"model_id": "L1", "gene_symbol": "GD", "truth": 2.0, "pred": 2.0},
        {"model_id": "L2", "gene_symbol": "GD", "truth": 3.0, "pred": 3.0},
        {"model_id": "L3", "gene_symbol": "GD", "truth": 4.0, "pred": 4.0},
    ]
    frame = pd.DataFrame(rows)

    per_gene = per_gene_spearman(frame, truth_col="truth", pred_col="pred")

    assert math.isnan(per_gene["GA"])  # too few observations
    assert math.isnan(per_gene["GB"])  # constant truth
    assert math.isnan(per_gene["GC"])  # constant prediction
    assert per_gene["GD"] == pytest.approx(1.0)  # control: well-defined

    score = score_predictions(frame, truth_col="truth", pred_col="pred")
    assert score.n_gene_undefined == 3
    assert score.n_genes == 1


def test_score_predictions_drops_nan_rows_before_correlating() -> None:
    rows = [
        {"model_id": "L0", "gene_symbol": "GE", "truth": 1.0, "pred": 1.0},
        {"model_id": "L1", "gene_symbol": "GE", "truth": 2.0, "pred": np.nan},
        {"model_id": "L2", "gene_symbol": "GE", "truth": np.nan, "pred": 3.0},
        {"model_id": "L3", "gene_symbol": "GE", "truth": 3.0, "pred": 3.0},
        {"model_id": "L4", "gene_symbol": "GE", "truth": 4.0, "pred": 4.0},
    ]
    frame = pd.DataFrame(rows)
    # Only 3 rows (L0, L3, L4) have both truth and pred finite -- exactly
    # MIN_OBSERVATIONS, and perfectly monotone among themselves.
    per_gene = per_gene_spearman(frame, truth_col="truth", pred_col="pred")
    assert per_gene["GE"] == pytest.approx(1.0)


def test_all_undefined_frame_yields_nan_macro_without_raising() -> None:
    rows = [
        {"model_id": "L0", "gene_symbol": "GA", "truth": 1.0, "pred": 1.0},
        {"model_id": "L1", "gene_symbol": "GA", "truth": 2.0, "pred": 2.0},
    ]
    frame = pd.DataFrame(rows)
    score = score_predictions(frame, truth_col="truth", pred_col="pred")
    assert math.isnan(score.macro_per_gene)
    assert score.n_gene_undefined == 1
    assert score.n_genes == 0


# --------------------------------------------------------------------------
# shuffled_context_control
# --------------------------------------------------------------------------

_SHUFFLE_TRUE_CONTEXT = {"L0": 0.0, "L1": 1.0, "L2": 2.0, "L3": 3.0, "L4": 4.0}
_SHUFFLE_GENES = ["G0", "G1", "G2", "G3"]
_SHUFFLE_MU = {"G0": -2.0, "G1": -1.0, "G2": 0.0, "G3": 1.0}
_SHUFFLE_SENS = {"G0": 1.0, "G1": 1.5, "G2": 2.0, "G3": 0.5}  # all non-zero


def _shuffle_fit_predict(context_df: pd.DataFrame) -> pd.DataFrame:
    """A toy context-conditioned model: pred depends on whatever context
    content ``context_df`` currently attaches to each line."""
    rows = []
    for line, true_ctx in _SHUFFLE_TRUE_CONTEXT.items():
        used_ctx = context_df.loc[line, "context_feature"]
        for gene in _SHUFFLE_GENES:
            rows.append(
                {
                    "model_id": line,
                    "gene_symbol": gene,
                    "truth": _SHUFFLE_MU[gene] + _SHUFFLE_SENS[gene] * true_ctx,
                    "pred": _SHUFFLE_MU[gene] + _SHUFFLE_SENS[gene] * used_ctx,
                }
            )
    return pd.DataFrame(rows)


def _shuffle_context() -> pd.DataFrame:
    lines = list(_SHUFFLE_TRUE_CONTEXT.keys())
    return pd.DataFrame(
        {"context_feature": [_SHUFFLE_TRUE_CONTEXT[line] for line in lines]},
        index=lines,
    )


def test_shuffled_context_control_reproducible_for_fixed_seed() -> None:
    context = _shuffle_context()
    kwargs = dict(
        fit_predict=_shuffle_fit_predict,
        context=context,
        baseline_score=0.0,
        axis="per_gene",
        truth_col="truth",
        pred_col="pred",
        n_repeats=25,
    )
    first = shuffled_context_control(seed=123, **kwargs)
    second = shuffled_context_control(seed=123, **kwargs)

    assert isinstance(first, ShuffleControl)
    assert first == second
    # The perfect (unshuffled) model gives per-gene rho == 1.0 for every
    # gene (all sensitivities are non-zero and the context is monotone in
    # line order), so observed_delta is unambiguously non-zero here.
    assert first.observed_delta == pytest.approx(1.0)
    assert not math.isnan(first.retained_gain_ratio)


def test_shuffled_context_control_differs_for_different_seed() -> None:
    context = _shuffle_context()
    kwargs = dict(
        fit_predict=_shuffle_fit_predict,
        context=context,
        baseline_score=0.0,
        axis="per_gene",
        truth_col="truth",
        pred_col="pred",
        n_repeats=25,
    )
    first = shuffled_context_control(seed=123, **kwargs)
    second = shuffled_context_control(seed=456, **kwargs)

    assert first.shuffled_delta_mean != second.shuffled_delta_mean


def test_shuffled_context_control_ratio_nan_when_observed_delta_zero() -> None:
    context = _shuffle_context()
    observed_frame = _shuffle_fit_predict(context)
    observed_macro = score_predictions(
        observed_frame, truth_col="truth", pred_col="pred"
    ).macro_per_gene

    result = shuffled_context_control(
        fit_predict=_shuffle_fit_predict,
        context=context,
        baseline_score=observed_macro,
        axis="per_gene",
        truth_col="truth",
        pred_col="pred",
        n_repeats=5,
        seed=1,
    )

    assert result.observed_delta == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(result.retained_gain_ratio)


# --------------------------------------------------------------------------
# bootstrap_delta
# --------------------------------------------------------------------------


def test_bootstrap_delta_ci_contains_point_estimate() -> None:
    paired = pd.Series([0.05, 0.02, 0.08, -0.01, 0.03, 0.06, 0.04, 0.01, 0.07, 0.02])
    point, ci_lo, ci_hi = bootstrap_delta(paired, n_resamples=5000, seed=20260804)

    assert ci_lo <= point <= ci_hi
    assert point == pytest.approx(paired.mean())


def test_bootstrap_delta_is_seed_reproducible() -> None:
    paired = pd.Series([0.1, -0.2, 0.3, 0.05, -0.05, 0.15, 0.0, 0.22])
    first = bootstrap_delta(paired, n_resamples=3000, seed=42)
    second = bootstrap_delta(paired, n_resamples=3000, seed=42)
    third = bootstrap_delta(paired, n_resamples=3000, seed=43)

    assert first == second
    assert first != third


def test_bootstrap_delta_drops_non_finite_entries() -> None:
    paired = pd.Series([1.0, 2.0, np.nan, 3.0, np.inf, 4.0])
    point, ci_lo, ci_hi = bootstrap_delta(paired, n_resamples=2000, seed=5)

    assert point == pytest.approx(2.5)  # mean of [1, 2, 3, 4]
    assert ci_lo <= point <= ci_hi


def test_bootstrap_delta_all_non_finite_returns_nan_triple() -> None:
    paired = pd.Series([np.nan, np.nan])
    point, ci_lo, ci_hi = bootstrap_delta(paired, n_resamples=100, seed=1)

    assert math.isnan(point)
    assert math.isnan(ci_lo)
    assert math.isnan(ci_hi)


# --------------------------------------------------------------------------
# per_line_spearman / per_gene_spearman: basic index / column contract.
# --------------------------------------------------------------------------


def test_per_line_and_per_gene_index_names() -> None:
    frame = _perfect_frame()
    per_line = per_line_spearman(frame, truth_col="truth", pred_col="pred")
    per_gene = per_gene_spearman(frame, truth_col="truth", pred_col="pred")

    assert set(per_line.index) == set(_LINES)
    assert set(per_gene.index) == set(_GENES)
