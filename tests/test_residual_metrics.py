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
import time

import numpy as np
import pandas as pd
import pytest

from src.eval.geneeffect import aggregate_geneeffect
from src.eval.metrics import (
    paired_line_bootstrap,
    per_gene_spearman,
)


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
# Undefined-unit causes: too few observations, constant truth, constant
# prediction -- each must land in the undefined counts without crashing.
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Selective-gene metrics and the paired line bootstrap.
# --------------------------------------------------------------------------

_EFFECT_LINES = [f"L{i}" for i in range(8)]


def _effect_frame(
    genes: list[str], *, seed: int, predict=None, noise: float = 0.3
) -> pd.DataFrame:
    """Aggregate-input frame; ``predict(truth, mean)`` defaults to a noisy truth."""
    rng = np.random.default_rng(seed)
    rows = []
    for gene_index, gene in enumerate(genes):
        mean = -0.2 - 0.1 * gene_index
        effect = mean + rng.normal(scale=0.8, size=len(_EFFECT_LINES))
        for line, value in zip(_EFFECT_LINES, effect):
            residual = value - mean
            guess = (
                residual + rng.normal(scale=noise)
                if predict is None
                else predict(residual, mean)
            )
            rows.append(
                dict(
                    model_id=line,
                    gene_symbol=gene,
                    gene_effect=value,
                    residual=residual,
                    residual_prediction=guess,
                    geneeffect_prediction=guess + mean,
                )
            )
    return pd.DataFrame(rows)


_AGG_GENES = ["A", "B", "C", "D", "E"]


def _aggregate(frame, *, variable, selective):
    return aggregate_geneeffect(
        frame,
        model_ids=_EFFECT_LINES,
        genes=_AGG_GENES,
        variable_genes=variable,
        selective_genes=selective,
    )


def test_selective_metrics_and_per_gene_table_cover_the_union_in_gene_order() -> None:
    frame = _effect_frame(_AGG_GENES, seed=3)
    # B is selective but not variable; E is variable but not selective.
    metrics, _, per_gene = _aggregate(
        frame, variable=["E", "A", "C"], selective=["C", "B"]
    )
    assert list(per_gene.gene_symbol) == ["A", "B", "C", "E"]
    assert per_gene.set_index("gene_symbol").variable.to_dict() == {
        "A": True,
        "B": False,
        "C": True,
        "E": True,
    }
    assert per_gene.set_index("gene_symbol").selective.to_dict() == {
        "A": False,
        "B": True,
        "C": True,
        "E": False,
    }
    spearman = per_gene.set_index("gene_symbol").spearman
    assert metrics["selective_spearman"] == pytest.approx(
        (spearman["B"] + spearman["C"]) / 2
    )
    assert metrics["selective_spearman_scored"] == 2
    assert metrics["selective_spearman_undefined"] == 0
    assert per_gene.loc[~per_gene.selective, "aupr_lift"].isna().all()


def test_residual_metrics_are_computed_over_variable_genes_only() -> None:
    frame = _effect_frame(_AGG_GENES, seed=5)
    variable = ["A", "B", "C"]
    baseline, _, _ = _aggregate(frame, variable=variable, selective=["A"])
    wider, _, per_gene = _aggregate(frame, variable=variable, selective=["D", "E"])
    # Selective genes outside the variable set must not move any residual metric.
    for key, value in baseline.items():
        if key.startswith(("residual_", "geneeffect_")):
            assert wider[key] == value, key
    # And they match a direct computation over the variable genes.
    sub = frame[frame.gene_symbol.isin(variable)]
    expected = per_gene_spearman(
        sub, truth_col="residual", pred_col="residual_prediction"
    )
    assert wider["residual_spearman_macro_per_gene"] == pytest.approx(expected.mean())
    assert wider["residual_spearman_per_gene_scored"] == 3


def test_constant_prediction_scores_exactly_zero_aupr_lift_and_undefined_spearman() -> (
    None
):
    frame = _effect_frame(_AGG_GENES, seed=9, predict=lambda residual, mean: 0.25)
    # Make every gene have both dependent and non-dependent lines.
    frame["gene_effect"] = np.where(frame.model_id.isin(_EFFECT_LINES[:3]), -1.0, 0.2)
    metrics, _, per_gene = _aggregate(frame, variable=_AGG_GENES, selective=_AGG_GENES)
    assert (per_gene.aupr_lift == 0.0).all()
    assert metrics["selective_aupr_lift"] == 0.0
    assert metrics["selective_aupr_lift_scored"] == 5
    assert metrics["selective_spearman"] is None
    assert metrics["selective_spearman_scored"] == 0
    assert metrics["selective_spearman_undefined"] == 5


def test_aupr_lift_perfect_ranking_and_single_class_genes_undefined() -> None:
    frame = _effect_frame(_AGG_GENES[:3], seed=1)
    # Gene A: dependent lines are exactly the three predicted lowest.
    # Gene B: no dependent line.  Gene C: every line dependent.
    effect = np.where(frame.model_id.isin(_EFFECT_LINES[:3]), -1.0, 0.2)
    frame["gene_effect"] = np.where(
        frame.gene_symbol == "A", effect, np.where(frame.gene_symbol == "B", 0.2, -1.0)
    )
    frame["geneeffect_prediction"] = frame.gene_effect + 0.01 * frame.model_id.str[
        1:
    ].astype(int)
    metrics, _, per_gene = aggregate_geneeffect(
        frame,
        model_ids=_EFFECT_LINES,
        genes=_AGG_GENES[:3],
        variable_genes=["A"],
        selective_genes=["A", "B", "C"],
    )
    by_gene = per_gene.set_index("gene_symbol").aupr_lift
    assert by_gene["A"] == pytest.approx(1.0 - 3 / 8)
    assert math.isnan(by_gene["B"]) and math.isnan(by_gene["C"])
    assert metrics["selective_aupr_lift"] == pytest.approx(1.0 - 3 / 8)
    assert metrics["selective_aupr_lift_scored"] == 1
    assert metrics["selective_aupr_lift_undefined"] == 2


def test_selective_genes_outside_gene_order_raise() -> None:
    frame = _effect_frame(_AGG_GENES, seed=2)
    with pytest.raises(ValueError, match="outside the gene order"):
        _aggregate(frame, variable=["A"], selective=["NOPE"])


def _reference_bootstrap_macro(frame, genes, lines, draw) -> float:
    """Macro selective Spearman of ``frame`` over the resampled ``draw`` of lines."""
    from src.eval.metrics import _unit_spearman

    values = []
    for gene in genes:
        sub = frame[frame.gene_symbol == gene].set_index("model_id")
        sub = sub.reindex(lines)
        truth = sub.residual.to_numpy()[draw]
        pred = sub.residual_prediction.to_numpy()[draw]
        values.append(_unit_spearman(truth, pred))
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def test_paired_line_bootstrap_matches_loop_reference_with_duplicates() -> None:
    genes = ["A", "B", "C", "D"]
    left = _effect_frame(genes, seed=11, noise=0.2)
    right = _effect_frame(genes, seed=11, noise=1.5)
    # D has a constant truth in some resamples (all-but-two lines equal) and a
    # NaN cell, so duplicated draws regularly leave it constant or too short.
    for frame in (left, right):
        mask = (frame.gene_symbol == "D") & ~frame.model_id.isin(["L0", "L1"])
        frame.loc[mask, "residual"] = 0.5
        frame.loc[
            (frame.gene_symbol == "C") & (frame.model_id == "L4"), "residual_prediction"
        ] = np.nan
    result = paired_line_bootstrap(left, right, genes, repeats=300, seed=7)

    lines = sorted(set(left.model_id))
    rng = np.random.default_rng(7)
    draws = rng.integers(0, len(lines), size=(300, len(lines)))
    assert any(len(set(draw)) < len(lines) for draw in draws)
    reference = []
    for draw in draws:
        a = _reference_bootstrap_macro(left, genes, lines, draw)
        b = _reference_bootstrap_macro(right, genes, lines, draw)
        reference.append(a - b)
    reference = np.asarray(reference)
    reference = reference[np.isfinite(reference)]
    identity = np.arange(len(lines))
    observed = _reference_bootstrap_macro(left, genes, lines, identity) - (
        _reference_bootstrap_macro(right, genes, lines, identity)
    )
    assert result["difference"] == pytest.approx(observed)
    assert result["interval"] == pytest.approx(
        list(np.percentile(reference, [2.5, 97.5]))
    )


def test_paired_line_bootstrap_identical_frames_have_zero_difference() -> None:
    genes = ["A", "B", "C"]
    frame = _effect_frame(genes, seed=4)
    result = paired_line_bootstrap(frame, frame.copy(), genes, repeats=50, seed=0)
    assert result == {"difference": 0.0, "interval": [0.0, 0.0]}


def test_paired_line_bootstrap_mismatched_keys_and_undefined_macro() -> None:
    genes = ["A", "B"]
    frame = _effect_frame(genes, seed=6)
    with pytest.raises(ValueError, match="same"):
        paired_line_bootstrap(frame, frame.iloc[1:], genes, repeats=5, seed=0)
    constant = frame.assign(residual_prediction=0.0)
    result = paired_line_bootstrap(constant, constant, genes, repeats=5, seed=0)
    assert math.isnan(result["difference"])
    assert all(math.isnan(value) for value in result["interval"])


def test_paired_line_bootstrap_is_fast_at_production_size() -> None:
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(3000)]
    lines = [f"L{i}" for i in range(27)]
    index = pd.MultiIndex.from_product(
        [lines, genes], names=["model_id", "gene_symbol"]
    )
    truth = rng.normal(size=len(index))
    frames = [
        pd.DataFrame(
            {
                "residual": truth,
                "residual_prediction": truth + rng.normal(scale=scale, size=len(index)),
            },
            index=index,
        ).reset_index()
        for scale in (0.5, 2.0)
    ]
    start = time.perf_counter()
    result = paired_line_bootstrap(*frames, genes, repeats=1000, seed=0)
    elapsed = time.perf_counter() - start
    assert result["difference"] > 0 and result["interval"][0] > 0
    assert elapsed < 30, elapsed
