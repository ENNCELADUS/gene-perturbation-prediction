"""Stage-one scoring: out-of-fold rank metrics within folds, and contrasts over the
genes defined on both sides."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.targets import Definitions
from src.eval.accuracy import (
    defined_genes,
    fold_accuracy,
    gain,
    line_spearman_gain,
    line_spearmans,
    wide_accuracy,
)

GENES = tuple(f"G{i}" for i in range(40))


def _definitions() -> Definitions:
    return Definitions(
        genes=GENES,
        gene_means=pd.Series(0.0, index=list(GENES)),
        selective=GENES,
        variable=GENES,
        residual_scale=pd.Series(1.0, index=list(GENES)),
    )


def _null_world(lines: int = 100, folds: int = 5, seed: int = 0):
    """Labels with no line signal; each fold's base is the gene's mean label over
    the other folds, as an out-of-fold base m_g^(-k) is."""
    rng = np.random.default_rng(seed)
    index = [f"L{i}" for i in range(lines)]
    truth = pd.DataFrame(
        rng.normal(size=(lines, len(GENES))), index=index, columns=GENES
    )
    fold = {line: i % folds for i, line in enumerate(index)}
    base = pd.DataFrame(0.0, index=index, columns=GENES)
    for k in range(folds):
        inside = [m for m in index if fold[m] == k]
        outside = [m for m in index if fold[m] != k]
        base.loc[inside] = truth.loc[outside].mean().to_numpy()
    return truth, base, fold


def test_a_pooled_out_of_fold_base_is_anti_correlated_by_construction():
    """The null the fold-wise scoring removes: the pooled base ranks lines against
    their truth, though it holds no line information."""
    truth, base, folds = _null_world()
    pooled = wide_accuracy(base, truth, _definitions())
    assert pooled["selective_spearman"] < -0.03
    scored = fold_accuracy(base, truth, _definitions(), folds)
    assert scored["selective_spearman"] is None  # constant within every fold
    assert set(scored["per_fold"]) == {str(k) for k in range(5)}
    assert all(s["lines"] == 20 for s in scored["per_fold"].values())
    assert scored["within_gene_r2"] == pytest.approx(pooled["within_gene_r2"])


def test_fold_wise_rank_metrics_are_centred_on_zero_under_the_null():
    truth, base, folds = _null_world(lines=200, seed=1)
    rng = np.random.default_rng(2)
    noisy = base + 0.3 * rng.normal(size=base.shape)
    pooled = wide_accuracy(noisy, truth, _definitions())
    scored = fold_accuracy(noisy, truth, _definitions(), folds)
    assert pooled["selective_spearman"] < -0.02
    assert abs(scored["selective_spearman"]) < 0.02


def test_contrasts_cover_the_genes_defined_on_both_sides():
    rng = np.random.default_rng(3)
    index = [f"L{i}" for i in range(30)]
    truth = pd.DataFrame(rng.normal(size=(30, len(GENES))), index=index, columns=GENES)
    left = truth + rng.normal(size=truth.shape)
    right = left.copy()
    right[["G0", "G1"]] = 0.25  # the base, as outside the decode panel
    assert set(GENES) - set(defined_genes(right, truth, GENES)) == {"G0", "G1"}
    common = [
        g
        for g in defined_genes(right, truth, GENES)
        if g in defined_genes(left, truth, GENES)
    ]
    same = gain(left, right, truth, _definitions(), 50, genes=common)
    assert same["difference"] == pytest.approx(0.0)
    unequal = gain(left, right, truth, _definitions(), 50)
    assert unequal["difference"] != pytest.approx(0.0)


def test_the_per_line_spearman_reads_each_line_on_its_own():
    """Each line's predicted residuals against its own truth across the selective
    genes: a perfect prediction scores 1 in every line, a line whose prediction is
    the same for every gene is undefined, and the mean is over defined lines."""
    rng = np.random.default_rng(4)
    index = [f"L{i}" for i in range(6)]
    truth = pd.DataFrame(rng.normal(size=(6, len(GENES))), index=index, columns=GENES)
    prediction = truth.copy()
    prediction.loc["L0"] = 0.5
    values = line_spearmans(prediction, truth, GENES)
    assert np.isnan(values[0]) and np.allclose(values[1:], 1.0)
    scores = wide_accuracy(prediction, truth, _definitions())
    assert scores["line_spearman"] == pytest.approx(1.0)
    assert scores["line_spearman_scored"] == 5


def test_the_per_line_spearman_has_no_out_of_fold_bias_under_the_null():
    """Unlike the within-gene rank metrics, a line is never ranked against the
    lines its fold's base was fitted without, so it is pooled out of fold."""
    truth, base, folds = _null_world(lines=200, seed=1)
    noisy = base + 0.3 * np.random.default_rng(2).normal(size=base.shape)
    pooled = fold_accuracy(noisy, truth, _definitions(), folds)
    assert abs(pooled["line_spearman"]) < 0.02


def test_the_per_line_contrast_pairs_lines_defined_on_both_sides():
    better = np.array([0.3, 0.2, np.nan, 0.4])
    reference = np.array([0.1, 0.1, 0.5, np.nan])
    result = line_spearman_gain(better, reference, repeats=200)
    assert result["lines"] == 2
    assert result["difference"] == pytest.approx(0.15)
    low, high = result["interval"]
    assert 0.1 <= low <= 0.15 <= high <= 0.2
