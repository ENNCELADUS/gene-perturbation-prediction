"""Stage-one scoring: out-of-fold rank metrics within folds, and contrasts over the
genes defined on both sides."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.targets import Definitions
from src.eval.geneeffect import defined_genes, fold_accuracy, gain, wide_accuracy

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
