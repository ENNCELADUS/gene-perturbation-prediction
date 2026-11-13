"""The Kebabci harness (hpc/paralog/kebabci.py): network features follow the official
notebooks (weighted shared-interactor means, the smallest shared GO term), and held-out
lines can read predicted GeneEffect."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.experiments.run_dir import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "kebabci", REPO_ROOT / "hpc" / "paralog" / "kebabci.py"
)
kebabci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kebabci)

PAIRS = pd.DataFrame({"gene_1": ["A", "A"], "gene_2": ["B", "C"]})
VALUES = pd.DataFrame(
    {"I": [1.0, 2.0], "J": [3.0, np.nan], "K": [5.0, 5.0]}, index=["L0", "L1"]
)


def test_weighted_ppi_weights_shared_interactors_by_both_scores():
    interactors = {
        "A": {"I": 2.0, "J": 1.0, "K": 4.0},
        "B": {"I": 3.0, "J": 2.0},
        "C": {"X": 1.0},  # X has no values: nothing shared with a value
    }
    out = kebabci.weighted_ppi(PAIRS, interactors, VALUES)
    # weights I 6, J 2; in L1, J is NaN and adds nothing (pandas sum), weight kept
    assert out[0] == pytest.approx([(6 * 1 + 2 * 3) / 8, (6 * 2) / 8])
    assert np.isnan(out[1]).all()


def test_smallest_go_term_takes_the_smallest_shared_term_and_ties_by_pick():
    terms = {
        "big": {"A", "B", "I", "J", "K"},
        "small1": {"A", "B", "I"},
        "small2": {"A", "B", "K"},
        "other": {"A", "C", "Z"},  # one gene with values: not a valid term
    }
    values = VALUES.assign(A=0.0, B=0.0)
    high = kebabci.smallest_go(PAIRS, terms, values, "max")
    low = kebabci.smallest_go(PAIRS, terms, values, "min")
    # small1 means (0 + 0 + I) / 3, small2 (0 + 0 + K) / 3
    assert high[0] == pytest.approx([5 / 3, 5 / 3])
    assert low[0] == pytest.approx([1 / 3, 2 / 3])
    assert np.isnan(high[1]).all()


def test_gene_effect_gives_held_lines_the_prediction_on_its_genes():
    measured = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [7.0, 8.0, 9.0]},
        index=["FIT", "HELD", "HELD_NO_PRED"],
    )
    predicted = pd.DataFrame(
        {"A": [-1.0, -2.0], "B": [-3.0, -4.0]}, index=["HELD", "X"]
    )
    out = kebabci.gene_effect(measured, predicted, ["HELD", "HELD_NO_PRED"])
    # fit lines keep measured values, held lines read the prediction, genes are cut
    assert list(out.columns) == ["A", "B"]
    assert out.loc["FIT"].tolist() == [1.0, 4.0]
    assert out.loc["HELD"].tolist() == [-1.0, -3.0]
    # a held line without a prediction is absent: its rows are not covered
    assert set(out.index) == {"FIT", "HELD"}
