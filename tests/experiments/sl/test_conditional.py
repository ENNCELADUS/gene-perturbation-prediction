"""The SL backbone on synthetic lines: an edit of A moves B by the reader's response
times A's fall, the context-ablated score is one value per pair, the head's weights
follow the benchmark, and the label-agreement ceiling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.accuracy import top_recall
from src.experiments.sl.benchmark_baselines import label_agreement
from src.experiments.sl.conditional import (
    FULL,
    check_affine,
    pair_responses,
    terms,
    weights,
)
from src.linear_model.linear_model import (
    LinearModelSpec,
    Stage,
    fit_linear_model,
    total,
)
from tests.linear_model.test_linear_model import synthetic

DEFAULT = LinearModelSpec(
    (Stage("expression_components", 0.01), Stage("data_selected", 0.01, selected=2))
)


def reader():
    inputs, ids = synthetic()
    fitted = fit_linear_model(DEFAULT, inputs, fit_lines=ids, encoder_lines=ids)
    return fitted, inputs.expression


def test_an_edit_of_a_moves_b_by_its_response_times_the_fall():
    fitted, expression = reader()
    origin = expression.mean(axis=0)
    check_affine(fitted, origin, expression.iloc[3], "E3")
    low = expression.quantile(0.1)
    lines = expression.iloc[:2].copy()
    lines.loc["L1", "E1"] = low["E1"] - 1.0  # already below the low level
    rows = pd.DataFrame(
        {"model_id": ["L0", "L1", "L0"], "design": ["combinatorial", "combinatorial",
         "anchor"], "gene_a": ["E1", "E1", "E1"], "gene_b": ["E0", "E0", "E0"]}
    )  # fmt: skip
    out = terms(rows, lines, low, {"L0": FULL, "L1": FULL}, {FULL: fitted}, origin)
    edited = lines.loc[["L0"]].copy()
    edited["E1"] = low["E1"]
    moved = total(fitted.predict(edited)) - total(fitted.predict(lines.loc[["L0"]]))
    assert np.isclose(out[0, 0], moved.loc["L0", "E0"])
    assert out[1, 0] == 0.0  # no fall: A is already low in L1
    assert out[2, 1] == 0.0  # an anchor list reads the first term only
    assert np.isfinite(out[0, 1])
    # a gene without an expression column has no response
    assert np.isnan(pair_responses(fitted, origin, np.array(["Q"]), np.array(["E0"])))[
        0
    ]


def test_the_context_ablated_terms_are_one_value_per_pair():
    fitted, expression = reader()
    origin = expression.mean(axis=0)
    ablated = pd.DataFrame([origin] * 3, index=["L0", "L1", "L2"])
    rows = pd.DataFrame(
        {"model_id": ["L0", "L1", "L2"], "design": ["combinatorial"] * 3,
         "gene_a": ["E0"] * 3, "gene_b": ["E1"] * 3}
    )  # fmt: skip
    readers = {FULL: fitted}
    out = terms(rows, ablated, expression.quantile(0.1), dict.fromkeys(ablated.index,
                FULL), readers, origin)  # fmt: skip
    assert np.ptp(out, axis=0).max() == 0.0


def test_weights_give_each_line_then_each_class_of_a_screen_equal_weight():
    rows = pd.DataFrame(
        {"split": "train", "design": "combinatorial", "condition": "genetic",
         "study_id": ["S"] * 6 + ["T"] * 2,
         "list_id": ["a", "a", "a", "b", "b", "b", "c", "c"],
         "model_id": ["L0"] * 3 + ["L1"] * 5,
         "label": [1, 0, 0, 1, 0, 0, 1, 0]}
    )  # fmt: skip
    w = pd.Series(weights(rows), index=rows.index)
    by_line = w.groupby(rows["model_id"]).sum()
    assert np.isclose(by_line["L0"], by_line["L1"])
    first = rows["list_id"] == "a"
    assert np.isclose(
        w[first & (rows["label"] == 1)].sum(), w[first & (rows["label"] == 0)].sum()
    )


def test_label_agreement_counts_label_pairs_within_a_line_and_across_lines():
    rows = pd.DataFrame(
        {"study_id": ["S", "S", "T", "S"], "model_id": ["L0", "L1", "L0", "L2"],
         "gene_a": "A", "gene_b": "B", "label": [1, 1, 0, 0]}
    )  # fmt: skip
    out = label_agreement(rows)
    # line L0 holds 1 (S) and 0 (T): they disagree
    assert out["same_line"] == {"pairs": 1, "agreement": 0.0, "positive_agreement": 0.0}
    # study S over lines L0, L1, L2 holds 1, 1, 0: one agreeing ordered pair in three
    assert out["same_study"]["agreement"] == 2 / 6
    assert out["same_study"]["positive_agreement"] == 0.5


def test_top_recall_is_the_share_of_the_strongest_dependencies_found():
    measured = np.array([[-3.0, -2.0, 0.0, 1.0], [-1.0, np.nan, 0.0, 2.0]])
    predicted = np.array([[-1.0, 0.0, -2.0, 1.0], [0.0, 5.0, -1.0, 1.0]])
    # line 0 finds one of its two; line 1 (two of three finite labels) finds both
    assert top_recall(predicted, measured, 2) == (0.5 + 1.0) / 2


def test_the_head_learns_a_planted_term_and_is_chosen_on_validation():
    from src.experiments.sl.conditional import decision, fit_head

    rng = np.random.default_rng(0)

    def side(split, lines):
        n = 40 * lines
        ids = np.repeat([f"{split}{i}" for i in range(lines)], 40)
        rows = pd.DataFrame(
            {"split": split, "design": "combinatorial", "condition": "genetic",
             "study_id": "S", "model_id": ids}
        )  # fmt: skip
        rows["list_id"] = rows["model_id"]
        x = rng.normal(size=(n, 7))
        rows["label"] = (x[:, 0] + 0.3 * rng.normal(size=n) > 1.0).astype(int)
        return rows, x

    train, x_train = side("train", 6)
    val, x_val = side("val", 3)
    x_train[0, 2] = np.nan  # a row without every input is left out
    head, records = fit_head(train, x_train, val, x_val, [0.1, 10.0], 5)
    assert [r["penalty"] for r in records] == [0.1, 10.0]
    assert max(r["val_recall_top10"] for r in records) > 0.5
    assert np.isnan(decision(head, x_train[:1]))[0]
