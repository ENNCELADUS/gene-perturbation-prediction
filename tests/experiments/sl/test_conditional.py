"""The SL backbone on synthetic lines: a term is the reader's change of B's prediction
when A is lost in the line's row (for an affine reader, A's fall times B's response
to A), the buffer-aware reader moves B more when A is B's only expressed paralog, the
context-ablated score is one value per pair, the partner flag, two DepMap screens of
a line, the head's weights follow the benchmark, and the label-agreement ceiling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.accuracy import top_recall
from src.experiments.sl.benchmark_baselines import label_agreement
from src.experiments.sl.conditional import (
    FULL,
    partner_rows,
    terms,
    two_screens,
    weights,
)
from src.linear_model.linear_model import (
    LinearModelInputs,
    LinearModelSpec,
    Stage,
    fit_linear_model,
    total,
)
from src.linear_model.reference import Reference
from src.linear_model.views import fit_expression_components
from tests.linear_model.test_linear_model import synthetic

DEFAULT = LinearModelSpec(
    (Stage("expression_components", 0.01), Stage("data_selected", 0.01, selected=2))
)
BUFFER = LinearModelSpec(
    (
        Stage("expression_components", 0.01),
        Stage("partners", 0.01),
        Stage("data_selected", 0.01, selected=2),
    )
)


def reader():
    inputs, ids = synthetic()
    fitted = fit_linear_model(DEFAULT, inputs, fit_lines=ids, encoder_lines=ids)
    return fitted, inputs.expression


def buffer_reader():
    """B's paralogs are A and C; B is a dependency (negative residual) when both are
    low: its residual is max(A, C), the sum of its paralogs minus the lowest."""
    rng = np.random.default_rng(0)
    ids = [f"L{i}" for i in range(300)]
    space = ["A", "B", "C", "N0", "N1", "N2", "N3", "N4"]
    expression = pd.DataFrame(rng.normal(size=(300, 8)), index=ids, columns=space)
    residual = pd.DataFrame(
        {"B": np.maximum(expression["A"], expression["C"])
         + 0.05 * rng.normal(size=300),
         "N0": expression["N1"] + 0.1 * rng.normal(size=300)},
        index=ids,
    )  # fmt: skip
    reference = Reference(
        paralogs=pd.DataFrame(
            {
                "gene": ["B", "B", "A"],
                "paralog": ["A", "C", "B"],
                "identity": [60.0, 50.0, 60.0],
            }
        ),  # fmt: skip
        complexes=pd.DataFrame({"complex_id": [1, 1], "gene": ["N3", "N4"]}),
        hallmark=pd.DataFrame({"gene_set": ["S"], "gene": ["N2"]}),
        progeny=pd.DataFrame({"pathway": ["P"], "gene": ["N2"], "weight": [1.0]}),
        drivers=pd.DataFrame({"KRAS": 0}, index=pd.Index(ids, name="model_id")),
        msi=pd.Series(0.0, index=pd.Index(ids, name="model_id")),
    )
    inputs = LinearModelInputs(
        expression=expression,
        residual=residual,
        components=fit_expression_components(expression, 4),
        reference=reference,
        lineage=pd.Series("Lung", index=ids),
        patients={m: m for m in ids},
    )
    fitted = fit_linear_model(BUFFER, inputs, fit_lines=ids, encoder_lines=ids)
    return fitted, expression, reference


def test_for_an_affine_reader_a_term_is_the_fall_times_the_response():
    fitted, expression = reader()
    origin = expression.mean(axis=0)
    low = expression.quantile(0.1)
    lines = expression.iloc[:3].copy()
    lines.loc["L1", "E1"] = low["E1"] - 1.0  # already below the low level
    rows = pd.DataFrame(
        {"model_id": ["L0", "L1", "L0", "L2", "L0"],
         "design": ["combinatorial", "combinatorial", "anchor", "combinatorial",
                    "anchor"],
         "gene_a": ["E1", "E1", "E1", "E0", "Q"], "gene_b": ["E0", "E0", "E0", "E1",
                                                            "E0"]}
    )  # fmt: skip
    out = terms(rows, lines, low, dict.fromkeys(lines.index, FULL), {FULL: fitted})

    def response(a, b):  # the backbone's affine shortcut, from the mean profile
        up = origin.copy()
        up[a] += 1.0
        moved = total(fitted.predict(pd.DataFrame([up, origin])))
        return moved.iloc[0][b] - moved.iloc[1][b]

    def fall(line, gene):
        return min(low[gene] - lines.loc[line, gene], 0.0)

    for i in (0, 3):
        line, a, b = rows.loc[i, ["model_id", "gene_a", "gene_b"]]
        assert np.isclose(out[i, 0], fall(line, a) * response(a, b))
        assert np.isclose(out[i, 1], fall(line, b) * response(b, a))
    assert out[1, 0] == 0.0  # no fall: A is already low in L1
    assert out[2, 1] == 0.0  # an anchor list reads the first term only
    assert np.isnan(out[4, 0])  # a gene without an expression column


def test_the_loss_of_the_only_expressed_paralog_moves_b_most():
    fitted, expression, _ = buffer_reader()
    low = expression.quantile(0.1)
    lines = pd.DataFrame(0.0, index=["only_a", "a_and_c"], columns=expression.columns)
    lines["A"] = 1.5
    lines["C"] = [-2.5, 1.5]  # C below its low level in the first line only
    rows = pd.DataFrame(
        {"model_id": ["only_a", "a_and_c"], "design": "anchor", "gene_a": "A",
         "gene_b": "B"}
    )  # fmt: skip
    out = terms(rows, lines, low, dict.fromkeys(lines.index, FULL), {FULL: fitted})
    only_a, a_and_c = out[:, 0]
    assert only_a < -1.0  # B becomes a dependency: its last buffer is lost
    assert abs(a_and_c) < 0.5 * abs(only_a)  # C still buffers B


def test_the_context_ablated_terms_are_one_value_per_pair():
    fitted, expression, _ = buffer_reader()
    origin = expression.mean(axis=0)
    ablated = pd.DataFrame([origin] * 3, index=["L0", "L1", "L2"])
    rows = pd.DataFrame(
        {"model_id": ["L0", "L1", "L2"], "design": ["combinatorial"] * 3,
         "gene_a": ["A"] * 3, "gene_b": ["B"] * 3}
    )  # fmt: skip
    out = terms(rows, ablated, expression.quantile(0.1),
                dict.fromkeys(ablated.index, FULL), {FULL: fitted})  # fmt: skip
    # only B has a prediction: the term of B's loss on A is undefined
    assert np.isfinite(out[:, 0]).all() and np.isnan(out[:, 1]).all()
    assert np.ptp(out[:, 0]) == 0.0


def test_the_partner_flag_reads_both_directions_of_a_pair_only():
    _, _, reference = buffer_reader()
    rows = pd.DataFrame(
        {"design": ["combinatorial", "anchor", "anchor", "combinatorial",
                    "combinatorial"],
         "gene_a": ["C", "A", "B", "N3", "A"], "gene_b": ["B", "B", "C", "N4", "N0"]}
    )  # fmt: skip
    # C is a paralog of B; B is not listed for C, so the anchor list B -> C is not
    assert partner_rows(rows, reference).tolist() == [True, True, False, True, False]


def test_two_screens_pairs_the_first_two_screens_of_a_line():
    screens = pd.DataFrame({"G": [1.0, 2.0, 3.0, 4.0]}, index=["S3", "S1", "S2", "S4"])
    screen_map = pd.DataFrame(
        {"ScreenID": ["S1", "S2", "S3", "S4"], "ModelID": ["M0", "M0", "M0", "M1"]}
    )
    first, second = two_screens(screens, screen_map)
    assert list(first.index) == ["M0"]  # M1 has one screen
    assert (first.loc["M0", "G"], second.loc["M0", "G"]) == (2.0, 3.0)


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
