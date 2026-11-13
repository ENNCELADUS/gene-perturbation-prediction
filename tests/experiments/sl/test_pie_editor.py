"""The PIE editor on synthetic lines: a zero change gives the line's own row and zero
terms, a change moves B through the reader, the unit slope, the store of changes, the
permutation of lines, PIE's input from the cell files, the measured change on the
panel, and the own-task scores against the other lines' mean response."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.experiments.sl.conditional import FULL, terms
from src.experiments.sl.pie_editor import (
    ChangeStore,
    ProfileEdit,
    derangement,
    queries,
    unit_slope,
)
from src.experiments.sl.pie_pass import (
    cell_inputs,
    measured_change,
    own_task_table,
    response_scores,
)
from src.linear_model.bridge import fit_bridge
from src.linear_model.space import quantile_normalize, quantile_reference
from tests.experiments.sl.test_conditional import reader


def world():
    """The fitted reader, raw pseudo-bulk of its lines, a normaliser and a bridge
    into the reader's space, and each line's bridged row."""
    fitted, expression = reader()
    rng = np.random.default_rng(1)
    raw = expression + 5.0 + 0.1 * rng.normal(size=expression.shape)
    profile = quantile_reference(raw)

    def normalize(frame):
        return quantile_normalize(frame, profile)

    bridge = fit_bridge(normalize(raw), expression)
    rows = bridge.apply(normalize(raw))
    return fitted, raw, normalize, {FULL: bridge}, rows


PAIRS = pd.DataFrame(
    {"model_id": ["L0", "L1", "L2"], "design": ["combinatorial", "anchor", "anchor"],
     "gene_a": ["E3", "E3", "E5"], "gene_b": ["E0", "E0", "Q"]}
)  # fmt: skip


def test_a_zero_change_gives_the_lines_row_and_zero_terms():
    fitted, raw, normalize, bridges, rows = world()
    panel = list(raw.columns[:8])
    zero = ProfileEdit(raw.columns, lambda m, g: np.zeros((len(m), len(panel))),
                       raw, panel, np.ones(len(panel)), normalize, bridges)  # fmt: skip
    edited = zero(np.arange(5), np.zeros(5, dtype=int), FULL)
    assert np.abs(edited - rows.iloc[:5].to_numpy()).max() < 1e-12
    pairs = PAIRS.assign(gene_a=["E1", "E3", "E5"])  # both genes predicted in row 0
    out = terms(pairs, rows, None, dict.fromkeys(rows.index, FULL), {FULL: fitted},
                zero)  # fmt: skip
    assert np.isfinite(out).all() and np.abs(out).max() < 1e-10


def test_a_change_reaches_b_through_the_reader_and_a_missing_gene_is_not_covered():
    fitted, raw, normalize, bridges, rows = world()
    panel = list(raw.columns)
    position = panel.index("E3")

    def change(models, genes):
        out = np.zeros((len(models), len(panel)))
        out[:, position] = -5.0  # E3 falls in every line that loses it
        return out

    edit = ProfileEdit(["E3"], change, raw, panel, np.ones(len(panel)), normalize,
                       bridges)  # fmt: skip
    out = terms(PAIRS, rows, None, dict.fromkeys(rows.index, FULL), {FULL: fitted},
                edit)  # fmt: skip
    # E0's residual follows E3: its prediction falls when E3 falls
    assert out[0, 0] < 0 and out[1, 0] < 0
    assert np.isnan(out[0, 1])  # E0 lost: the edit cannot remove it
    assert np.isnan(out[2, 0])  # E5 lost: not a gene the edit knows


def test_the_unit_slope_maps_pies_input_to_the_raw_row():
    rng = np.random.default_rng(0)
    pie_input = pd.DataFrame(rng.normal(size=(30, 3)), columns=["A", "B", "C"])
    raw = pd.DataFrame(
        {"A": 2.0 * pie_input["A"] + 1.0, "B": 0.5 * pie_input["B"],
         "C": pie_input["C"], "D": 0.0}
    )  # fmt: skip
    assert np.allclose(unit_slope(pie_input, raw), [2.0, 0.5, 1.0])


def test_the_raw_row_moves_by_the_slope_times_the_change():
    _, raw, normalize, bridges, _ = world()
    panel = ["E1", "E2"]

    def change(models, genes):
        return np.array([[1.0, np.nan]] * len(models))

    edit = ProfileEdit(["E9"], change, raw, panel, np.array([3.0, 2.0]), normalize,
                       bridges)  # fmt: skip
    after = edit.after(np.array([4]), np.array([0]))
    moved = after.iloc[0] - raw.iloc[4]
    assert moved["E1"] == pytest.approx(3.0)
    assert (moved.drop("E1") == 0.0).all()  # a NaN change is no change


def test_the_store_reads_each_asked_change(tmp_path):
    lines = ["L0", "L1", "L2"]
    for g, gene in enumerate(["A", "B"]):
        np.save(tmp_path / f"{gene}.npy",
                (np.arange(6).reshape(3, 2) + 10 * g).astype(np.float16))  # fmt: skip
    pairs = pd.DataFrame({"source": ["L2", "L0", "L2"], "gene": ["A", "B", "A"]})
    store = ChangeStore(tmp_path, lines, pairs, 2)
    out = store(np.array(["L0", "L2"]), np.array(["B", "A"]))
    assert np.array_equal(out, [[10.0, 11.0], [4.0, 5.0]])
    with pytest.raises(KeyError):
        store(np.array(["L1"]), np.array(["A"]))


def test_the_permutation_keeps_no_line_and_is_fixed_by_the_seed():
    lines = [f"L{i}" for i in range(86)]
    other = derangement(lines, 0)
    assert sorted(other) == sorted(other.values()) == sorted(lines)
    assert all(m != other[m] for m in lines)
    assert other == derangement(list(reversed(lines)), 0)
    assert other != derangement(lines, 1)


def test_queries_edit_a_of_every_row_and_b_of_a_pair():
    asked = queries(PAIRS)
    assert set(map(tuple, asked.to_numpy())) == {
        ("L0", "E3"),
        ("L1", "E3"),
        ("L2", "E5"),
        ("L0", "E0"),
    }


def test_pies_input_fills_an_unlisted_value_and_refuses_another_gene_list(tmp_path):
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "genes.json").write_text(json.dumps({"genes": ["A", "B", "C"]}))
    for model_id, values, listed in (("L0", [1.0, 2.0, 3.0], [1, 1, 0]),
                                     ("L1", [3.0, 4.0, 9.0], [1, 1, 1])):  # fmt: skip
        np.savez(cells / f"{model_id}.npz", pie_input=np.array(values),
                 listed=np.array(listed, dtype=bool))  # fmt: skip
    frame = cell_inputs(tmp_path, ["L0", "L1"], ["C", "A"])
    assert frame.loc["L0", "C"] == 9.0  # the mean of the lines listing C
    assert frame.loc["L1", "A"] == 3.0
    np.savez(cells / "L2.npz", pie_input=np.zeros(2), listed=np.ones(2, dtype=bool))
    with pytest.raises(ValueError, match="genes in cells/genes.json"):
        cell_inputs(tmp_path, ["L2"], ["A"])


def test_the_measured_change_is_read_on_the_panel_by_symbol():
    delta = np.array([[1.0, 2.0, 3.0]])
    out = measured_change(delta, ["ensg1", "aars1", "Kras"], ["KRAS", "AARS1", "TP53"])
    assert np.array_equal(out[0, :2], [3.0, 2.0]) and np.isnan(out[0, 2])


def test_pie_against_the_other_lines_mean_response():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=50)
    own = rng.normal(size=50)
    delta = np.stack([shared + own, shared - 0.5 * own, shared + rng.normal(size=50)])
    index = {("k562", "G"): 0, ("rpe1", "G"): 1, ("jurkat", "G"): 2,
             ("k562", "H"): 3}  # fmt: skip
    delta = np.vstack([delta, rng.normal(size=(1, 50))])
    predictions = pd.DataFrame(
        {"context": ["k562", "k562"], "perturbation": ["G", "H"],
         "delta_p_pred": [shared + own, np.zeros(50)]}
    )  # fmt: skip
    table = own_task_table(predictions, delta, index)
    assert list(table["perturbation"]) == ["G"]  # H: no other line measured it
    row = table.iloc[0]
    assert row["others"] == 2
    assert row["pie_pearson"] == pytest.approx(1.0)
    assert row["pie_deviation_pearson"] == pytest.approx(1.0)
    assert row["other_lines_mean_pearson"] < 0.9
    assert row["one_other_line_deviation_pearson"] < 1.0
    flat = response_scores(np.ones(50), shared, shared, shared)
    assert np.isnan(flat["pie_pearson"])  # a constant prediction has no correlation
