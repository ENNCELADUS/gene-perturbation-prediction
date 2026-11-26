"""The MVGCN-iSL harness inputs (hpc/mvgcn_isl/prepare.py): held-out lines read stage
one's prediction, never measured GeneEffect; node features follow the official order
and zero fill."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from src.experiments.run_dir import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "hpc" / "common"))
spec = importlib.util.spec_from_file_location(
    "mvgcn_prepare", REPO_ROOT / "hpc" / "mvgcn_isl" / "prepare.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def frame(**lines):
    return pd.DataFrame(lines, index=["G1", "G2"]).T


def test_held_out_line_reads_predicted_gene_effect_only():
    measured = pd.DataFrame(
        {"G1": [-1.0, -2.0], "G2": [0.0, 0.5], "G3": [9.0, 9.0]}, index=["FIT", "HELD"]
    )
    predicted = pd.DataFrame({"G1": [-0.3], "G2": [0.1]}, index=["HELD"])
    table = prepare.gene_effect_input(measured, predicted, held_lines=["HELD"])
    assert list(table.columns) == ["G1", "G2"]  # cut to the predicted genes
    assert table.loc["HELD"].tolist() == [-0.3, 0.1]
    assert table.loc["FIT"].tolist() == [-1.0, 0.0]


def test_a_measured_held_out_profile_is_refused():
    measured = pd.DataFrame({"G1": [-1.0, -2.0]}, index=["FIT", "HELD"])
    with pytest.raises(ValueError, match="HELD"):
        prepare.assert_no_held_gene_effect(measured, held_lines=["HELD"])
    prepare.assert_no_held_gene_effect(measured.loc[["FIT"]], held_lines=["HELD"])


def test_node_features_follow_the_official_order_and_zero_fill():
    tables = {
        "exp": frame(L=[3.0, 0.0]),
        "mut": frame(L=[1.0, 0.0]),
        "cnv": frame(L=[1.0, 2.0]),
        "ess": pd.DataFrame({"G1": [-0.5]}, index=["L"]),  # G2 has no GeneEffect
    }
    x = prepare.node_features(tables, "L", pd.Index(["G1", "G2", "G9"]))
    assert x.tolist() == [
        [3.0, 1.0, 1.0, -0.5],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]


def test_a_line_a_table_lacks_reads_zero_for_that_feature():
    tables = {k: frame(L=[1.0, 1.0]) for k in ("exp", "mut", "ess")}
    tables["cnv"] = frame(OTHER=[5.0, 5.0])
    x = prepare.node_features(tables, "L", pd.Index(["G1", "G2"]))
    assert x[:, 2].tolist() == [0.0, 0.0] and x[:, 0].tolist() == [1.0, 1.0]


def test_line_graph_keeps_edges_between_expressed_genes():
    edges = np.array([[0, 1, 2], [1, 2, 0]])
    kept = prepare.line_edges(edges, expressed=np.array([True, True, False]))
    assert kept.tolist() == [[0], [1]]


def test_graph_edges_drop_benchmark_pairs_and_index_genes():
    edges = pd.DataFrame({"gene1": ["a", "b", "c"], "gene2": ["b", "a", "d"]})
    rows = pd.DataFrame({"gene_1": ["A"], "gene_2": ["B"]})
    index = {"A": 0, "B": 1, "C": 2, "D": 3}
    out = prepare.graph_edges(edges, rows, index)
    assert out.tolist() == [[2, 3], [3, 2]]  # symmetric, upper-case, pair edge gone


SPLIT = {
    "train": ["A"],
    "bulk_labelled": ["B"],
    "val": ["V"],
    "test": ["T"],
    "unlabeled_train": ["X"],
}


def test_coessentiality_reads_listed_fit_lines_with_gene_effect_only():
    measured = pd.DataFrame({"G1": range(7)}, index=["A", "B", "V", "T", "X", "Z", "C"])
    table = prepare.coessentiality_input(measured, SPLIT)
    # not the held-out lines, nor a line the split does not list as train or bulk
    assert sorted(table.index) == ["A", "B"]


def test_coessentiality_refuses_a_line_that_is_fit_and_held_out():
    measured = pd.DataFrame({"G1": [0.0]}, index=["A"])
    with pytest.raises(ValueError, match="A"):
        prepare.coessentiality_input(measured, {**SPLIT, "val": ["A"]})
