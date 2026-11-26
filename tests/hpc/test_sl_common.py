"""The shared pieces of the published-baseline harnesses (hpc/common/sl_common.py):
rows of the fit and held-out lines, pair labels, and score tables that hold each stored
orientation and tie the rows a model cannot score."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd

from src.experiments.run_dir import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "sl_common", REPO_ROOT / "hpc" / "common" / "sl_common.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

ROWS = pd.DataFrame(
    {
        "model_id": ["T1", "T1", "T2", "V1", "V1", "X1"],
        "gene_a": ["B", "A", "A", "B", "C", "A"],
        "gene_b": ["A", "C", "B", "A", "Z", "B"],
        "label": [1.0, 0.0, 0.0, 1.0, 0.0, None],
        "condition": ["genetic"] * 6,
    }
)
SPLIT = {
    "train": ["T1"],
    "bulk_labelled": ["T2"],
    "val": ["V1"],
    "test": [],
    "unlabeled_train": ["X1"],
}


def rows_of(tmp_path) -> pd.DataFrame:
    ROWS.to_parquet(tmp_path / "labels.parquet")
    (tmp_path / "split.json").write_text(json.dumps(SPLIT))
    return common.benchmark_rows(tmp_path / "labels.parquet", tmp_path / "split.json")


def test_rows_keep_labelled_split_lines_and_sort_the_pair(tmp_path):
    rows = rows_of(tmp_path)
    assert list(rows["side"]) == ["train", "train", "bulk_labelled", "val", "val"]
    assert list(rows["gene_1"] + rows["gene_2"]) == ["AB", "AC", "AB", "AB", "CZ"]


def test_pair_is_sl_when_half_of_its_fit_rows_are_sl(tmp_path):
    pairs = common.pair_labels(rows_of(tmp_path), common.FIT_SIDES)
    assert dict(zip(pairs["gene_1"] + pairs["gene_2"], pairs["label"])) == {
        "AB": 1,
        "AC": 0,
    }


def test_balanced_keeps_all_sl_rows_and_as_many_non_sl_rows():
    pairs = pd.DataFrame(
        {
            "gene_1": list("AAAAAAA"),
            "gene_2": list("BCDEFGH"),
            "label": [1, 1, 0, 0, 0, 0, 0],
        }
    )
    out = common.balanced(pairs, np.random.default_rng(0))
    assert len(out) == 4 and int(out["label"].sum()) == 2


def test_drop_pair_edges_removes_both_orders(tmp_path):
    edges = pd.DataFrame({"gene1": ["A", "B", "A"], "gene2": ["B", "A", "Q"]})
    assert common.drop_pair_edges(edges, rows_of(tmp_path)).values.tolist() == [
        ["A", "Q"]
    ]


def test_pair_scores_cover_each_stored_orientation_and_tie_the_rest(tmp_path):
    rows = rows_of(tmp_path)
    scores = pd.Series(
        {("A", "B"): 0.9, ("A", "C"): 0.1}
    )  # (C, Z): a gene outside the model
    report = common.write_pair_scores(tmp_path, rows, scores)
    table = pd.read_parquet(tmp_path / "scores.parquet").set_index(["gene_a", "gene_b"])
    assert (
        table.loc[("B", "A"), "score"] == 0.9 and table.loc[("A", "C"), "score"] == 0.1
    )
    assert ("A", "B") not in table.index  # stored by a bulk-only line alone: not scored
    assert not table.loc[("C", "Z"), "covered"]
    assert (
        table.loc[("C", "Z"), "score"] == 0.9
    )  # the median of the covered held-out rows
    assert report["held_rows_covered_share"] == 0.5


def test_line_scores_keep_the_line(tmp_path):
    rows = rows_of(tmp_path)
    scores = pd.Series({("T1", "A", "B"): 0.2, ("V1", "A", "B"): 0.7})
    common.write_line_scores(tmp_path, rows, scores, name="own")
    table = pd.read_parquet(tmp_path / "own.parquet").set_index(
        ["model_id", "gene_a", "gene_b"]
    )
    assert (
        table.loc[("V1", "B", "A"), "score"] == 0.7
        and table.loc[("T1", "B", "A"), "score"] == 0.2
    )
    assert (
        not table.loc[("V1", "C", "Z"), "covered"]
        and not table.loc[("T1", "A", "C"), "covered"]
    )
