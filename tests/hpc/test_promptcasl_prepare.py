"""The PromptCASL harness (hpc/promptcasl/prepare.py): omics tables in the official
file format, prompts from the vendored verbalizer, label tables, and the fit-only
statistics. The vendored package is a top-level ``src`` like the repository's own, so
the fixture swaps it in for the duration of the module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED = REPO_ROOT / "third_party" / "promptcasl"


@pytest.fixture(scope="module")
def prepare():
    def foreign(name: str) -> bool:
        return name.split(".")[0] in ("src", "scripts")

    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if foreign(k)}
    sys.path.insert(0, str(VENDORED))
    sys.path.insert(0, str(REPO_ROOT / "hpc" / "common"))
    try:
        spec = importlib.util.spec_from_file_location(
            "promptcasl_prepare", REPO_ROOT / "hpc" / "promptcasl" / "prepare.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.verbalizer()  # imports the vendored src.data_prepare now
        yield module
    finally:
        sys.path.remove(str(VENDORED))
        sys.path.remove(str(REPO_ROOT / "hpc" / "common"))
        for k in [k for k in sys.modules if foreign(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


GENES = ["G1", "G2", "G3"]


def tables() -> dict[str, pd.DataFrame]:
    """L1 has everything; L2 lacks copy number; L3 lacks DepMap expression."""
    return {
        "expression": pd.DataFrame(
            [[1.0, 3.0, 0.0], [2.0, 4.0, 6.0]], index=["L1", "L2"], columns=GENES
        ),
        # DepMap copy number is log2(relative copy number + 1): 1 and 2 copies -> 1 and
        # log2 3
        "copy_number": pd.DataFrame(
            [[1.0, np.log2(3.0), 1.0]], index=["L1"], columns=GENES
        ),
        "mutation": pd.DataFrame(
            [[0, 1, 0], [1, 0, 0]], index=["L1", "L2"], columns=GENES
        ),
        "pseudo_bulk": pd.DataFrame([[5.0, 5.0, 5.0]], index=["L3"], columns=GENES),
    }


def test_line_without_copy_number_keeps_nan_and_still_gets_a_prompt_per_gene(prepare):
    out = prepare.omics_tables(tables(), ["L1", "L2", "L3"])
    assert out["CNGene"].loc["L1"].tolist() == pytest.approx([1.0, 2.0, 1.0])
    assert out["CNGene"].loc["L2"].isna().all()
    prompts = prepare.prompts(
        "L2", "L2 name", out["GE"], out["CNGene"], out["MU"], GENES
    )
    assert set(prompts) == set(GENES)
    # the cut points of copy number need the line's own values: none, so the verbalizer
    # reads a missing value, which it writes as no alteration
    assert "no copy-number alteration" in prompts["G1"]
    assert prompts["G1"].startswith("In the L2 name cell line, G1 is ")
    assert "mutated" in prompts["G1"] and "wild-type" in prompts["G2"]


def test_line_without_depmap_expression_reads_its_pseudo_bulk(prepare):
    out = prepare.omics_tables(tables(), ["L1", "L3"])
    assert out["GE"].loc["L3"].tolist() == [5.0, 5.0, 5.0]
    assert out["GE"].loc["L1"].tolist() == [1.0, 3.0, 0.0]


def test_mutations_are_the_official_long_table_and_a_missing_line_has_none(prepare):
    out = prepare.omics_tables(tables(), ["L1", "L2", "L3"])
    got = sorted(map(tuple, out["MU"][["ModelID", "HugoSymbol"]].to_numpy()))
    assert got == [("L1", "G2"), ("L2", "G1")]


def test_mean_input_line_is_the_mean_of_the_fit_lines(prepare):
    out = prepare.omics_tables(tables(), ["L1", "L2"], fit_lines=["L1", "L2"])
    assert out["GE"].loc[prepare.MEAN_INPUT].tolist() == [1.5, 3.5, 3.0]
    # copy number: the mean of the lines that have it (L2 has none)
    assert out["CNGene"].loc[prepare.MEAN_INPUT].tolist() == pytest.approx(
        [1.0, 2.0, 1.0]
    )
    assert prepare.MEAN_INPUT not in set(out["MU"]["ModelID"])


def test_pseudo_bulk_is_log2_counts_per_million_of_the_summed_cells(prepare):
    counts = np.array([[1, 0, 3], [1, 0, 3]])  # two cells, summed: 2, 0, 6
    out = prepare.pseudo_bulk(counts, GENES)
    assert out.tolist() == pytest.approx(np.log2(np.array([2, 0, 6]) / 8 * 1e6 + 1))


def test_label_table_has_the_official_columns_and_one_row_per_line_pair(prepare):
    rows = pd.DataFrame(
        {
            "model_id": ["L1", "L1", "L1"],
            "gene_1": ["A", "A", "B"],
            "gene_2": ["B", "B", "C"],
            "label": [1, 0, 0],
        }
    )
    out = prepare.label_table(rows)
    assert list(out["gene_pair"]) == ["A|B", "B|C"]
    assert list(out["SL_or_not"]) == ["SL", "Not SL"]  # A|B: mean 0.5 is SL
    assert set(out.columns) >= {
        "gene_pair",
        "cell_line_origin",
        "gene_1",
        "gene_2",
        "fold",
    }


def test_fit_samples_are_the_sl_pairs_and_as_many_non_sl_pairs_of_each_line(prepare):
    rows = pd.DataFrame(
        {
            "model_id": ["L1"] * 5 + ["L2"] * 3,
            "side": ["train"] * 8,
            "condition": ["genetic"] * 8,
            "label": [1, 0, 0, 0, 0, 1, 1, 0],
            "gene_1": list("AAAAABBB"),
            "gene_2": list("BCDEFCDE"),
        }
    )
    out = prepare.fit_samples(rows, ("train",), np.random.default_rng(0))
    got = out.groupby("model_id")["label"].agg(["sum", "size"]).values.tolist()
    assert got == [[1, 2], [2, 3]]


def test_fit_statistics_ignore_the_held_out_lines(prepare):
    ge = pd.DataFrame({"G": [1.0, 3.0, 100.0]}, index=["F1", "F2", "HELD"])
    z = prepare.fit_only_zscore(ge, ["F1", "F2"])(ge)
    assert z["G"].tolist() == pytest.approx([-0.7071, 0.7071, 69.296], abs=1e-3)


def test_lineage_means_come_from_fit_lines_and_fall_back_to_all_fit_lines(prepare):
    activity = {
        "F1": np.array([[1.0], [3.0]]),
        "F2": np.array([[3.0], [5.0]]),
        "HELD": np.array([[100.0], [100.0]]),
    }
    lineage = {"F1": "lung", "F2": "lung", "HELD": "skin", "MEAN": "all"}
    mean = prepare.fit_lineage_means(activity, lineage, ["F1", "F2"])
    assert mean["lung"].tolist() == [2.0, 4.0]
    assert mean["skin"].tolist() == [2.0, 4.0] and mean["all"].tolist() == [2.0, 4.0]


def test_neighbour_ids_follow_the_official_selection(prepare):
    pytest.importorskip("pykeen")  # the official neighbour functions import it
    # 10 entities in a ring plus a hub: pairs share neighbours through the hub
    adj = {i: {(i - 1) % 10, (i + 1) % 10, 10} for i in range(10)}
    adj[10] = set(range(10))
    ids = prepare.neighbor_ids(adj, 0, 5, min_len=4)
    assert ids is not None and ids.shape == (2, 4)
    assert len(set(ids[0])) == 4 and 5 not in ids[0]  # the partner is not a neighbour
    assert prepare.neighbor_ids(adj, 0, 5, min_len=4).tolist() == ids.tolist()
    # an entity with too few neighbours even after two hops gives no row
    assert prepare.neighbor_ids({0: {1}, 1: {0}, 2: set()}, 0, 2, min_len=4) is None


def test_kg_triples_drop_rev_copies_and_keep_gene_symbols(prepare):
    edges = {
        ("gene/protein", "protein_protein", "gene/protein"): np.array([[0, 1], [1, 0]]),
        ("drug", "drug_protein", "gene/protein"): np.array([[0], [1]]),
        ("gene/protein", "rev_drug_protein", "drug"): np.array([[1], [0]]),
    }
    names = {
        "gene/protein": np.array(["tp53", "MDM2"], dtype=object),
        "drug": np.array(["tp53"]),
    }
    out = prepare.kg_triples(edges, names)
    assert sorted(map(tuple, out.to_numpy())) == [
        ("MDM2", "protein_protein", "TP53"),
        ("TP53", "protein_protein", "MDM2"),
        ("drug:tp53", "drug_protein", "MDM2"),
    ]


def test_pairs_are_matched_in_upper_case_and_sorted_again(prepare):
    rows = pd.DataFrame({"gene_a": ["C1orf112", "C1QA"], "gene_b": ["C1QA", "A1BG"]})
    out = prepare.upper_pairs(rows)
    # C1ORF112 sorts before C1QA once both are upper case
    assert out[["gene_1", "gene_2"]].values.tolist() == [
        ["C1ORF112", "C1QA"],
        ["A1BG", "C1QA"],
    ]
    assert out["gene_a"].tolist() == ["C1orf112", "C1QA"]
