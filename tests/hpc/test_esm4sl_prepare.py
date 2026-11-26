"""The ESM4SL harness (hpc/esm4sl): protein sequences per benchmark gene, the official
three-column pair files, and the score table read back from the official test pass."""

from __future__ import annotations

import importlib.util
import sys

import pandas as pd

from src.experiments.run_dir import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "hpc" / "common"))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "hpc" / "esm4sl" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare, collect = load("prepare"), load("collect")


def test_gene_takes_its_longest_reviewed_sequence_cut_to_the_limit():
    table = pd.DataFrame(
        {
            "Entry": ["P1", "P2", "P3"],
            "Gene Names (primary)": ["TP53", "TP53", "BRCA1"],
            "Sequence": ["MEE", "MEEPQ", "MDL"],
        }
    )
    out = prepare.gene_sequences(table, pd.Index(["BRCA1", "NOGENE", "TP53"]), limit=4)
    assert out.to_dict() == {"BRCA1": "MDL", "TP53": "MEEP"}


def test_pair_file_has_the_official_columns_and_drops_uncovered_pairs():
    pairs = pd.DataFrame({"gene_1": ["A", "A"], "gene_2": ["B", "Z"], "label": [1, 0]})
    out = prepare.pair_file(pairs, pd.Series({"A": 0, "B": 1}))
    assert out.to_dict("list") == {"0": [0], "1": [1], "2": [1]}


def test_logits_map_back_to_the_sorted_symbol_pair():
    genes = pd.DataFrame({"id": [0, 1, 2], "symbol": ["A", "B", "C"]})
    logits = pd.DataFrame(
        {"gene1": [2, 0, 0], "gene2": [0, 1, 1], "probs": [0.3, 0.9, 0.9]}
    )
    out = collect.scores_from_logits(logits, genes)
    assert out.to_dict() == {("A", "C"): 0.3, ("A", "B"): 0.9}
