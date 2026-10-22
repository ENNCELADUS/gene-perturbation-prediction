from __future__ import annotations

import pandas as pd
import pytest

from src.data.extra_lines import ExtraSingleCellLines
from src.data.build.build_sl_screen_benchmark import (
    finish_rows,
    geneeffect_namer,
    kebabci_labels,
    line_sides,
    symbol_mapper,
)
from src.data.splits import FixedSplit


def test_kebabci_rule_needs_top_decile_gemini_and_low_lfc() -> None:
    gemini = pd.Series([float(value) for value in range(10)])
    lfc = pd.Series([-1.0] * 9 + [-1.0])
    lfc[8] = 0.0
    label, cut = kebabci_labels(gemini, lfc, -0.51)
    assert cut == pytest.approx(8.1)
    assert label.tolist() == [0] * 9 + [1]


def test_kebabci_rule_leaves_rows_without_lfc_unlabelled() -> None:
    gemini = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    lfc = pd.Series([-1.0] * 9 + [float("nan")])
    label, _ = kebabci_labels(gemini, lfc, -0.51)
    assert label.isna().tolist() == [False] * 9 + [True]


def test_kebabci_rule_rejects_unscored_rows() -> None:
    with pytest.raises(ValueError, match="scored rows"):
        kebabci_labels(pd.Series([1.0, float("nan")]), pd.Series([0.0, 0.0]), -0.5)


def _mapper() -> object:
    hgnc = pd.DataFrame(
        {
            "symbol": ["MAP3K21", "MAP3K10", "GENEX", "GENEY"],
            "prev_symbol": ["KIAA1804", None, "OLDX", "OLDX"],
            "alias_symbol": [None, "ALIAS10", None, None],
        }
    )
    return symbol_mapper(hgnc)


def test_symbol_mapper_prefers_approved_then_unique_previous_then_alias() -> None:
    current = _mapper()
    assert current("MAP3K10") == "MAP3K10"
    assert current("KIAA1804") == "MAP3K21"
    assert current("ALIAS10") == "MAP3K10"
    assert current("OLDX") == "OLDX"  # ambiguous previous symbol stays as published
    assert current("UNKNOWN") == "UNKNOWN"


def test_genes_take_their_geneeffect_symbol_when_hgnc_has_renamed_them() -> None:
    hgnc = pd.DataFrame(
        {
            "symbol": ["SACK1A", "MAP3K21"],
            "prev_symbol": ["FAM83A", "KIAA1804"],
            "alias_symbol": [None, None],
        }
    )
    name = geneeffect_namer(symbol_mapper(hgnc), {"FAM83A", "MAP3K10"})
    assert name("FAM83A") == "FAM83A"  # DepMap 26Q1 still uses the old symbol
    assert name("SACK1A") == "FAM83A"
    assert name("KIAA1804") == "MAP3K21"  # outside GeneEffect: current HGNC symbol
    assert name("MAP3K10") == "MAP3K10"


def _sides() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "screen": ["kl", "kl"],
            "screen_line": ["L1", "L2"],
            "model_id": ["ACH-1", "ACH-2"],
            "split": ["train", "test"],
            "stratum": ["single_cell", "single_cell"],
            "line_reason": ["", ""],
        }
    )


def test_constructs_and_old_symbols_aggregate_to_one_pair_and_self_pairs_drop() -> None:
    frame = pd.DataFrame(
        {
            "screen_line": ["L1", "L1", "L1", "L2", "L2"],
            "screen_gene_a": ["KIAA1804", "MAP3K21", "KIAA1804", "MAP3K21", "MAP3K21"],
            "screen_gene_b": ["MAP3K10", "MAP3K10", "MAP3K21", "MAP3K10", "MAP3K10"],
            "gemini": [0.9, 0.2, 0.5, 0.1, 0.3],
            "lfc": [-2.0, -0.1, -1.0, -0.2, float("nan")],
            "fdr": [0.01, 0.5, 0.2, 0.6, 0.7],
            "n_constructs": [1, 1, 1, 1, 1],
            "label": pd.array([1, 0, 0, 0, pd.NA], dtype="Int8"),
        }
    )
    rows = finish_rows(frame, "kl", _sides(), _mapper(), {"MAP3K10", "MAP3K21"})
    assert rows.attrs["same_gene_rows"] == 1
    assert len(rows) == 2
    first = rows.set_index("screen_line").loc["L1"]
    assert (first["gene_a"], first["gene_b"]) == ("MAP3K10", "MAP3K21")
    assert first["label"] == 1 and first["n_constructs"] == 2
    assert first["gemini"] == 0.9 and first["lfc"] == -2.0
    assert first["screen_gene_b"] == "KIAA1804|MAP3K21"
    second = rows.set_index("screen_line").loc["L2"]
    assert pd.isna(second["label"]) and second["unlabelled_reason"] == "no LFC"
    assert second["split"] == "test" and bool(second["both_in_geneeffect"])


def test_single_depleted_non_hit_is_unlabelled_with_its_reason() -> None:
    frame = pd.DataFrame(
        {
            "screen_line": ["L1", "L2"],
            "screen_gene_a": ["MAP3K10", "MAP3K10"],
            "screen_gene_b": ["MAP3K21", "MAP3K21"],
            "gi": [-1.0, 0.1],
            "fdr": [0.001, 0.9],
            "n_constructs": [1, 1],
            "label": pd.array([1, pd.NA], dtype="Int8"),
            "single_depleted": [False, True],
        }
    )
    rows = finish_rows(frame, "kl", _sides(), _mapper(), {"MAP3K10"})
    assert rows["unlabelled_reason"].tolist() == ["", "single-depleted gene"]
    assert not rows["both_in_geneeffect"].any()


def test_finish_rows_refuses_a_line_missing_from_the_map() -> None:
    frame = pd.DataFrame(
        {
            "screen_line": ["L9"],
            "screen_gene_a": ["MAP3K10"],
            "screen_gene_b": ["MAP3K21"],
            "n_constructs": [1],
            "label": pd.array([0], dtype="Int8"),
        }
    )
    with pytest.raises(ValueError, match="absent from the line map"):
        finish_rows(frame, "kl", _sides(), _mapper(), set())


def test_line_sides_follow_the_nested_split_and_the_extra_lines() -> None:
    split = FixedSplit(
        train=("T1", "PC9"), val=("V1",), test=("E1",), unlabeled_train=("PC9",)
    )
    lines = pd.DataFrame(
        {
            "screen": ["s"] * 8,
            "screen_line": list("abcdefgh"),
            "model_id": ["V1", "E1", "T1", "PC9", "SC1", "B1", "SCU", ""],
        }
    )
    sides = line_sides(
        lines,
        split,
        extra_bulk_labelled={"B1", "SC1"},
        extra_single_cell=ExtraSingleCellLines(
            labelled={"SC1": {}}, unlabelled=("SCU",), pending={}, excluded={}
        ),
    )
    assert sides["split"].tolist() == [
        "val",
        "test",
        "train",
        "none",
        "train",
        "train",
        "none",
        "none",
    ]
    assert sides["stratum"].tolist() == [
        "single_cell",
        "single_cell",
        "single_cell",
        "unused",
        "single_cell",
        "bulk",
        "unused",
        "unused",
    ]
    assert sides.loc[7, "line_reason"] == "no DepMap ModelID"
