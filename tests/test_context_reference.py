"""Reference-table parsers and the leakage guard of the loader."""

from __future__ import annotations

import pandas as pd
import pytest

from src.context_prior.reference import load_reference
from src.data.prepare.build_context_reference import (
    closest_paralogs,
    driver_status,
    parse_corum,
    parse_gmt,
    parse_paralogs,
    parse_progeny,
)

BIOMART = (
    "Gene name\tHuman paralogue associated gene name\t"
    "Paralogue %id. target Human gene identical to query gene\t"
    "Paralogue %id. query gene identical to target Human gene\n"
    "ARID1A\tARID1B\t60\t55\n"
    "ARID1A\tARID1A\t100\t100\n"
    "ARID1A\tARID2\t15\t30\n"
    "ARID1A\t\t\t\n"
    "SMARCA4\tSMARCA2\t75\t74\n"
)


def test_paralogs_take_the_lower_identity_and_drop_self_pairs():
    pairs = parse_paralogs(BIOMART)
    row = pairs.set_index(["gene", "paralog"]).loc[("ARID1A", "ARID1B"), "identity"]
    assert row == 55
    assert not ((pairs.gene == pairs.paralog).any())
    assert len(pairs) == 3


def test_biomart_error_text_raises():
    with pytest.raises(ValueError, match="BioMart"):
        parse_paralogs("Query ERROR: caught BioMart::Exception\n")


def test_closest_paralogs_threshold_cap_and_order():
    pairs = pd.DataFrame(
        {
            "gene": ["A", "A", "A", "B"],
            "paralog": ["X", "Y", "Z", "W"],
            "identity": [30.0, 50.0, 10.0, 25.0],
        }
    )
    kept = closest_paralogs(pairs, min_identity=20.0, per_gene=1)
    assert kept.values.tolist() == [["A", "Y", 50.0], ["B", "W", 25.0]]


def test_corum_keeps_human_complexes_as_long_rows():
    text = (
        "complex_id\torganism\tsubunits_gene_name\n"
        "1\tHuman\tBCL6;HDAC4\n2\tMouse\tA;B\n3\tHuman\tsmarca4;ARID1A\n"
    )
    rows = parse_corum(text)
    assert rows.values.tolist() == [
        [1, "BCL6"],
        [1, "HDAC4"],
        [3, "ARID1A"],
        [3, "SMARCA4"],
    ]


def test_gmt_and_progeny():
    sets = parse_gmt("HALLMARK_X\turl\tA\tb\nHALLMARK_Y\turl\tC\n")
    assert sets.values.tolist() == [
        ["HALLMARK_X", "A"],
        ["HALLMARK_X", "B"],
        ["HALLMARK_Y", "C"],
    ]
    text = (
        "uniprot\tgenesymbol\tentity_type\tsource\tlabel\tvalue\trecord_id\n"
        + "".join(
            f"P\t{gene}\tprotein\tPROGENy\t{label}\t{value}\t{record}\n"
            for gene, record, pathway, weight, p in (
                ("A", 1, "EGFR", 0.5, 0.01),
                ("B", 2, "EGFR", -0.2, 0.2),
                ("C", 3, "EGFR", 0.9, 0.001),
            )
            for label, value in (
                ("pathway", pathway),
                ("weight", weight),
                ("p_value", p),
            )
        )
    )
    top = parse_progeny(text, top=2)
    assert top.values.tolist() == [["EGFR", "C", 0.9], ["EGFR", "A", 0.5]]


def test_driver_rule_reads_only_given_lines():
    hotspot = pd.DataFrame(
        {"KRAS": [1, 0, 0, 0], "BRAF": [0, 0, 0, 0]}, index=["L1", "L2", "L3", "L4"]
    )
    damaging = pd.DataFrame(
        {"KRAS": [0, 0, 0, 0], "BRAF": [0, 1, 0, 0], "TTN": [2, 2, 2, 2]},
        index=["L1", "L2", "L3", "L4"],
    )
    status = driver_status(hotspot, damaging, ["L1", "L2", "L3"], min_fraction=0.3)
    assert list(status.columns) == ["KRAS", "BRAF"]  # TTN is not a hotspot gene
    assert list(status.index) == ["L1", "L2", "L3"]
    assert status.loc["L2", "BRAF"] == 1


def test_loader_refuses_blocked_lines(tmp_path):
    pd.DataFrame({"gene": ["A"], "paralog": ["B"], "identity": [50.0]}).to_csv(
        tmp_path / "paralogs.tsv.gz", sep="\t", index=False
    )
    pd.DataFrame({"complex_id": [1], "gene": ["A"]}).to_csv(
        tmp_path / "complexes.tsv", sep="\t", index=False
    )
    pd.DataFrame({"gene_set": ["S"], "gene": ["A"]}).to_csv(
        tmp_path / "hallmark.tsv", sep="\t", index=False
    )
    pd.DataFrame({"pathway": ["P"], "gene": ["A"], "weight": [1.0]}).to_csv(
        tmp_path / "progeny.tsv", sep="\t", index=False
    )
    pd.DataFrame({"model_id": ["L1", "V1"], "KRAS": [1, 0]}).to_csv(
        tmp_path / "drivers.tsv", sep="\t", index=False
    )
    pd.DataFrame({"model_id": ["L1"], "msi_score": [3.0]}).to_csv(
        tmp_path / "msi.tsv", sep="\t", index=False
    )
    assert load_reference(tmp_path, blocked=set()).drivers.shape == (2, 1)
    with pytest.raises(ValueError, match="held-out"):
        load_reference(tmp_path, blocked={"V1"})
