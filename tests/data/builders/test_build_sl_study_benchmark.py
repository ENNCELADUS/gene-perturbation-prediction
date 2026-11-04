"""The multi-study SL benchmark builder: lines map to reviewed ModelIDs or line_ids,
symbols to GeneEffect names, pairs are ordered, and constructs of one pair merge."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.builders import build_sl_study_benchmark as builder
from src.data.builders.sl_study_readers import COLUMNS, READERS
from src.data.depmap import IDENTITY_COLUMNS, read_line_identities


def reader(raw):
    rows = [  # line, gene_a, gene_b, label
        # one pair, two constructs
        ("A549", "OLD1", "G0", 1), ("A549", "G0", "OLD1", 0),
        ("A549", "G2", "G0", 0), ("H1299", "G0", "G2", pd.NA), ("Odd", "G0", "G2", 1),
    ]  # fmt: skip
    frame = pd.DataFrame(rows, columns=["line", "gene_a", "gene_b", "label"])
    return frame.assign(
        study_id="S", list_id="S|" + frame["line"], anchor_gene="", anchor_genotype="",
        score=1.0, fdr=0.01, label=frame["label"].astype("Int8"),
        unlabelled_reason=frame["label"].isna().map({True: "depleted", False: ""}),
    )[list(COLUMNS)]  # fmt: skip


def identities(path, spellings):
    rows = {column: [""] for column in IDENTITY_COLUMNS}
    rows |= {"line_id": ["CVCL_TEST"], "spellings": [spellings],
             "patient_id": ["CVCL_TEST"], "lineage": ["Eye"],
             "non_cancer": ["true"]}  # fmt: skip
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build(tmp_path, monkeypatch):
    monkeypatch.setitem(builder.READERS, "S", reader)
    rules = pd.DataFrame(
        {"study_id": ["S", "Q"], "reader": ["S", ""], "design": ["combinatorial", ""],
         "status": ["ready", "unusable"],
         "lines": ["A549 [ACH-000681]; H1299 [ACH-000510]; Odd", ""]}
    )  # fmt: skip
    rules.to_csv(tmp_path / "rules.csv", index=False)
    pd.DataFrame(columns=["ModelID", "G0 (1)", "G1 (2)", "G2 (3)"]).to_csv(
        tmp_path / "ge.csv", index=False
    )
    pd.DataFrame(
        {
            "symbol": ["G0", "G1", "G2"],
            "prev_symbol": ["", "OLD1", ""],
            "alias_symbol": "",
        }
    ).to_csv(tmp_path / "hgnc.txt", sep="\t", index=False)
    out = tmp_path / "bench"
    manifest = builder.build(
        tmp_path / "rules.csv",
        tmp_path,
        tmp_path / "ge.csv",
        tmp_path / "hgnc.txt",
        out,
        identities(tmp_path / "ids.csv", "S:Odd"),
    )
    rows = pd.read_parquet(out / "labels.parquet").set_index(
        ["line", "gene_a", "gene_b"]
    )
    merged = rows.loc[("A549", "G0", "G1")]
    assert merged["label"] == 1 and merged["n_constructs"] == 2
    assert merged["screen_gene_b"] == "OLD1"
    assert rows.loc[("A549", "G0", "G2"), "model_id"] == "ACH-000681"
    assert pd.isna(rows.loc[("H1299", "G0", "G2"), "label"])
    assert rows.loc[("Odd", "G0", "G2"), "model_id"] == "CVCL_TEST"
    assert manifest["studies"]["S"]["lists_without_model_id"] == 0
    assert manifest["rules_not_ready"] == {"Q": "unusable"}
    lists = pd.read_csv(out / "lists.csv")
    assert lists.set_index("line").loc[
        "A549", ["n_positive", "n_negative"]
    ].tolist() == [1, 1]
    with pytest.raises(FileExistsError):
        builder.build(tmp_path / "rules.csv", tmp_path, tmp_path / "ge.csv",
                      tmp_path / "hgnc.txt", out, tmp_path / "ids.csv")  # fmt: skip
    json.loads((out / "manifest.json").read_text())
    with pytest.raises(ValueError, match="no reader returns"):
        builder.build(tmp_path / "rules.csv", tmp_path, tmp_path / "ge.csv",
                      tmp_path / "hgnc.txt", tmp_path / "other",
                      identities(tmp_path / "lost.csv", "S:Lost"))  # fmt: skip


def test_identity_spellings_name_ready_studies():
    rules = pd.read_csv(
        "configs/benchmarks/sl_label_rules.csv", dtype=str, keep_default_na=False
    )
    ready = set(rules.loc[rules["status"] == "ready", "study_id"])
    spellings = read_line_identities()["spellings"].str.split("; ").explode()
    assert set(spellings.str.split(":").str[0]) <= ready


def test_every_ready_rule_has_its_own_reader():
    rules = pd.read_csv(
        "configs/benchmarks/sl_label_rules.csv", dtype=str, keep_default_na=False
    )
    ready = rules[rules["status"] == "ready"]
    assert ready["reader"].is_unique and set(ready["reader"]) == set(READERS)
