"""The SL split: the placed lines are the held-out sides, every other line trains,
and a held-out line's patient siblings follow it or leave every fit."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.builders.build_sl_split import build_sl_split, main
from src.data.extra_lines import load_bulk_lines
from src.data.splits import FixedSplit, load_geneeffect_split

TRAIN = [f"T{i}" for i in range(8)]
SPLIT = FixedSplit(
    train=(*TRAIN, "U0"), val=("V0",), test=("E0",), unlabeled_train=("U0",)
)
FINAL = {
    "train": [*TRAIN, "U0"], "val": ["V0"], "test": ["E0"], "unlabeled_train": ["U0"],
    "bulk_labelled": ["B0", "B1"], "bulk_unlabelled": [], "excluded": {"X0": "old"},
}  # fmt: skip
TABLE = pd.DataFrame(
    {
        "split": ["train"] * 9 + ["val", "test"],
        "model_id": [*TRAIN, "U0", "V0", "E0"],
        "patient_id": [f"P{i}" for i in range(8)] + ["PU", "PV", "PE"],
        "lineage": ["Lung", "Lung", "Lung", "Lung", "Skin", "Skin", "Skin", "Bowel",
                    "Lung", "Lung", "Skin"],
        "expression_source": "kinker_sccle",
    }
)  # fmt: skip
# B0 shares T-lines' patients so a move must exclude it; B1 does not.
PATIENTS = pd.Series(
    {
        **dict(zip(TABLE["model_id"], TABLE["patient_id"])),
        "B0": "P4",
        "B1": "PB",
        "X0": "PX",
    }
)


def lists(lines: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"study_id": "S", "list_id": lines, "model_id": lines,
         "n_positive": 3, "n_negative": 30}
    )  # fmt: skip


def test_placed_lines_are_held_out_and_stage_one_sides_do_not_bind():
    placed = {"T1": ("val", "r"), "U0": ("test", "r")}
    payload, table = build_sl_split(
        FINAL, SPLIT, TABLE, lists(["T1", "U0", "V0"]), PATIENTS, placed=placed
    )
    assert (payload["val"], payload["test"]) == (["T1"], ["U0"])
    assert {"V0", "E0"} <= set(payload["train"])  # stage one's held-out lines train
    assert "U0" not in payload["unlabeled_train"]
    assert list(table["model_id"]) == [*payload["train"], "T1", "U0"]
    groups = table.set_index("model_id")["group"]
    assert groups["U0"] == "without_geneeffect" and groups["T0"] == "with_geneeffect"


def test_a_held_lines_unscreened_patient_siblings_are_excluded():
    payload, _ = build_sl_split(
        FINAL, SPLIT, TABLE, lists(["T4"]), PATIENTS, placed={"T4": ("test", "r")}
    )
    assert "B0" in payload["excluded"] and "B0" not in payload["bulk_labelled"]
    assert "X0" in payload["excluded"] and "B1" in payload["bulk_labelled"]


def test_a_screened_patient_sibling_must_be_placed_on_the_same_side():
    patients = PATIENTS.copy()
    patients["U0"] = patients["T4"]
    table = TABLE.assign(patient_id=TABLE["model_id"].map(patients))
    with pytest.raises(ValueError, match="not placed on test"):
        build_sl_split(FINAL, SPLIT, table, lists(["T4", "U0"]), patients,
                       placed={"T4": ("test", "r")})  # fmt: skip
    payload, _ = build_sl_split(
        FINAL, SPLIT, table, lists(["T4", "U0"]), patients,
        placed={"T4": ("test", "r"), "U0": ("test", "r")},
    )  # fmt: skip
    assert payload["test"] == ["T4", "U0"]


def test_main_writes_a_split_the_loaders_accept(tmp_path):
    final = tmp_path / "final.json"
    final.write_text(
        json.dumps({"schema_version": "cell-line-geneeffect-split-v1", **FINAL})
    )
    TABLE.to_csv(final.with_suffix(".csv"), index=False)
    lists([*TRAIN, "V0"]).to_csv(tmp_path / "lists.csv", index=False)
    pd.DataFrame(
        {"ModelID": PATIENTS.index, "PatientID": PATIENTS.values,
         "OncotreeLineage": "Lung"}
    ).to_csv(tmp_path / "Model.csv", index=False)  # fmt: skip
    (tmp_path / "placed.csv").write_text(
        "line_id,side,reason\nT1,val,reviewed\nT2,test,reviewed\n"
    )
    out = tmp_path / "sl.json"
    main(["--final", str(final), "--lists", str(tmp_path / "lists.csv"),
          "--model", str(tmp_path / "Model.csv"), "--placements",
          str(tmp_path / "placed.csv"), "--out", str(out)])  # fmt: skip
    split = load_geneeffect_split(out)
    load_bulk_lines(out, split)
    assert (split.val, split.test) == (("T1",), ("T2",))


def test_tracked_placements_hold_only_held_out_lines():
    from src.data.builders.build_sl_split import read_placements

    placed = read_placements("configs/benchmarks/sl_split_placements.csv")
    sides = [side for side, _ in placed.values()]
    assert sides.count("val") >= 15 and sides.count("test") >= 15
