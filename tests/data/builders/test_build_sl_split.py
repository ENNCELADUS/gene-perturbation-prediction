"""The SL split: held-out lines keep their side, quotas fill across lineages, and a
moved line's patient siblings leave every fit."""

from __future__ import annotations

import json

import pandas as pd

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


def test_held_lines_stay_and_quotas_fill_across_lineages():
    scored = [*TRAIN, "U0", "V0"]  # 10 scored lines, E0 has no list
    payload, table = build_sl_split(
        FINAL, SPLIT, TABLE, lists(scored), PATIENTS, fraction=0.2, seed=0, placed={}
    )
    assert payload["val"][0] == "V0" and payload["test"][0] == "E0"
    scored_held = lambda side: [m for m in payload[side] if m in scored]  # noqa: E731
    assert len(scored_held("val")) == len(scored_held("test")) == 2
    moved = set(payload["val"] + payload["test"]) - {"V0", "E0"}
    assert len(moved) == 3 and moved <= {*TRAIN, "U0"}  # U0 has no GeneEffect
    assert len({TABLE.set_index("model_id").loc[m, "lineage"] for m in moved}) > 1
    assert not moved & set(payload["train"])
    assert list(table["model_id"]) == [
        *payload["train"],
        *payload["val"],
        *payload["test"],
    ]


def test_a_line_without_geneeffect_can_move_and_records_its_group():
    payload, table = build_sl_split(
        FINAL, SPLIT, TABLE, lists(["U0", "V0"]), PATIENTS, fraction=1.0, seed=0,
        placed={},
    )  # fmt: skip
    assert payload["test"] == ["E0", "U0"] and "U0" not in payload["unlabeled_train"]
    groups = table.set_index("model_id")["group"]
    assert groups["U0"] == "without_geneeffect" and groups["T0"] == "with_geneeffect"


def test_a_moved_lines_patient_siblings_are_excluded():
    payload, _ = build_sl_split(
        FINAL,
        SPLIT,
        TABLE,
        lists(["T4", "V0"]),
        PATIENTS,
        fraction=1.0,
        seed=0,
        placed={},
    )
    assert payload["test"] == ["E0", "T4"]
    assert "B0" in payload["excluded"] and "B0" not in payload["bulk_labelled"]
    assert "X0" in payload["excluded"] and "B1" in payload["bulk_labelled"]


def test_a_placed_line_and_its_patient_stay_on_the_training_side():
    payload, _ = build_sl_split(
        FINAL, SPLIT, TABLE, lists(["T4", "V0"]), PATIENTS, fraction=1.0, seed=0,
        placed={"T4": "reviewed"},
    )  # fmt: skip
    assert "T4" in payload["train"] and payload["placed"] == {"T4": "reviewed"}
    assert "B0" in payload["bulk_labelled"]  # T4's patient sibling is not excluded


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
    (tmp_path / "placed.csv").write_text("line_id,side,reason\nT1,train,reviewed\n")
    out = tmp_path / "sl.json"
    main(["--final", str(final), "--lists", str(tmp_path / "lists.csv"),
          "--model", str(tmp_path / "Model.csv"), "--placements",
          str(tmp_path / "placed.csv"), "--out", str(out)])  # fmt: skip
    split = load_geneeffect_split(out)
    load_bulk_lines(out, split)
    assert len(split.val) + len(split.test) > 2 and "T1" in split.train


def test_tracked_placements_are_reviewed_train_placements():
    from src.data.builders.build_sl_split import read_placements

    assert set(read_placements("configs/benchmarks/sl_split_placements.csv")) == {
        "CVCL_4388"
    }


def test_a_screened_patient_sibling_moves_with_the_drawn_line():
    patients = PATIENTS.copy()
    patients["U0"] = patients["T4"]  # U0 (no GeneEffect) and T4 share a patient
    payload, table = build_sl_split(
        FINAL, SPLIT, TABLE.assign(patient_id=TABLE["model_id"].map(patients)),
        lists(["T4", "U0", "V0"]), patients, fraction=1.0, seed=0, placed={},
    )  # fmt: skip
    held = {*payload["val"], *payload["test"]}
    assert {"T4", "U0"} <= held and "U0" not in payload["excluded"]
    side = {m: s for s in ("val", "test") for m in payload[s]}
    assert side["T4"] == side["U0"]
