"""The final split: one role per line, the single-cell extras as training lines,
bulk-only lines without built cells, the held-out lines unchanged; and the tracked
file is what the builder writes from the tracked inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.extra_lines import (
    ExtraLines,
    ExtraSingleCellLines,
    load_bulk_lines,
    load_extra_lines,
    load_extra_single_cell_lines,
)
from src.data.build.build_cell_line_geneeffect_split import build_split, main
from src.data.splits import FixedSplit, load_geneeffect_split

SPLIT = FixedSplit(
    train=("T0", "T1", "U0"), val=("V0",), test=("E0",), unlabeled_train=("U0",)
)
TABLE = pd.DataFrame(
    {
        "split": ["train", "train", "train", "val", "test"],
        "model_id": ["T0", "T1", "U0", "V0", "E0"],
        "patient_id": ["P0", "P1", "P2", "PV", "PE"],
        "lineage": ["Lung"] * 5,
        "expression_source": ["kinker_sccle"] * 5,
        "cohort": ["atlas"] * 5,
    }
)


def record(patient: str) -> dict:
    return {"patient_id": patient, "lineage": "Skin", "source": "gse1"}


CELLS = ExtraSingleCellLines(
    {"X1": record("PX1"), "X0": record("PX0")}, ("Y0",), {}, {"Z1": "shares PV"}
)
BULK = ExtraLines(
    ("B0", "X0", "X1"), ("Y0", "Y1"), {"Z0": "shares PE", "Z1": "shares PV"}
)


def test_extras_train_and_bulk_only_lines_keep_one_role():
    payload, table = build_split(SPLIT, TABLE, CELLS, BULK)
    assert payload["train"] == ["T0", "T1", "U0", "X0", "X1"]
    assert payload["val"] == ["V0"] and payload["test"] == ["E0"]
    assert payload["unlabeled_train"] == ["U0"]
    assert payload["bulk_labelled"] == ["B0"]  # X0, X1 train through their cells
    assert payload["bulk_unlabelled"] == ["Y0", "Y1"]
    assert payload["excluded"] == {"Z0": "shares PE", "Z1": "shares PV"}
    assert list(table.columns) == [
        "split",
        "model_id",
        "patient_id",
        "lineage",
        "expression_source",
    ]
    assert list(table["model_id"]) == ["T0", "T1", "U0", "X0", "X1", "V0", "E0"]
    extra = table.set_index("model_id").loc["X0"]
    assert (extra["split"], extra["patient_id"], extra["expression_source"]) == (
        "train",
        "PX0",
        "gse1",
    )


def test_build_refuses_unbuilt_leaking_or_contradicted_extras():
    pending = ExtraSingleCellLines(CELLS.labelled, (), {"X2": "not built"}, {})
    with pytest.raises(ValueError, match="not built yet"):
        build_split(SPLIT, TABLE, pending, BULK)
    leaking = ExtraSingleCellLines({"X0": record("PV")}, (), {}, {})
    with pytest.raises(ValueError, match="sharing a patient"):
        build_split(SPLIT, TABLE, leaking, BULK)
    excluded = ExtraLines(("B0",), (), {"X0": "shares PE"})
    with pytest.raises(ValueError, match="calls excluded"):
        build_split(SPLIT, TABLE, CELLS, excluded)


def test_bulk_lines_must_not_overlap_the_split(tmp_path):
    payload, _ = build_split(SPLIT, TABLE, CELLS, BULK)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload))
    split = load_geneeffect_split(path)
    assert split.supervised_train == ("T0", "T1", "X0", "X1")
    assert load_bulk_lines(path, split).labelled == ("B0",)
    payload["bulk_labelled"].append("X0")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="overlap"):
        load_bulk_lines(path, split)


def test_tracked_final_split_is_what_the_builder_writes(tmp_path):
    benchmarks = Path("configs/benchmarks")
    out = tmp_path / "cell_line_geneeffect_split.json"
    main(
        [
            "--split-226",
            str(benchmarks / "cell_line_geneeffect_226_split.json"),
            "--single-cell",
            str(benchmarks / "extra_single_cell_lines_26Q1.json"),
            "--bulk",
            str(benchmarks / "extra_bulk_lines_26Q1.json"),
            "--out",
            str(out),
        ]
    )
    tracked = benchmarks / "cell_line_geneeffect_split.json"
    assert json.loads(out.read_text()) == json.loads(tracked.read_text())
    assert (
        out.with_suffix(".csv").read_text() == tracked.with_suffix(".csv").read_text()
    )
    split = load_geneeffect_split(tracked)
    old = load_geneeffect_split(benchmarks / "cell_line_geneeffect_226_split.json")
    cells = load_extra_single_cell_lines(
        benchmarks / "extra_single_cell_lines_26Q1.json", old
    )
    assert (split.val, split.test) == (old.val, old.test)
    assert split.train == (*old.train, *sorted(cells.labelled))
    bulk = load_bulk_lines(tracked, split)
    every = load_extra_lines(benchmarks / "extra_bulk_lines_26Q1.json", old)
    assert set(bulk.labelled) == set(every.labelled) - set(cells.labelled)
