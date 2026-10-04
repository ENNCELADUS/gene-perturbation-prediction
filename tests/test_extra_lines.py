"""Extra-line membership: exclusions by patient, labelled vs unlabelled, loader."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.extra_lines import SCHEMA_VERSION, load_extra_lines
from src.data.prepare.build_extra_bulk_lines import build_extra_lines
from src.data.splits import FixedSplit

SPLIT = FixedSplit(train=("T1",), val=("V1",), test=("S1",))
MODELS = pd.DataFrame(
    {
        "patient_id": {
            "T1": "P1",
            "V1": "P2",
            "S1": "P3",
            "E1": "P2",
            "E2": "P1",
            "E3": "P4",
            "E4": "P5",
            "E5": "P3",
        },
        "lineage": "Lung",
    }
).rename_axis("model_id")


def payload():
    return build_extra_lines(
        MODELS,
        labelled_ids={"T1", "V1", "S1", "E1", "E2", "E4"},
        bulk_ids={"T1", "V1", "E1", "E2", "E3", "E5"},
        split=SPLIT,
    )


def test_membership_rules():
    built = payload()
    assert built["schema_version"] == SCHEMA_VERSION
    assert built["labelled"] == ["E2"]  # shares a patient with a train line: kept
    assert built["unlabelled"] == ["E3"]  # bulk RNA without GeneEffect
    assert sorted(built["excluded"]) == ["E1", "E5"]  # share a held-out patient
    assert "V1" in built["excluded"]["E1"]


def test_line_missing_from_model_csv_raises():
    with pytest.raises(ValueError, match="absent from Model.csv"):
        build_extra_lines(MODELS, labelled_ids={"E9"}, bulk_ids={"E9"}, split=SPLIT)


def test_loader_round_trip_and_overlap_guard(tmp_path):
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(payload()))
    lines = load_extra_lines(path, SPLIT)
    assert lines.labelled == ("E2",) and lines.unlabelled == ("E3",)
    bad = payload()
    bad["unlabelled"] = ["E3", "T1"]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="overlap"):
        load_extra_lines(path, SPLIT)
