"""Extra single-cell lines: labelled, outside the 226, no held-out patient."""

import pandas as pd
import pytest

from src.data.prepare.build_extra_single_cell_lines import build_extra_single_cell_lines
from src.data.splits import FixedSplit


def test_membership_excludes_held_patients_and_our_lines():
    models = pd.DataFrame(
        {"patient_id": ["P1", "P2", "P3", "P4", "P5"]},
        index=["ACH-T", "ACH-V", "ACH-A", "ACH-B", "ACH-C"],
    )
    models.loc["ACH-D"] = {"patient_id": "P2"}
    split = FixedSplit(train=("ACH-T",), val=("ACH-V",), test=())
    candidates = pd.DataFrame(
        {
            "source": ["mixseq"] * 4 + ["tahoe"],
            "model_id": ["ACH-T", "ACH-A", "ACH-B", "ACH-D", "ACH-A"],
        }
    )
    payload = build_extra_single_cell_lines(
        candidates, models, labelled_ids={"ACH-A", "ACH-D"}, split=split
    )
    assert payload["labelled"] == ["ACH-A"]
    assert payload["unlabelled"] == ["ACH-B"]
    assert "ACH-T" in payload["excluded"] and "226" in payload["excluded"]["ACH-T"]
    assert "ACH-V" in payload["excluded"]["ACH-D"]
    assert payload["sources"]["ACH-A"] == ["mixseq", "tahoe"]


def test_membership_refuses_unknown_model_ids():
    models = pd.DataFrame({"patient_id": ["P1"]}, index=["ACH-T"])
    split = FixedSplit(train=("ACH-T",), val=(), test=())
    candidates = pd.DataFrame({"source": ["mixseq"], "model_id": ["ACH-NOPE"]})
    with pytest.raises(ValueError, match="absent from Model.csv"):
        build_extra_single_cell_lines(
            candidates, models, labelled_ids=set(), split=split
        )
