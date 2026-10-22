"""DepMap readers: Omics default entries, CRISPR layout, symbols, Model.csv."""

from __future__ import annotations

import math

import pytest

from src.data.depmap import read_depmap_matrix, read_model_ids, read_models

OMICS = (
    ",SequencingID,ModelConditionID,ModelID,IsDefaultEntryForMC,"
    "IsDefaultEntryForModel,TP53 (7157),ZNF781 (Unknown)\n"
    "0,S1,MC1,ACH-1,Yes,Yes,1.5,0\n"
    "1,S2,MC2,ACH-1,No,No,9.0,9\n"
    "2,S3,MC3,ACH-2,Yes,Yes,0.5,2\n"
)
CRISPR = ",A1BG (1),A2M (2)\nACH-1,0.1,\nACH-2,-1.0,0.3\n"


def write(path, text):
    path.write_text(text)
    return path


def test_omics_matrix_keeps_each_models_default_entry(tmp_path):
    frame = read_depmap_matrix(write(tmp_path / "omics.csv", OMICS))
    assert list(frame.index) == ["ACH-1", "ACH-2"]
    assert list(frame.columns) == ["TP53", "ZNF781"]
    assert frame.loc["ACH-1", "TP53"] == 1.5
    assert frame.index.name == "model_id"


def test_crispr_matrix_is_indexed_by_its_first_column(tmp_path):
    frame = read_depmap_matrix(write(tmp_path / "crispr.csv", CRISPR))
    assert list(frame.columns) == ["A1BG", "A2M"]
    assert math.isnan(frame.loc["ACH-1", "A2M"])


def test_duplicate_symbols_raise(tmp_path):
    path = write(tmp_path / "dup.csv", ",TP53 (7157),tp53 (1)\nACH-1,1,2\n")
    with pytest.raises(ValueError, match="duplicate symbols"):
        read_depmap_matrix(path)


def test_model_ids_without_values(tmp_path):
    assert read_model_ids(write(tmp_path / "o.csv", OMICS)) == {"ACH-1", "ACH-2"}
    assert read_model_ids(write(tmp_path / "c.csv", CRISPR)) == {"ACH-1", "ACH-2"}


def test_missing_patient_becomes_its_own_patient(tmp_path):
    path = write(
        tmp_path / "Model.csv",
        "ModelID,PatientID,OncotreeLineage,Other\nACH-1,PT-1,Lung,x\nACH-2,,Bowel,y\n",
    )
    models = read_models(path)
    assert models.loc["ACH-1", "patient_id"] == "PT-1"
    assert models.loc["ACH-2", "patient_id"] == "ACH-2"
    assert models.loc["ACH-2", "lineage"] == "Bowel"
