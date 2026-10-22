"""Quantile normalisation to a reference and patient-grouped folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.folds import patient_folds
from src.context_prior.space import quantile_normalize, quantile_reference


def test_quantile_normalisation_maps_ranks_to_the_reference():
    frame = pd.DataFrame([[1.0, 5.0, 3.0], [10.0, 0.0, 20.0]], columns=list("abc"))
    reference = quantile_reference(frame)
    assert np.allclose(reference, [0.5, 6.5, 12.5])
    normalized = quantile_normalize(frame, reference)
    assert np.allclose(normalized.iloc[0], [0.5, 12.5, 6.5])
    assert np.allclose(normalized.iloc[1], [6.5, 0.5, 12.5])


def test_ties_share_one_value_and_nan_raises():
    frame = pd.DataFrame([[0.0, 0.0, 2.0]], columns=list("abc"))
    normalized = quantile_normalize(frame, np.array([0.0, 1.0, 2.0]))
    assert normalized.iloc[0, 0] == normalized.iloc[0, 1] == 0.5
    with pytest.raises(ValueError, match="finite"):
        quantile_normalize(frame.assign(a=np.nan), np.array([0.0, 1.0, 2.0]))


def test_patient_folds_keep_patients_together_and_are_seeded():
    lines = [f"L{i}" for i in range(12)]
    patients = {line: f"P{i // 2}" for i, line in enumerate(lines)}
    folds = patient_folds(lines, patients, n_folds=3, seed=0)
    assert folds == patient_folds(lines, patients, n_folds=3, seed=0)
    assert all(folds[f"L{2 * i}"] == folds[f"L{2 * i + 1}"] for i in range(6))
    assert set(folds.values()) == {0, 1, 2}


def test_space_keeps_genes_every_scored_line_measures():
    from src.context_prior.space import measured_genes

    frame = pd.DataFrame(
        {"A": [1.0, 2.0, np.nan], "B": [1.0, np.nan, 3.0], "C": [1.0, 2.0, 3.0]},
        index=["T1", "V1", "S1"],
    )
    assert measured_genes(frame, ["A", "B", "C"], ["V1", "S1"]) == ["C"]
    assert measured_genes(frame, ["A", "B", "C"], ["V1"]) == ["A", "C"]


def test_unmeasured_values_take_the_reference_lines_mean():
    from src.context_prior.space import fill_unmeasured

    frame = pd.DataFrame(
        {"A": [1.0, 3.0, np.nan], "B": [np.nan, 4.0, 6.0]}, index=["T1", "T2", "T3"]
    )
    filled, counts = fill_unmeasured(frame, ["T1", "T2", "T3"])
    assert filled.loc["T3", "A"] == 2.0 and filled.loc["T1", "B"] == 5.0
    assert counts.to_dict() == {"T1": 1, "T2": 0, "T3": 1}
    with pytest.raises(ValueError, match="none of the reference lines"):
        fill_unmeasured(frame.assign(B=np.nan), ["T1"])
