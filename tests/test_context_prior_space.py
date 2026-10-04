"""Quantile normalisation to a reference and patient-grouped folds and subsets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.folds import patient_folds, patient_subset
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


def test_patient_subset_takes_whole_patients():
    lines = [f"L{i}" for i in range(10)]
    patients = {line: f"P{i // 2}" for i, line in enumerate(lines)}
    subset = patient_subset(lines, patients, size=5, seed=3)
    assert 5 <= len(subset) <= 6
    chosen = {patients[m] for m in subset}
    assert sum(patients[m] in chosen for m in lines) == len(subset)
    with pytest.raises(ValueError):
        patient_subset(lines, patients, size=11, seed=0)
