"""Patient-grouped, seeded folds of training-side lines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def patient_folds(
    model_ids: Sequence[str], patients: Mapping[str, str], *, n_folds: int, seed: int
) -> dict[str, int]:
    """Fold of each line; every patient's lines share a fold."""
    groups = sorted({patients[m] for m in model_ids})
    if n_folds < 2 or len(groups) < n_folds:
        raise ValueError(f"{len(groups)} patients cannot fill {n_folds} folds")
    order = np.random.default_rng(seed).permutation(len(groups))
    fold = {groups[g]: rank % n_folds for rank, g in enumerate(order)}
    return {m: fold[patients[m]] for m in model_ids}


def training_side_folds(
    single_cell: Mapping[str, int],
    lines: Sequence[str],
    patients: Mapping[str, object],
) -> dict[str, int]:
    """Folds over the whole training side for cross-fitting the single-cell lines.

    A single-cell training line keeps its fold; any other line takes the fold of
    the single-cell training lines that share its patient, and -1 when none does
    (or its patient is unknown), so cross-fitting always fits on it and never
    holds it out.
    """
    by_patient = {
        patients[m]: fold
        for m, fold in single_cell.items()
        if isinstance(patients[m], str)
    }
    folds = dict(single_cell)
    for model_id in lines:
        if model_id not in folds:
            patient = patients[model_id]
            folds[model_id] = (
                by_patient.get(patient, -1) if isinstance(patient, str) else -1
            )
    return folds
