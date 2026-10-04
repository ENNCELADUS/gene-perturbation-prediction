"""Patient-grouped, seeded folds and subsets of training-side lines."""

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


def patient_subset(
    model_ids: Sequence[str], patients: Mapping[str, str], *, size: int, seed: int
) -> tuple[str, ...]:
    """Whole patients in seeded random order until at least ``size`` lines."""
    by_patient: dict[str, list[str]] = {}
    for m in model_ids:
        by_patient.setdefault(patients[m], []).append(m)
    groups = sorted(by_patient)
    chosen: list[str] = []
    for g in np.random.default_rng(seed).permutation(len(groups)):
        if len(chosen) >= size:
            break
        chosen.extend(by_patient[groups[g]])
    if len(chosen) < size:
        raise ValueError(f"only {len(chosen)} lines for a subset of {size}")
    return tuple(sorted(chosen))
