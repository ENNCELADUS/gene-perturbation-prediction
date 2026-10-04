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
