"""One expression space for bulk and pseudo-bulk: quantile normalisation per line."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _finite(expression: pd.DataFrame) -> np.ndarray:
    values = expression.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("expression must be finite")
    return values


def quantile_reference(expression: pd.DataFrame) -> np.ndarray:
    """The mean sorted profile of ``expression`` rows (lines x genes)."""
    return np.sort(_finite(expression), axis=1).mean(axis=0)


def quantile_normalize(expression: pd.DataFrame, reference: np.ndarray) -> pd.DataFrame:
    """Give each line the reference value at each gene's rank; ties share a value."""
    values = _finite(expression)
    if values.shape[1] != reference.size:
        raise ValueError("expression and reference widths differ")
    ranks = rankdata(values, axis=1, method="average") - 1.0
    normalized = np.interp(ranks, np.arange(reference.size), reference)
    return pd.DataFrame(normalized, index=expression.index, columns=expression.columns)
