"""One expression space for bulk and pseudo-bulk: quantile normalisation per line."""

from __future__ import annotations

from collections.abc import Sequence

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


def measured_genes(
    pseudobulk: pd.DataFrame, genes: Sequence[str], lines: Sequence[str]
) -> list[str]:
    """The genes of ``genes`` with a finite pseudo-bulk value in every one of
    ``lines`` (the scored validation and test lines)."""
    values = pseudobulk.loc[list(lines), list(genes)].to_numpy(dtype=np.float64)
    finite = np.isfinite(values).all(axis=0)
    return [gene for gene, ok in zip(genes, finite, strict=True) if ok]


def fill_unmeasured(
    pseudobulk: pd.DataFrame, reference_lines: Sequence[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Give each unmeasured (NaN) value its gene's mean over the reference lines
    that measured it; return the filled frame and the count filled per line.

    Single-cell sources differ in gene vocabulary; a training line whose source
    lacks a gene of the space takes that gene's training mean before quantile
    normalisation, as an undefined feature does elsewhere in the prior.
    """
    means = pseudobulk.loc[list(reference_lines)].mean(axis=0, skipna=True)
    if means.isna().any():
        missing = list(means.index[means.isna()])[:10]
        raise ValueError(f"genes measured in none of the reference lines: {missing}")
    counts = pseudobulk.isna().sum(axis=1)
    return pseudobulk.fillna(means), counts
