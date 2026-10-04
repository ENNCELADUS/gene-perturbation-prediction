"""A per-gene affine map from quantile-normalised pseudo-bulk to bulk.

3' UMI counts carry no gene-length normalisation and TPM does, so the offset and
slope differ by gene; one least-squares line per gene, fitted on lines with both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Bridge:
    genes: tuple[str, ...]
    slope: np.ndarray
    intercept: np.ndarray

    def apply(self, pseudobulk: pd.DataFrame) -> pd.DataFrame:
        values = pseudobulk.loc[:, list(self.genes)].to_numpy(dtype=np.float64)
        return pd.DataFrame(
            values * self.slope + self.intercept,
            index=pseudobulk.index,
            columns=list(self.genes),
        )


def _aligned(left: pd.DataFrame, right: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if list(left.index) != list(right.index) or list(left.columns) != list(
        right.columns
    ):
        raise ValueError("pseudo-bulk and bulk must be aligned on lines and genes")
    return left.to_numpy(dtype=np.float64), right.to_numpy(dtype=np.float64)


def fit_bridge(pseudobulk: pd.DataFrame, bulk: pd.DataFrame) -> Bridge:
    """Least squares per gene; a gene constant in pseudo-bulk maps to its bulk mean."""
    x, y = _aligned(pseudobulk, bulk)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    variance = ((x - x_mean) ** 2).mean(axis=0)
    covariance = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    varies = np.ptp(x, axis=0) > 0
    slope = np.where(varies, covariance / np.where(varies, variance, 1.0), 0.0)
    return Bridge(tuple(pseudobulk.columns), slope, y_mean - slope * x_mean)


def bridge_quality(bridged: pd.DataFrame, bulk: pd.DataFrame) -> pd.Series:
    """Per-gene Pearson across lines; NaN where either side is constant."""
    x, y = _aligned(bridged, bulk)
    varies = (np.ptp(x, axis=0) > 0) & (np.ptp(y, axis=0) > 0)
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    denominator = np.sqrt((x**2).sum(axis=0) * (y**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(varies, (x * y).sum(axis=0) / denominator, np.nan)
    return pd.Series(rho, index=bridged.columns, name="bridge_pearson")
