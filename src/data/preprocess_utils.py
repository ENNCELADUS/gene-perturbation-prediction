"""
Preprocessing utilities aligned with the scGPT pipeline.

Counts -> normalize_total -> log1p -> per-cell quantile binning.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def normalize_log1p(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Normalize a counts vector to target_sum and apply log1p."""
    counts = counts.astype(np.float32, copy=False)
    total = float(counts.sum())
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float32)
    normed = counts / total * float(target_sum)
    return np.log1p(normed).astype(np.float32, copy=False)


def _digitize(
    x: np.ndarray, bins: np.ndarray, rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Digitize with scGPT-style tie handling (random between left/right)."""
    left_digits = np.digitize(x, bins)
    right_digits = np.digitize(x, bins, right=True)
    rng = rng or np.random
    rands = rng.rand(len(x))
    digits = rands * (right_digits - left_digits) + left_digits
    return np.ceil(digits).astype(np.int64)


def scgpt_binning(
    row: np.ndarray,
    n_bins: int,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Bin a row into n_bins using scGPT's per-cell quantile binning."""
    if row.max() == 0:
        return np.zeros_like(row, dtype=np.int64)

    if row.min() <= 0:
        non_zero_ids = row.nonzero()[0]
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins, rng=rng)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
        return binned_row

    bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
    return _digitize(row, bins, rng=rng)


def preprocess_counts_to_bins(
    counts: np.ndarray,
    n_bins: int,
    target_sum: float = 1e4,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Full scGPT preprocessing for a single cell counts vector."""
    log1p = normalize_log1p(counts, target_sum=target_sum)
    return scgpt_binning(log1p, n_bins=n_bins, rng=rng)
