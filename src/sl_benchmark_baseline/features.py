"""Symmetric pair features and a train-fit standardizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "f_min",
    "f_max",
    "f_sum",
    "f_product",
    "f_absdiff",
)


def build_pair_features(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """Build swap-invariant features from two gene-effect vectors.

    Args:
        ea: Gene-effect values for gene a, shape ``(n,)``.
        eb: Gene-effect values for gene b, shape ``(n,)``.

    Returns:
        Feature matrix of shape ``(n, 5)`` ordered by ``FEATURE_NAMES``.
    """
    ea = np.asarray(ea, dtype=float)
    eb = np.asarray(eb, dtype=float)
    return np.column_stack(
        [
            np.minimum(ea, eb),
            np.maximum(ea, eb),
            ea + eb,
            ea * eb,
            np.abs(ea - eb),
        ]
    )


@dataclass(frozen=True)
class Standardizer:
    """Zero-mean unit-variance standardizer fit on training data only."""

    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardizer":
        """Fit per-column mean and std; zero-std columns map to std 1.0."""
        features = np.asarray(features, dtype=float)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        return cls(mean_=mean, std_=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the fitted standardization to a feature matrix."""
        features = np.asarray(features, dtype=float)
        return (features - self.mean_) / self.std_
