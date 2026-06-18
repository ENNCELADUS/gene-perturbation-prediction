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


def transcript_feature_names(dim: int, include_coverage_flag: bool) -> tuple[str, ...]:
    """Column names for the transcript pair-feature block."""
    names = (
        [f"sum_{i}" for i in range(dim)]
        + [f"absdiff_{i}" for i in range(dim)]
        + [f"prod_{i}" for i in range(dim)]
    )
    if include_coverage_flag:
        names += ["cov_min", "cov_max"]
    return tuple(names)


def build_transcript_pair_features(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    flag_a: np.ndarray,
    flag_b: np.ndarray,
    include_coverage_flag: bool,
) -> np.ndarray:
    """Build swap-invariant transcript features from two per-gene embeddings.

    Args:
        emb_a: Per-pair embedding for gene a, shape ``(n, dim)``.
        emb_b: Per-pair embedding for gene b, shape ``(n, dim)``.
        flag_a: Coverage indicator for gene a, shape ``(n,)``.
        flag_b: Coverage indicator for gene b, shape ``(n,)``.
        include_coverage_flag: Whether to append swap-invariant coverage columns.

    Returns:
        Feature matrix of shape ``(n, 3*dim + (2 if include_coverage_flag else 0))``.
    """
    emb_a = np.asarray(emb_a, dtype=float)
    emb_b = np.asarray(emb_b, dtype=float)
    blocks = [emb_a + emb_b, np.abs(emb_a - emb_b), emb_a * emb_b]
    if include_coverage_flag:
        flag_a = np.asarray(flag_a, dtype=float)
        flag_b = np.asarray(flag_b, dtype=float)
        blocks.append(
            np.column_stack([np.minimum(flag_a, flag_b), np.maximum(flag_a, flag_b)])
        )
    return np.column_stack(blocks)


def build_augmented_pair_features(
    ea: np.ndarray,
    eb: np.ndarray,
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    flag_a: np.ndarray,
    flag_b: np.ndarray,
    include_coverage_flag: bool,
) -> np.ndarray:
    """Concatenate the GeneEffect block and the transcript block."""
    return np.column_stack(
        [
            build_pair_features(ea, eb),
            build_transcript_pair_features(
                emb_a, emb_b, flag_a, flag_b, include_coverage_flag
            ),
        ]
    )


SELECTIVITY_FEATURE_NAMES: tuple[str, ...] = (
    "sel_mean",
    "sel_absdiff",
    "pan_essential_min",
)


def build_selectivity_pair_features(
    sel_ab: np.ndarray,
    sel_ba: np.ndarray,
    pan_a: np.ndarray,
    pan_b: np.ndarray,
) -> np.ndarray:
    """Build swap-invariant selectivity features from directional contrasts.

    Args:
        sel_ab: Directional selectivity ``sel(a -> b)``, shape ``(n,)``.
        sel_ba: Directional selectivity ``sel(b -> a)``, shape ``(n,)``.
        pan_a: Pan-essentiality of gene a (mean GeneEffect), shape ``(n,)``.
        pan_b: Pan-essentiality of gene b (mean GeneEffect), shape ``(n,)``.

    Returns:
        Feature matrix of shape ``(n, 3)`` ordered by
        ``SELECTIVITY_FEATURE_NAMES``.
    """
    sel_ab = np.asarray(sel_ab, dtype=float)
    sel_ba = np.asarray(sel_ba, dtype=float)
    pan_a = np.asarray(pan_a, dtype=float)
    pan_b = np.asarray(pan_b, dtype=float)
    return np.column_stack(
        [
            (sel_ab + sel_ba) / 2.0,
            np.abs(sel_ab - sel_ba),
            np.minimum(pan_a, pan_b),
        ]
    )
