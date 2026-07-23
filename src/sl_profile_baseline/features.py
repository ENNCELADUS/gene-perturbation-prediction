"""Swap-invariant features derived only from two GeneEffect profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

_SUMMARY_NAMES = (
    "mean",
    "std",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "dependent_fraction",
    "strong_dependency_fraction",
)
_PAIR_NAMES = (
    "pearson",
    "spearman",
    "cosine",
    "mean_abs_difference",
    "rms_difference",
    "mean_product",
    "both_dependent_fraction",
    "exclusive_dependent_fraction",
    "coverage_both",
)
PROFILE_FEATURE_NAMES = tuple(
    [f"{name}_sum" for name in _SUMMARY_NAMES]
    + [f"{name}_absdiff" for name in _SUMMARY_NAMES]
    + list(_PAIR_NAMES)
)


@dataclass(frozen=True)
class ProfileFeatureCache:
    """Per-gene quantities reused across pair-feature batches."""

    profiles: np.ndarray
    coverage: np.ndarray
    summaries: np.ndarray
    centered: np.ndarray
    centered_norm: np.ndarray
    rank_centered: np.ndarray
    rank_norm: np.ndarray
    line_residual_centered: np.ndarray
    line_residual_norm: np.ndarray
    raw_norm: np.ndarray
    dependency_mask: np.ndarray


def _safe_norm(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1)
    return np.where(norm > 0.0, norm, 1.0).astype(np.float32)


def prepare_feature_cache(
    profiles: np.ndarray,
    coverage: np.ndarray,
    dependency_threshold: float,
) -> ProfileFeatureCache:
    """Precompute summary, correlation, and dependency-profile quantities."""
    profiles = np.asarray(profiles, dtype=np.float32)
    coverage = np.asarray(coverage, dtype=bool)
    quantiles = np.quantile(profiles, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    summaries = np.column_stack(
        [
            profiles.mean(axis=1),
            profiles.std(axis=1),
            quantiles,
            (profiles < dependency_threshold).mean(axis=1),
            (profiles < -1.0).mean(axis=1),
        ]
    ).astype(np.float32)
    centered = profiles - profiles.mean(axis=1, keepdims=True)
    ranks = rankdata(profiles, axis=1).astype(np.float32)
    rank_centered = ranks - ranks.mean(axis=1, keepdims=True)
    line_median = np.median(profiles[coverage], axis=0)
    line_residual = profiles - line_median
    line_residual[~coverage] = 0.0
    line_residual_centered = line_residual - line_residual.mean(
        axis=1, keepdims=True
    )
    return ProfileFeatureCache(
        profiles=profiles,
        coverage=coverage,
        summaries=summaries,
        centered=centered,
        centered_norm=_safe_norm(centered),
        rank_centered=rank_centered,
        rank_norm=_safe_norm(rank_centered),
        line_residual_centered=line_residual_centered,
        line_residual_norm=_safe_norm(line_residual_centered),
        raw_norm=_safe_norm(profiles),
        dependency_mask=profiles < dependency_threshold,
    )


def _batch_features(cache: ProfileFeatureCache, pairs: np.ndarray) -> np.ndarray:
    a_index = pairs[:, 0]
    b_index = pairs[:, 1]
    a = cache.profiles[a_index]
    b = cache.profiles[b_index]
    a_summary = cache.summaries[a_index]
    b_summary = cache.summaries[b_index]
    centered_product = (cache.centered[a_index] * cache.centered[b_index]).sum(axis=1)
    rank_product = (
        cache.rank_centered[a_index] * cache.rank_centered[b_index]
    ).sum(axis=1)
    pearson = centered_product / (
        cache.centered_norm[a_index] * cache.centered_norm[b_index]
    )
    spearman = rank_product / (cache.rank_norm[a_index] * cache.rank_norm[b_index])
    cosine = (a * b).sum(axis=1) / (
        cache.raw_norm[a_index] * cache.raw_norm[b_index]
    )
    difference = a - b
    dep_a = cache.dependency_mask[a_index]
    dep_b = cache.dependency_mask[b_index]
    pair_features = np.column_stack(
        [
            pearson,
            spearman,
            cosine,
            np.abs(difference).mean(axis=1),
            np.sqrt(np.square(difference).mean(axis=1)),
            (a * b).mean(axis=1),
            (dep_a & dep_b).mean(axis=1),
            np.logical_xor(dep_a, dep_b).mean(axis=1),
            cache.coverage[a_index] & cache.coverage[b_index],
        ]
    )
    return np.column_stack(
        [a_summary + b_summary, np.abs(a_summary - b_summary), pair_features]
    ).astype(np.float32)


def build_cached_pair_features(
    cache: ProfileFeatureCache, pairs: np.ndarray, chunk_size: int = 2048
) -> np.ndarray:
    """Build compact profile features in bounded-memory batches."""
    pairs = np.asarray(pairs, dtype=int)
    output = np.empty((len(pairs), len(PROFILE_FEATURE_NAMES)), dtype=np.float32)
    for start in range(0, len(pairs), chunk_size):
        stop = min(start + chunk_size, len(pairs))
        output[start:stop] = _batch_features(cache, pairs[start:stop])
    return output


def build_profile_pair_features(
    profiles: np.ndarray,
    coverage: np.ndarray,
    pairs: np.ndarray,
    dependency_threshold: float,
) -> np.ndarray:
    """Convenience wrapper for compact two-profile features."""
    cache = prepare_feature_cache(profiles, coverage, dependency_threshold)
    return build_cached_pair_features(cache, pairs)


def build_raw_pair_features(profiles: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Return ``[x_a+x_b, |x_a-x_b|]`` for unordered gene pairs."""
    profiles = np.asarray(profiles, dtype=np.float32)
    pairs = np.asarray(pairs, dtype=int)
    a = profiles[pairs[:, 0]]
    b = profiles[pairs[:, 1]]
    return np.concatenate([a + b, np.abs(a - b)], axis=1).astype(np.float32)
