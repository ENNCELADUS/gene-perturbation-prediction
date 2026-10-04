"""Contrastive alignment of pseudo-bulk and bulk before the bridge.

Celligner's contrastive PCA adapted to paired profiles: on lines measured both
ways, the directions with more variance in one source than in the other are the
top eigenvectors of the difference of their covariance matrices. Their union is
projected out of every row of both sources, each about its own paired mean. Both
covariances are over the same lines, so no cluster-mean removal or nearest-
neighbour matching is needed. Reads no label; fitted once on the paired lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Below this fraction of the largest singular value a column is dependent; below
#: this fraction of the two sources' total variance an excess variance is zero.
_RANK_TOLERANCE = 1e-8


def contrastive_directions(
    first: np.ndarray, second: np.ndarray, count: int
) -> np.ndarray:
    """Genes x k orthonormal directions (k <= ``count``) with the largest positive
    excess variance of ``first`` over ``second``, computed in the span of the data
    (no genes x genes covariance)."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.zeros((first.shape[1], 0))
    a = first - first.mean(axis=0)
    b = second - second.mean(axis=0)
    basis, triangle = np.linalg.qr(np.vstack([a, b]).T)
    left, right = triangle[:, : len(a)], triangle[:, len(a) :]
    values, vectors = np.linalg.eigh(left @ left.T / len(a) - right @ right.T / len(b))
    total = (a**2).sum() / len(a) + (b**2).sum() / len(b)
    order = np.argsort(values)[::-1][:count]
    order = order[values[order] > _RANK_TOLERANCE * total]
    return basis @ vectors[:, order]


@dataclass(frozen=True, eq=False)
class Alignment:
    """Orthonormal directions (genes x count) removed from both sources, and each
    source's mean over the paired lines."""

    genes: tuple[str, ...]
    directions: np.ndarray
    pseudo_mean: np.ndarray
    bulk_mean: np.ndarray

    @property
    def count(self) -> int:
        return int(self.directions.shape[1])

    def apply(self, frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
        """``frame`` with its rows' components along the directions, about the
        ``source``'s paired mean, removed; the frame itself when there are none."""
        if source == "pseudo":
            mean = self.pseudo_mean
        elif source == "bulk":
            mean = self.bulk_mean
        else:
            raise ValueError(f"source must be 'pseudo' or 'bulk', not {source!r}")
        if self.count == 0:
            return frame
        values = frame.loc[:, list(self.genes)].to_numpy(dtype=np.float64)
        projected = ((values - mean) @ self.directions) @ self.directions.T
        return pd.DataFrame(
            values - projected, index=frame.index, columns=list(self.genes)
        )


def fit_alignment(
    pseudobulk: pd.DataFrame,
    bulk: pd.DataFrame,
    *,
    pseudo_components: int,
    bulk_components: int,
) -> Alignment:
    """The union of the top ``pseudo_components`` pseudo-bulk-excess and
    ``bulk_components`` bulk-excess directions over the paired rows, made
    orthonormal."""
    if list(pseudobulk.index) != list(bulk.index) or list(pseudobulk.columns) != list(
        bulk.columns
    ):
        raise ValueError("pseudo-bulk and bulk must be paired on lines and genes")
    p, b = pseudobulk.to_numpy(dtype=np.float64), bulk.to_numpy(dtype=np.float64)
    union = np.hstack(
        [
            contrastive_directions(p, b, pseudo_components),
            contrastive_directions(b, p, bulk_components),
        ]
    )
    if union.shape[1]:
        left, singular, _ = np.linalg.svd(union, full_matrices=False)
        union = left[:, singular > _RANK_TOLERANCE * singular[0]]
    return Alignment(tuple(pseudobulk.columns), union, p.mean(axis=0), b.mean(axis=0))
