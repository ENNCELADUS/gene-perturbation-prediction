"""Train-fit principal components of the Tx1 line context ``z_c``.

``z_c`` (per-dimension mean and population variance of a line's Tx1 cell
embeddings) is z-scored per dimension and projected on its leading principal
components, all fitted on the labelled training lines. Every score is divided by
one constant, the first component's score SD over those lines, so component one
has unit variance and the tail keeps its smaller scale (eigen-scaled, not
whitened). Numpy only: preparation workers import this module without torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

STATE_KEYS = ("mean", "scale", "components", "score_scale")


def pooled_context(controls_tx1: np.ndarray) -> np.ndarray:
    """``z_c``: per-dimension mean and population variance of the cells, float64."""
    cells = np.asarray(controls_tx1, dtype=np.float64)
    mean = cells.mean(axis=0)
    return np.concatenate((mean, np.square(cells - mean).mean(axis=0)))


@dataclass(frozen=True)
class ContextPCA:
    """Fitted context compression ``z_c -> z~_c``.

    Attributes:
        mean: Per-dimension training mean of ``z_c``.
        scale: Per-dimension training population SD; 1 where the column is constant.
        components: ``[n_components, width]`` unit loadings, each signed so its
            largest-magnitude loading is positive.
        score_scale: The first component's training score SD, ``sqrt(lambda_1)``.
    """

    mean: np.ndarray = field(repr=False)
    scale: np.ndarray = field(repr=False)
    components: np.ndarray = field(repr=False)
    score_scale: float

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def width(self) -> int:
        return int(self.components.shape[1])

    def transform(self, contexts: np.ndarray) -> np.ndarray:
        """``[n, width]`` pooled contexts to ``[n, n_components]`` scaled scores."""
        contexts = np.asarray(contexts, dtype=np.float64)
        if contexts.ndim != 2 or contexts.shape[1] != self.width:
            raise ValueError(
                f"contexts must be shaped [n, {self.width}], got {contexts.shape}"
            )
        standardized = (contexts - self.mean) / self.scale
        return standardized @ self.components.T / self.score_scale

    def to_state(self) -> dict[str, object]:
        """Arrays and the scaling constant, for the checkpoint's preprocessing."""
        return {
            "mean": self.mean,
            "scale": self.scale,
            "components": self.components,
            "score_scale": float(self.score_scale),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> ContextPCA:
        """Restore without refitting; array-likes (including CPU tensors) accepted."""
        missing = [key for key in STATE_KEYS if key not in state]
        if missing:
            raise ValueError(f"context_pca state has no {', '.join(missing)}")
        mean = np.asarray(state["mean"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        components = np.asarray(state["components"], dtype=np.float64)
        score_scale = float(state["score_scale"])
        if (
            mean.ndim != 1
            or scale.shape != mean.shape
            or components.ndim != 2
            or components.shape[0] < 1
            or components.shape[1] != mean.size
            or not np.isfinite(mean).all()
            or not np.isfinite(components).all()
            or not (np.isfinite(scale).all() and (scale > 0).all())
            or not (np.isfinite(score_scale) and score_scale > 0)
        ):
            raise ValueError("invalid context_pca state")
        return cls(mean, scale, components, score_scale)


def fit_context_pca(contexts: np.ndarray, n_components: int) -> ContextPCA:
    """Fit on ``[n_lines, width]`` training-line contexts.

    Raises:
        ValueError: If ``n_components`` is not positive or exceeds the rank of
            the z-scored training contexts.
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    if contexts.ndim != 2 or contexts.shape[0] < 2:
        raise ValueError("context PCA needs a [n_lines >= 2, width] array")
    if not np.isfinite(contexts).all():
        raise ValueError("context PCA fit data must be finite")
    if n_components < 1:
        raise ValueError(f"n_components must be positive, got {n_components}")
    mean = contexts.mean(axis=0)
    scale = contexts.std(axis=0, ddof=0)
    # A constant column can carry an SD of ~1e-16 from float rounding of its mean;
    # its zero range marks it, so a query off the constant cannot blow up.
    scale[(scale == 0.0) | (np.ptp(contexts, axis=0) == 0.0)] = 1.0
    standardized = (contexts - mean) / scale
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    tolerance = singular[0] * max(standardized.shape) * np.finfo(np.float64).eps
    rank = int((singular > tolerance).sum())
    if n_components > rank:
        raise ValueError(
            f"context PCA asks for {n_components} components but the "
            f"{contexts.shape[0]} training contexts have rank {rank}"
        )
    components = vt[:n_components].copy()
    signs = np.sign(components[np.arange(n_components), np.abs(components).argmax(1)])
    components *= signs[:, None]
    score_scale = float(singular[0] / np.sqrt(contexts.shape[0]))
    return ContextPCA(mean, scale, components, score_scale)


__all__ = ["ContextPCA", "fit_context_pca", "pooled_context"]
