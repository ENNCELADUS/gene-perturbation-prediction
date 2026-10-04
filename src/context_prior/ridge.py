"""Closed-form ridge stages of the linear context prior.

Each stage predicts lines x genes and is fitted to what earlier stages left.
Penalties are multiplied by the number of fitted lines, so one grid serves every
training-set size. Features are standardised with the fitted lines' statistics; a
constant column gets scale 1 and contributes nothing. Every stage has a per-gene
intercept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def column_scale(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD along ``axis``; a constant column gets scale 1.

    Constancy is tested on the range, not the SD: the float mean of a constant
    column can miss the constant by rounding, leaving an SD of ~1e-17 that would
    blow a query off the constant up to ~1e17.
    """
    center = values.mean(axis=axis)
    scale = values.std(axis=axis)
    varies = (np.ptp(values, axis=axis) > 0) & (scale > 0)
    return center, np.where(varies, scale, 1.0)


@dataclass(frozen=True)
class LinearFit:
    """One feature matrix shared by every gene."""

    center: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # features x genes
    intercept: np.ndarray  # genes

    def predict(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.center) / self.scale) @ self.coef + self.intercept


def shared_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    penalties: Sequence[float],
    *,
    basis: np.ndarray | None = None,
) -> list[LinearFit]:
    """Ridge of every target column on the same features, one fit per penalty.

    With ``basis`` (genes x k, orthonormal columns) the targets are projected on
    it and the coefficients mapped back: a reduced-rank context term.
    """
    n = features.shape[0]
    center, scale = column_scale(features)
    z = (features - center) / scale
    intercept = targets.mean(axis=0)
    centered = targets - intercept
    if basis is not None:
        centered = centered @ basis
    eigenvalues, eigenvectors = np.linalg.eigh(z.T @ z)
    projected = eigenvectors.T @ (z.T @ centered)
    fits = []
    for penalty in penalties:
        if not penalty > 0:
            raise ValueError("shared ridge penalties must be positive")
        coef = eigenvectors @ (projected / (eigenvalues + penalty * n)[:, None])
        if basis is not None:
            coef = coef @ basis.T
        fits.append(LinearFit(center, scale, coef, intercept))
    return fits


@dataclass(frozen=True)
class GeneFit:
    """Gene-specific features (lines x genes x k): pooled weights plus deviations."""

    pooled: np.ndarray  # k
    deviation: np.ndarray  # genes x k
    intercept: np.ndarray  # genes

    def predict(self, features: np.ndarray) -> np.ndarray:
        weights = self.pooled[None, :] + self.deviation
        return np.einsum("ngk,gk->ng", features, weights) + self.intercept


def gene_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    shrinkages: Sequence[float],
    *,
    pooled: bool,
) -> list[GeneFit]:
    """Per-gene ridge on gene-specific features already standardised per gene.

    With ``pooled``, weights shared by every gene are fitted first by least squares
    over all (line, gene) rows, and each gene's deviation from them is shrunk by
    ``shrinkage * n``; an infinite shrinkage keeps the pooled weights only.
    """
    n, genes, k = features.shape
    intercept = targets.mean(axis=0)
    centered = targets - intercept
    gram = np.einsum("ngk,ngl->gkl", features, features, dtype=np.float64)
    cross = np.einsum("ngk,ng->gk", features, centered, dtype=np.float64)
    shared = np.zeros(k)
    if pooled:
        shared = np.linalg.lstsq(gram.sum(axis=0), cross.sum(axis=0), rcond=None)[0]
        cross = cross - np.einsum("gkl,l->gk", gram, shared)
    fits = []
    for shrinkage in shrinkages:
        if np.isinf(shrinkage):
            deviation = np.zeros((genes, k))
        elif shrinkage > 0:
            system = gram + shrinkage * n * np.eye(k)
            deviation = np.linalg.solve(system, cross[..., None])[..., 0]
        else:
            raise ValueError("gene ridge shrinkages must be positive or infinite")
        fits.append(GeneFit(shared, deviation, intercept))
    return fits


@dataclass(frozen=True)
class SelectedFit:
    """Per-gene ridge on the expression of each gene's selected columns."""

    selection: np.ndarray  # genes x count
    coef: np.ndarray  # genes x count
    intercept: np.ndarray  # genes

    def predict(self, expression: np.ndarray, chunk: int = 128) -> np.ndarray:
        genes = self.selection.shape[0]
        columns = np.ascontiguousarray(expression.T)
        out = np.empty((expression.shape[0], genes))
        for start in range(0, genes, chunk):
            stop = min(start + chunk, genes)
            gathered = columns[self.selection[start:stop]]  # genes x count x lines
            out[:, start:stop] = np.einsum(
                "cmn,cm->nc", gathered, self.coef[start:stop]
            )
        return out + self.intercept


def selected_ridge(
    expression: np.ndarray,
    selection: np.ndarray,
    targets: np.ndarray,
    penalties: Sequence[float],
    *,
    chunk: int = 128,
) -> list[SelectedFit]:
    """One ridge per gene on its selected expression columns, which the caller
    standardises on the fitted lines (mean zero), so the intercept is the mean.

    Selected columns are gathered as rows of the transposed expression and each
    chunk's Gram matrices are one batched BLAS product.
    """
    n = expression.shape[0]
    genes, count = selection.shape
    intercept = targets.mean(axis=0)
    centered = targets - intercept
    columns = np.ascontiguousarray(expression.T)
    coefs = [np.empty((genes, count)) for _ in penalties]
    for start in range(0, genes, chunk):
        stop = min(start + chunk, genes)
        gathered = columns[selection[start:stop]]  # genes x count x lines
        gram = gathered @ gathered.transpose(0, 2, 1)
        cross = np.einsum("cmn,nc->cm", gathered, centered[:, start:stop])
        for coef, penalty in zip(coefs, penalties, strict=True):
            if not penalty > 0:
                raise ValueError("selected ridge penalties must be positive")
            system = gram + penalty * n * np.eye(count)
            coef[start:stop] = np.linalg.solve(system, cross[..., None])[..., 0]
    return [SelectedFit(selection, coef, intercept) for coef in coefs]
