"""Features of gene g in line c: own, paralog and complex-partner expression, the
low-partner fraction (ISLE's cSL score), and data-selected genes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.context_prior.reference import Reference

KNOWLEDGE_FEATURES = (
    "own",
    "paralog_min",
    "paralog_sum",
    "paralog_closest",
    "complex_mean",
    "low_partner_fraction",
)


@dataclass(frozen=True)
class PartnerIndex:
    """Column positions in the expression space per target gene (-1: absent)."""

    own: np.ndarray
    paralogs: list[np.ndarray]  # closest first
    complex_members: list[np.ndarray]
    partners: list[np.ndarray]  # paralogs and complex members


def partner_index(
    genes: Sequence[str], space: Sequence[str], reference: Reference
) -> PartnerIndex:
    """Positions of each gene's own column, its measured paralogs in the table's
    order (closest first) and its measured CORUM co-members over every complex."""
    position = {gene: i for i, gene in enumerate(space)}
    paralogs = {
        gene: [position[p] for p in rows["paralog"] if p in position]
        for gene, rows in reference.paralogs.groupby("gene", sort=False)
    }
    members: dict[str, set[str]] = {}
    for _, rows in reference.complexes.groupby("complex_id"):
        group = set(rows["gene"])
        for gene in group:
            members.setdefault(gene, set()).update(group - {gene})
    complex_members = {
        gene: sorted(position[m] for m in others if m in position)
        for gene, others in members.items()
    }
    own = np.array([position.get(g, -1) for g in genes], dtype=int)
    paralog_list = [np.array(paralogs.get(g, []), dtype=int) for g in genes]
    complex_list = [np.array(complex_members.get(g, []), dtype=int) for g in genes]
    partners = [
        np.unique(np.concatenate([p, c])).astype(int)
        for p, c in zip(paralog_list, complex_list, strict=True)
    ]
    return PartnerIndex(own, paralog_list, complex_list, partners)


def knowledge_features(
    expression: np.ndarray, index: PartnerIndex, low_threshold: np.ndarray
) -> np.ndarray:
    """Lines x genes x 6 raw features; NaN where a gene has no such partner.

    A partner counts as low when its expression is at or below its training-side
    10th percentile (``low_threshold``, one value per expression column).
    """
    # Built gene-major (contiguous per-gene gathers and writes), then laid out
    # lines x genes x features: about 9x faster than strided columns at full size.
    by_gene = np.ascontiguousarray(np.asarray(expression).T)
    lines, genes = by_gene.shape[1], len(index.own)
    out = np.full((genes, len(KNOWLEDGE_FEATURES), lines), np.nan, dtype=np.float32)
    for g in range(genes):
        if index.own[g] >= 0:
            out[g, 0] = by_gene[index.own[g]]
        paralogs = index.paralogs[g]
        if paralogs.size:
            values = by_gene[paralogs]
            out[g, 1] = values.min(axis=0)
            out[g, 2] = values.sum(axis=0)
            out[g, 3] = values[0]
        members = index.complex_members[g]
        if members.size:
            out[g, 4] = by_gene[members].mean(axis=0)
        partners = index.partners[g]
        if partners.size:
            out[g, 5] = (by_gene[partners] <= low_threshold[partners, None]).mean(
                axis=0
            )
    return np.ascontiguousarray(out.transpose(2, 0, 1))


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    return (values - center) / np.where(scale > 0, scale, 1.0)


def select_genes(
    expression: np.ndarray,
    residual: np.ndarray,
    count: int,
    own: np.ndarray,
    chunk: int = 1024,
) -> np.ndarray:
    """Genes x ``count`` expression columns with the largest absolute Pearson
    correlation with each gene's residual across rows; a gene's own column is
    excluded. Missing residuals count as zero."""
    if not 0 < count < expression.shape[1]:
        raise ValueError(
            f"count must lie in [1, {expression.shape[1] - 1}] so a gene's own column "
            f"can be left out, got {count}"
        )
    x = _standardize_columns(expression)
    y = _standardize_columns(np.where(np.isfinite(residual), residual, 0.0))
    genes = y.shape[1]
    out = np.empty((genes, count), dtype=int)
    for start in range(0, genes, chunk):
        stop = min(start + chunk, genes)
        strength = np.abs(x.T @ y[:, start:stop]) / x.shape[0]
        for local, g in enumerate(range(start, stop)):
            if own[g] >= 0:
                strength[own[g], local] = -1.0
        out[start:stop] = np.argpartition(-strength, count - 1, axis=0)[:count].T
    return out
