"""The linear context prior: a stagewise ridge over context views and gene features.

``fit_prior`` fits the stages of a :class:`PriorSpec` in order, each on the residual
the earlier stages leave, from training-side bulk rows. ``FittedPrior.predict`` maps
any expression rows (bulk, or bridged pseudo-bulk) to per-stage predictions in units
of the per-gene residual SD, keyed by the rows given. A missing training label
counts as the residual's expected value, zero, in fitting. Everything that reads
labels is fitted on the rows it is given, so cross-fitting is ``fit_prior`` on the
rows outside a fold; the expression components are label-free and fitted once,
outside.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.context_prior.folds import patient_folds
from src.context_prior.gene_features import (
    KNOWLEDGE_FEATURES,
    PartnerIndex,
    knowledge_features,
    partner_index,
    select_genes,
)
from src.context_prior.reference import Reference
from src.context_prior.ridge import (
    column_scale,
    gene_ridge,
    selected_ridge,
    shared_ridge,
)
from src.context_prior.views import expression_components, fit_genotype, pathway_scores
from src.data.context_pca import ContextPCA

BLOCKS = (
    "expression_components",
    "pathway_scores",
    "predicted_genotype",
    "own_expression",
    "partners",
    "data_selected",
)
CONTEXT_BLOCKS = BLOCKS[:3]
#: Features of ``knowledge_features`` read by each knowledge block: own expression,
#: then the paralog, complex and low-partner features as one group.
_KNOWLEDGE_FEATURES = {
    "own_expression": slice(0, 1),
    "partners": slice(1, len(KNOWLEDGE_FEATURES)),
}
#: Folds of the predicted-genotype encoders, inside whatever lines a fit is given.
INNER_FOLDS, INNER_FOLD_SEED = 5, 1
#: A partner is low at or below this training-side percentile.
LOW_PERCENTILE = 10

Features = Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class Stage:
    """One block; ``penalty`` is the ridge strength times n, or for a knowledge
    block the shrinkage toward the pooled weights (infinite: pooled only)."""

    block: str
    penalty: float
    selected: int = 0

    def __post_init__(self) -> None:
        if self.block not in BLOCKS:
            raise ValueError(f"unknown block {self.block!r}")
        if (self.block == "data_selected") != (self.selected > 0):
            raise ValueError("data_selected, and only it, needs a positive count")


@dataclass(frozen=True)
class PriorSpec:
    """Stages in fitting order; ``rank`` restricts the context stages to that many
    residual factors (None: per gene)."""

    stages: tuple[Stage, ...]
    rank: int | None = None

    def __post_init__(self) -> None:
        blocks = [stage.block for stage in self.stages]
        if len(set(blocks)) != len(blocks):
            raise ValueError("a block appears twice")
        if self.rank is not None and self.rank < 1:
            raise ValueError(f"rank must be positive, got {self.rank}")


@dataclass(frozen=True)
class PriorInputs:
    """Everything a fit reads, keyed by ModelID.

    Attributes:
        expression: Quantile-normalised bulk rows of the training side.
        residual: Labelled training-side lines x genes, in residual-SD units.
        components: Expression components, fitted once, label-free.
        reference: Pinned reference tables.
        lineage: Training-side lineage labels.
        patients: PatientID of every line.
        gene_space: Expression genes the gene-level blocks may read; a gene
            outside it is an undefined (zero) feature. None: every gene.
        gene_rows: Expression rows the gene-level blocks are fitted on, with
            labels in ``residual``, instead of the fit lines' rows; they fit to
            what the context stages leave on these rows. None: the fit lines.
    """

    expression: pd.DataFrame
    residual: pd.DataFrame
    components: ContextPCA
    reference: Reference
    lineage: pd.Series
    patients: Mapping[str, str]
    gene_space: tuple[str, ...] | None = None
    gene_rows: pd.DataFrame | None = None


@dataclass
class FittedPrior:
    genes: tuple[str, ...]
    space: tuple[str, ...]
    stages: list[tuple[str, Features, Any]]

    def predict(self, expression: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Per-stage predictions of ``expression``'s rows, keyed by its index."""
        rows = expression.loc[:, list(self.space)]
        return {
            block: pd.DataFrame(
                model.predict(features(rows)),
                index=rows.index,
                columns=list(self.genes),
            )
            for block, features, model in self.stages
        }


def total(predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = list(predictions.values())
    if not frames:
        raise ValueError("a prior needs at least one stage")
    out = frames[0].copy()
    for frame in frames[1:]:
        out = out + frame
    return out


def _context_view(
    block: str, inputs: PriorInputs, fit_rows: pd.DataFrame, encoder_rows: pd.DataFrame
) -> tuple[Features, np.ndarray]:
    """Features of any rows, and the fit rows' training features.

    The predicted-genotype encoders that serve queries are fitted on every encoder
    row; a fit row's training features come from the inner fold that excludes it,
    so the stage learns at the noise level a query will have.
    """
    if block == "expression_components":

        def features(rows: pd.DataFrame) -> np.ndarray:
            return inputs.components.transform(rows.to_numpy(dtype=np.float64))

        return features, features(fit_rows)
    if block == "pathway_scores":

        def features(rows: pd.DataFrame) -> np.ndarray:
            reference = inputs.reference
            return pathway_scores(
                rows, reference.hallmark, reference.progeny
            ).to_numpy()

        return features, features(fit_rows)
    components = expression_components(inputs.components, encoder_rows)
    folds = patient_folds(
        list(encoder_rows.index),
        inputs.patients,
        n_folds=INNER_FOLDS,
        seed=INNER_FOLD_SEED,
    )
    encoders, own = fit_genotype(components, inputs.reference, inputs.lineage, folds)
    columns = list(own.columns)

    def features(rows: pd.DataFrame) -> np.ndarray:
        query = expression_components(inputs.components, rows)
        return encoders.predict(query)[columns].to_numpy()

    return features, own.loc[list(fit_rows.index), columns].to_numpy()


def _feature_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per gene and feature over the fit rows: the mean, and the inverse SD.

    A feature whose range on the fit rows is zero (constant, or undefined: no such
    partner) gets inverse 0 and reads zero for every row, training or query. A
    pooled weight would otherwise move a query off the constant, and a float mean
    that misses the constant by rounding would leave an SD near 0 to divide by.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN: no such partner
        mean = np.nanmean(values, axis=0, dtype=np.float64)
        std = np.nanstd(values, axis=0, dtype=np.float64)
        varies = np.nanmax(values, axis=0) > np.nanmin(values, axis=0)
    inverse = np.zeros_like(std)
    np.divide(1.0, std, out=inverse, where=varies)
    return np.where(varies, mean, 0.0).astype(np.float32), inverse.astype(np.float32)


def _standardize(
    values: np.ndarray, mean: np.ndarray, inverse: np.ndarray
) -> np.ndarray:
    """``(values - mean) * inverse`` in place; an undefined feature reads zero."""
    values -= mean
    values *= inverse
    np.copyto(values, 0.0, where=np.isnan(values))
    return values


def _knowledge_view(
    features_of_block: slice,
    index: PartnerIndex,
    low: np.ndarray,
    fit_rows: pd.DataFrame,
) -> tuple[Features, np.ndarray]:
    """Features z-scored per gene with the fit rows' statistics (lines x genes x k).

    The block's features are a view into ``knowledge_features``' fresh array and
    are standardised in place, so no full-size copy is made.
    """

    def raw(rows: pd.DataFrame) -> np.ndarray:
        values = knowledge_features(rows.to_numpy(dtype=np.float64), index, low)
        return values[:, :, features_of_block]

    train = raw(fit_rows)
    mean, inverse = _feature_scale(train)

    def features(rows: pd.DataFrame) -> np.ndarray:
        return _standardize(raw(rows), mean, inverse)

    return features, _standardize(train, mean, inverse)


def _selected_view(fit_rows: pd.DataFrame) -> tuple[Features, np.ndarray]:
    """Expression standardised per column with the fit rows' statistics."""
    center, scale = column_scale(fit_rows.to_numpy(dtype=np.float64))

    def features(rows: pd.DataFrame) -> np.ndarray:
        return (rows.to_numpy(dtype=np.float64) - center) / scale

    return features, features(fit_rows)


def _residual_basis(residual: np.ndarray, rank: int) -> np.ndarray:
    """Genes x ``rank``: the leading right singular vectors of the centred residual,
    the targets ``shared_ridge`` projects on them."""
    if rank > min(residual.shape):
        raise ValueError(f"rank {rank} exceeds the residual's shape {residual.shape}")
    centered = residual - residual.mean(axis=0)
    return np.linalg.svd(centered, full_matrices=False)[2][:rank].T


def fit_prior(
    spec: PriorSpec,
    inputs: PriorInputs,
    *,
    fit_lines: Sequence[str],
    encoder_lines: Sequence[str],
) -> FittedPrior:
    """Fit every stage in order on labelled ``fit_lines``; encoders that read no
    GeneEffect (predicted genotype, low-expression thresholds) use ``encoder_lines``.

    The reduced-rank basis, when ``spec.rank`` is set, comes from the fit lines'
    residual and restricts the context stages only. Context stages come before
    gene-level stages. The gene-level stages read the columns of
    ``inputs.gene_space`` and fit on ``inputs.gene_rows`` when they are set; the
    data-selected stage selects at most every readable column but a gene's own.
    """
    if not set(fit_lines) <= set(encoder_lines):
        raise ValueError("fit lines must be encoder lines too")
    genes = tuple(inputs.residual.columns)
    space = tuple(inputs.expression.columns)
    fit_rows = inputs.expression.loc[list(fit_lines)]
    encoder_rows = inputs.expression.loc[list(encoder_lines)]
    remaining = inputs.residual.loc[list(fit_lines)].to_numpy(dtype=np.float64)
    remaining = np.where(np.isfinite(remaining), remaining, 0.0)
    basis = None if spec.rank is None else _residual_basis(remaining, spec.rank)
    readable = None if inputs.gene_space is None else set(inputs.gene_space)
    gene_columns = (
        space if readable is None else tuple(g for g in space if g in readable)
    )
    gene_rows = (
        None if inputs.gene_rows is None else inputs.gene_rows.loc[:, list(space)]
    )
    gene_fit_rows = fit_rows if gene_rows is None else gene_rows
    gene_remaining: np.ndarray | None = None
    index: PartnerIndex | None = None
    low: np.ndarray | None = None
    stages = []
    for stage in spec.stages:
        if stage.block in CONTEXT_BLOCKS:
            if gene_remaining is not None:
                raise ValueError("context blocks must come before gene-level blocks")
            features, train = _context_view(stage.block, inputs, fit_rows, encoder_rows)
            model = shared_ridge(train, remaining, [stage.penalty], basis=basis)[0]
            remaining -= model.predict(train)
            stages.append((stage.block, features, model))
            continue
        if gene_remaining is None:
            # Gene-level blocks fit on their own rows: what the context stages
            # leave on those rows (the same rows as the context stages by default).
            if gene_rows is None:
                gene_remaining = remaining
            else:
                residual = inputs.residual.loc[list(gene_rows.index)].to_numpy(
                    dtype=np.float64
                )
                gene_remaining = np.where(np.isfinite(residual), residual, 0.0)
                for _, context_features, context_model in stages:
                    gene_remaining = gene_remaining - context_model.predict(
                        context_features(gene_rows)
                    )
        rows = gene_fit_rows.loc[:, list(gene_columns)]
        if stage.block in _KNOWLEDGE_FEATURES:
            if index is None:
                index = partner_index(genes, gene_columns, inputs.reference)
                source = encoder_rows if gene_rows is None else gene_rows
                low = np.percentile(
                    source.loc[:, list(gene_columns)].to_numpy(dtype=np.float64),
                    LOW_PERCENTILE,
                    axis=0,
                )
            inner, train = _knowledge_view(
                _KNOWLEDGE_FEATURES[stage.block], index, low, rows
            )
            model = gene_ridge(train, gene_remaining, [stage.penalty], pooled=True)[0]
        else:
            inner, train = _selected_view(rows)
            position = {gene: i for i, gene in enumerate(gene_columns)}
            own = np.array([position.get(gene, -1) for gene in genes], dtype=int)
            # A gene space narrower than the count yields every column but a
            # gene's own; one with no column to spare selects none (intercepts).
            count = min(stage.selected, len(gene_columns) - 1)
            selection = (
                select_genes(train, gene_remaining, count, own)
                if count > 0
                else np.empty((len(genes), 0), dtype=int)
            )
            model = selected_ridge(train, selection, gene_remaining, [stage.penalty])[0]
        # Reassigned, never updated in place: without gene rows it aliases
        # ``remaining`` until here.
        gene_remaining = gene_remaining - model.predict(train)

        def features(rows_in: pd.DataFrame, inner=inner) -> np.ndarray:
            return inner(rows_in.loc[:, list(gene_columns)])

        stages.append((stage.block, features, model))
    return FittedPrior(genes, space, stages)


def crossfit(
    spec: PriorSpec,
    inputs: PriorInputs,
    *,
    folds: Mapping[str, int],
    queries: Mapping[int, pd.DataFrame],
    labelled: Sequence[str],
    encoder_lines: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Per-stage predictions of each fold's query rows by the prior fitted on the
    lines outside that fold, keyed by the query rows' index."""
    if inputs.gene_rows is not None:
        raise ValueError("cross-fitting does not split gene rows by fold")
    parts: dict[str, list[pd.DataFrame]] = {}
    for fold, query in sorted(queries.items()):
        strays = [m for m in query.index if folds[m] != fold]
        if strays:
            raise ValueError(f"fold {fold} queries lines of other folds: {strays[:10]}")
        fitted = fit_prior(
            spec,
            inputs,
            fit_lines=[m for m in labelled if folds[m] != fold],
            encoder_lines=[m for m in encoder_lines if folds[m] != fold],
        )
        for block, frame in fitted.predict(query).items():
            parts.setdefault(block, []).append(frame)
    return {
        block: pd.concat(frames, verify_integrity=True)
        for block, frames in parts.items()
    }
