"""Per-gene single-cell statistics of a line's basal cells (the prepared ``q_sc``
caches: log-space mean, detected fraction, log-space variance) stacked on the prior
as a two-stage ridge (protocol §13)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.context_prior.gene_features import select_genes
from src.context_prior.prior import feature_scale, standardize
from src.context_prior.ridge import column_scale, gene_ridge, selected_ridge
from src.context_prior.space import quantile_normalize, quantile_reference

#: The ``q_sc`` columns, in the cache's order.
STATISTICS = ("mean", "detected", "variance")


def read_statistics(
    prepared_root: Path, lines: Sequence[str], genes: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Lines x genes per statistic from ``lines/<ModelID>.npz``; NaN where a
    line's source does not measure the gene. ``genes`` is the prepared panel."""
    from src.data.prepared import read_prepared_line

    values = np.stack(
        [
            read_prepared_line(
                Path(prepared_root) / "lines" / f"{model_id}.npz", tuple(genes)
            ).q_sc.values
            for model_id in lines
        ]
    )  # lines x genes x statistics
    if values.shape[1:] != (len(genes), len(STATISTICS)):
        raise ValueError(
            f"q_sc caches hold {values.shape[1:]}, not {len(genes)} genes x "
            f"{len(STATISTICS)} statistics"
        )
    return {
        name: pd.DataFrame(values[:, :, i], index=list(lines), columns=list(genes))
        for i, name in enumerate(STATISTICS)
    }


def normalize_statistics(
    statistics: Mapping[str, pd.DataFrame],
    train: Sequence[str],
    measured_in: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Every statistic on the genes ``measured_in`` lines all measure, a gap
    filled with the gene's training-line mean, then quantile-normalised per line
    to the training lines' mean sorted profile (sources differ in depth and
    vocabulary; ranks within a line do not). Fitted on ``train`` only."""
    first = next(iter(statistics.values()))
    measured = first.loc[list(measured_in)].notna().all(axis=0)
    genes = list(first.columns[measured])
    out = {}
    for name, frame in statistics.items():
        frame = frame.loc[:, genes]
        means = frame.loc[list(train)].mean(axis=0, skipna=True)
        if means.isna().any():
            raise ValueError(
                f"{name}: genes no training line measures: "
                f"{list(means.index[means.isna()])[:10]}"
            )
        filled = frame.fillna(means)
        profile = quantile_reference(filled.loc[list(train)])
        out[name] = quantile_normalize(filled, profile)
    return out


def _own_features(
    statistics: Mapping[str, pd.DataFrame],
    chosen: Sequence[str],
    genes: Sequence[str],
    lines: Sequence[str],
) -> np.ndarray:
    """Lines x genes x len(chosen): each gene's own statistics, NaN where the
    gene is not a candidate column."""
    columns = list(statistics[chosen[0]].columns)
    position = {gene: i for i, gene in enumerate(columns)}
    own = np.array([position.get(gene, -1) for gene in genes])
    out = np.full((len(lines), len(genes), len(chosen)), np.nan, dtype=np.float32)
    present = own >= 0
    for k, name in enumerate(chosen):
        values = statistics[name].loc[list(lines)].to_numpy(dtype=np.float32)
        out[:, present, k] = values[:, own[present]]
    return out


def _pool(
    statistics: Mapping[str, pd.DataFrame],
    chosen: Sequence[str],
    genes: Sequence[str],
    lines: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """The pool (lines x chosen statistics' columns, statistic-major) and each
    gene's own pool columns (genes x len(chosen), -1 where it has none)."""
    columns = list(statistics[chosen[0]].columns)
    position = {gene: i for i, gene in enumerate(columns)}
    own = np.array([position.get(gene, -1) for gene in genes])
    pool = np.hstack(
        [
            statistics[name].loc[list(lines)].to_numpy(dtype=np.float64)
            for name in chosen
        ]
    )
    own_columns = np.stack(
        [np.where(own >= 0, own + k * len(columns), -1) for k in range(len(chosen))],
        axis=1,
    )
    return pool, own_columns


def stack_predictions(
    statistics: Mapping[str, pd.DataFrame],
    target: pd.DataFrame,
    queries: Sequence[str],
    *,
    chosen: Sequence[str],
    own_shrinkages: Sequence[float],
    selected_penalties: Sequence[float],
    count: int,
) -> dict[tuple[float, float], pd.DataFrame]:
    """Query predictions (queries x target genes) of the two-stage stack, per
    (own shrinkage, selected penalty), fitted on ``target``'s rows only.

    ``target`` is what the prior leaves on the training lines (lines x genes,
    residual-SD units); a missing label counts as zero, as in the prior. The own
    stage reads the gene's own statistics (pooled weights plus shrunk per-gene
    deviations); the selected stage, per gene, the ``count`` pool columns (every
    gene's chosen statistics, its own left out) most correlated with what the own
    stage leaves.
    """
    train, genes = list(target.index), list(target.columns)
    remaining = np.where(np.isfinite(target.to_numpy()), target.to_numpy(), 0.0)

    raw = _own_features(statistics, chosen, genes, train)
    mean, inverse = feature_scale(raw)
    own_train = standardize(raw, mean, inverse)
    own_query = standardize(
        _own_features(statistics, chosen, genes, queries), mean, inverse
    )

    pool_train, own_columns = _pool(statistics, chosen, genes, train)
    pool_query, _ = _pool(statistics, chosen, genes, queries)
    center, scale = column_scale(pool_train)
    pool_train = (pool_train - center) / scale
    pool_query = (pool_query - center) / scale

    out = {}
    own_fits = gene_ridge(own_train, remaining, list(own_shrinkages), pooled=True)
    for shrinkage, own_fit in zip(own_shrinkages, own_fits, strict=True):
        left = remaining - own_fit.predict(own_train)
        selection = select_genes(pool_train, left, count, own_columns)
        selected_fits = selected_ridge(
            pool_train, selection, left, list(selected_penalties)
        )
        for penalty, selected_fit in zip(
            selected_penalties, selected_fits, strict=True
        ):
            prediction = own_fit.predict(own_query) + selected_fit.predict(pool_query)
            out[(float(shrinkage), float(penalty))] = pd.DataFrame(
                prediction, index=list(queries), columns=genes
            )
    return out
