"""Cell-line-context-free residual targets for DepMap GeneEffect.

DepMap GeneEffect(g, c) for gene g in cell line c decomposes as
``GeneEffect(g, c) = mu_g + delta(g, c)``, where ``mu_g`` is the gene main
effect (how essential g is on average across lines) and ``delta`` is the
line-specific deviation. Because ``Var(mu_g) >> Var(delta)``, a model scored
on raw GeneEffect is mostly graded on recovering ``mu_g`` — a context-blind
per-gene mean already captures most of the achievable score. This module
estimates ``mu_hat_g`` from train lines only and exposes the residual
``gene_effect - mu_hat_g`` as the prediction target, so a model is scored on
the line-specific deviation it is actually meant to explain.

The correctness-critical property is that ``mu_hat_g`` must never be fit
using a held-out line's own label: :func:`fit_gene_means` only ever sees
``train_lines``, and :func:`build_residual_targets` applies that train-fit
mean to every row (train and held-out alike) without ever refitting on the
full dataset.

This module is intentionally plain: pure pandas/numpy, no torch, no file
I/O, no hashing, and no provenance bookkeeping.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_LABEL_COLUMNS = ("model_id", "gene_symbol", "gene_effect")


@dataclass(frozen=True)
class ResidualTargets:
    """Long-form residual targets plus the train-fit gene means used to build them.

    Attributes:
        long: One row per surviving (model_id, gene_symbol) observation, with
            columns model_id, gene_symbol, gene_effect, gene_mean, residual.
        gene_mean: Train-fit ``mu_hat_g``, indexed by gene_symbol.
        train_lines: The train line IDs used to fit ``gene_mean``, as given.
        n_genes: Number of distinct genes present in ``long``.
        n_lines: Number of distinct model_id values present in ``long``.
    """

    long: pd.DataFrame
    gene_mean: pd.Series
    train_lines: tuple[str, ...]
    n_genes: int
    n_lines: int


def _validate_labels_schema(labels: pd.DataFrame) -> None:
    """Raise ValueError if ``labels`` is missing any required column."""
    missing = [c for c in _REQUIRED_LABEL_COLUMNS if c not in labels.columns]
    if missing:
        raise ValueError(f"labels is missing required columns: {missing}")


def _validate_train_lines(labels: pd.DataFrame, train_lines: Sequence[str]) -> None:
    """Raise ValueError on empty, duplicate, or unknown train_lines."""
    if len(train_lines) == 0:
        raise ValueError("train_lines must not be empty")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for line in train_lines:
        if line in seen:
            duplicates.add(line)
        seen.add(line)
    if duplicates:
        raise ValueError(f"train_lines contains duplicates: {sorted(duplicates)}")
    known_lines = set(labels["model_id"])
    unknown = sorted(set(train_lines) - known_lines)
    if unknown:
        raise ValueError(
            f"train_lines contains model_id(s) absent from labels: {unknown}"
        )


def fit_gene_means(
    labels: pd.DataFrame,
    train_lines: Sequence[str],
    *,
    min_lines: int = 3,
) -> pd.Series:
    """Estimate mu_hat_g from train lines only.

    Computes the mean of ``gene_effect`` (skipping NaN) over rows whose
    ``model_id`` is in ``train_lines``, grouped by ``gene_symbol``. Genes with
    fewer than ``min_lines`` non-NaN train observations are dropped from the
    result rather than kept with a noisy mean or zero-filled.

    Args:
        labels: Long-form DataFrame with columns model_id, gene_symbol,
            gene_effect.
        train_lines: Model IDs to fit the per-gene mean over. Must be
            non-empty, duplicate-free, and a subset of labels["model_id"].
        min_lines: Minimum number of non-NaN train observations a gene needs
            to be retained. Genes below this threshold are dropped with a
            logged warning naming the count.

    Returns:
        Series of mu_hat_g indexed by gene_symbol (name "gene_mean"),
        containing only genes that met ``min_lines``.

    Raises:
        ValueError: If labels lacks required columns, or train_lines is
            empty, contains duplicates, or contains a model_id absent from
            labels.
    """
    _validate_labels_schema(labels)
    _validate_train_lines(labels, train_lines)

    train_rows = labels[labels["model_id"].isin(set(train_lines))]
    all_genes = pd.Index(sorted(labels["gene_symbol"].unique()), name="gene_symbol")
    grouped = train_rows.groupby("gene_symbol")["gene_effect"]
    # Reindex over every gene seen anywhere in labels (not just genes with at
    # least one train row) so genes absent from train_lines entirely — or
    # present only with NaN gene_effect there — are also counted as dropped
    # rather than silently omitted from the warning.
    counts = grouped.count().reindex(all_genes, fill_value=0)
    means = grouped.mean().reindex(all_genes)

    keep_mask = counts >= min_lines
    n_dropped = int((~keep_mask).sum())
    if n_dropped:
        logger.warning(
            "fit_gene_means: dropping %d gene(s) with fewer than min_lines=%d "
            "non-NaN train observations",
            n_dropped,
            min_lines,
        )

    gene_mean = means[keep_mask].sort_index()
    gene_mean.name = "gene_mean"
    gene_mean.index.name = "gene_symbol"
    return gene_mean


def build_residual_targets(
    labels: pd.DataFrame,
    train_lines: Sequence[str],
) -> ResidualTargets:
    """Attach a train-fit gene_mean and residual to every row, train and held-out alike.

    ``gene_mean`` is fit via :func:`fit_gene_means` using only ``train_lines``
    (with its default ``min_lines``), then broadcast onto every row of
    ``labels``, including rows whose model_id is not in ``train_lines``. This
    is the leakage-safe construction: a held-out line's own labels never
    participate in estimating the mean subtracted from it.

    Rows whose gene was dropped by the ``min_lines`` filter are removed from
    the returned ``long`` entirely (their count is logged). Rows with NaN
    gene_effect are retained with NaN residual — they are not filled.

    Args:
        labels: Long-form DataFrame with columns model_id, gene_symbol,
            gene_effect.
        train_lines: Model IDs to fit the per-gene mean over. See
            :func:`fit_gene_means` for validation rules.

    Returns:
        A ResidualTargets bundle whose ``long`` frame has columns model_id,
        gene_symbol, gene_effect, gene_mean, residual.
    """
    _validate_labels_schema(labels)
    gene_mean = fit_gene_means(labels, train_lines)

    n_rows_before = len(labels)
    long = labels.merge(
        gene_mean.rename("gene_mean"),
        left_on="gene_symbol",
        right_index=True,
        how="inner",
    )
    n_dropped_rows = n_rows_before - len(long)
    if n_dropped_rows:
        logger.warning(
            "build_residual_targets: dropped %d row(s) whose gene did not "
            "survive the min_lines filter",
            n_dropped_rows,
        )

    long = long.copy()
    long["residual"] = long["gene_effect"] - long["gene_mean"]
    long = long[["model_id", "gene_symbol", "gene_effect", "gene_mean", "residual"]]
    long = long.reset_index(drop=True)

    return ResidualTargets(
        long=long,
        gene_mean=gene_mean,
        train_lines=tuple(train_lines),
        n_genes=long["gene_symbol"].nunique(),
        n_lines=long["model_id"].nunique(),
    )


def to_matrix(labels: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot a long-form frame to wide, genes x lines.

    Args:
        labels: Long-form DataFrame containing at least model_id,
            gene_symbol, and ``value_col``.
        value_col: Name of the column to place in the matrix cells.

    Returns:
        DataFrame indexed by gene_symbol (sorted) with columns model_id
        (sorted), containing ``value_col`` values.

    Raises:
        ValueError: If ``labels`` contains duplicate (gene_symbol, model_id)
            pairs, or is missing model_id, gene_symbol, or ``value_col``.
    """
    required = {"model_id", "gene_symbol", value_col}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels is missing required columns: {missing}")

    key = labels[["gene_symbol", "model_id"]]
    duplicated = key.duplicated(keep=False)
    if duplicated.any():
        dup_pairs = (
            key[duplicated].drop_duplicates().itertuples(index=False, name=None)
        )
        raise ValueError(
            f"labels contains duplicate (gene_symbol, model_id) pairs: "
            f"{sorted(dup_pairs)[:10]}"
        )

    matrix = labels.pivot(index="gene_symbol", columns="model_id", values=value_col)
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    matrix.columns.name = "model_id"
    matrix.index.name = "gene_symbol"
    return matrix


__all__ = [
    "ResidualTargets",
    "fit_gene_means",
    "build_residual_targets",
    "to_matrix",
]
