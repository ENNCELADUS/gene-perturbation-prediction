"""DepMap GeneEffect labels by ModelID and what the training lines fit from them:
gene means and residual targets, variable and selective genes, the residual SD.

A held-out line's label never enters its own centring: ``fit_gene_means`` sees only
training lines, and every row (held out or not) is centred on that fit."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.data.splits import FixedSplit

DEPENDENCY_THRESHOLD: Final = -0.5


_GENE_EFFECT_COLUMN_RE: Final = re.compile(r"^(?P<symbol>\S+) \((?P<entrez>\d+)\)$")


def fit_variable_gene_membership(
    labels: pd.DataFrame,
    train_lines: Sequence[str],
    genes: Sequence[str],
    *,
    min_observations: int = 5,
    percentile: float = 75.0,
) -> frozenset[str]:
    """Fit residual-variance membership using labeled training rows only."""
    required = {"model_id", "gene_symbol", "residual"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels is missing columns: {missing}")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = _unique_strings(genes, "variable-gene panel")
    train_set = set(train_lines)
    if not train_set:
        raise ValueError("train_lines must not be empty")
    train = labels.loc[labels["model_id"].isin(train_set)]
    grouped = train.groupby("gene_symbol")["residual"]
    counts = grouped.count().reindex(ordered, fill_value=0)
    eligible = counts[counts >= min_observations].index
    variance = grouped.var(ddof=0).reindex(eligible)
    if variance.empty or not bool(np.isfinite(variance.to_numpy()).all()):
        raise ValueError("variable-gene fit has no finite eligible variances")
    threshold = float(np.percentile(variance.to_numpy(), percentile, method="linear"))
    return frozenset(variance.index[variance >= threshold].astype(str))


def fit_selective_genes(
    labels: pd.DataFrame,
    train_lines: Sequence[str],
    genes: Sequence[str],
    *,
    min_lines: int,
    max_fraction: float,
) -> frozenset[str]:
    """Fit selective-dependency membership using labeled training rows only.

    A gene is selective when at least ``min_lines`` training lines depend on it
    (``gene_effect < DEPENDENCY_THRESHOLD``) and fewer than ``max_fraction`` of
    its labeled training lines do.
    """
    required = {"model_id", "gene_symbol", "gene_effect"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels is missing columns: {missing}")
    if min_lines <= 0:
        raise ValueError("min_lines must be positive")
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError("max_fraction must be in (0, 1]")
    ordered = _unique_strings(genes, "selective-gene panel")
    train_set = set(train_lines)
    if not train_set:
        raise ValueError("train_lines must not be empty")
    train = labels.loc[
        labels["model_id"].isin(train_set) & np.isfinite(labels["gene_effect"])
    ]
    dependent = train["gene_effect"] < DEPENDENCY_THRESHOLD
    grouped = dependent.groupby(train["gene_symbol"])
    counts = grouped.sum().reindex(ordered, fill_value=0)
    rows = grouped.count().reindex(ordered, fill_value=0)
    selective = (counts >= min_lines) & (counts / rows < max_fraction)
    return frozenset(counts.index[selective].astype(str))


def fit_residual_scale(
    labels: pd.DataFrame,
    train_lines: Sequence[str],
    genes: Sequence[str],
    *,
    floor_percentile: float,
) -> pd.Series:
    """Fit the per-gene residual SD on labeled training rows only.

    Population SD (ddof 0) per gene, indexed in ``genes`` order. The floor is
    ``floor_percentile`` over the finite SDs; a gene with fewer than two rows or
    an SD below the floor takes the floor, so every value is finite and positive.
    """
    required = {"model_id", "gene_symbol", "residual"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels is missing columns: {missing}")
    if not 0.0 <= floor_percentile <= 100.0:
        raise ValueError("floor_percentile must be between 0 and 100")
    ordered = _unique_strings(genes, "residual-scale panel")
    train_set = set(train_lines)
    if not train_set:
        raise ValueError("train_lines must not be empty")
    train = labels.loc[
        labels["model_id"].isin(train_set) & np.isfinite(labels["residual"])
    ]
    grouped = train.groupby("gene_symbol")["residual"]
    counts = grouped.count().reindex(ordered, fill_value=0)
    sd = grouped.std(ddof=0).reindex(ordered).where(counts >= 2)
    finite = sd.to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("residual-scale fit has no gene with two training rows")
    floor = float(np.percentile(finite, floor_percentile, method="linear"))
    if not floor > 0.0:
        raise ValueError(f"residual-scale floor must be positive; got {floor}")
    scale = sd.fillna(floor).clip(lower=floor)
    scale.index.name = "gene_symbol"
    scale.name = "residual_scale"
    return scale.astype(np.float64)


def _unique_strings(values: Sequence[object], label: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ValueError(f"{label} contains an empty value")
    duplicates = sorted(value for value, count in Counter(result).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicates: {duplicates[:10]}")
    return result


def parse_gene_symbol(column: object) -> str:
    """Parse and normalize DepMap's ``SYMBOL (EntrezID)`` header."""
    text = str(column).strip()
    match = _GENE_EFFECT_COLUMN_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid DepMap GeneEffect column: {text!r}")
    return match.group("symbol").upper()


def load_geneeffect_long(path: Path, split: FixedSplit) -> pd.DataFrame:
    """Load CRISPRGeneEffect.csv into canonical ModelID/symbol long form."""
    wide = pd.read_csv(path, index_col=0)
    wide.index = wide.index.astype(str)
    if not wide.index.is_unique:
        duplicates = sorted(wide.index[wide.index.duplicated()].unique())
        raise ValueError(f"GeneEffect contains duplicate ModelIDs: {duplicates[:10]}")
    symbols = [parse_gene_symbol(column) for column in wide.columns]
    if len(symbols) != len(set(symbols)):
        duplicates = sorted(
            symbol for symbol, count in Counter(symbols).items() if count > 1
        )
        raise ValueError(
            f"GeneEffect columns map to duplicate symbols: {duplicates[:10]}"
        )
    expected_missing = set(split.unlabeled_train)
    observed_missing = set(split.all_model_ids) - set(wide.index)
    if observed_missing != expected_missing:
        raise ValueError(
            "split label coverage does not match unlabeled_train: "
            f"missing={sorted(observed_missing)}"
        )
    selected_ids = [
        model_id for model_id in split.all_model_ids if model_id in wide.index
    ]
    selected = wide.loc[selected_ids]
    numeric = selected.apply(pd.to_numeric, errors="coerce")
    invalid = selected.notna() & numeric.isna()
    if invalid.any().any():
        invalid_positions = np.argwhere(invalid.to_numpy())[:10]
        examples = [
            (
                str(selected.index[row]),
                str(selected.columns[column]),
                str(selected.iloc[row, column]),
            )
            for row, column in invalid_positions
        ]
        raise ValueError(
            f"GeneEffect contains nonnumeric nonmissing values: {examples}"
        )
    infinite = np.isinf(numeric.to_numpy())
    if infinite.any():
        raise ValueError(f"GeneEffect contains {int(infinite.sum())} infinite values")
    numeric.columns = symbols
    long = (
        numeric.rename_axis("model_id")
        .reset_index()
        .melt(id_vars="model_id", var_name="gene_symbol", value_name="gene_effect")
    )
    return long


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


__all__ = [
    "ResidualTargets",
    "fit_gene_means",
    "build_residual_targets",
]
