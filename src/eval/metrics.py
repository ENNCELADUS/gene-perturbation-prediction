"""eval / metrics."""

from __future__ import annotations
from typing import Sequence
import pandas as pd
import logging
import math
import numpy as np
from scipy.stats import pearsonr, spearmanr


_LOGGER = logging.getLogger(__name__)


MIN_OBSERVATIONS = 3


def unit_pearson(truth: np.ndarray, pred: np.ndarray) -> float:
    """Pearson on finite pairs, with the same undefined policy as Spearman."""
    finite = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = truth[finite], pred[finite]
    if truth.size < MIN_OBSERVATIONS:
        return float("nan")
    if np.all(truth == truth[0]) or np.all(pred == pred[0]):
        return float("nan")
    return float(pearsonr(truth, pred).statistic)


def unit_spearman(truth: np.ndarray, pred: np.ndarray) -> float:
    """Spearman rho for one unit's (truth, pred) pairs, or NaN if undefined.

    Rows where either value is NaN are dropped first. The remaining pair is
    undefined (returns NaN without calling ``scipy.stats.spearmanr``) if
    fewer than :data:`MIN_OBSERVATIONS` usable pairs remain, or if the truth
    or prediction vector is constant. Checking constancy ourselves before
    calling ``spearmanr`` avoids SciPy's ``ConstantInputWarning`` path,
    whose NaN output would otherwise be indistinguishable from a genuine
    "too few observations" NaN.

    Args:
        truth: Raw (possibly NaN-containing) truth values for one unit.
        pred: Raw (possibly NaN-containing) prediction values for one unit,
            aligned with ``truth``.

    Returns:
        Spearman rank correlation (average-rank ties, as
        ``scipy.stats.spearmanr``), or NaN if the unit is undefined.
    """
    finite = np.isfinite(truth) & np.isfinite(pred)
    truth = truth[finite]
    pred = pred[finite]
    if truth.size < MIN_OBSERVATIONS:
        return float("nan")
    if np.all(truth == truth[0]) or np.all(pred == pred[0]):
        return float("nan")
    return float(spearmanr(truth, pred).statistic)


def _spearman_by_unit(
    frame: pd.DataFrame, *, unit_col: str, truth_col: str, pred_col: str
) -> pd.Series:
    """Compute :func:`unit_spearman` within each group of ``unit_col``.

    Args:
        frame: Long-form predictions frame.
        unit_col: Column to group by (``model_id`` or ``gene_symbol``).
        truth_col: Column holding the true values.
        pred_col: Column holding the predicted values.

    Returns:
        A ``float`` Series indexed by ``unit_col``, with one entry per
        distinct value of ``unit_col`` present in ``frame`` (NaN where
        undefined; never dropped from the index).

    Raises:
        KeyError: If ``unit_col``, ``truth_col``, or ``pred_col`` is not a
            column of ``frame``.
    """
    required = {unit_col, truth_col, pred_col}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"frame is missing required column(s): {sorted(missing)}")

    values: dict[object, float] = {}
    for unit, group in frame.groupby(unit_col, sort=False):
        truth = group[truth_col].to_numpy(dtype=float)
        pred = group[pred_col].to_numpy(dtype=float)
        values[unit] = unit_spearman(truth, pred)

    series = pd.Series(values, dtype=float, name="spearman")
    series.index.name = unit_col
    return series


def per_gene_spearman(
    frame: pd.DataFrame, *, truth_col: str, pred_col: str
) -> pd.Series:
    """Spearman across lines within each gene. Index=gene_symbol.

    This is the honest complementary axis introduced by R1: for one gene,
    it correlates predicted vs. true GeneEffect across cell lines, which is
    invariant to subtracting any per-gene constant (Fact 1 in the module
    docstring) and undefined for a context-blind (per-gene-constant)
    predictor (Fact 2).

    Args:
        frame: Long-form predictions frame with at least ``gene_symbol``,
            ``truth_col``, and ``pred_col`` columns.
        truth_col: Column holding the true GeneEffect values.
        pred_col: Column holding the predicted GeneEffect values.

    Returns:
        A ``float`` Series indexed by ``gene_symbol``, one entry per gene
        present in ``frame``. A gene yields NaN if it has fewer than
        :data:`MIN_OBSERVATIONS` usable (finite truth, finite pred) lines,
        or a constant truth or prediction vector across lines -- notably
        including a context-blind predictor, whose prediction is constant
        across lines by construction.
    """
    return _spearman_by_unit(
        frame, unit_col="gene_symbol", truth_col=truth_col, pred_col=pred_col
    )


# Elements of a bootstrap chunk (genes x lines x repeats); bounds peak memory.
_BOOTSTRAP_CHUNK_ELEMENTS = 8_000_000

# A rank variance below this is a constant column (the smallest real one is O(1)).
_RANK_VARIANCE_FLOOR = 1e-9


def _gene_line_arrays(
    frame: pd.DataFrame, genes: Sequence[str], lines: pd.Index
) -> tuple[np.ndarray, np.ndarray]:
    """Gene x line arrays of residual and residual prediction (NaN if absent)."""
    gene_position = pd.Index(genes).get_indexer(frame["gene_symbol"])
    line_position = lines.get_indexer(frame["model_id"])
    keep = (gene_position >= 0) & (line_position >= 0)
    gene_position, line_position = gene_position[keep], line_position[keep]
    if len(set(zip(gene_position.tolist(), line_position.tolist()))) != len(
        gene_position
    ):
        raise ValueError("duplicate GeneEffect rows")
    arrays = []
    for column in ("residual", "residual_prediction"):
        values = np.full((len(genes), len(lines)), np.nan)
        values[gene_position, line_position] = frame[column].to_numpy(dtype=float)[keep]
        arrays.append(values)
    return arrays[0], arrays[1]


class _ResampledSpearman:
    """Per-gene Spearman of one gene x line pair of arrays under line resamples.

    A resample is a vector of line multiplicities ``c`` (a line drawn twice
    counts twice), which gives the exact Spearman of the expanded sample without
    materialising it: the average rank of line ``k`` is
    ``sum_j c_j * ([x_j < x_k] + [x_j == x_k] / 2) + 1/2`` over valid lines, so
    all resamples' ranks are one matrix product with a comparison tensor built
    once, and the correlation is the ``c``-weighted Pearson of those ranks.
    Entries where either value is not finite are dropped; a gene is NaN with
    fewer than :data:`MIN_OBSERVATIONS` usable pairs or a constant truth or
    prediction in the resample, as in :func:`unit_spearman`.
    """

    def __init__(self, truth: np.ndarray, pred: np.ndarray) -> None:
        self.valid = np.isfinite(truth) & np.isfinite(pred)
        self.shape = truth.shape
        self.comparison = [self._comparison(x) for x in (truth, pred)]

    def _comparison(self, values: np.ndarray) -> np.ndarray:
        """``[genes * lines, lines]``: valid ``[x_j < x_k] + [x_j == x_k] / 2``."""
        less = (values[:, None, :] < values[:, :, None]).astype(float)
        less += 0.5 * (values[:, None, :] == values[:, :, None])
        less *= self.valid[:, None, :]
        return less.reshape(-1, values.shape[1])

    def __call__(self, counts: np.ndarray) -> np.ndarray:
        """Spearman per gene for each row of ``counts`` ``[repeats, lines]``."""
        genes, lines = self.shape
        weight = self.valid[:, :, None] * counts.T[None, :, :]
        total = weight.sum(axis=1)
        moments = []
        for comparison in self.comparison:
            rank = (comparison @ counts.T).reshape(genes, lines, -1) + 0.5
            mean = (weight * rank).sum(axis=1) / np.maximum(total, 1)
            moments.append(rank - mean[:, None, :])
        truth, pred = moments
        variance_truth = (weight * truth * truth).sum(axis=1)
        variance_pred = (weight * pred * pred).sum(axis=1)
        covariance = (weight * truth * pred).sum(axis=1)
        undefined = (
            (total < MIN_OBSERVATIONS)
            | (variance_truth < _RANK_VARIANCE_FLOOR)
            | (variance_pred < _RANK_VARIANCE_FLOOR)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = covariance / np.sqrt(variance_truth * variance_pred)
        return np.where(undefined, np.nan, rho).T


def paired_line_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    selective_genes: Sequence[str],
    *,
    repeats: int,
    seed: int,
) -> dict[str, float | list[float]]:
    """Paired cell-line bootstrap of the macro selective-gene Spearman difference.

    ``left`` and ``right`` hold ``model_id``, ``gene_symbol``, ``residual`` and
    ``residual_prediction`` over the same (line, gene) keys. Lines are resampled
    with replacement, the same draw for both frames, and each frame's macro
    Spearman over ``selective_genes`` is recomputed. A gene whose resampled truth
    or prediction is constant is undefined and left out of that resample's macro
    mean; a resample with no defined gene is left out of the interval.

    Returns:
        ``{"difference": left - right observed, "interval": [2.5%, 97.5%]}``; NaN
        where no resample (or no observed gene) is defined.
    """
    keys = [
        set(zip(frame["model_id"], frame["gene_symbol"])) for frame in (left, right)
    ]
    if keys[0] != keys[1]:
        raise ValueError("left and right must hold the same (line, gene) keys")
    genes = sorted(selective_genes)
    lines = pd.Index(sorted(set(left["model_id"])))
    n_lines = len(lines)
    scorers = [
        _ResampledSpearman(*_gene_line_arrays(frame, genes, lines))
        for frame in (left, right)
    ]

    def difference(counts: np.ndarray) -> np.ndarray:
        """Left minus right macro Spearman for each row of ``counts``."""
        macros = []
        for scorer in scorers:
            rho = scorer(counts)
            defined = np.isfinite(rho)
            total = np.where(defined, rho, 0.0).sum(axis=1)
            macros.append(
                np.where(
                    defined.any(axis=1),
                    total / np.maximum(defined.sum(axis=1), 1),
                    np.nan,
                )
            )
        return macros[0] - macros[1]

    if n_lines == 0:
        return {"difference": math.nan, "interval": [math.nan, math.nan]}
    observed = float(difference(np.ones((1, n_lines)))[0])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_lines, size=(repeats, n_lines))
    step = max(1, _BOOTSTRAP_CHUNK_ELEMENTS // max(1, len(genes) * n_lines))
    resampled = [
        difference(
            (draws[start : start + step, :, None] == np.arange(n_lines)).sum(axis=1)
        )
        for start in range(0, repeats, step)
    ]
    resampled = np.concatenate(resampled) if resampled else np.empty(0)
    resampled = resampled[np.isfinite(resampled)]
    low, high = (
        np.percentile(resampled, [2.5, 97.5])
        if resampled.size
        else (math.nan, math.nan)
    )
    return {"difference": observed, "interval": [float(low), float(high)]}


def calibration_slope(predicted: np.ndarray, measured: np.ndarray) -> float:
    """OLS slope of the measured residual on the predicted residual, pooled over
    the columns (genes) of two lines x genes arrays, each gene's own offset removed.

    Each column is centred on its own mean over the pairs where both values are
    finite (a gene fixed effect), so the slope reads how the line-to-line spread of
    the prediction scales against the truth's: 1 is calibrated, above 1 shrunk.
    Undefined (NaN) when the prediction is constant within every gene, as a
    context-blind prediction is.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if predicted.shape != measured.shape or predicted.ndim != 2:
        raise ValueError("predicted and measured must be lines x genes of one shape")
    valid = np.isfinite(predicted) & np.isfinite(measured)
    count = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        x_mean = np.where(valid, predicted, 0.0).sum(axis=0) / count
        y_mean = np.where(valid, measured, 0.0).sum(axis=0) / count
    x = np.where(valid, predicted - x_mean, 0.0)
    y = np.where(valid, measured - y_mean, 0.0)
    spread = float((x * x).sum())
    if not spread > _RANK_VARIANCE_FLOOR:
        return math.nan
    return float((x * y).sum() / spread)
