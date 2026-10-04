"""eval / metrics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, Sequence
import pandas as pd
import torch
import logging
import math
from collections.abc import Callable
from typing import Literal
import numpy as np
from scipy.stats import pearsonr, spearmanr


class MacroPerGeneSpearman(NamedTuple):
    """Result of :func:`macro_per_gene_spearman`.

    Attributes:
        macro: ``nanmean`` of ``per_gene`` -- NaN (never 0.0, never raised)
            if every gene is undefined.
        per_gene: Spearman rho per gene (index: ``gene_ids`` or a default
            positional index), NaN where undefined.
        n_scored: Number of genes with a defined correlation.
        n_undefined: Number of genes with an undefined (NaN) correlation.
    """

    macro: float
    per_gene: pd.Series
    n_scored: int
    n_undefined: int


def macro_per_gene_spearman(
    pred: torch.Tensor,
    target: torch.Tensor,
    gene_ids: Sequence[str] | None = None,
) -> MacroPerGeneSpearman:
    """Macro per-gene Spearman across contexts for residual reporting.

    This is a thin torch-tensor adapter over
    :func:`src.eval.metrics.per_gene_spearman`: it reshapes
    ``[n_genes, n_contexts]`` into that module's long-form frame and reuses
    its per-gene grouping, NaN-on-undefined convention (constant truth or
    prediction, or fewer than
    ``src.eval.metrics.MIN_OBSERVATIONS`` finite pairs), and
    ``nanmean`` aggregation, rather than reimplementing them. Joint-training
    checkpoint selection uses GeneEffect Huber loss, not this correlation.

    Args:
        pred: Predictions, shape ``[n_genes, n_contexts]``.
        target: Targets, shape ``[n_genes, n_contexts]``, same shape as
            ``pred``.
        gene_ids: Optional length-``n_genes`` gene labels for the returned
            ``per_gene`` index. Defaults to ``"0", "1", ...`` positional ids.

    Returns:
        A :class:`MacroPerGeneSpearman`.

    Raises:
        ValueError: If ``pred``/``target`` are not both 2-D of equal shape,
            or ``gene_ids`` is given with the wrong length.
    """
    if pred.dim() != 2 or tuple(pred.shape) != tuple(target.shape):
        raise ValueError(
            "pred and target must both be 2-D [n_genes, n_contexts] of equal "
            f"shape; got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    n_genes, n_contexts = pred.shape
    if gene_ids is None:
        gene_ids = [str(i) for i in range(n_genes)]
    elif len(gene_ids) != n_genes:
        raise ValueError(
            f"gene_ids must have length n_genes={n_genes}, got {len(gene_ids)}"
        )

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    frame = pd.DataFrame(
        {
            "gene_symbol": [g for g in gene_ids for _ in range(n_contexts)],
            "context_id": list(range(n_contexts)) * n_genes,
            "truth": target_np.reshape(-1),
            "pred": pred_np.reshape(-1),
        }
    )
    per_gene = per_gene_spearman(frame, truth_col="truth", pred_col="pred")
    finite = per_gene.dropna()
    macro = float(finite.mean()) if len(finite) else float("nan")
    return MacroPerGeneSpearman(
        macro=macro,
        per_gene=per_gene,
        n_scored=int(len(finite)),
        n_undefined=int(per_gene.isna().sum()),
    )


_LOGGER = logging.getLogger(__name__)


MIN_OBSERVATIONS = 3


_ZERO_DELTA_EPS = 1e-12


_Axis = Literal["per_line", "per_gene"]


def _unit_pearson(truth: np.ndarray, pred: np.ndarray) -> float:
    """Pearson on finite pairs, with the same undefined policy as Spearman."""
    finite = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = truth[finite], pred[finite]
    if truth.size < MIN_OBSERVATIONS:
        return float("nan")
    if np.all(truth == truth[0]) or np.all(pred == pred[0]):
        return float("nan")
    return float(pearsonr(truth, pred).statistic)


def _unit_spearman(truth: np.ndarray, pred: np.ndarray) -> float:
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
    """Compute :func:`_unit_spearman` within each group of ``unit_col``.

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
        values[unit] = _unit_spearman(truth, pred)

    series = pd.Series(values, dtype=float, name="spearman")
    series.index.name = unit_col
    return series


def per_line_spearman(
    frame: pd.DataFrame, *, truth_col: str, pred_col: str
) -> pd.Series:
    """Spearman across genes within each line. Index=model_id.

    This is the historical "macro per-line" axis: within one cell line, it
    ranks genes by predicted vs. true GeneEffect. It is dominated by the
    per-gene main effect ``mu_g`` (see module docstring) and is retained
    here only as the complement of :func:`per_gene_spearman`, not as the
    honest generalization metric.

    Args:
        frame: Long-form predictions frame with at least ``model_id``,
            ``truth_col``, and ``pred_col`` columns.
        truth_col: Column holding the true GeneEffect values.
        pred_col: Column holding the predicted GeneEffect values.

    Returns:
        A ``float`` Series indexed by ``model_id``, one entry per line
        present in ``frame``. A line yields NaN if it has fewer than
        :data:`MIN_OBSERVATIONS` usable (finite truth, finite pred) genes,
        or a constant truth or prediction vector.
    """
    return _spearman_by_unit(
        frame, unit_col="model_id", truth_col=truth_col, pred_col=pred_col
    )


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


def _nanmean(values: pd.Series | np.ndarray) -> float:
    """Mean of the finite entries of ``values``, or NaN if none are finite.

    Deliberately not ``numpy.nanmean``: an all-NaN input to ``nanmean``
    still returns NaN but also emits a "Mean of empty slice" RuntimeWarning,
    which this avoids entirely.

    Args:
        values: Per-unit values, some possibly NaN (undefined units).

    Returns:
        The unweighted mean of the finite entries, or NaN if ``values`` is
        empty or entirely NaN.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


@dataclass(frozen=True)
class ResidualScore:
    """Both scoring axes for one predictions frame.

    Attributes:
        macro_per_line: ``nanmean`` of ``per_line`` (historical axis).
        macro_per_gene: ``nanmean`` of ``per_gene`` (R1's honest axis).
        per_line: Per-line Spearman, see :func:`per_line_spearman`.
        per_gene: Per-gene Spearman, see :func:`per_gene_spearman`.
        n_lines: Number of lines with a defined (non-NaN) correlation.
        n_genes: Number of genes with a defined (non-NaN) correlation.
        n_line_undefined: Number of lines that returned NaN.
        n_gene_undefined: Number of genes that returned NaN.
    """

    macro_per_line: float
    macro_per_gene: float
    per_line: pd.Series
    per_gene: pd.Series
    n_lines: int
    n_genes: int
    n_line_undefined: int
    n_gene_undefined: int


def score_predictions(
    frame: pd.DataFrame, *, truth_col: str, pred_col: str
) -> ResidualScore:
    """Score a predictions frame on both the per-line and per-gene axes.

    Args:
        frame: Long-form predictions frame with ``model_id``,
            ``gene_symbol``, ``truth_col``, and ``pred_col`` columns.
        truth_col: Column holding the true GeneEffect values.
        pred_col: Column holding the predicted GeneEffect values.

    Returns:
        A :class:`ResidualScore` combining both axes. Macro aggregates are
        NaN (never raised, never coerced to 0.0) if every unit on that axis
        is undefined.
    """
    per_line = per_line_spearman(frame, truth_col=truth_col, pred_col=pred_col)
    per_gene = per_gene_spearman(frame, truth_col=truth_col, pred_col=pred_col)
    n_line_undefined = int(per_line.isna().sum())
    n_gene_undefined = int(per_gene.isna().sum())
    return ResidualScore(
        macro_per_line=_nanmean(per_line),
        macro_per_gene=_nanmean(per_gene),
        per_line=per_line,
        per_gene=per_gene,
        n_lines=int(per_line.size) - n_line_undefined,
        n_genes=int(per_gene.size) - n_gene_undefined,
        n_line_undefined=n_line_undefined,
        n_gene_undefined=n_gene_undefined,
    )


def _macro_score(
    frame: pd.DataFrame, *, axis: _Axis, truth_col: str, pred_col: str
) -> float:
    """Macro (nanmean) score for one axis of one predictions frame.

    Args:
        frame: Long-form predictions frame.
        axis: ``"per_line"`` or ``"per_gene"``.
        truth_col: Column holding the true GeneEffect values.
        pred_col: Column holding the predicted GeneEffect values.

    Returns:
        The requested macro score.

    Raises:
        ValueError: If ``axis`` is neither ``"per_line"`` nor ``"per_gene"``.
    """
    if axis == "per_line":
        series = per_line_spearman(frame, truth_col=truth_col, pred_col=pred_col)
    elif axis == "per_gene":
        series = per_gene_spearman(frame, truth_col=truth_col, pred_col=pred_col)
    else:
        raise ValueError(f"axis must be 'per_line' or 'per_gene', got {axis!r}.")
    return _nanmean(series)


def _permute_row_index(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of ``frame`` with a permuted row index, values fixed.

    Breaks the row-label -> row-content correspondence while exactly
    preserving the marginal distribution of row content: the same multiset
    of rows is reassigned to (randomly) different labels, rather than
    resampled or perturbed.

    Args:
        frame: A DataFrame indexed by the unit whose correspondence to its
            content should be broken (e.g. ``model_id`` -> context vector).
        rng: Seeded generator driving the permutation.

    Returns:
        A new DataFrame with ``frame``'s rows in their original physical
        order but a permuted index, so ``result.loc[label]`` returns a
        (uniformly randomly chosen) different original row's content.
    """
    permuted = frame.copy()
    permuted.index = frame.index[rng.permutation(len(frame))]
    return permuted


@dataclass(frozen=True)
class ShuffleControl:
    """Result of :func:`shuffled_context_control`.

    Attributes:
        observed_delta: Real (unpermuted-context) macro score minus
            ``baseline_score``.
        shuffled_delta_mean: Mean, over ``n_repeats``, of the shuffled
            macro score minus ``baseline_score``.
        shuffled_delta_std: Sample standard deviation (``ddof=1``) of the
            per-repeat shuffled deltas; NaN if fewer than 2 finite repeats.
        retained_gain_ratio: ``shuffled_delta_mean / observed_delta``; NaN
            if ``observed_delta`` is ~0 (division would be undefined/
            meaningless), never a division by zero.
        n_repeats: Number of shuffle repeats performed.
        seed: Seed used to drive every permutation in this run.
    """

    observed_delta: float
    shuffled_delta_mean: float
    shuffled_delta_std: float
    retained_gain_ratio: float
    n_repeats: int
    seed: int


def shuffled_context_control(
    fit_predict: Callable[[pd.DataFrame], pd.DataFrame],
    context: pd.DataFrame,
    *,
    baseline_score: float,
    axis: _Axis,
    truth_col: str,
    pred_col: str,
    n_repeats: int = 20,
    seed: int = 20260804,
) -> ShuffleControl:
    """Quantify how much apparent context gain survives a broken correspondence.

    Computes the real (unpermuted) macro score via ``fit_predict(context)``,
    then repeatedly permutes ``context``'s row index (see
    :func:`_permute_row_index`: the line -> context-vector correspondence is
    destroyed but the marginal distribution of context vectors is exactly
    preserved), refits via ``fit_predict``, and rescores. If most of the
    apparent gain over ``baseline_score`` survives this destruction
    (``retained_gain_ratio`` close to 1), the gain is not actually coming
    from the true line-context correspondence.

    Args:
        fit_predict: Callable that takes a context frame (indexed by the
            same unit as ``context``, e.g. ``model_id``) and returns a
            long-form predictions frame scoreable by ``truth_col``/
            ``pred_col`` on the requested ``axis``.
        context: Context frame indexed by the unit whose correspondence to
            its content is to be shuffled (e.g. one row per cell line).
        baseline_score: Reference macro score (e.g. a context-blind
            baseline's) that both the observed and shuffled scores are
            compared against.
        axis: ``"per_line"`` or ``"per_gene"``, forwarded to
            :func:`_macro_score`.
        truth_col: Column holding the true GeneEffect values in
            ``fit_predict``'s output.
        pred_col: Column holding the predicted GeneEffect values in
            ``fit_predict``'s output.
        n_repeats: Number of independent context-index permutations to
            average over.
        seed: Seed for the single ``numpy.random.default_rng`` driving
            every permutation in this call; the whole run is reproducible
            from this one seed.

    Returns:
        A :class:`ShuffleControl` summarizing the observed and shuffled
        deltas.
    """
    observed_frame = fit_predict(context)
    observed_score = _macro_score(
        observed_frame, axis=axis, truth_col=truth_col, pred_col=pred_col
    )
    observed_delta = observed_score - baseline_score

    rng = np.random.default_rng(seed)
    shuffled_deltas = np.empty(n_repeats, dtype=float)
    for repeat in range(n_repeats):
        shuffled_context = _permute_row_index(context, rng)
        shuffled_frame = fit_predict(shuffled_context)
        shuffled_score = _macro_score(
            shuffled_frame, axis=axis, truth_col=truth_col, pred_col=pred_col
        )
        shuffled_deltas[repeat] = shuffled_score - baseline_score

    shuffled_delta_mean = _nanmean(shuffled_deltas)
    finite_deltas = shuffled_deltas[np.isfinite(shuffled_deltas)]
    shuffled_delta_std = (
        float(finite_deltas.std(ddof=1)) if finite_deltas.size > 1 else float("nan")
    )

    if not np.isfinite(observed_delta) or abs(observed_delta) < _ZERO_DELTA_EPS:
        retained_gain_ratio = float("nan")
    else:
        retained_gain_ratio = shuffled_delta_mean / observed_delta

    return ShuffleControl(
        observed_delta=float(observed_delta),
        shuffled_delta_mean=float(shuffled_delta_mean),
        shuffled_delta_std=shuffled_delta_std,
        retained_gain_ratio=float(retained_gain_ratio),
        n_repeats=n_repeats,
        seed=seed,
    )


def bootstrap_delta(
    paired: pd.Series,
    *,
    n_resamples: int = 10000,
    seed: int = 20260804,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of per-unit paired differences.

    Args:
        paired: Per-unit ``(candidate - reference)`` differences (e.g. one
            entry per gene). Non-finite entries are dropped before
            resampling (with a warning), rather than raised or coerced.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the ``numpy.random.default_rng`` driving the
            resampling; a fresh generator is created from this seed on
            every call, so repeated calls with the same inputs and seed are
            bit-identical.

    Returns:
        ``(point_estimate, ci_lo, ci_hi)``: the observed mean of the finite
        entries of ``paired``, and the two-sided 95% ([2.5, 97.5]
        percentile) bootstrap CI of the resample means. All three are NaN
        if ``paired`` has no finite entries.
    """
    values = np.asarray(paired, dtype=float)
    finite = values[np.isfinite(values)]
    n_dropped = values.size - finite.size
    if n_dropped:
        _LOGGER.warning(
            "bootstrap_delta: dropping %d/%d non-finite paired difference(s) "
            "before resampling.",
            n_dropped,
            values.size,
        )
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(finite.mean())
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, finite.size, size=(n_resamples, finite.size))
    resample_means = finite[resample_idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(resample_means, [2.5, 97.5])
    return point, float(ci_lo), float(ci_hi)


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
    prediction in the resample, as in :func:`_unit_spearman`.
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


def macro_gene_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Macro mean over genes (rows) of the Spearman across lines (columns).

    The same quantity as ``aggregate_geneeffect``'s ``selective_spearman``:
    non-finite pairs are dropped, and a gene with fewer than
    :data:`MIN_OBSERVATIONS` pairs or a constant side is undefined and left out.
    """
    if truth.ndim != 2 or truth.shape != prediction.shape:
        raise ValueError("truth and prediction must be equal genes x lines arrays")
    rho = _ResampledSpearman(truth, prediction)(np.ones((1, truth.shape[1])))[0]
    defined = rho[np.isfinite(rho)]
    return float(defined.mean()) if defined.size else math.nan
