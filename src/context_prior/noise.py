"""Statistics of the noise diagnosis: agreement between two readings of the same
lines, the share of a variable's variance a grouping explains, nested patient
fractions and cell-state features. Pure functions; nothing reads labels here."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def column_pearson(left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
    """Per column, Pearson across rows over the rows finite in both; NaN where
    fewer than three such rows or either side is constant on them (tested on the
    range, as the bridge's quality is: a float SD of a constant need not be 0)."""
    if list(left.index) != list(right.index) or list(left.columns) != list(
        right.columns
    ):
        raise ValueError("the two readings must share rows and columns")
    x = left.to_numpy(dtype=np.float64)
    y = right.to_numpy(dtype=np.float64)
    both = np.isfinite(x) & np.isfinite(y)
    count = both.sum(axis=0)
    varies = [
        np.where(both, v, -np.inf).max(axis=0) > np.where(both, v, np.inf).min(axis=0)
        for v in (x, y)
    ]
    x, y = np.where(both, x, 0.0), np.where(both, y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = x.sum(axis=0) / count, y.sum(axis=0) / count
        dx, dy = np.where(both, x - mx, 0.0), np.where(both, y - my, 0.0)
        sxx, syy = (dx**2).sum(axis=0), (dy**2).sum(axis=0)
        rho = (dx * dy).sum(axis=0) / np.sqrt(sxx * syy)
    defined = (count >= 3) & varies[0] & varies[1]
    return pd.Series(np.where(defined, rho, np.nan), index=left.columns)


def spearman_brown(r: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Reliability of the full sample from a split-half correlation."""
    return 2 * r / (1 + r)


def quartiles(values: pd.Series | np.ndarray) -> dict[str, float]:
    """Median, quartiles and count of the defined values."""
    defined = pd.Series(values, dtype=np.float64).dropna()
    if defined.empty:
        return {"median": math.nan, "q25": math.nan, "q75": math.nan, "count": 0}
    return {
        "median": float(defined.median()),
        "q25": float(defined.quantile(0.25)),
        "q75": float(defined.quantile(0.75)),
        "count": int(defined.size),
    }


def nested_fractions(
    lines: Sequence[str],
    patients: Mapping[str, str],
    fractions: Sequence[float],
    *,
    seed: int,
) -> dict[float, list[str]]:
    """Per fraction, the lines of the first ``ceil(f * patients)`` patients in a
    seeded order, in ``lines`` order; larger fractions keep every smaller one's
    lines."""
    groups = sorted({patients[m] for m in lines})
    order = [groups[i] for i in np.random.default_rng(seed).permutation(len(groups))]
    out = {}
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f"fractions must lie in (0, 1], got {fraction}")
        kept = set(order[: math.ceil(fraction * len(groups))])
        out[float(fraction)] = [m for m in lines if patients[m] in kept]
    return out


def _between_share(
    values: np.ndarray, finite: np.ndarray, onehot: np.ndarray
) -> np.ndarray:
    """Per column, the between-group sum of squares over the total, over the
    finite rows (``values`` is zero where not finite)."""
    count = onehot.T @ finite  # groups x columns
    total = onehot.T @ values
    n = finite.sum(axis=0)
    grand = values.sum(axis=0) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        means = total / count
        between = np.nansum(count * (means - grand) ** 2, axis=0)
        overall = ((values - grand) ** 2 * finite).sum(axis=0)
        return np.where(overall > 0, between / overall, np.nan)


def group_share(
    frame: pd.DataFrame,
    groups: pd.Series,
    strata: pd.Series | None,
    *,
    permutations: int,
    seed: int,
) -> dict:
    """η² of ``groups`` on each column of ``frame`` (lines x columns), macro mean
    over columns, against a null that permutes group labels among the lines of
    each stratum (lineage; None: all lines). Missing values are left out per
    column."""
    lines = list(frame.index)
    labels = groups.loc[lines].astype(str).to_numpy()
    levels = sorted(set(labels))
    finite = np.isfinite(frame.to_numpy(dtype=np.float64)).astype(np.float64)
    values = np.where(finite > 0, frame.to_numpy(dtype=np.float64), 0.0)

    def onehot(assigned: np.ndarray) -> np.ndarray:
        return (assigned[:, None] == np.asarray(levels)[None, :]).astype(np.float64)

    observed = float(np.nanmean(_between_share(values, finite, onehot(labels))))
    rng = np.random.default_rng(seed)
    if strata is None:
        blocks = [np.arange(len(lines))]
    else:
        blocks = [
            np.flatnonzero(strata.loc[lines].to_numpy() == stratum)
            for stratum in sorted(set(strata.loc[lines]))
        ]
    null = np.empty(permutations)
    for k in range(permutations):
        shuffled = labels.copy()
        for block in blocks:
            shuffled[block] = labels[rng.permutation(block)]
        null[k] = np.nanmean(_between_share(values, finite, onehot(shuffled)))
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + (null >= observed).sum()) / (1 + permutations)),
        "groups": len(levels),
        "lines": len(lines),
    }


def weighted_quantile(values: Sequence[np.ndarray], quantile: float) -> np.ndarray:
    """Column quantiles of the rows of every array pooled, each array weighted
    equally (each row by 1 / its array's rows)."""
    pooled = np.vstack(values)
    weights = np.concatenate([np.full(len(v), 1.0 / len(v)) for v in values])
    out = np.empty(pooled.shape[1])
    for j in range(pooled.shape[1]):
        order = np.argsort(pooled[:, j], kind="stable")
        cumulative = np.cumsum(weights[order])
        cut = quantile * cumulative[-1]
        out[j] = pooled[order, j][np.searchsorted(cumulative, cut, side="left")]
    return out


def program_features(
    scores: Mapping[str, np.ndarray],
    training: Sequence[str],
    programs: Sequence[str],
    percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Per line, the fraction of its cells above each program's threshold (the
    ``percentile`` of the training lines' pooled scores, each line weighted
    equally) and its mean score; the thresholds."""
    thresholds = weighted_quantile([scores[m] for m in training], percentile / 100)
    lines = list(scores)
    fractions = pd.DataFrame(
        [(scores[m] > thresholds).mean(axis=0) for m in lines],
        index=lines,
        columns=list(programs),
    )
    means = pd.DataFrame(
        [scores[m].mean(axis=0) for m in lines], index=lines, columns=list(programs)
    )
    return fractions, means, thresholds


def cluster_proportions(
    cells: Mapping[str, np.ndarray],
    training: Sequence[str],
    *,
    components: int,
    counts: Sequence[int],
    seed: int,
) -> dict[int, pd.DataFrame]:
    """Per cluster count ``k``, each line's share of cells nearest each centroid.

    Each line's cells are centred on its own mean (line identity removed),
    projected on the leading principal components of the training lines'
    centred cells, and K-means is fitted on the training cells only."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    centred = {m: x - x.mean(axis=0) for m, x in cells.items()}
    pca = PCA(n_components=components, random_state=seed)
    pca.fit(np.vstack([centred[m] for m in training]))
    projected = {m: pca.transform(x) for m, x in centred.items()}
    fitted_on = np.vstack([projected[m] for m in training])
    out = {}
    for k in counts:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(fitted_on)
        out[int(k)] = pd.DataFrame(
            [
                np.bincount(model.predict(projected[m]), minlength=k) / len(cells[m])
                for m in cells
            ],
            index=list(cells),
            columns=[f"cluster_{i}" for i in range(k)],
        )
    return out
