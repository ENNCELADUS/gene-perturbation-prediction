"""R1 residual baseline ladder for DepMap GeneEffect.

GeneEffect(g, c) = mu_g + delta(g, c) with Var(mu_g) >> Var(delta). This
ladder scores every method on a *residual* (delta-only) prediction, never
re-adding a mu term to the emitted prediction, and reports both the
historical per-line axis and the honest per-gene across-line axis:
    gene_mean          -> delta_hat identically 0.0
    copy_prior          -> delta_hat = prior_g - mu_bar_g (fold-independent)
    nearest_line         -> delta_hat = y_neighbor - mu_bar_g (fold-independent)
    context_pca_ridge -> ridge(residual ~ context PCs), already a residual

## The leave-one-out centering artifact (why mu_bar, not mu_hat_g^(-c))

Two per-gene means appear here. ``mu_hat_g^(-c)`` is *fold-fit* (only the
lines currently in train, excluding evaluated line c) and builds the
**target**: ``delta(g,c) = y_gc - mu_hat_g^(-c)``. ``mu_bar_g`` is
**fold-independent** (LOLO: every line; fixed-split: train only -- either
way the same number for every evaluated line) and centers **predictions**
(``copy_prior``, ``nearest_line``). They must not be swapped.

For a fixed gene g, ``mu_hat_g^(-c) = (S_g - y_gc) / (n-1)`` (``S_g`` the
fixed total over all n lines) is an *exact affine function of the held-out
label itself*. Two algebraic (not approximate) consequences:

1. Safe for the target: ``y_gc - mu_hat_g^(-c) == (n/(n-1)) * (y_gc -
   mean_g)`` -- a positive scalar multiple of the ordinary fixed-centered
   residual, so it never changes any Spearman ranking. Fold-fit centering
   is fine for the truth.
2. Unsafe for a prediction: ``constant_g - mu_hat_g^(-c)`` (constant_g not
   depending on c -- an external prior, or a neighbor's own value) has its
   entire c-dependence collapse to ``+y_gc / (n-1)``: an exact monotonic
   function of the very label being predicted. Its per-gene Spearman
   (across the n evaluated lines, for one gene) is +1.0 *by construction*,
   for any n, any data, any constant_g -- a mechanical leakage artifact,
   not a result. Verified in ``tests/test_run_r1_residual_ladder.py``.

A fold-independent ``mu_bar_g`` removes the artifact: being the same
number for every evaluated line, it contributes zero variance along the
axis the correlation is taken over, so it cannot manufacture a spurious
relationship with the held-out label. ``context_pca_ridge`` never had this
problem: its prediction is ``Z_test @ W`` with no mu term at all.

## Two outer protocols

- ``outer="fixed"`` (default): a fixed train/val/test partition
  (:class:`FixedSplit`). mu_hat and mu_bar are each fit *once*, on
  ``split.train`` only -- disjoint from every evaluated line by
  construction, so the whole artifact family above cannot arise: there is
  no fold-to-fold variation for a fold-independent quantity to leak
  through. Standard protocol, the primary path.
- ``outer="lolo"``: strict leave-one-line-out over every line in
  ``labels`` (the 29-line diagnostic cohort); mu_hat refits per line, but
  mu_bar is fit once over all n lines (fold-independent by definition).

This module is pure numpy/pandas/sklearn: no torch, no file I/O, no
hashing/provenance machinery.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from aivc_model.residual_metrics import (
    ResidualScore,
    bootstrap_delta,
    score_predictions,
)
from aivc_model.residual_target import ResidualTargets, fit_gene_means

_LOGGER = logging.getLogger(__name__)

GENE_MEAN: str = "gene_mean"
COPY_PRIOR: str = "copy_prior"
NEAREST_LINE: str = "nearest_line"
CONTEXT_PCA_RIDGE: str = "context_pca_ridge"

LOLO_SLICE: str = "lolo"
VAL_SLICE: str = "val"
TEST_SLICE: str = "test"

_MIN_TRAIN_LINES_FOR_CONTEXT: int = 2
_FEATURE_CONSTANT_EPS: float = 1e-12
_AXIS_INVARIANCE_TOL: float = 1e-6


@dataclass(frozen=True)
class FixedSplit:
    """A fixed train/val/test line partition for ``outer="fixed"``.

    Attributes:
        train: Lines mu_hat/mu_bar and every context method are fit on.
        val: Lines evaluated as the "val" slice (may be empty).
        test: Lines evaluated as the "test" slice (may be empty).
        unlabeled_train: Declared train members without GeneEffect supervision.
    """

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    unlabeled_train: tuple[str, ...] = ()


@dataclass(frozen=True)
class _LadderConfig:
    """Ladder-wide tuning knobs, bundled to shorten call sites below."""

    pca_components: int
    ridge_alpha: float
    min_lines: int


@dataclass(frozen=True)
class R1Result:
    """Artifacts produced by :func:`run_r1_ladder`.

    Attributes:
        predictions: slice, model_id, gene_symbol, method, gene_effect,
            residual, residual_prediction -- one row per (slice, line,
            gene, method).
        per_line: slice, method, model_id, spearman.
        per_gene: slice, method, gene_symbol, spearman.
        summary: Plain-JSON-serializable summary dict.
    """

    predictions: pd.DataFrame
    per_line: pd.DataFrame
    per_gene: pd.DataFrame
    summary: dict[str, object]


def _view_method_name(base: str, view_name: str) -> str:
    """Per-context-view method name, e.g. ``nearest_line[tx1]``."""
    return f"{base}[{view_name}]"


def _row(
    slice_name: str, model_id: str, method: str, held: pd.DataFrame, gene: str, pred
) -> dict:
    """One predictions.csv row. ``held`` is this line's rt.long, indexed by gene."""
    return {
        "slice": slice_name,
        "model_id": model_id,
        "gene_symbol": gene,
        "method": method,
        "gene_effect": float(held.loc[gene, "gene_effect"]),
        "residual": float(held.loc[gene, "residual"]),
        "residual_prediction": float(pred),
    }


def _build_fold_targets(
    labels: pd.DataFrame, train_lines: Sequence[str], min_lines: int
) -> ResidualTargets:
    """Fit mu_hat_g on ``train_lines`` only; attach it to every label row.

    This is the TARGET-centering mean -- safe to fit per-fold/per-slice
    (see module docstring, fact 1). Mirrors
    ``residual_target.build_residual_targets``, but forwards ``min_lines``
    to :func:`fit_gene_means` -- ``build_residual_targets`` always uses
    that function's own default (3) with no override.
    """
    gene_mean = fit_gene_means(labels, train_lines, min_lines=min_lines)
    n_before = len(labels)
    long = labels.merge(
        gene_mean.rename("gene_mean"),
        left_on="gene_symbol",
        right_index=True,
        how="inner",
    )
    n_dropped = n_before - len(long)
    if n_dropped:
        _LOGGER.warning(
            "dropped %d label row(s) whose gene did not survive min_lines=%d",
            n_dropped,
            min_lines,
        )
    missing_labels = int(long["gene_effect"].isna().sum())
    if missing_labels:
        _LOGGER.warning("dropped %d row(s) without GeneEffect labels", missing_labels)
        long = long.loc[long["gene_effect"].notna()]
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


def _gene_mean_rows(
    slice_name: str, rt: ResidualTargets, held_out_id: str
) -> list[dict]:
    """gene_mean rows: delta_hat identically 0.0 for every gene."""
    held = rt.long.loc[rt.long["model_id"] == held_out_id].set_index("gene_symbol")
    return [_row(slice_name, held_out_id, GENE_MEAN, held, g, 0.0) for g in held.index]


def _copy_prior_rows(
    slice_name: str,
    rt: ResidualTargets,
    held_out_id: str,
    prior: pd.Series,
    mu_bar: pd.Series,
) -> list[dict]:
    """copy_prior rows: delta_hat = prior_g - mu_bar_g (fold-independent)."""
    held = rt.long.loc[rt.long["model_id"] == held_out_id].set_index("gene_symbol")
    genes = held.index.intersection(prior.index).intersection(mu_bar.index)
    if len(genes) == 0:
        _LOGGER.warning("copy_prior: no overlapping genes for line %s", held_out_id)
        return []
    delta = prior.loc[genes] - mu_bar.loc[genes]
    return [
        _row(slice_name, held_out_id, COPY_PRIOR, held, g, delta.loc[g]) for g in genes
    ]


def _prepare_fold_context(
    view: pd.DataFrame, fold_train_ids: list[str], held_out_id: str
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Drop within-fold-constant features, then standardize on train stats.

    Returns (train_scaled, held_scaled, feature_columns_kept), or None if
    every feature has zero spread across fold_train_ids.
    """
    feature_columns = list(view.columns)
    train_raw = view.loc[fold_train_ids, feature_columns].to_numpy(dtype=float)
    spread = train_raw.max(axis=0) - train_raw.min(axis=0)
    keep = spread > _FEATURE_CONSTANT_EPS
    n_dropped = int((~keep).sum())
    if n_dropped:
        _LOGGER.info(
            "line %s: dropped %d fold-constant context feature(s)",
            held_out_id,
            n_dropped,
        )
    if not keep.any():
        _LOGGER.warning(
            "line %s: no non-constant context features; skipping context methods",
            held_out_id,
        )
        return None
    feature_columns = [c for c, k in zip(feature_columns, keep, strict=True) if k]
    train_raw = train_raw[:, keep]
    held_raw = view.loc[[held_out_id], feature_columns].to_numpy(dtype=float)

    scaler = StandardScaler().fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    held_scaled = scaler.transform(held_raw)
    if not np.isfinite(train_scaled).all() or not np.isfinite(held_scaled).all():
        raise ValueError(f"non-finite standardized context for line {held_out_id}")
    return train_scaled, held_scaled, feature_columns


def _prepare_fixed_context(
    view: pd.DataFrame,
    train_ids: list[str],
    eval_ids: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit one train-only scaler and transform every fixed-split eval line."""
    feature_columns = list(view.columns)
    train_raw = view.loc[train_ids, feature_columns].to_numpy(dtype=float)
    spread = train_raw.max(axis=0) - train_raw.min(axis=0)
    keep = spread > _FEATURE_CONSTANT_EPS
    if not keep.any():
        _LOGGER.warning("fixed split: no non-constant context features")
        return None
    train_raw = train_raw[:, keep]
    eval_raw = view.loc[eval_ids, feature_columns].to_numpy(dtype=float)[:, keep]
    scaler = StandardScaler().fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    eval_scaled = scaler.transform(eval_raw)
    if not np.isfinite(train_scaled).all() or not np.isfinite(eval_scaled).all():
        raise ValueError("non-finite standardized context in fixed split")
    return train_scaled, eval_scaled


def _run_fixed_evals(
    labels: pd.DataFrame,
    train_lines: Sequence[str],
    eval_specs: Sequence[tuple[str, str]],
    mu_bar: pd.Series,
    context_views: Mapping[str, pd.DataFrame],
    copy_prior: pd.Series | None,
    config: _LadderConfig,
    effective_components: dict[str, dict[str, set[int]]],
) -> list[dict]:
    """Fit each fixed-split baseline once and batch-predict val plus test."""
    rt = _build_fold_targets(labels, train_lines, config.min_lines)
    train_ids = list(train_lines)
    eval_ids = [held_out_id for _slice_name, held_out_id in eval_specs]
    held_by_id = {
        model_id: frame.set_index("gene_symbol")
        for model_id, frame in rt.long.loc[
            rt.long["model_id"].isin(eval_ids)
        ].groupby("model_id", sort=False)
    }
    missing_labels = sorted(set(eval_ids) - set(held_by_id))
    if missing_labels:
        raise ValueError(f"fixed split has no surviving labels for {missing_labels}")

    rows: list[dict] = []
    for slice_name, held_out_id in eval_specs:
        rows.extend(_gene_mean_rows(slice_name, rt, held_out_id))
        if copy_prior is not None:
            rows.extend(
                _copy_prior_rows(slice_name, rt, held_out_id, copy_prior, mu_bar)
            )

    train_label_rows = rt.long.loc[rt.long["model_id"].isin(train_ids)]
    train_by_gene = {
        gene: frame.set_index("model_id")[["gene_effect", "residual"]]
        for gene, frame in train_label_rows.groupby("gene_symbol", sort=False)
    }
    eval_row_by_id = {model_id: index for index, model_id in enumerate(eval_ids)}

    for view_name, view in context_views.items():
        view_ids = set(view.index)
        missing_eval = sorted(set(eval_ids) - view_ids)
        if missing_eval:
            raise ValueError(
                f"context view {view_name!r} lacks fixed-split eval rows: "
                f"{missing_eval}"
            )
        context_train_ids = [model_id for model_id in train_ids if model_id in view_ids]
        if len(context_train_ids) < _MIN_TRAIN_LINES_FOR_CONTEXT:
            raise ValueError(
                f"context view {view_name!r} has too few fixed-split train rows"
            )
        prepared = _prepare_fixed_context(view, context_train_ids, eval_ids)
        if prepared is None:
            continue
        train_scaled, eval_scaled = prepared
        context_row_by_id = {
            model_id: index for index, model_id in enumerate(context_train_ids)
        }

        distances = np.linalg.norm(
            eval_scaled[:, np.newaxis, :] - train_scaled[np.newaxis, :, :], axis=2
        )
        donor_orders = np.argsort(distances, axis=1, kind="stable")
        nearest_method = _view_method_name(NEAREST_LINE, view_name)
        for slice_name, held_out_id in eval_specs:
            held = held_by_id[held_out_id]
            donor_order = donor_orders[eval_row_by_id[held_out_id]]
            for gene in held.index:
                gene_train = train_by_gene[gene]
                donor_id = next(
                    (
                        context_train_ids[index]
                        for index in donor_order
                        if context_train_ids[index] in gene_train.index
                    ),
                    None,
                )
                if donor_id is None:
                    raise ValueError(
                        f"nearest_line[{view_name}]: no labeled train donor for {gene}"
                    )
                prediction = float(gene_train.loc[donor_id, "gene_effect"]) - float(
                    mu_bar.loc[gene]
                )
                rows.append(
                    _row(
                        slice_name,
                        held_out_id,
                        nearest_method,
                        held,
                        gene,
                        prediction,
                    )
                )

        n_train, n_features = train_scaled.shape
        n_components = min(config.pca_components, n_train - 1, n_features)
        if n_components < 1:
            continue
        pca = PCA(n_components=n_components, svd_solver="full").fit(train_scaled)
        train_pcs = pca.transform(train_scaled)
        eval_pcs = pca.transform(eval_scaled)
        ridge_method = _view_method_name(CONTEXT_PCA_RIDGE, view_name)
        eval_genes = set().union(*(set(held.index) for held in held_by_id.values()))
        for gene in rt.gene_mean.index:
            if gene not in eval_genes:
                continue
            gene_rows = train_by_gene[gene]
            gene_rows = gene_rows.loc[gene_rows.index.isin(context_row_by_id)]
            if len(gene_rows) < _MIN_TRAIN_LINES_FOR_CONTEXT:
                raise ValueError(
                    f"context_pca_ridge[{view_name}]: gene {gene} has fewer than "
                    f"{_MIN_TRAIN_LINES_FOR_CONTEXT} labeled context-train lines"
                )
            indices = [
                context_row_by_id[model_id] for model_id in gene_rows.index
            ]
            targets = gene_rows["residual"].to_numpy(dtype=float)
            ridge = Ridge(alpha=config.ridge_alpha).fit(train_pcs[indices], targets)
            gene_predictions = np.asarray(ridge.predict(eval_pcs), dtype=float)
            if not np.isfinite(gene_predictions).all():
                raise ValueError(
                    f"invalid fixed-split ridge prediction: {view_name}/{gene}"
                )
            for slice_name, held_out_id in eval_specs:
                held = held_by_id[held_out_id]
                if gene in held.index:
                    rows.append(
                        _row(
                            slice_name,
                            held_out_id,
                            ridge_method,
                            held,
                            gene,
                            gene_predictions[eval_row_by_id[held_out_id]],
                        )
                    )
        for slice_name, _held_out_id in eval_specs:
            effective_components.setdefault(slice_name, {}).setdefault(
                view_name, set()
            ).add(n_components)
    return rows


def _nearest_line_rows(
    slice_name: str,
    rt: ResidualTargets,
    view_name: str,
    held_out_id: str,
    fold_train_ids: list[str],
    train_scaled: np.ndarray,
    held_scaled: np.ndarray,
    mu_bar: pd.Series,
) -> list[dict]:
    """Use the nearest context among the labeled donors for each held gene."""
    distances = np.linalg.norm(train_scaled - held_scaled, axis=1)
    held = rt.long.loc[rt.long["model_id"] == held_out_id].set_index("gene_symbol")
    train_labels = rt.long.loc[rt.long["model_id"].isin(fold_train_ids)].set_index(
        ["model_id", "gene_symbol"]
    )
    method = _view_method_name(NEAREST_LINE, view_name)
    rows: list[dict] = []
    for gene in held.index:
        if gene not in mu_bar.index:
            raise ValueError(f"nearest_line[{view_name}]: missing mu_bar for {gene}")
        donor_indices = [
            index
            for index, model_id in enumerate(fold_train_ids)
            if (model_id, gene) in train_labels.index
        ]
        if not donor_indices:
            raise ValueError(
                f"nearest_line[{view_name}]: no labeled train donor for {gene}"
            )
        nearest_index = min(donor_indices, key=lambda index: distances[index])
        nearest_id = fold_train_ids[nearest_index]
        neighbor_effect = float(train_labels.loc[(nearest_id, gene), "gene_effect"])
        rows.append(
            _row(
                slice_name,
                held_out_id,
                method,
                held,
                gene,
                neighbor_effect - float(mu_bar.loc[gene]),
            )
        )
    return rows


def _context_pca_ridge_rows(
    slice_name: str,
    rt: ResidualTargets,
    view_name: str,
    held_out_id: str,
    fold_train_ids: list[str],
    train_scaled: np.ndarray,
    held_scaled: np.ndarray,
    config: _LadderConfig,
) -> tuple[list[dict], int] | None:
    """context_pca_ridge rows: ridge(residual ~ context PCs) -- no mu term.

    Returns (rows, effective_pca_components), or None if the fold's rank
    cannot support even one component.
    """
    n_train, n_features = train_scaled.shape
    n_components = min(config.pca_components, n_train - 1, n_features)
    if n_components < 1:
        _LOGGER.warning(
            "context_pca_ridge[%s]: no usable PCA rank for line %s; skipping",
            view_name,
            held_out_id,
        )
        return None

    pca = PCA(n_components=n_components, svd_solver="full").fit(train_scaled)
    train_pcs = pca.transform(train_scaled)
    held_pcs = pca.transform(held_scaled)
    if not np.isfinite(train_pcs).all() or not np.isfinite(held_pcs).all():
        raise ValueError(f"context PCA is degenerate for line {held_out_id}")

    held = rt.long.loc[rt.long["model_id"] == held_out_id].set_index("gene_symbol")
    gene_order = [gene for gene in rt.gene_mean.index if gene in held.index]
    fold_train_rows = rt.long.loc[rt.long["model_id"].isin(fold_train_ids)]
    method = _view_method_name(CONTEXT_PCA_RIDGE, view_name)
    row_by_model_id = {model_id: index for index, model_id in enumerate(fold_train_ids)}
    rows: list[dict] = []
    for gene in gene_order:
        gene_rows = fold_train_rows.loc[
            fold_train_rows["gene_symbol"] == gene,
            ["model_id", "residual"],
        ]
        gene_rows = gene_rows.loc[gene_rows["model_id"].isin(row_by_model_id)]
        if len(gene_rows) < _MIN_TRAIN_LINES_FOR_CONTEXT:
            raise ValueError(
                f"context_pca_ridge[{view_name}]: gene {gene} has fewer than "
                f"{_MIN_TRAIN_LINES_FOR_CONTEXT} labeled context-train lines"
            )
        indices = [row_by_model_id[model_id] for model_id in gene_rows["model_id"]]
        targets = gene_rows["residual"].to_numpy(dtype=float)
        ridge = Ridge(alpha=config.ridge_alpha).fit(train_pcs[indices], targets)
        prediction = float(np.asarray(ridge.predict(held_pcs)).reshape(-1)[0])
        if not np.isfinite(prediction):
            raise ValueError(
                f"invalid ridge prediction: {held_out_id}/{view_name}/{gene}"
            )
        rows.append(_row(slice_name, held_out_id, method, held, gene, prediction))
    return rows, n_components


def _check_axis_invariance(
    residual_score: ResidualScore, raw_score: ResidualScore, method: str
) -> None:
    """Assert the per-gene axis agrees whether scored on residual or raw truth.

    Per-gene Spearman is invariant to subtracting any per-gene constant
    from the truth column, so scoring against ``residual`` and against raw
    ``gene_effect`` must agree on the per-gene axis. Free correctness check
    on the whole fold-refit-and-score pipeline: divergence means the
    residual construction leaked.
    """
    residual_series = residual_score.per_gene.sort_index()
    raw_series = raw_score.per_gene.sort_index()
    if not residual_series.index.equals(raw_series.index):
        raise ValueError(f"{method}: per-gene axis index mismatch")
    residual_nan = residual_series.isna()
    raw_nan = raw_series.isna()
    if (residual_nan != raw_nan).any():
        bad = residual_series.index[residual_nan != raw_nan].tolist()
        raise ValueError(f"{method}: per-gene axis NaN pattern differs for {bad[:10]}")
    finite = ~residual_nan
    if finite.any() and not np.allclose(
        residual_series[finite].to_numpy(),
        raw_series[finite].to_numpy(),
        atol=_AXIS_INVARIANCE_TOL,
    ):
        raise ValueError(
            f"{method}: per-gene axis is not invariant to truth-column choice "
            "(residual vs. raw gene_effect) -- the residual construction leaked"
        )


def _combined_score(frame: pd.DataFrame, method: str) -> ResidualScore:
    """Score one method: per-line vs. delta (the coherent pairing), per-gene vs. raw.

    Per-line truth is the fold-fit ``residual`` column (delta), paired with
    the delta-only prediction -- the mu-free quantity on both sides. Truth
    on raw ``gene_effect`` would put mu_g back into only one side of the
    per-line comparison and measure accidental alignment with the gene
    main effect instead. Per-gene truth is raw ``gene_effect``, which
    :func:`_check_axis_invariance` proves must equal scoring against
    ``residual`` (the axis is invariant to a per-gene truth shift).
    """
    residual_score = score_predictions(
        frame, truth_col="residual", pred_col="residual_prediction"
    )
    raw_score = score_predictions(
        frame, truth_col="gene_effect", pred_col="residual_prediction"
    )
    _check_axis_invariance(residual_score, raw_score, method)
    return ResidualScore(
        macro_per_line=residual_score.macro_per_line,
        macro_per_gene=raw_score.macro_per_gene,
        per_line=residual_score.per_line,
        per_gene=raw_score.per_gene,
        n_lines=residual_score.n_lines,
        n_genes=raw_score.n_genes,
        n_line_undefined=residual_score.n_line_undefined,
        n_gene_undefined=raw_score.n_gene_undefined,
    )


def _run_eval(
    labels: pd.DataFrame,
    train_lines: Sequence[str],
    held_out_id: str,
    slice_name: str,
    mu_bar: pd.Series,
    context_views: Mapping[str, pd.DataFrame],
    copy_prior: pd.Series | None,
    config: _LadderConfig,
    effective_components: dict[str, set[int]],
) -> list[dict]:
    """Run every method for one evaluated line against a given train set.

    Never lets ``held_out_id`` enter any fit: mu_hat_g is fit on
    ``train_lines`` only (caller guarantees this excludes ``held_out_id``),
    and every context method is fit on ``fold_train_ids`` (train lines
    present in that view), which also excludes ``held_out_id`` by
    construction. ``mu_bar`` must be fold-independent -- see the module
    docstring -- the caller supplies it; this function only consumes it.
    """
    if held_out_id in train_lines:
        raise ValueError("internal error: held-out line in its own train set")

    rt = _build_fold_targets(labels, train_lines, config.min_lines)
    if held_out_id in rt.train_lines:
        raise ValueError(f"internal error: line {held_out_id} leaked into train_lines")
    if held_out_id not in set(rt.long["model_id"]):
        _LOGGER.warning("line %s: no surviving genes this fold; skipping", held_out_id)
        return []

    rows = _gene_mean_rows(slice_name, rt, held_out_id)
    if copy_prior is not None:
        rows.extend(_copy_prior_rows(slice_name, rt, held_out_id, copy_prior, mu_bar))

    for view_name, view in context_views.items():
        view_ids = set(view.index)
        if held_out_id not in view_ids:
            _LOGGER.warning(
                "context view %r has no row for line %s; skipping this fold",
                view_name,
                held_out_id,
            )
            continue
        fold_train_ids = [line for line in train_lines if line in view_ids]
        if len(fold_train_ids) < _MIN_TRAIN_LINES_FOR_CONTEXT:
            _LOGGER.warning(
                "context view %r: too few train lines for %s; skipping",
                view_name,
                held_out_id,
            )
            continue

        prepared = _prepare_fold_context(view, fold_train_ids, held_out_id)
        if prepared is None:
            continue
        train_scaled, held_scaled, _features = prepared

        rows.extend(
            _nearest_line_rows(
                slice_name,
                rt,
                view_name,
                held_out_id,
                fold_train_ids,
                train_scaled,
                held_scaled,
                mu_bar,
            )
        )
        ridge_result = _context_pca_ridge_rows(
            slice_name,
            rt,
            view_name,
            held_out_id,
            fold_train_ids,
            train_scaled,
            held_scaled,
            config,
        )
        if ridge_result is not None:
            ridge_rows, n_components = ridge_result
            rows.extend(ridge_rows)
            effective_components.setdefault(view_name, set()).add(n_components)

    return rows


def _validate_split(labels: pd.DataFrame, split: FixedSplit) -> tuple[str, ...]:
    """Validate ``split`` and return its supervised train subset.

    Train membership may include contexts without GeneEffect labels. They remain
    registered train contexts but are excluded from every supervised fit and
    label-donor set. Validation and test must be fully labeled.
    """
    known = set(labels["model_id"])
    for name, ids in (("val", split.val), ("test", split.test)):
        unknown = sorted(set(ids) - known)
        if unknown:
            raise ValueError(f"split {name!r} has unknown model_id(s): {unknown}")
    overlaps = {
        "train/val": set(split.train) & set(split.val),
        "train/test": set(split.train) & set(split.test),
        "val/test": set(split.val) & set(split.test),
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"split partitions overlap: {bad}")
    if not split.train:
        raise ValueError("split 'train' must be non-empty")
    declared_unlabeled = set(split.unlabeled_train)
    outside_train = sorted(declared_unlabeled - set(split.train))
    if outside_train:
        raise ValueError(
            f"split 'unlabeled_train' contains non-train model_id(s): {outside_train}"
        )
    unknown_train = set(split.train) - known
    if unknown_train != declared_unlabeled:
        raise ValueError(
            "split train label coverage does not match declared unlabeled_train: "
            f"unknown={sorted(unknown_train)}, "
            f"declared={sorted(declared_unlabeled)}"
        )
    supervised_train = tuple(model_id for model_id in split.train if model_id in known)
    if not supervised_train:
        raise ValueError("split 'train' has no GeneEffect-labeled model_id")
    if not split.val and not split.test:
        raise ValueError("split must have at least one of 'val' or 'test' to evaluate")
    return supervised_train


def run_r1_ladder(
    labels: pd.DataFrame,
    context_views: Mapping[str, pd.DataFrame],
    copy_prior: pd.Series | None,
    *,
    pca_components: int = 8,
    ridge_alpha: float = 1.0,
    seed: int = 20260804,
    min_lines: int = 3,
    outer: str = "fixed",
    split: FixedSplit | None = None,
) -> R1Result:
    """Run the R1 residual baseline ladder under a fixed split or LOLO.

    Args:
        labels, context_views, copy_prior: See the matching CLI flags
            ``--labels``/``--context``/``--copy-prior``. Context views run
            ``nearest_line``/``context_pca_ridge`` once each.
        pca_components, ridge_alpha, seed, min_lines: See the matching CLI
            flags.
        outer: ``"fixed"`` (default; see :class:`FixedSplit`) or
            ``"lolo"`` (leave-one-line-out diagnostic over every line).
        split: Required when ``outer="fixed"``.

    Returns:
        An :class:`R1Result` with one summary entry per evaluated slice.

    Raises:
        ValueError: On malformed inputs, an internal leakage/consistency
            violation, or if no predictions were produced.
    """
    if outer not in ("fixed", "lolo"):
        raise ValueError(f"outer must be 'fixed' or 'lolo', got {outer!r}")
    config = _LadderConfig(pca_components, ridge_alpha, min_lines)

    slice_specs: dict[str, list[tuple[list[str], str]]]
    fixed_train_lines: tuple[str, ...] | None = None
    if outer == "lolo":
        lines = sorted(labels["model_id"].unique())
        if len(lines) < 3:
            raise ValueError("outer='lolo' requires at least 3 lines")
        mu_bar = fit_gene_means(labels, lines, min_lines=min_lines)
        slice_specs = {
            LOLO_SLICE: [
                ([line for line in lines if line != held], held) for held in lines
            ]
        }
    else:
        if split is None:
            raise ValueError("outer='fixed' requires a split")
        supervised_train = _validate_split(labels, split)
        fixed_train_lines = supervised_train
        mu_bar = fit_gene_means(labels, supervised_train, min_lines=min_lines)
        slice_specs = {}
        if split.val:
            slice_specs[VAL_SLICE] = [
                (list(supervised_train), held) for held in split.val
            ]
        if split.test:
            slice_specs[TEST_SLICE] = [
                (list(supervised_train), held) for held in split.test
            ]

    effective_components: dict[str, dict[str, set[int]]] = {}
    if fixed_train_lines is not None:
        eval_specs = [
            (slice_name, held_out_id)
            for slice_name, evals in slice_specs.items()
            for _train_lines, held_out_id in evals
        ]
        all_rows = _run_fixed_evals(
            labels,
            fixed_train_lines,
            eval_specs,
            mu_bar,
            context_views,
            copy_prior,
            config,
            effective_components,
        )
    else:
        all_rows = []
        for slice_name, evals in slice_specs.items():
            components_this_slice: dict[str, set[int]] = {}
            for train_lines, held_out_id in evals:
                all_rows.extend(
                    _run_eval(
                        labels,
                        train_lines,
                        held_out_id,
                        slice_name,
                        mu_bar,
                        context_views,
                        copy_prior,
                        config,
                        components_this_slice,
                    )
                )
            effective_components[slice_name] = components_this_slice

    if not all_rows:
        raise ValueError("R1 ladder produced no predictions; check inputs")

    predictions = (
        pd.DataFrame(all_rows)
        .sort_values(["slice", "method", "model_id", "gene_symbol"], kind="stable")
        .reset_index(drop=True)
    )
    if predictions.duplicated(["slice", "method", "model_id", "gene_symbol"]).any():
        raise ValueError("internal error: duplicate prediction keys")
    _validate_exact_method_coverage(predictions, context_views, copy_prior)

    return _summarize(
        predictions, effective_components, config, seed, context_views, outer, split
    )


def _validate_exact_method_coverage(
    predictions: pd.DataFrame,
    context_views: Mapping[str, pd.DataFrame],
    copy_prior: pd.Series | None,
) -> None:
    """Require every configured baseline on the same observable truth keys."""
    expected_methods = {GENE_MEAN}
    if copy_prior is not None:
        expected_methods.add(COPY_PRIOR)
    for view_name in context_views:
        expected_methods.add(_view_method_name(NEAREST_LINE, view_name))
        expected_methods.add(_view_method_name(CONTEXT_PCA_RIDGE, view_name))

    key_columns = ["slice", "model_id", "gene_symbol"]
    truth_keys = set(
        predictions.loc[predictions["method"] == GENE_MEAN, key_columns].itertuples(
            index=False, name=None
        )
    )
    actual_methods = set(predictions["method"])
    missing_methods = sorted(expected_methods - actual_methods)
    unexpected_methods = sorted(actual_methods - expected_methods)
    if missing_methods or unexpected_methods:
        raise ValueError(
            "baseline method set mismatch: "
            f"missing={missing_methods}, unexpected={unexpected_methods}"
        )
    for method in sorted(expected_methods):
        method_keys = set(
            predictions.loc[predictions["method"] == method, key_columns].itertuples(
                index=False, name=None
            )
        )
        if method_keys != truth_keys:
            missing = sorted(truth_keys - method_keys)
            extra = sorted(method_keys - truth_keys)
            raise ValueError(
                f"{method}: evaluated-key coverage differs from common truth mask: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )


def _method_entry(score: ResidualScore, seed: int) -> dict[str, object]:
    """Build one method's summary entry: macros, undefined counts, self-CIs.

    The CI is a bootstrap over the method's own per-unit values (resample
    units, not a paired delta) -- see :func:`_pairwise_deltas` for the
    cross-method comparisons that matter.
    """
    point_l, lo_l, hi_l = bootstrap_delta(score.per_line, seed=seed)
    point_g, lo_g, hi_g = bootstrap_delta(score.per_gene, seed=seed)
    return {
        "macro_per_line": score.macro_per_line,
        "macro_per_gene": score.macro_per_gene,
        "n_lines": score.n_lines,
        "n_genes": score.n_genes,
        "n_line_undefined": score.n_line_undefined,
        "n_gene_undefined": score.n_gene_undefined,
        "ci_per_line": {"point": point_l, "ci_lo": lo_l, "ci_hi": hi_l},
        "ci_per_gene": {"point": point_g, "ci_lo": lo_g, "ci_hi": hi_g},
    }


def _pairwise_deltas(
    scores: Mapping[str, ResidualScore], view_names: list[str], seed: int
) -> list[dict[str, object]]:
    """Paired bootstrap deltas between the same method across context views.

    gene_mean/copy_prior are undefined by construction and cannot anchor a
    delta (see the ``reference_undefined`` summary note), so the only
    well-posed comparisons are between context-conditioned methods run on
    different representations of the same evaluated lines, e.g.
    ``context_pca_ridge[hvg]`` vs. ``context_pca_ridge[tx1]``. Views are
    compared in the order they were supplied (each later view as
    "candidate" against each earlier view as "reference").
    """
    comparisons: list[dict[str, object]] = []
    for base in (NEAREST_LINE, CONTEXT_PCA_RIDGE):
        available = [v for v in view_names if _view_method_name(base, v) in scores]
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                reference_view, candidate_view = available[i], available[j]
                reference_score = scores[_view_method_name(base, reference_view)]
                candidate_score = scores[_view_method_name(base, candidate_view)]
                paired_line = candidate_score.per_line - reference_score.per_line
                paired_gene = candidate_score.per_gene - reference_score.per_gene
                point_l, lo_l, hi_l = bootstrap_delta(paired_line, seed=seed)
                point_g, lo_g, hi_g = bootstrap_delta(paired_gene, seed=seed)
                comparisons.append(
                    {
                        "method": base,
                        "candidate_view": candidate_view,
                        "reference_view": reference_view,
                        "per_line": {"point": point_l, "ci_lo": lo_l, "ci_hi": hi_l},
                        "per_gene": {"point": point_g, "ci_lo": lo_g, "ci_hi": hi_g},
                    }
                )
    return comparisons


_REFERENCE_UNDEFINED_NOTE = (
    "gene_mean's delta_hat is identically 0.0 (a context-blind method makes "
    "no line-specific claim) and copy_prior's delta_hat = prior_g - mu_bar_g "
    "is a pure per-gene constant carrying no line-specific information: both "
    "are undefined (NaN) on one or both axes by construction, so neither can "
    "anchor a paired bootstrap delta for another method -- that is a result, "
    "not a defect. See each method's own ci_per_line/ci_per_gene for a "
    "self-contained CI, and pairwise_deltas for comparisons between the "
    "scoreable (context-conditioned) methods."
)


def _summarize(
    predictions: pd.DataFrame,
    effective_components: dict[str, dict[str, set[int]]],
    config: _LadderConfig,
    seed: int,
    context_views: Mapping[str, pd.DataFrame],
    outer: str,
    split: FixedSplit | None,
) -> R1Result:
    """Score every method in every slice and assemble the R1Result."""
    view_names = list(context_views)
    slice_summaries: dict[str, dict[str, object]] = {}
    per_line_frames: list[pd.DataFrame] = []
    per_gene_frames: list[pd.DataFrame] = []

    for slice_name, slice_predictions in predictions.groupby("slice", sort=False):
        scores: dict[str, ResidualScore] = {}
        for method in sorted(slice_predictions["method"].unique()):
            subframe = slice_predictions.loc[slice_predictions["method"] == method]
            scores[method] = _combined_score(subframe, method)

        method_summaries: dict[str, dict[str, object]] = {}
        for method, score in scores.items():
            entry = _method_entry(score, seed)
            base_method, _, view_suffix = method.partition("[")
            if base_method == CONTEXT_PCA_RIDGE and view_suffix:
                view_name = view_suffix.rstrip("]")
                comps = effective_components.get(slice_name, {}).get(view_name)
                if comps:
                    entry["pca_components_effective"] = sorted(comps)
            method_summaries[method] = entry

        slice_summaries[slice_name] = {
            "n_lines_evaluated": int(slice_predictions["model_id"].nunique()),
            "n_genes_evaluated": int(slice_predictions["gene_symbol"].nunique()),
            "methods": method_summaries,
            "pairwise_deltas": _pairwise_deltas(scores, view_names, seed),
        }
        for method, score in scores.items():
            per_line_frames.append(
                score.per_line.rename("spearman")
                .reset_index()
                .assign(method=method, slice=slice_name)
            )
            per_gene_frames.append(
                score.per_gene.rename("spearman")
                .reset_index()
                .assign(method=method, slice=slice_name)
            )

    per_line_cols = ["slice", "method", "model_id", "spearman"]
    per_gene_cols = ["slice", "method", "gene_symbol", "spearman"]
    per_line = (
        pd.concat(per_line_frames, ignore_index=True)[per_line_cols]
        .sort_values(["slice", "method", "model_id"], kind="stable")
        .reset_index(drop=True)
    )
    per_gene = (
        pd.concat(per_gene_frames, ignore_index=True)[per_gene_cols]
        .sort_values(["slice", "method", "gene_symbol"], kind="stable")
        .reset_index(drop=True)
    )

    summary: dict[str, object] = {
        "outer": outer,
        "reference_undefined": _REFERENCE_UNDEFINED_NOTE,
        "config": {
            "pca_components_requested": config.pca_components,
            "ridge_alpha": config.ridge_alpha,
            "seed": seed,
            "min_lines": config.min_lines,
            "context_views": sorted(context_views),
        },
        "slices": slice_summaries,
    }
    if outer == "fixed" and split is not None:
        summary["split"] = {
            "train": list(split.train),
            "supervised_train": [
                model_id
                for model_id in split.train
                if model_id not in set(split.unlabeled_train)
            ],
            "unlabeled_train": list(split.unlabeled_train),
            "val": list(split.val),
            "test": list(split.test),
        }
    return R1Result(
        predictions=predictions, per_line=per_line, per_gene=per_gene, summary=summary
    )
