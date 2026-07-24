"""Per-line few-shot RE-RANKING calibration ("Phase E") for the T2 Tx1 head.

Frozen spec §4 ("Few-shot adaptation"): a light per-line calibration fit on
``k`` labeled genes of a held-out cell line, scored on the remaining
DISJOINT genes of that line, with no full-model fine-tune.

Why this module exists (and why ``affine_kshot_calibrate`` in
``tx1_geneeffect_eval.py`` cannot do this job)
--------------------------------------------------------------------------
The per-line metric is Spearman rank correlation, which is invariant to any
monotone-increasing transform of the predictions. A scalar-affine map,
``y = a + b * base_pred``, is monotone in ``base_pred`` whenever ``b > 0``
(and reverses ranks uniformly whenever ``b < 0``) — either way it cannot
change *which* genes are ranked above which others. So fitting ``a`` and
``b`` on k labeled genes and applying the same map to all genes is INERT
for Spearman: it can only leave the score unchanged (or flip its sign),
never improve it. That is by design in ``tx1_geneeffect_eval.py`` — the
affine seam is a deliberately weak reference calibrator for baselines.

To move the Spearman curve, an adapter must RE-RANK genes, and re-ranking
requires information the scalar ``base_pred`` does not carry: a per-gene
FEATURE VECTOR ``Φ[g, :]`` (e.g. moment-pooled response/basal summaries, or
any other per-gene representation available at inference time). Given such
features, a (regularized) linear/ridge fit on the k labeled genes can
reorder genes relative to ``base_pred`` alone, because different genes with
the same scalar ``base_pred`` can have different feature vectors and hence
different adapted scores.

This module implements exactly that: closed-form ridge regression from
per-gene features (optionally on the RESIDUAL of ``base_pred``) fit on k
labeled genes, applied to all genes. See ``calibrate_line`` for the
high-level per-line entry point and the module docstring section
"Phase-F integration" below for how this feeds the evaluator.

Phase-F integration
--------------------------------------------------------------------------
``tx1_geneeffect_eval.py`` scores a tidy long-format predictions table with
one ``base_pred`` scalar per (line, gene, method) row, and applies its
pluggable ``KShotCalibrator`` seam (``affine_kshot_calibrate`` by default)
identically to every method in that table, including the k=0 baseline
column. That seam only ever sees the scalar ``base_pred`` — it has no
access to per-gene features, so it is structurally incapable of
re-ranking. The integration decision is:

* Baselines (``copy_k562``, ``cross_line_mean``, ``nearest_line``,
  ``lineage_only``, ``ccle_bulk``, ``pseudobulk_basal``) do not have
  per-gene feature vectors defined in the same feature space as the T2
  head, and are scored with Phase F's affine ``"+k"`` seam unchanged. That
  seam is provably inert for them too, which is expected and correct: it
  is a smoke test that the k-shot machinery does not spuriously help a
  method that has no re-ranking mechanism.
* The method under test (``tx1_3b_st``) does its few-shot re-ranking HERE,
  upstream of Phase F, using :func:`calibrate_line` on the head's per-gene
  moment-pooled features. The resulting already-adapted scalar is written
  into the ``base_pred`` column of the method's rows in the predictions
  table for a given k, and Phase F's ``per_line_metric``/``evaluate`` is
  then run with ``calibrate=lambda base_pred, y_true, label_mask:
  base_pred`` (an identity calibrator) for that method's rows, since the
  re-ranking already happened. Concretely, for each held-out line and each
  frozen panel/k, call :func:`calibrate_line` once per panel with that
  panel's label_mask, and place the returned per-gene vector as the
  method's ``base_pred`` for that (line, panel, k) before Phase F computes
  ``panel_spearman`` on the panel's scored (non-label) genes. See
  :func:`make_predictions_long` for a helper that assembles that
  per-(line, panel, k) predictions frame directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)

#: Numerical floor for feature standard deviations, so a constant (zero
#: variance) feature column does not produce a division-by-zero.
_STD_EPS: float = 1e-8

#: Default ridge regularization strength. Fixed, not tuned on held-out
#: lines: Phase A will freeze this (or a small grid around it) as the
#: adapter hyperparameter, selected by cross-validation on TRAINING lines
#: only, never on the held-out test lines this module is applied to. A
#: value of 1.0 is the conventional ridge default (equal prior variance to
#: noise variance under the Bayesian/MAP reading of ridge regression) and
#: is a reasonable starting point once features are standardized to unit
#: variance (this module always standardizes before fitting, so the
#: regularization strength is comparable across feature sets and across
#: line-specific k-shot fits).
DEFAULT_ALPHA: float = 1.0


@dataclass(frozen=True)
class RidgeCalibrator:
    """A fitted per-line closed-form ridge calibrator.

    Attributes:
        weights: Ridge coefficients on STANDARDIZED features, shape (F,).
        intercept: Ridge intercept (fit on the standardized-feature,
            possibly-residual target).
        feature_mean: Per-feature mean used to standardize, shape (F,),
            computed from the k labeled genes only.
        feature_std: Per-feature standard deviation used to standardize
            (floored at :data:`_STD_EPS`), shape (F,), computed from the k
            labeled genes only.
        residual: If True, :meth:`apply` adds the ridge prediction to the
            supplied ``base_pred_all`` (residual-correction mode); if
            False, the ridge prediction is used directly as the adapted
            score.
        alpha: The ridge regularization strength used to fit this
            calibrator (stored for provenance/logging only).
    """

    weights: np.ndarray = field(repr=False)
    intercept: float
    feature_mean: np.ndarray = field(repr=False)
    feature_std: np.ndarray = field(repr=False)
    residual: bool
    alpha: float

    def apply(
        self,
        features_all: np.ndarray,
        base_pred_all: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply this calibrator to every gene's feature vector.

        Args:
            features_all: Per-gene feature matrix for ALL genes to be
                scored (labeled and disjoint-scored alike), shape (G, F).
                ``F`` must match the ``F`` this calibrator was fit with.
            base_pred_all: Base scalar predictions for the same ``G``
                genes, shape (G,). Required (and added to the ridge output)
                when ``self.residual`` is True; ignored if ``self.residual``
                is False.

        Returns:
            Adapted per-gene predictions, shape (G,).

        Raises:
            ValueError: If ``features_all`` is not 2-D with ``F`` columns
                matching this calibrator, or if ``residual`` is True but
                ``base_pred_all`` is not supplied.
        """
        features_all = np.asarray(features_all, dtype=float)
        if features_all.ndim != 2 or features_all.shape[1] != self.weights.shape[0]:
            msg = (
                "features_all must be 2-D with "
                f"{self.weights.shape[0]} columns, got shape "
                f"{features_all.shape}"
            )
            raise ValueError(msg)

        standardized = (features_all - self.feature_mean) / self.feature_std
        ridge_out = standardized @ self.weights + self.intercept

        if not self.residual:
            return ridge_out

        if base_pred_all is None:
            msg = "residual=True calibrator requires base_pred_all in apply()"
            raise ValueError(msg)
        base_pred_all = np.asarray(base_pred_all, dtype=float).reshape(-1)
        if base_pred_all.shape[0] != features_all.shape[0]:
            msg = (
                "base_pred_all must have one entry per row of features_all: "
                f"got {base_pred_all.shape[0]} vs {features_all.shape[0]}"
            )
            raise ValueError(msg)
        return base_pred_all + ridge_out


def fit_ridge_calibration(
    features_labeled: np.ndarray,
    y_labeled: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    base_pred_labeled: np.ndarray | None = None,
) -> RidgeCalibrator:
    """Fit a closed-form ridge calibrator on k labeled genes.

    Standardizes ``features_labeled`` using the labeled set's own mean/std
    (the only statistics available at k-shot time for a held-out line),
    then solves ridge regression with an unpenalized intercept:
    ``target = intercept + standardized_features @ weights``, minimizing
    ``||target - intercept - Z @ weights||^2 + alpha * ||weights||^2``.

    If ``base_pred_labeled`` is given, ``target = y_labeled -
    base_pred_labeled`` (a RESIDUAL correction on top of the existing base
    score); otherwise ``target = y_labeled`` directly.

    Args:
        features_labeled: Per-gene feature matrix for the k labeled genes,
            shape (k, F).
        y_labeled: True GeneEffect values for the same k genes, shape (k,).
        alpha: Ridge regularization strength (see :data:`DEFAULT_ALPHA`).
            Must be >= 0.
        base_pred_labeled: Optional base scalar predictions for the same k
            genes, shape (k,); enables residual-correction mode.

    Returns:
        A fitted :class:`RidgeCalibrator`.

    Raises:
        ValueError: If shapes are inconsistent, ``k == 0``, or ``alpha`` is
            negative.
    """
    features_labeled = np.asarray(features_labeled, dtype=float)
    y_labeled = np.asarray(y_labeled, dtype=float).reshape(-1)
    if features_labeled.ndim != 2:
        msg = f"features_labeled must be 2-D [k, F], got {features_labeled.shape}"
        raise ValueError(msg)
    k, n_features = features_labeled.shape
    if k == 0:
        raise ValueError("fit_ridge_calibration requires at least one labeled gene")
    if y_labeled.shape[0] != k:
        msg = f"y_labeled has {y_labeled.shape[0]} entries, expected {k}"
        raise ValueError(msg)
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")

    residual_mode = base_pred_labeled is not None
    if residual_mode:
        base_pred_labeled = np.asarray(base_pred_labeled, dtype=float).reshape(-1)
        if base_pred_labeled.shape[0] != k:
            msg = (
                f"base_pred_labeled has {base_pred_labeled.shape[0]} entries, "
                f"expected {k}"
            )
            raise ValueError(msg)
        target = y_labeled - base_pred_labeled
    else:
        target = y_labeled

    feature_mean = features_labeled.mean(axis=0)
    feature_std = features_labeled.std(axis=0, ddof=0)
    feature_std = np.maximum(feature_std, _STD_EPS)
    standardized = (features_labeled - feature_mean) / feature_std

    # Center the target so the intercept is simply its mean and the ridge
    # penalty only shrinks the (standardized-feature) slope weights, not
    # the intercept — the standard ridge-with-unpenalized-intercept trick.
    target_mean = target.mean()
    target_centered = target - target_mean

    # lstsq (SVD-based) rather than solve: at alpha=0 with more features than
    # labeled genes (F > k), the Gram matrix is exactly singular, and solve
    # raises LinAlgError. lstsq is always well-defined (returns the min-norm
    # least-squares solution) and agrees with solve whenever the system is
    # actually well-posed (alpha > 0, or alpha=0 with a full-rank Gram).
    gram = standardized.T @ standardized + alpha * np.eye(n_features)
    weights, *_ = np.linalg.lstsq(gram, standardized.T @ target_centered, rcond=None)

    return RidgeCalibrator(
        weights=weights,
        intercept=float(target_mean),
        feature_mean=feature_mean,
        feature_std=feature_std,
        residual=residual_mode,
        alpha=float(alpha),
    )


def calibrate_line(
    features: np.ndarray,
    base_pred: np.ndarray,
    y_true: np.ndarray,
    label_mask: np.ndarray,
    k: int,
    alpha: float = DEFAULT_ALPHA,
    residual: bool = True,
) -> np.ndarray:
    """High-level per-line few-shot calibration entry point.

    Selects the first ``k`` labeled genes from ``label_mask`` (in the order
    they appear), fits a :class:`RidgeCalibrator` on those genes' features
    and labels, and applies it to ALL ``G`` genes. Mirrors the contract of
    ``tx1_geneeffect_eval.affine_kshot_calibrate`` (same
    base_pred/y_true/label_mask inputs, k=0 is identity) but additionally
    consumes per-gene ``features`` so it can re-rank.

    Args:
        features: Per-gene feature matrix for all G genes, shape (G, F).
        base_pred: Base scalar predictions for all G genes, shape (G,).
        y_true: True GeneEffect values for all G genes, shape (G,). Only
            the k labeled entries are used to fit; the rest are for the
            caller's later scoring and are not read here.
        label_mask: Boolean mask selecting the AVAILABLE label-gene pool
            for this panel, shape (G,). ``k`` of the ``True`` entries are
            used for fitting; the remainder of ``label_mask`` (beyond the
            first k) and every ``False`` entry are left for the caller to
            score, i.e. the fit and score gene sets are always disjoint.
        k: Number of labeled genes to use for the fit, taken from the
            first ``k`` `True` entries of ``label_mask`` in array order.
            ``k == 0`` returns ``base_pred`` unchanged (identity), matching
            the frozen affine-calibrator contract.
        alpha: Ridge regularization strength, forwarded to
            :func:`fit_ridge_calibration`.
        residual: If True (default), fit/apply a residual correction on
            top of ``base_pred``; if False, the ridge output replaces
            ``base_pred`` entirely.

    Returns:
        Adapted per-gene predictions for all G genes, shape (G,). Entries
        outside the k selected labeled genes are the ones a caller should
        score (per the disjointness contract above).

    Raises:
        ValueError: If ``k`` is negative, larger than the number of
            available labels, or shapes are inconsistent.
    """
    features = np.asarray(features, dtype=float)
    base_pred = np.asarray(base_pred, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    label_mask = np.asarray(label_mask, dtype=bool).reshape(-1)

    n_genes = features.shape[0]
    if not (base_pred.shape[0] == y_true.shape[0] == label_mask.shape[0] == n_genes):
        msg = (
            "features, base_pred, y_true, and label_mask must share the "
            f"same gene count G; got {n_genes}, {base_pred.shape[0]}, "
            f"{y_true.shape[0]}, {label_mask.shape[0]}"
        )
        raise ValueError(msg)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    if k == 0:
        return base_pred.copy()

    label_indices = np.flatnonzero(label_mask)
    if k > label_indices.size:
        msg = f"k={k} exceeds available labeled genes ({label_indices.size})"
        raise ValueError(msg)
    fit_indices = label_indices[:k]

    calibrator = fit_ridge_calibration(
        features_labeled=features[fit_indices],
        y_labeled=y_true[fit_indices],
        alpha=alpha,
        base_pred_labeled=base_pred[fit_indices] if residual else None,
    )
    return calibrator.apply(
        features_all=features,
        base_pred_all=base_pred if residual else None,
    )


def make_predictions_long(
    model_id: str,
    genes: np.ndarray,
    features: np.ndarray,
    base_pred: np.ndarray,
    y_true: np.ndarray,
    panels_for_line: pd.DataFrame,
    k_schedule: list[int],
    method: str,
    alpha: float = DEFAULT_ALPHA,
    residual: bool = True,
) -> pd.DataFrame:
    """Assemble Phase-F-compatible per-(panel, k) rows for the METHOD.

    For each frozen panel and each k in ``k_schedule``, fits
    :func:`calibrate_line` on that panel's first-k label genes and writes
    the resulting already-re-ranked scores as the ``base_pred`` column for
    that (panel, k) slice, so that when ``tx1_geneeffect_eval.py`` later
    runs its own k-shot loop over these rows with an IDENTITY calibrator
    (``lambda base_pred, y_true, label_mask: base_pred``), it scores this
    method's few-shot re-ranking rather than re-deriving an affine map from
    a scalar it never sees a feature-augmented view of. This keeps the
    disjointness/panel-aggregation/bootstrap logic in
    ``tx1_geneeffect_eval.py`` as the single source of truth; only the
    ``base_pred`` values differ per k, computed here.

    Args:
        model_id: The held-out line identifier (Phase-F's ``model_id``).
        genes: Gene identifiers, shape (G,), matching
            ``panels_for_line["depmap_column"]`` values.
        features: Per-gene feature matrix, shape (G, F).
        base_pred: Base (k=0) scalar predictions, shape (G,).
        y_true: True GeneEffect values, shape (G,).
        panels_for_line: Frozen nested k-label panels for this line, with
            columns ``[panel, label_order, depmap_column]`` (see
            ``tx1_geneeffect_eval.load_panels``).
        k_schedule: k values to emit rows for (typically
            ``tx1_geneeffect_eval.K_SCHEDULE``).
        method: Method identifier to stamp into the ``method`` column
            (e.g. ``"tx1_3b_st"``).
        alpha: Ridge regularization strength, forwarded to
            :func:`calibrate_line`.
        residual: Forwarded to :func:`calibrate_line`.

    Returns:
        Tidy long-format DataFrame with columns ``[model_id, panel,
        depmap_column, method, k, base_pred, y_true]`` — one row per
        (panel, gene, k). Downstream scoring must restrict each panel/k
        slice to genes with ``label_order > k`` (or absent from the
        panel), matching ``tx1_geneeffect_eval.per_line_metric``'s
        ``scored_mask = ~label_mask`` convention.
    """
    genes = np.asarray(genes)
    rows: list[dict[str, object]] = []
    panel_ids = sorted(panels_for_line["panel"].unique())
    for panel_id in panel_ids:
        panel_rows = panels_for_line[panels_for_line["panel"] == panel_id]
        for k in k_schedule:
            if k > 0:
                label_genes = set(
                    panel_rows.loc[panel_rows["label_order"] <= k, "depmap_column"]
                )
            else:
                label_genes = set()
            label_mask = (
                np.isin(genes, list(label_genes))
                if label_genes
                else (np.zeros(genes.shape[0], dtype=bool))
            )
            adapted = calibrate_line(
                features, base_pred, y_true, label_mask, k, alpha, residual
            )
            for gene, pred, truth in zip(genes, adapted, y_true):
                rows.append(
                    {
                        "model_id": model_id,
                        "panel": panel_id,
                        "depmap_column": gene,
                        "method": method,
                        "k": k,
                        "base_pred": pred,
                        "y_true": truth,
                    }
                )
    return pd.DataFrame(rows)
