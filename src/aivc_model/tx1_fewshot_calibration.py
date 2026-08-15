"""Low-dimensional per-line few-shot re-ranking for the T2 Tx1 head.

The adapter fits a label-free PCA view once from all gene features in a held-out
line, then fits a bounded residual ridge on each panel's k labeled genes.
``make_predictions_long`` emits the adapted panel/k scores consumed by the
Phase-F evaluator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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

#: Maximum number of unsupervised per-line principal components exposed to
#: the supervised k-shot ridge.  The actual fit uses at most ``k_finite // 2``
#: components, so the supervised problem is always low-dimensional relative
#: to the labels available to it.
DEFAULT_MAX_COMPONENTS: int = 8

#: Provenance label for the per-line adapter implemented here.
CALIBRATION_SCHEMA: str = "per_line_transductive_pca_residual_ridge_v1"

#: Human-readable form of the supervised dimension cap recorded in manifests.
CALIBRATION_DIMENSION_RULE: str = "min(max_components, floor(k_finite / 2))"

#: Minimum finite labeled genes required to fit a ridge calibrator. Held-out
#: lines are NOT completeness-guaranteed (the differential slice was frozen
#: on TRAINING lines), so a k-shot panel can land on a gene with a missing
#: GeneEffect label; below this minimum, :func:`calibrate_line` falls back
#: to identity (``base_pred``) rather than fit on too little (or no) signal.
#: A ridge fit is technically well-posed from a single finite point (the
#: intercept alone), but 2 is the smallest count that lets the fit reflect
#: any actual feature-target relationship rather than a single anecdote.
MIN_FINITE_LABELS: int = 2


class InsufficientFiniteLabelsError(ValueError):
    """Raised when too few labeled genes have finite target/features to fit."""


@dataclass(frozen=True)
class FeatureReducer:
    """Label-free, deterministic PCA reducer fit once per held-out line."""

    feature_mean: np.ndarray = field(repr=False)
    feature_std: np.ndarray = field(repr=False)
    standardized_mean: np.ndarray = field(repr=False)
    components: np.ndarray = field(repr=False)
    component_std: np.ndarray = field(repr=False)

    @property
    def input_dim(self) -> int:
        """Raw pooled-feature width."""
        return int(self.feature_mean.shape[0])

    @property
    def output_dim(self) -> int:
        """Maximum reduced width available to a k-shot fit."""
        return int(self.components.shape[0])

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Project raw features into standardized, label-free PCA scores."""
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "features must be 2-D with "
                f"{self.input_dim} columns, got shape {features.shape}"
            )
        standardized = (features - self.feature_mean) / self.feature_std
        standardized = standardized - self.standardized_mean
        return (standardized @ self.components.T) / self.component_std


def fit_feature_reducer(
    features_reference: np.ndarray,
    max_components: int = DEFAULT_MAX_COMPONENTS,
) -> FeatureReducer:
    """Fit a PCA reducer from features only, without consulting labels.

    Non-finite rows are excluded using feature values alone.  In the Phase-E
    path ``features_reference`` is the full per-line gene matrix, so this
    representation is fitted once before any panel labels are selected.
    """
    features_reference = np.asarray(features_reference, dtype=float)
    if features_reference.ndim != 2:
        raise ValueError(
            "features_reference must be 2-D, got "
            f"{features_reference.shape}"
        )
    if max_components < 1:
        raise ValueError(f"max_components must be >= 1, got {max_components}")
    finite_rows = np.all(np.isfinite(features_reference), axis=1)
    finite_features = features_reference[finite_rows]
    if finite_features.shape[0] < 2 or finite_features.shape[1] < 1:
        raise ValueError("feature reducer requires at least two finite rows")

    feature_mean = finite_features.mean(axis=0)
    feature_std = np.maximum(finite_features.std(axis=0, ddof=0), _STD_EPS)
    standardized = (finite_features - feature_mean) / feature_std
    n_components = min(
        int(max_components),
        standardized.shape[0] - 1,
        standardized.shape[1],
    )
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(standardized)
    component_std = np.maximum(
        np.sqrt(np.asarray(pca.explained_variance_, dtype=float)), _STD_EPS
    )
    return FeatureReducer(
        feature_mean=feature_mean,
        feature_std=feature_std,
        standardized_mean=np.asarray(pca.mean_, dtype=float),
        components=np.asarray(pca.components_, dtype=float),
        component_std=component_std,
    )


@dataclass(frozen=True)
class RidgeCalibrator:
    """A fitted per-line closed-form ridge calibrator.

    Attributes:
        weights: Ridge coefficients on reduced PCA scores, shape (D,), where
            ``D <= min(DEFAULT_MAX_COMPONENTS, k_finite // 2)``.
        intercept: Ridge intercept (fit on the standardized-feature,
            possibly-residual target).
        feature_reducer: Label-free PCA reducer fitted from the reference
            feature population, normally all genes of the held-out line.
        reduced_mean: Per-fit mean of the selected PCA scores, used to keep
            the ridge intercept unpenalized without rescaling each component.
        residual: If True, :meth:`apply` adds the ridge prediction to the
            supplied ``base_pred_all`` (residual-correction mode); if
            False, the ridge prediction is used directly as the adapted
            score.
        alpha: The ridge regularization strength used to fit this
            calibrator (stored for provenance/logging only).
    """

    weights: np.ndarray = field(repr=False)
    intercept: float
    feature_reducer: FeatureReducer = field(repr=False)
    reduced_mean: np.ndarray = field(repr=False)
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
        if (
            features_all.ndim != 2
            or features_all.shape[1] != self.feature_reducer.input_dim
        ):
            msg = (
                "features_all must be 2-D with "
                f"{self.feature_reducer.input_dim} columns, got shape "
                f"{features_all.shape}"
            )
            raise ValueError(msg)

        reduced = self.feature_reducer.transform(features_all)[:, : self.weights.size]
        reduced = reduced - self.reduced_mean
        ridge_out = reduced @ self.weights + self.intercept

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
    feature_reducer: FeatureReducer | None = None,
    max_components: int = DEFAULT_MAX_COMPONENTS,
) -> RidgeCalibrator:
    """Fit a closed-form ridge calibrator on k labeled genes.

    Projects features through a label-free PCA reducer, then solves ridge
    regression with an unpenalized intercept.  The supervised fit receives
    at most ``min(max_components, k_finite // 2)`` dimensions, preventing the
    former thousands-of-features-from-at-most-50-labels interpolation path.
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
        feature_reducer: Optional reducer already fit from the full unlabeled
            per-line feature matrix. If omitted, a reducer is fit from the
            supplied feature rows without reading targets.
        max_components: Hard cap on PCA dimensions exposed to ridge.

    Returns:
        A fitted :class:`RidgeCalibrator`.

    Raises:
        ValueError: If shapes are inconsistent, ``k == 0``, or ``alpha`` is
            negative.
        InsufficientFiniteLabelsError: If fewer than :data:`MIN_FINITE_LABELS`
            labeled genes have a finite target (after any residual
            subtraction) AND finite features; a held-out line's labels are
            not completeness-guaranteed, and fitting on a non-finite target
            would silently poison every weight with NaN.
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
    if max_components < 1:
        raise ValueError(f"max_components must be >= 1, got {max_components}")

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

    # Held-out lines are not completeness-guaranteed: drop labeled genes
    # with a non-finite target or any non-finite feature before fitting, so
    # one NaN GeneEffect cannot poison target.mean() (and hence every
    # weight) for the whole panel.
    finite_mask = np.isfinite(target) & np.all(np.isfinite(features_labeled), axis=1)
    n_finite = int(finite_mask.sum())
    if n_finite < MIN_FINITE_LABELS:
        msg = (
            f"Only {n_finite} finite labeled gene(s) out of {k} (minimum "
            f"{MIN_FINITE_LABELS}); refusing to fit a ridge calibrator on "
            "non-finite targets/features."
        )
        raise InsufficientFiniteLabelsError(msg)
    if n_finite < k:
        _LOGGER.warning(
            "Dropping %d of %d labeled gene(s) with non-finite target/features "
            "before fitting ridge calibration.",
            k - n_finite,
            k,
        )
        features_labeled = features_labeled[finite_mask]
        target = target[finite_mask]

    if feature_reducer is None:
        feature_reducer = fit_feature_reducer(
            features_labeled[np.all(np.isfinite(features_labeled), axis=1)],
            max_components=max_components,
        )
    elif feature_reducer.input_dim != n_features:
        raise ValueError(
            "feature_reducer input width does not match features_labeled: "
            f"{feature_reducer.input_dim} vs {n_features}"
        )
    n_components = min(
        feature_reducer.output_dim,
        int(max_components),
        max(1, n_finite // 2),
    )
    reduced = feature_reducer.transform(features_labeled)[:, :n_components]
    reduced_mean = reduced.mean(axis=0)
    reduced = reduced - reduced_mean

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
    gram = reduced.T @ reduced + alpha * np.eye(n_components)
    weights, *_ = np.linalg.lstsq(gram, reduced.T @ target_centered, rcond=None)

    return RidgeCalibrator(
        weights=weights,
        intercept=float(target_mean),
        feature_reducer=feature_reducer,
        reduced_mean=reduced_mean,
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
    feature_reducer: FeatureReducer | None = None,
    max_components: int = DEFAULT_MAX_COMPONENTS,
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
        feature_reducer: Label-free reducer fit from all genes in this line.
            If omitted, it is fit from ``features`` without reading labels.
        max_components: Hard cap on reduced dimensions exposed to ridge.

    Returns:
        Adapted per-gene predictions for all G genes, shape (G,). Entries
        outside the k selected labeled genes are the ones a caller should
        score (per the disjointness contract above). Held-out lines are not
        completeness-guaranteed: if fewer than :data:`MIN_FINITE_LABELS` of
        the k selected genes have a finite target/features, this falls back
        to ``base_pred`` unchanged (with a logged warning) instead of
        raising or returning NaN; see :func:`fit_ridge_calibration`.

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

    if feature_reducer is None:
        feature_reducer = fit_feature_reducer(features, max_components)

    label_indices = np.flatnonzero(label_mask)
    if k > label_indices.size:
        msg = f"k={k} exceeds available labeled genes ({label_indices.size})"
        raise ValueError(msg)
    fit_indices = label_indices[:k]

    try:
        calibrator = fit_ridge_calibration(
            features_labeled=features[fit_indices],
            y_labeled=y_true[fit_indices],
            alpha=alpha,
            base_pred_labeled=base_pred[fit_indices] if residual else None,
            feature_reducer=feature_reducer,
            max_components=max_components,
        )
    except InsufficientFiniteLabelsError as exc:
        _LOGGER.warning(
            "k=%d labeled gene(s): %s Falling back to identity (base_pred) "
            "calibration for this panel/k rather than emitting NaN.",
            k,
            exc,
        )
        return base_pred.copy()

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
    feature_reducer = (
        fit_feature_reducer(features, DEFAULT_MAX_COMPONENTS)
        if any(k > 0 for k in k_schedule)
        else None
    )
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
                features,
                base_pred,
                y_true,
                label_mask,
                k,
                alpha,
                residual,
                feature_reducer,
                DEFAULT_MAX_COMPONENTS,
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
