"""Phase-F statistical evaluator and gate for cross-line GeneEffect transfer.

Implements, without modification, the evaluation contract frozen in
``results/phase_a_tx1_20260724/phase_a_registration.json``: per-line Spearman
correlation on the 589-gene differentially-essential slice, a k-shot affine
calibration evaluated over a fixed nested-panel schedule, panel-then-line
aggregation, a paired difference against the copy-K562 baseline, and a
two-sided 95% percentile bootstrap over the 9 held-out test-line means. The
gate is: at k=10, the bootstrap lower bound of the macro-mean paired
difference must exceed ``rho_min``.

This module is pure numpy/scipy/pandas. It consumes a tidy long-format
predictions table with columns ``[model_id, depmap_column, method,
base_pred, y_true]`` (one row per test line x gene x method) and the three
frozen Phase-A artifacts: the cell-line manifest, the differential-essential
slice, and the k-label panels. It does not produce predictions itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_LOGGER = logging.getLogger(__name__)

#: Frozen k-shot schedule (Phase A contract).
K_SCHEDULE: list[int] = [0, 5, 10, 25, 50]
#: Frozen gate k (Phase A contract).
GATE_K: int = 10
#: Frozen gate threshold (Phase A contract).
RHO_MIN: float = 0.05
#: Frozen bootstrap repetition count (Phase A contract).
BOOTSTRAP_REPS: int = 10_000
#: Frozen bootstrap seed (Phase A contract).
BOOTSTRAP_SEED: int = 20260725
#: Frozen minimum pairwise-complete scored genes per panel (Phase A contract).
MIN_SCORED_GENES: int = 100
#: Frozen number of nested label panels per held-out line (Phase A contract).
N_PANELS: int = 20
#: Frozen number of label genes per panel (Phase A contract: k_label_panels.csv
#: has label_order 1..50 for every (model_id, panel)).
N_LABELS: int = 50
#: Frozen paired-difference baseline method (Phase A contract).
DEFAULT_BASELINE_METHOD: str = "copy_k562"
#: Primary candidate method under test (Phase A contract).
DEFAULT_PRIMARY_METHOD: str = "tx1_3b_st"
#: Non-exhaustive, extensible set of recognized method identifiers.
KNOWN_METHODS: frozenset[str] = frozenset(
    {
        "tx1_3b_st",
        "copy_k562",
        "cross_line_mean",
        "nearest_line",
        "lineage_only",
        "ccle_bulk",
        "pseudobulk_basal",
    }
)

#: A pluggable k-shot calibrator: (base_pred, y_true, is_label_mask) -> adjusted_pred.
KShotCalibrator = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


class EvaluationContractError(ValueError):
    """Raised when inputs violate the frozen Phase-A evaluation contract."""


def load_manifest(path: Path) -> pd.DataFrame:
    """Load the frozen cell-line manifest.

    Args:
        path: Path to ``cell_line_manifest.csv``.

    Returns:
        DataFrame with a ``role`` column identifying held-out test lines.
    """
    return pd.read_csv(path)


def load_slice(path: Path) -> pd.DataFrame:
    """Load the frozen differentially-essential gene slice.

    Args:
        path: Path to ``differentially_essential_slice.csv``.

    Returns:
        DataFrame with a ``depmap_column`` column (e.g. "ACLY (47)").
    """
    return pd.read_csv(path)


def load_panels(path: Path) -> pd.DataFrame:
    """Load the frozen nested k-label panels.

    Args:
        path: Path to ``k_label_panels.csv``.

    Returns:
        DataFrame with columns
        ``[model_id, panel, panel_seed, label_order, depmap_column]``.
    """
    return pd.read_csv(path)


def load_predictions(path: Path) -> pd.DataFrame:
    """Load a tidy long-format predictions table (CSV or parquet).

    Args:
        path: Path to a predictions file with columns
            ``[model_id, depmap_column, method, base_pred, y_true]``.

    Returns:
        The predictions table as a DataFrame.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def affine_kshot_calibrate(
    base_pred: np.ndarray,
    y_true: np.ndarray,
    label_mask: np.ndarray,
) -> np.ndarray:
    """Fit an affine map on label genes and apply it to all predictions.

    Least-squares fits ``y_true = a + b * base_pred`` on the genes selected
    by ``label_mask`` (the panel's k label genes), then applies ``a + b *
    base_pred`` to every element of ``base_pred``. With no label genes
    (k=0) this returns ``base_pred`` unchanged, per the frozen contract.
    This is the reference default calibrator; the ``calibrate`` parameter
    threaded through the rest of this module lets a future frozen adapter
    replace it without changing the scoring logic.

    Args:
        base_pred: Pre-calibration scores, shape (n_genes,).
        y_true: True GeneEffect values, shape (n_genes,), aligned with
            ``base_pred``.
        label_mask: Boolean mask selecting the k label genes used to fit
            the affine map, shape (n_genes,).

    Returns:
        Calibrated predictions, shape (n_genes,), aligned with ``base_pred``.
    """
    base_pred = np.asarray(base_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    label_mask = np.asarray(label_mask, dtype=bool)
    if label_mask.sum() == 0:
        return base_pred.copy()

    x = base_pred[label_mask]
    y = y_true[label_mask]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        _LOGGER.warning(
            "Only %d finite label pair(s) available for affine calibration; "
            "falling back to identity.",
            x.size,
        )
        return base_pred.copy()

    design = np.column_stack([np.ones_like(x), x])
    (intercept, slope), *_ = np.linalg.lstsq(design, y, rcond=None)
    return intercept + slope * base_pred


def panel_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scored_mask: np.ndarray,
    min_genes: int = MIN_SCORED_GENES,
) -> float:
    """Spearman rho over pairwise-complete scored genes, or NaN if too few.

    Args:
        y_true: True GeneEffect values, shape (n_genes,).
        y_pred: Calibrated predictions, shape (n_genes,), aligned with
            ``y_true``.
        scored_mask: Boolean mask selecting scored genes (slice genes minus
            the panel's k label genes), shape (n_genes,).
        min_genes: Minimum pairwise-complete (y_true, y_pred) cases required;
            below this the panel is dropped (returns NaN), per the frozen
            contract.

    Returns:
        Spearman rank correlation (average-rank ties, as
        ``scipy.stats.spearmanr``) over the pairwise-complete scored cases,
        or NaN if fewer than ``min_genes`` such cases exist.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scored_mask = np.asarray(scored_mask, dtype=bool)
    yt = y_true[scored_mask]
    yp = y_pred[scored_mask]
    complete = np.isfinite(yt) & np.isfinite(yp)
    n_complete = int(complete.sum())
    if n_complete < min_genes:
        return float("nan")
    result = spearmanr(yt[complete], yp[complete])
    return float(result.statistic)


def _label_genes_for_panel(panel_rows: pd.DataFrame, k: int) -> set[str]:
    """Return the set of first-``k`` label genes for one frozen panel.

    Args:
        panel_rows: Rows of the panels table restricted to a single
            ``(model_id, panel)``, with columns ``[label_order,
            depmap_column]``.
        k: Number of nested label genes to select (``k <= 0`` -> empty set).

    Returns:
        The set of ``depmap_column`` values with ``label_order <= k``.
    """
    if k <= 0:
        return set()
    return set(panel_rows.loc[panel_rows["label_order"] <= k, "depmap_column"])


def _schema_is_panel_aware(preds: pd.DataFrame) -> bool:
    """Detect the Phase-E per-(panel, k) predictions schema for ONE method.

    Column presence alone is not enough: when a panel-aware method (e.g.
    ``tx1_3b_st``, from :func:`aivc_model.tx1_fewshot_calibration.
    make_predictions_long`) is concatenated in the same predictions table
    with a simple-schema method (e.g. ``copy_k562``), pandas fills the
    simple-schema method's ``panel``/``k`` cells with NaN rather than
    dropping the columns. ``preds`` must therefore already be restricted to
    a single method's rows; panel-awareness is then decided from whether
    THIS method's ``panel``/``k`` values are populated, not merely present.

    Args:
        preds: A predictions frame already restricted to one method (or a
            further slice of one, e.g. one line's rows).

    Returns:
        True if ``preds`` is non-empty and every row has non-null
        ``panel``/``k`` (the Phase-E schema); False if the columns are
        absent, or present but every row is null (the plain one-row-per-gene
        schema scored with a pluggable ``calibrate``).

    Raises:
        EvaluationContractError: If ``panel``/``k`` are null for some rows
            and non-null for others -- a malformed mix of the two schemas
            within what should be a single method's predictions.
    """
    if not ({"panel", "k"} <= set(preds.columns)) or preds.empty:
        return False

    panel_null = preds["panel"].isna()
    k_null = preds["k"].isna()
    if not panel_null.equals(k_null):
        raise EvaluationContractError(
            "panel/k null-ness disagrees within these rows (some rows have "
            "panel set but not k, or vice versa); malformed predictions "
            "schema."
        )

    n_null = int(panel_null.sum())
    if n_null == 0:
        return True
    if n_null == len(preds):
        return False
    raise EvaluationContractError(
        f"Mixed panel-aware and simple-schema rows within one method's "
        f"predictions: {len(preds) - n_null} row(s) have panel/k set and "
        f"{n_null} row(s) do not. A method's predictions must be entirely "
        "panel-aware or entirely simple-schema."
    )


def _panel_aware_rho(
    preds_for_line: pd.DataFrame,
    panel_rows: pd.DataFrame,
    panel_id: object,
    k: int,
    min_genes: int,
) -> float:
    """Score one already-adapted (panel, k) prediction slice with IDENTITY.

    Args:
        preds_for_line: Panel/k-aware predictions for one line and method,
            with columns ``[panel, k, depmap_column, base_pred, y_true]``.
        panel_rows: This panel's rows of the frozen k-label panel table.
        panel_id: The panel identifier to select.
        k: The k value to select; also determines the panel's label genes.
        min_genes: Minimum pairwise-complete scored genes, forwarded to
            :func:`panel_spearman`.

    Returns:
        Spearman rho for this (panel, k) slice, or NaN if the slice is
        empty or under-powered. ``base_pred`` is used AS-IS: Phase E
        already applied the re-ranking adapter upstream of this call.
    """
    rows = preds_for_line[
        (preds_for_line["panel"] == panel_id) & (preds_for_line["k"] == k)
    ]
    # Defensive no-op: _validate_no_duplicate_prediction_keys (Finding 3)
    # already guarantees no (model_id, method, panel, k, depmap_column)
    # duplicates for a formally validated run.
    rows = rows.drop_duplicates("depmap_column").set_index("depmap_column")
    if rows.empty:
        return float("nan")
    genes = rows.index.to_numpy()
    y_true = rows["y_true"].to_numpy(dtype=float)
    base_pred = rows["base_pred"].to_numpy(dtype=float)
    label_genes = _label_genes_for_panel(panel_rows, k)
    label_mask = (
        np.isin(genes, list(label_genes)) if label_genes else np.zeros(len(genes), bool)
    )
    return panel_spearman(y_true, base_pred, ~label_mask, min_genes=min_genes)


def _simple_schema_rho(
    genes: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    panel_rows: pd.DataFrame,
    k: int,
    calibrate: KShotCalibrator,
    min_genes: int,
) -> float:
    """Score one panel with the pluggable scalar ``calibrate`` seam.

    Args:
        genes: Gene identifiers for the line/method's predictions.
        y_true: True GeneEffect values, aligned with ``genes``.
        base_pred: Pre-calibration scores, aligned with ``genes``.
        panel_rows: This panel's rows of the frozen k-label panel table.
        k: Number of nested label genes to use for calibration.
        calibrate: Pluggable k-shot calibrator.
        min_genes: Minimum pairwise-complete scored genes, forwarded to
            :func:`panel_spearman`.

    Returns:
        Spearman rho for this panel, or NaN if under-powered.
    """
    label_genes = _label_genes_for_panel(panel_rows, k)
    label_mask = (
        np.isin(genes, list(label_genes)) if label_genes else np.zeros(len(genes), bool)
    )
    adjusted = calibrate(base_pred, y_true, label_mask)
    return panel_spearman(y_true, adjusted, ~label_mask, min_genes=min_genes)


def per_line_metric(
    preds_for_line: pd.DataFrame,
    panels_for_line: pd.DataFrame,
    method: str,
    k: int,
    calibrate: KShotCalibrator = affine_kshot_calibrate,
    min_genes: int = MIN_SCORED_GENES,
) -> float:
    """Mean Spearman rho for one (line, method, k) across the 20 panels.

    Two predictions schemas are supported (see
    :func:`_schema_is_panel_aware`): the plain one-row-per-gene schema,
    scored by applying ``calibrate`` to a single shared ``base_pred`` per
    panel; and the Phase-E per-(panel, k) schema (one already-adapted
    ``base_pred`` per panel AND k, as emitted by
    ``tx1_fewshot_calibration.make_predictions_long``), scored with
    IDENTITY since the re-ranking already happened upstream. The schema is
    detected once from ``preds_for_line`` and applied to every panel.

    Args:
        preds_for_line: Predictions for a single test line and method,
            already restricted to the differential-essential slice gene
            set. Simple schema: columns ``[depmap_column, base_pred,
            y_true]``. Panel-aware schema: columns ``[panel, k,
            depmap_column, base_pred, y_true]``.
        panels_for_line: The frozen k-label panels for the same line, with
            columns ``[panel, label_order, depmap_column]``.
        method: Method identifier, used only for logging (``preds_for_line``
            is assumed pre-filtered to this method).
        k: Number of nested label genes to use for calibration (0 disables
            calibration; see :func:`affine_kshot_calibrate`).
        calibrate: Pluggable k-shot calibrator; ignored for the panel-aware
            schema, where the ``base_pred`` values are already adapted.
        min_genes: Minimum pairwise-complete scored genes per panel, passed
            through to :func:`panel_spearman`.

    Returns:
        Mean Spearman rho across panels with a defined (non-NaN) score, or
        NaN if every panel was dropped for insufficient scored genes.
    """
    panel_aware = _schema_is_panel_aware(preds_for_line)
    if not panel_aware:
        # Defensive no-op: _validate_no_duplicate_prediction_keys (Finding 3)
        # already guarantees no (model_id, method, depmap_column) duplicates
        # for a formally validated run.
        preds = preds_for_line.drop_duplicates("depmap_column").set_index(
            "depmap_column"
        )
        genes = preds.index.to_numpy()
        y_true = preds["y_true"].to_numpy(dtype=float)
        base_pred = preds["base_pred"].to_numpy(dtype=float)

    panel_ids = sorted(panels_for_line["panel"].unique())
    if len(panel_ids) != N_PANELS:
        _LOGGER.warning(
            "method=%s k=%d: expected %d panels, found %d.",
            method,
            k,
            N_PANELS,
            len(panel_ids),
        )

    rhos = np.full(len(panel_ids), np.nan, dtype=float)
    for i, panel_id in enumerate(panel_ids):
        panel_rows = panels_for_line[panels_for_line["panel"] == panel_id]
        if panel_aware:
            rhos[i] = _panel_aware_rho(
                preds_for_line, panel_rows, panel_id, k, min_genes
            )
        else:
            rhos[i] = _simple_schema_rho(
                genes, y_true, base_pred, panel_rows, k, calibrate, min_genes
            )

    if np.all(np.isnan(rhos)):
        _LOGGER.warning(
            "All %d panels dropped for method=%s k=%d; returning NaN.",
            len(panel_ids),
            method,
            k,
        )
        return float("nan")
    return float(np.nanmean(rhos))


def paired_differences(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    slice_df: pd.DataFrame,
    panels: pd.DataFrame,
    methods: Sequence[str],
    k_schedule: Sequence[int] = K_SCHEDULE,
    baseline_method: str = DEFAULT_BASELINE_METHOD,
    calibrate: KShotCalibrator = affine_kshot_calibrate,
    min_genes: int = MIN_SCORED_GENES,
) -> pd.DataFrame:
    """Per-line, per-k paired difference of each method against the baseline.

    Args:
        predictions: Tidy long-format predictions with columns
            ``[model_id, depmap_column, method, base_pred, y_true]``.
        manifest: Cell-line manifest with a ``role`` column; only rows with
            ``role == "test"`` are scored.
        slice_df: Differential-essential slice with a ``depmap_column``
            column; predictions are restricted to these genes.
        panels: Frozen nested k-label panels.
        methods: Method identifiers to evaluate (each is paired against
            ``baseline_method``; ``baseline_method`` itself may be included
            for a trivial zero-diff sanity row).
        k_schedule: k values to evaluate.
        baseline_method: The method every other method is compared against.
        calibrate: Pluggable k-shot calibrator.
        min_genes: Minimum pairwise-complete scored genes per panel.

    Returns:
        Tidy DataFrame with columns ``[model_id, method, k,
        method_mean_spearman, baseline_mean_spearman, paired_diff]``.
    """
    test_lines = manifest.loc[manifest["role"] == "test", "model_id"].tolist()
    if len(test_lines) != 9:
        _LOGGER.warning(
            "Expected 9 held-out test lines per the frozen contract; found %d.",
            len(test_lines),
        )
    slice_genes = set(slice_df["depmap_column"])

    rows: list[dict[str, object]] = []
    for model_id in test_lines:
        line_panels = panels[panels["model_id"] == model_id]
        line_preds = predictions[
            (predictions["model_id"] == model_id)
            & (predictions["depmap_column"].isin(slice_genes))
        ]
        baseline_preds = line_preds[line_preds["method"] == baseline_method]
        for k in k_schedule:
            baseline_mean = per_line_metric(
                baseline_preds,
                line_panels,
                baseline_method,
                k,
                calibrate,
                min_genes,
            )
            for method in methods:
                method_preds = line_preds[line_preds["method"] == method]
                method_mean = per_line_metric(
                    method_preds, line_panels, method, k, calibrate, min_genes
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "method": method,
                        "k": k,
                        "method_mean_spearman": method_mean,
                        "baseline_mean_spearman": baseline_mean,
                        "paired_diff": method_mean - baseline_mean,
                    }
                )
    return pd.DataFrame(rows)


def line_bootstrap_ci(
    line_means: np.ndarray | Sequence[float],
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Two-sided percentile bootstrap CI for the macro mean of line-level values.

    Args:
        line_means: Line-level values to resample with replacement (e.g. the
            9 per-line paired differences at one k).
        reps: Bootstrap resample count (frozen contract: 10000).
        seed: RNG seed (frozen contract: 20260725); a fresh generator is
            created from this seed on every call, so repeated calls with the
            same inputs are bit-identical.
        alpha: Two-sided miscoverage rate (0.05 -> 95% CI).

    Returns:
        ``(point_estimate, ci_lower, ci_upper)``: the observed unweighted
        macro mean of ``line_means``, and the
        ``[100*alpha/2, 100*(1-alpha/2)]`` percentiles of the bootstrap
        resample means.

    Raises:
        ValueError: If ``line_means`` is empty or contains non-finite
            values (a dropped line must be resolved upstream, not silently
            excluded from the bootstrap).
    """
    values = np.asarray(line_means, dtype=float)
    if values.size == 0:
        raise ValueError("line_means must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "line_means contains non-finite value(s); a dropped/NaN line "
            "must be resolved upstream, not silently excluded here."
        )

    point = float(np.mean(values))
    rng = np.random.default_rng(seed)
    n = values.size
    idx = rng.integers(0, n, size=(reps, n))
    resample_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(resample_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def _validate_panel_label_rows(
    panels: pd.DataFrame, test_set: set[str], n_labels: int
) -> None:
    """Require exactly ``n_labels`` unique, well-formed label rows per panel.

    Args:
        panels: Frozen nested k-label panels.
        test_set: Held-out test-line ``model_id`` set.
        n_labels: Required label-row count per ``(model_id, panel)`` (Phase
            A contract: 50, ``label_order`` 1..50).

    Raises:
        EvaluationContractError: If any ``(model_id, panel)`` has a label
            row count other than ``n_labels``, a duplicate ``label_order``,
            a duplicate ``depmap_column``, or a ``label_order`` set other
            than ``{1, ..., n_labels}`` (e.g. shifted to 51..100, which would
            pass the count/dup checks but make every k-selection empty).
    """
    line_panels = panels[panels["model_id"].isin(test_set)]
    grouped = line_panels.groupby(["model_id", "panel"])

    counts = grouped.size()
    bad_counts = counts[counts != n_labels]
    if not bad_counts.empty:
        raise EvaluationContractError(
            f"Expected {n_labels} label rows per (line, panel); found "
            f"{bad_counts.to_dict()}."
        )

    dup_order = grouped["label_order"].apply(lambda s: s.duplicated().any())
    bad_order = sorted(dup_order[dup_order].index.tolist())
    if bad_order:
        raise EvaluationContractError(
            f"Duplicate label_order within (line, panel): {bad_order}."
        )

    dup_gene = grouped["depmap_column"].apply(lambda s: s.duplicated().any())
    bad_gene = sorted(dup_gene[dup_gene].index.tolist())
    if bad_gene:
        raise EvaluationContractError(
            f"Duplicate depmap_column within (line, panel): {bad_gene}."
        )

    expected_orders = set(range(1, n_labels + 1))
    bad_range = grouped["label_order"].apply(
        lambda s: set(s.tolist()) != expected_orders
    )
    bad_range = sorted(bad_range[bad_range].index.tolist())
    if bad_range:
        raise EvaluationContractError(
            f"label_order must be exactly 1..{n_labels} within (line, "
            f"panel); violated for: {bad_range}."
        )


def _validate_simple_coverage(
    line_preds: pd.DataFrame,
    model_id: str,
    method: str,
    slice_genes: set[str],
    max_listed: int,
) -> None:
    """Require full slice-gene coverage for one (method, line), simple schema.

    Args:
        line_preds: This method's predictions for one test line.
        model_id: The test line identifier (for the error message).
        method: The method identifier (for the error message).
        slice_genes: The full differential-essential slice gene set.
        max_listed: Maximum number of missing genes to name in the error.

    Raises:
        EvaluationContractError: If any slice gene has no prediction row.
    """
    missing = slice_genes - set(line_preds["depmap_column"])
    if missing:
        raise EvaluationContractError(
            f"method={method!r} line={model_id!r} missing {len(missing)} "
            f"slice gene(s), e.g. {sorted(missing)[:max_listed]}."
        )


def _missing_panel_k_cells(
    line_preds: pd.DataFrame,
    required_panels: set[object],
    k_schedule: Sequence[int],
) -> list[tuple[object, int]]:
    """Return required (panel, k) cells absent from ``line_preds``.

    Args:
        line_preds: This method's panel-aware predictions for one test
            line, with columns ``[panel, k]``.
        required_panels: The frozen panel-id set for this line (every
            registered panel, not just the ones observed in predictions).
        k_schedule: The k values every registered panel must be scored at.

    Returns:
        Sorted list of ``(panel, k)`` cells in ``required_panels x
        k_schedule`` that have no row in ``line_preds``.
    """
    required = {(panel_id, int(k)) for panel_id in required_panels for k in k_schedule}
    present = {
        (panel_id, int(k))
        for panel_id, k in line_preds[["panel", "k"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    return sorted(required - present, key=lambda cell: (str(cell[0]), cell[1]))


def _validate_panel_aware_coverage(
    line_preds: pd.DataFrame,
    line_panels: pd.DataFrame,
    model_id: str,
    method: str,
    slice_genes: set[str],
    k_schedule: Sequence[int],
    max_listed: int,
) -> None:
    """Require every (panel, k) cell and its full SCORED-gene coverage.

    A prior version only checked (panel, k) combinations PRESENT in
    ``line_preds``; if an entire registered panel or k value were absent,
    validation passed silently and ``per_line_metric`` would nanmean-average
    over fewer than the frozen 20 panels. This requires the COMPLETE grid:
    every panel registered for this line in ``line_panels``, crossed with
    every k in ``k_schedule``.

    Args:
        line_preds: This method's panel-aware predictions for one test
            line, with columns ``[panel, k, depmap_column]``.
        line_panels: The frozen k-label panels for the same line.
        model_id: The test line identifier (for the error message).
        method: The method identifier (for the error message).
        slice_genes: The full differential-essential slice gene set.
        k_schedule: The k values every registered panel must be scored at.
        max_listed: Maximum number of missing genes/cells to name in one
            error.

    Raises:
        EvaluationContractError: If any (panel, k) cell required by the
            frozen panel set x ``k_schedule`` grid is entirely absent from
            ``line_preds``, or if a present cell is missing predictions for
            a scored gene (a slice gene not among that panel's first-``k``
            labels).
    """
    required_panels = set(line_panels["panel"].unique())
    missing_cells = _missing_panel_k_cells(line_preds, required_panels, k_schedule)
    if missing_cells:
        raise EvaluationContractError(
            f"method={method!r} line={model_id!r} missing {len(missing_cells)} "
            f"(panel, k) cell(s) required by k_schedule={list(k_schedule)}, "
            f"e.g. {missing_cells[:max_listed]}."
        )

    for panel_id in required_panels:
        panel_rows = line_panels[line_panels["panel"] == panel_id]
        for k in k_schedule:
            scored_genes = slice_genes - _label_genes_for_panel(panel_rows, k)
            slice_rows = line_preds[
                (line_preds["panel"] == panel_id) & (line_preds["k"] == k)
            ]
            missing = scored_genes - set(slice_rows["depmap_column"])
            if missing:
                raise EvaluationContractError(
                    f"method={method!r} line={model_id!r} panel={panel_id} "
                    f"k={k} missing {len(missing)} scored gene(s), e.g. "
                    f"{sorted(missing)[:max_listed]}."
                )


def _validate_slice_coverage(
    predictions: pd.DataFrame,
    panels: pd.DataFrame,
    slice_genes: set[str],
    test_set: set[str],
    methods: set[str],
    k_schedule: Sequence[int] = K_SCHEDULE,
    max_listed: int = 20,
) -> None:
    """Require full (scored-)slice-gene coverage for every method x line.

    Args:
        predictions: Tidy long-format predictions, already known to have at
            least one row per (method, line).
        panels: Frozen nested k-label panels.
        slice_genes: The full differential-essential slice gene set.
        test_set: Held-out test-line ``model_id`` set.
        methods: Method identifiers to validate (candidates + baseline).
        k_schedule: k values every panel-aware method x line must cover for
            every registered panel (frozen contract default: ``K_SCHEDULE``).
        max_listed: Maximum number of missing genes to name in one error.

    Raises:
        EvaluationContractError: See :func:`_validate_simple_coverage` and
            :func:`_validate_panel_aware_coverage`.
    """
    for method in methods:
        method_preds = predictions[
            (predictions["method"] == method) & (predictions["model_id"].isin(test_set))
        ]
        panel_aware = _schema_is_panel_aware(method_preds)
        for model_id in sorted(test_set):
            line_preds = method_preds[method_preds["model_id"] == model_id]
            if panel_aware:
                line_panels = panels[panels["model_id"] == model_id]
                _validate_panel_aware_coverage(
                    line_preds,
                    line_panels,
                    model_id,
                    method,
                    slice_genes,
                    k_schedule,
                    max_listed,
                )
            else:
                _validate_simple_coverage(
                    line_preds, model_id, method, slice_genes, max_listed
                )


def _validate_no_duplicate_prediction_keys(
    predictions: pd.DataFrame,
    slice_genes: set[str],
    test_set: set[str],
    methods: set[str],
    max_listed: int = 20,
) -> None:
    """Reject duplicate prediction keys among the rows that will be scored.

    ``per_line_metric`` (via ``drop_duplicates(...)``) keeps the first row
    of a repeated key, so a duplicate key with a different ``base_pred``
    would otherwise silently depend on row order rather than raising. This
    makes duplication a hard contract violation instead, restricted to rows
    actually in scope for scoring: scored methods x test lines x slice
    genes.

    Args:
        predictions: Tidy long-format predictions.
        slice_genes: The full differential-essential slice gene set.
        test_set: Held-out test-line ``model_id`` set.
        methods: Method identifiers to be scored (candidates + baseline).
        max_listed: Maximum number of duplicate keys to name in one error.

    Raises:
        EvaluationContractError: If any ``(model_id, method, depmap_column)``
            key (simple schema) or ``(model_id, method, panel, k,
            depmap_column)`` key (panel-aware schema) repeats.
    """
    scoped = predictions[
        predictions["method"].isin(methods)
        & predictions["model_id"].isin(test_set)
        & predictions["depmap_column"].isin(slice_genes)
    ]
    for method in sorted(methods):
        method_rows = scoped[scoped["method"] == method]
        if method_rows.empty:
            continue
        if _schema_is_panel_aware(method_rows):
            key_cols = ["model_id", "method", "panel", "k", "depmap_column"]
        else:
            key_cols = ["model_id", "method", "depmap_column"]
        dup_mask = method_rows.duplicated(subset=key_cols, keep=False)
        if dup_mask.any():
            dup_keys = (
                method_rows.loc[dup_mask, key_cols].drop_duplicates().to_dict("records")
            )
            raise EvaluationContractError(
                f"method={method!r} has duplicate prediction key(s) "
                f"{key_cols}: {dup_keys[:max_listed]}."
            )


def _y_true_group_is_inconsistent(y_true: pd.Series, tol: float) -> bool:
    """Flag a (line, gene) group whose y_true mixes NaN and finite values.

    A group is consistent if EVERY method reports the same finite value
    (within ``tol``), or EVERY method reports NaN; it is inconsistent if
    some methods are NaN and others finite (a silent way past the old
    ``nanmax - nanmin`` check, which ignored NaN and so reported zero
    spread), or if the finite values disagree by more than ``tol``.

    Args:
        y_true: The ``y_true`` values for one ``(model_id, depmap_column)``
            group, one entry per method.
        tol: Max allowed spread among the finite values.

    Returns:
        True if the group violates the consistency contract.
    """
    finite = y_true[np.isfinite(y_true)]
    if finite.empty:
        return False
    if finite.shape[0] != y_true.shape[0]:
        return True
    return bool((finite.max() - finite.min()) > tol)


def _validate_evaluation_inputs(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    slice_df: pd.DataFrame,
    panels: pd.DataFrame,
    methods: Sequence[str],
    baseline_method: str,
    k_schedule: Sequence[int] = K_SCHEDULE,
    n_panels: int = N_PANELS,
    n_labels: int = N_LABELS,
    y_true_tol: float = 1e-8,
    expected_test_lines: int | None = None,
) -> list[str]:
    """Enforce frozen-contract invariants before a formal gate verdict.

    Closes ways a silent evaluator could pass the gate on a different
    estimator than the registered one: a changed inferential line sample,
    an altered panel/label schedule, incomplete prediction coverage of the
    frozen gene slice (including, for panel-aware methods, the complete
    registered-panel x ``k_schedule`` grid), duplicate prediction keys that
    would make scoring depend on row order, and paired methods scored
    against inconsistent targets.

    Args:
        predictions: Tidy long-format predictions.
        manifest: Cell-line manifest with a ``role`` column.
        slice_df: Differential-essential slice with a ``depmap_column``
            column.
        panels: Frozen nested k-label panels.
        methods: Method identifiers to be scored.
        baseline_method: Paired-difference baseline (also must be covered).
        k_schedule: k values every panel-aware method x line must cover for
            every registered panel (frozen contract default: ``K_SCHEDULE``).
        n_panels: Required panel count per held-out line.
        n_labels: Required label-row count per (line, panel) (Phase A
            contract: 50, ``label_order`` 1..50).
        y_true_tol: Max allowed spread of ``y_true`` across methods per gene
            (only among methods that report a finite value; see
            :func:`_y_true_group_is_inconsistent`). Checked only over rows
            actually in scope for scoring: ``methods`` (plus
            ``baseline_method``) x held-out test lines x slice genes, so an
            unselected method or an out-of-slice gene cannot fail a valid
            ``methods`` subset run.
        expected_test_lines: If given, the exact required test-line count.

    Returns:
        The validated list of held-out test-line ``model_id``s.

    Raises:
        EvaluationContractError: On duplicate/absent test lines, a per-line
            panel or label-row schedule that deviates from the frozen
            contract, prediction-coverage gaps against the full slice (or,
            for panel-aware methods, the full panel x k grid) for any scored
            method, duplicate prediction keys within a scored method, or
            per-gene ``y_true`` disagreement among evaluated rows.
    """
    test_lines = manifest.loc[manifest["role"] == "test", "model_id"].tolist()
    if not test_lines:
        raise EvaluationContractError("Manifest declares no held-out test lines.")
    if len(test_lines) != len(set(test_lines)):
        raise EvaluationContractError("Manifest repeats held-out test line(s).")
    if expected_test_lines is not None and len(test_lines) != expected_test_lines:
        raise EvaluationContractError(
            f"Expected {expected_test_lines} held-out test lines; "
            f"manifest declares {len(test_lines)}."
        )
    test_set = set(test_lines)

    counts = panels[panels["model_id"].isin(test_set)].groupby("model_id")["panel"]
    counts = counts.nunique()
    missing = test_set - set(counts.index)
    if missing:
        raise EvaluationContractError(f"No panels for test line(s): {sorted(missing)}.")
    bad = counts[counts != n_panels]
    if not bad.empty:
        raise EvaluationContractError(
            f"Expected {n_panels} panels per line; found {bad.to_dict()}."
        )
    _validate_panel_label_rows(panels, test_set, n_labels)

    all_methods = set(methods) | {baseline_method}
    for method in all_methods:
        covered = predictions.loc[predictions["method"] == method, "model_id"]
        gap = test_set - set(covered)
        if gap:
            raise EvaluationContractError(
                f"method={method!r} missing predictions for line(s): {sorted(gap)}."
            )
    slice_genes = set(slice_df["depmap_column"])
    _validate_slice_coverage(
        predictions, panels, slice_genes, test_set, all_methods, k_schedule=k_schedule
    )
    _validate_no_duplicate_prediction_keys(
        predictions, slice_genes, test_set, all_methods
    )

    # Restrict the y_true-consistency check to rows actually in scope for
    # scoring (Finding 5): grouping ALL prediction rows would wrongly reject
    # a valid `methods` subset run over an unselected method, or over a gene
    # outside the frozen slice, that happens to carry inconsistent/missing
    # y_true.
    evaluated = predictions[
        predictions["model_id"].isin(test_set)
        & predictions["method"].isin(all_methods)
        & predictions["depmap_column"].isin(slice_genes)
    ]
    spread_bad = evaluated.groupby(["model_id", "depmap_column"])["y_true"].apply(
        lambda s: _y_true_group_is_inconsistent(s, y_true_tol)
    )
    if spread_bad.any():
        n_bad = int(spread_bad.sum())
        raise EvaluationContractError(
            f"y_true disagrees across methods for {n_bad} (line, gene) pair(s); "
            "paired differences would compare against different targets."
        )
    return test_lines


def evaluate(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    slice_df: pd.DataFrame,
    panels: pd.DataFrame,
    methods: Sequence[str],
    k_schedule: Sequence[int] = K_SCHEDULE,
    baseline_method: str = DEFAULT_BASELINE_METHOD,
    primary_method: str = DEFAULT_PRIMARY_METHOD,
    gate_k: int = GATE_K,
    rho_min: float = RHO_MIN,
    calibrate: KShotCalibrator = affine_kshot_calibrate,
    min_genes: int = MIN_SCORED_GENES,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    strict: bool = True,
    expected_test_lines: int | None = None,
) -> dict[str, object]:
    """Run the full frozen evaluation contract and the k=10 gate.

    Args:
        predictions: Tidy long-format predictions with columns
            ``[model_id, depmap_column, method, base_pred, y_true]``.
        manifest: Cell-line manifest with a ``role`` column.
        slice_df: Differential-essential slice with a ``depmap_column``
            column.
        panels: Frozen nested k-label panels.
        methods: Method identifiers to evaluate.
        k_schedule: k values to evaluate (frozen contract: [0, 5, 10, 25, 50]).
        baseline_method: Paired-difference baseline (frozen contract:
            "copy_k562").
        primary_method: The candidate method the gate verdict is computed
            for (frozen contract: "tx1_3b_st").
        gate_k: The k at which the gate is evaluated (frozen contract: 10).
        rho_min: The gate threshold (frozen contract: 0.05).
        calibrate: Pluggable k-shot calibrator.
        min_genes: Minimum pairwise-complete scored genes per panel.
        bootstrap_reps: Bootstrap resample count.
        bootstrap_seed: Bootstrap RNG seed.

    Returns:
        Dict with keys:
            "per_line": DataFrame of per-line, per-method, per-k means and
                paired differences (see :func:`paired_differences`).
            "curve": DataFrame with one row per (method, k) giving the
                bootstrapped macro-mean paired difference and its 95% CI.
            "gate": Dict ``{rho_min, k, method, baseline_method, macro_mean,
                ci_lo, ci_hi, passes}`` for ``primary_method`` at ``gate_k``.
        strict: When True (the default, required for a formal verdict),
            validate the frozen-contract invariants and raise
            ``EvaluationContractError`` on any violation before scoring.
    """
    if strict:
        test_lines = _validate_evaluation_inputs(
            predictions,
            manifest,
            slice_df,
            panels,
            methods,
            baseline_method,
            k_schedule=k_schedule,
            expected_test_lines=expected_test_lines,
        )
    else:
        test_lines = manifest.loc[manifest["role"] == "test", "model_id"].tolist()
    per_line = paired_differences(
        predictions,
        manifest,
        slice_df,
        panels,
        methods,
        k_schedule,
        baseline_method,
        calibrate,
        min_genes,
    )

    curve_rows: list[dict[str, object]] = []
    for method in methods:
        for k in k_schedule:
            subset = per_line[(per_line["method"] == method) & (per_line["k"] == k)]
            point, lo, hi = line_bootstrap_ci(
                subset["paired_diff"].to_numpy(),
                reps=bootstrap_reps,
                seed=bootstrap_seed,
            )
            curve_rows.append(
                {
                    "method": method,
                    "k": k,
                    "n_lines": int(subset.shape[0]),
                    "macro_mean": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "passes_rho_min": bool(lo > rho_min),
                }
            )
    curve = pd.DataFrame(curve_rows)

    gate_rows = curve[(curve["method"] == primary_method) & (curve["k"] == gate_k)]
    if gate_rows.empty:
        raise ValueError(
            f"No curve row for primary_method={primary_method!r} at "
            f"gate_k={gate_k}; check the methods/k_schedule inputs."
        )
    gate_row = gate_rows.iloc[0]
    if strict and int(gate_row["n_lines"]) != len(test_lines):
        raise EvaluationContractError(
            f"Gate computed on {int(gate_row['n_lines'])} line(s) but "
            f"{len(test_lines)} held-out line(s) are registered."
        )
    gate = {
        "rho_min": rho_min,
        "k": gate_k,
        "method": primary_method,
        "baseline_method": baseline_method,
        "macro_mean": float(gate_row["macro_mean"]),
        "ci_lo": float(gate_row["ci_lo"]),
        "ci_hi": float(gate_row["ci_hi"]),
        "passes": bool(gate_row["ci_lo"] > rho_min),
    }
    return {"per_line": per_line, "curve": curve, "gate": gate}


#: Frozen Phase-A artifact filenames, mapped to their
#: ``phase_a_registration.json`` ``artifacts.*_sha256`` key.
_REGISTERED_ARTIFACT_FILES: dict[str, str] = {
    "cell_line_manifest.csv": "cell_line_manifest_sha256",
    "differentially_essential_slice.csv": "differential_slice_sha256",
    "k_label_panels.csv": "k_label_panels_sha256",
}

#: SHA-256 of ``results/phase_a_tx1_20260724/phase_a_registration.json``, the
#: frozen Phase-A registration itself. This is the out-of-band trust anchor:
#: without it, ``verify_artifact_hashes`` only checks that each artifact
#: matches the hash recorded BESIDE it, so a matched swap of an artifact and
#: its recorded hash in the registration file would pass (self-consistency,
#: not identity with the frozen registration).
FROZEN_REGISTRATION_SHA256: str = (
    "63fb8f2c9e749b9fcfb2b54d605aa4b71fd740faa8871069d9a39dec983092b9"
)


def verify_artifact_hashes(
    phase_a_dir: Path,
    expected_registration_sha256: str = FROZEN_REGISTRATION_SHA256,
) -> None:
    """Verify frozen Phase-A artifact bytes against the registered SHA-256s.

    A same-shaped but silently modified manifest/slice/panels file would
    otherwise pass every other check in this module and still produce a
    ``formal: true`` verdict. This closes that gap by recomputing each
    file's SHA-256 and comparing it to the value recorded at Phase-A freeze
    time in ``phase_a_registration.json``.

    That registration file is itself untrusted input: a matched swap of an
    artifact and its recorded hash would leave the registration
    self-consistent. So before trusting any hash it records, this first
    verifies the registration file's own bytes against
    ``expected_registration_sha256``, an out-of-band trust anchor pinned in
    this module.

    Args:
        phase_a_dir: Directory holding ``phase_a_registration.json`` and the
            frozen ``cell_line_manifest.csv``,
            ``differentially_essential_slice.csv``, and ``k_label_panels.csv``
            artifacts.
        expected_registration_sha256: The trusted SHA-256 of
            ``phase_a_registration.json`` itself (default:
            :data:`FROZEN_REGISTRATION_SHA256`). Overridable so tests can pin
            a synthetic digest instead of the real frozen one.

    Raises:
        EvaluationContractError: If ``phase_a_registration.json`` is
            missing or its own SHA-256 does not match
            ``expected_registration_sha256``, if it has no recorded hash for
            one of the three artifacts, if an artifact file is missing, or
            if a file's computed SHA-256 does not match its recorded hash.
    """
    phase_a_dir = Path(phase_a_dir)
    registration_path = phase_a_dir / "phase_a_registration.json"
    if not registration_path.exists():
        raise EvaluationContractError(
            f"Missing registration file: {registration_path}."
        )
    registration_bytes = registration_path.read_bytes()
    actual_registration_hash = hashlib.sha256(registration_bytes).hexdigest()
    if actual_registration_hash != expected_registration_sha256:
        raise EvaluationContractError(
            f"{registration_path} SHA-256 mismatch: expected "
            f"{expected_registration_sha256!r}, computed "
            f"{actual_registration_hash!r}. The frozen Phase-A registration "
            "itself has changed; its recorded artifact hashes cannot be "
            "trusted."
        )
    registration = json.loads(registration_bytes)
    artifacts = registration.get("artifacts", {})

    for filename, hash_key in _REGISTERED_ARTIFACT_FILES.items():
        expected = artifacts.get(hash_key)
        if not expected:
            raise EvaluationContractError(
                f"{registration_path} has no {hash_key!r} entry under 'artifacts'."
            )
        artifact_path = phase_a_dir / filename
        if not artifact_path.exists():
            raise EvaluationContractError(f"Missing frozen artifact: {artifact_path}.")
        actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual != expected:
            raise EvaluationContractError(
                f"{filename} SHA-256 mismatch: registered {expected!r}, "
                f"computed {actual!r}. The frozen Phase-A artifact has "
                "changed since registration."
            )
