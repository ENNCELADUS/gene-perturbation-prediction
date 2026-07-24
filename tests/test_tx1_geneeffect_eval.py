"""Unit tests for the frozen Phase-A GeneEffect evaluator, on synthetic data.

None of these tests touch real prediction files; every manifest, slice,
panel, and prediction table is built in-process so the suite is fully
self-contained and independent of ``results/phase_a_tx1_20260724``.
"""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from aivc_model.tx1_fewshot_calibration import make_predictions_long
from aivc_model.tx1_geneeffect_eval import (
    EvaluationContractError,
    affine_kshot_calibrate,
    evaluate,
    line_bootstrap_ci,
    load_manifest,
    load_panels,
    load_predictions,
    load_slice,
    panel_spearman,
    paired_differences,
    per_line_metric,
    verify_artifact_hashes,
)
from aivc_model.tx1_geneeffect_eval import _validate_slice_coverage

N_GENES = 150
N_PANELS = 20
N_LABELS = 50
SLICE_GENES = [f"GENE{i} ({i})" for i in range(N_GENES)]


def _build_panels(
    model_ids: list[str],
    slice_genes: list[str],
    seed: int,
    n_panels: int = N_PANELS,
    n_labels: int = N_LABELS,
) -> pd.DataFrame:
    """Build a synthetic k_label_panels.csv-shaped DataFrame."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for model_id in model_ids:
        for panel in range(n_panels):
            panel_seed = int(rng.integers(0, 2**31))
            order = rng.permutation(len(slice_genes))[:n_labels]
            for label_order, gene_idx in enumerate(order, start=1):
                rows.append(
                    {
                        "model_id": model_id,
                        "panel": panel,
                        "panel_seed": panel_seed,
                        "label_order": label_order,
                        "depmap_column": slice_genes[gene_idx],
                    }
                )
    return pd.DataFrame(rows)


class Fixture(NamedTuple):
    manifest: pd.DataFrame
    slice_df: pd.DataFrame
    panels: pd.DataFrame
    predictions: pd.DataFrame


def _build_fixture(seed: int = 20260701) -> Fixture:
    """Nine test lines + two train lines; tx1_3b_st strongly beats copy_k562."""
    rng = np.random.default_rng(seed)
    slice_df = pd.DataFrame({"depmap_column": SLICE_GENES})

    test_lines = [f"TEST_{i}" for i in range(9)]
    train_lines = ["TRAIN_0", "TRAIN_1"]
    manifest = pd.DataFrame(
        {
            "model_id": test_lines + train_lines,
            "role": ["test"] * len(test_lines) + ["train"] * len(train_lines),
        }
    )

    all_lines = test_lines + train_lines
    panels = _build_panels(all_lines, SLICE_GENES, seed=seed + 1)

    pred_rows: list[dict[str, object]] = []
    for line in all_lines:
        y_true = rng.normal(size=N_GENES)
        tx1_pred = y_true + 0.15 * rng.normal(size=N_GENES)
        copy_pred = rng.normal(size=N_GENES)
        for gene, yt, tp, cp in zip(SLICE_GENES, y_true, tx1_pred, copy_pred):
            pred_rows.append(
                {
                    "model_id": line,
                    "depmap_column": gene,
                    "method": "tx1_3b_st",
                    "base_pred": tp,
                    "y_true": yt,
                }
            )
            pred_rows.append(
                {
                    "model_id": line,
                    "depmap_column": gene,
                    "method": "copy_k562",
                    "base_pred": cp,
                    "y_true": yt,
                }
            )
    predictions = pd.DataFrame(pred_rows)
    return Fixture(manifest, slice_df, panels, predictions)


# --------------------------------------------------------------------------
# affine_kshot_calibrate
# --------------------------------------------------------------------------


def test_affine_calibrate_recovers_known_linear_relationship() -> None:
    rng = np.random.default_rng(1)
    base_pred = rng.normal(size=200)
    y_true = 2.0 * base_pred + 3.0
    label_mask = np.zeros(200, dtype=bool)
    label_mask[:20] = True

    adjusted = affine_kshot_calibrate(base_pred, y_true, label_mask)

    assert np.allclose(adjusted, y_true, atol=1e-8)


def test_affine_calibrate_k0_is_identity() -> None:
    rng = np.random.default_rng(2)
    base_pred = rng.normal(size=50)
    y_true = rng.normal(size=50)
    label_mask = np.zeros(50, dtype=bool)

    adjusted = affine_kshot_calibrate(base_pred, y_true, label_mask)

    assert np.array_equal(adjusted, base_pred)
    assert adjusted is not base_pred  # returns a copy, not an alias


def test_affine_calibrate_falls_back_to_identity_with_too_few_labels() -> None:
    base_pred = np.array([1.0, 2.0, 3.0, 4.0])
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    label_mask = np.array([True, False, False, False])  # only one label gene

    adjusted = affine_kshot_calibrate(base_pred, y_true, label_mask)

    assert np.array_equal(adjusted, base_pred)


# --------------------------------------------------------------------------
# panel_spearman
# --------------------------------------------------------------------------


def test_panel_spearman_excludes_label_genes() -> None:
    rng = np.random.default_rng(3)
    n = 150
    y_true = rng.normal(size=n)
    y_pred = y_true.copy()
    # Corrupt the first 60 "label" genes so including them would break rho.
    y_pred[:60] = rng.normal(size=60) * 100
    scored_mask = np.zeros(n, dtype=bool)
    scored_mask[60:] = True  # 90 scored genes, below the default min of 100

    rho_below_min = panel_spearman(y_true, y_pred, scored_mask, min_genes=100)
    assert np.isnan(rho_below_min)

    rho_above_min = panel_spearman(y_true, y_pred, scored_mask, min_genes=90)
    assert rho_above_min == pytest.approx(1.0)


def test_panel_spearman_min_genes_boundary_is_inclusive() -> None:
    rng = np.random.default_rng(4)
    n = 150
    y_true = rng.normal(size=n)
    y_pred = y_true.copy()
    scored_mask = np.zeros(n, dtype=bool)
    scored_mask[:100] = True  # exactly the minimum

    rho = panel_spearman(y_true, y_pred, scored_mask, min_genes=100)
    assert rho == pytest.approx(1.0)

    scored_mask[:100] = False
    scored_mask[:99] = True  # one short of the minimum
    rho_short = panel_spearman(y_true, y_pred, scored_mask, min_genes=100)
    assert np.isnan(rho_short)


def test_panel_spearman_drops_non_finite_pairs_before_counting() -> None:
    n = 150
    y_true = np.linspace(0, 1, n)
    y_pred = np.linspace(0, 1, n)
    y_pred[:60] = np.nan  # 60 genes have no prediction
    scored_mask = np.ones(n, dtype=bool)

    rho = panel_spearman(y_true, y_pred, scored_mask, min_genes=100)
    assert np.isnan(rho)  # only 90 finite pairs remain, below the minimum


# --------------------------------------------------------------------------
# per_line_metric
# --------------------------------------------------------------------------


def test_per_line_metric_averages_across_20_panels() -> None:
    rng = np.random.default_rng(7)
    y_true = rng.normal(size=N_GENES)
    base_pred = y_true + 0.3 * rng.normal(size=N_GENES)
    preds_for_line = pd.DataFrame(
        {"depmap_column": SLICE_GENES, "base_pred": base_pred, "y_true": y_true}
    )
    panels_for_line = _build_panels(["LINE_X"], SLICE_GENES, seed=99)

    k = 10
    result = per_line_metric(preds_for_line, panels_for_line, method="dummy", k=k)

    expected_rhos = []
    for panel_id in sorted(panels_for_line["panel"].unique()):
        panel_rows = panels_for_line[panels_for_line["panel"] == panel_id]
        label_genes = set(
            panel_rows.loc[panel_rows["label_order"] <= k, "depmap_column"]
        )
        label_mask = np.array([g in label_genes for g in SLICE_GENES])
        adjusted = affine_kshot_calibrate(base_pred, y_true, label_mask)
        scored_mask = ~label_mask
        rho = spearmanr(y_true[scored_mask], adjusted[scored_mask]).statistic
        expected_rhos.append(rho)

    assert len(expected_rhos) == N_PANELS
    assert len(set(np.round(expected_rhos, 6))) > 1  # panels genuinely differ
    assert result == pytest.approx(float(np.mean(expected_rhos)), abs=1e-9)


def test_per_line_metric_returns_nan_when_every_panel_is_dropped() -> None:
    rng = np.random.default_rng(8)
    y_true = rng.normal(size=N_GENES)
    base_pred = y_true.copy()
    preds_for_line = pd.DataFrame(
        {"depmap_column": SLICE_GENES, "base_pred": base_pred, "y_true": y_true}
    )
    panels_for_line = _build_panels(["LINE_Y"], SLICE_GENES, seed=100)

    # k=50 (of 150 genes) leaves 100 scored; require more than that so every
    # panel is under-powered and dropped.
    result = per_line_metric(
        preds_for_line, panels_for_line, method="dummy", k=50, min_genes=101
    )
    assert np.isnan(result)


def test_per_line_metric_is_panel_k_aware_and_measures_the_reranker() -> None:
    """Finding 1 regression: per_line_metric must not dedup away (panel, k).

    Phase E's ``make_predictions_long`` emits one already-adapted
    ``base_pred`` per (panel, gene, k); dropping the panel/k dimensions
    (the old ``drop_duplicates("depmap_column")`` behavior) collapses every
    k onto the first (panel 0, k=0) row, so the few-shot curve never
    actually measures the re-ranking adapter. Build a fixture where
    increasing k genuinely re-ranks genes (base_pred is dominated by
    feature directions irrelevant to y_true; the ridge adapter recovers the
    true direction from labeled genes), and assert the resulting per-k
    curve differs meaningfully across k -- i.e. is not flat / not silently
    reusing the k=0 value.
    """
    n_genes = 400
    n_features = 10
    rng = np.random.default_rng(2026)
    features = rng.normal(size=(n_genes, n_features))
    true_w = np.array([1.5, -1.0, 0.8] + [0.0] * (n_features - 3))
    y_true = features @ true_w + 0.2 * rng.normal(size=n_genes)
    base_pred = features[:, 3:].sum(axis=1) + 0.1 * features[:, 0]
    genes = [f"RGENE{i}" for i in range(n_genes)]

    panels_for_line = _build_panels(
        ["LINE_RERANK"], genes, seed=2027, n_panels=4, n_labels=50
    )
    k_schedule = [0, 5, 25]

    long_df = make_predictions_long(
        model_id="LINE_RERANK",
        genes=np.array(genes),
        features=features,
        base_pred=base_pred,
        y_true=y_true,
        panels_for_line=panels_for_line,
        k_schedule=k_schedule,
        method="tx1_3b_st",
    )
    assert {"panel", "k"} <= set(long_df.columns)  # sanity: panel-aware schema

    rho_by_k = {
        k: per_line_metric(
            long_df, panels_for_line, method="tx1_3b_st", k=k, min_genes=100
        )
        for k in k_schedule
    }

    # The curve must genuinely move with k (the reranker is being measured),
    # not stay pinned at the k=0 value as the pre-fix dedup bug would give.
    assert not any(np.isnan(v) for v in rho_by_k.values())
    assert rho_by_k[25] > rho_by_k[0] + 0.2
    assert rho_by_k[5] != pytest.approx(rho_by_k[0], abs=1e-9)


def test_per_line_metric_simple_schema_path_is_unaffected_by_panel_awareness() -> None:
    """Finding 1 regression: the pre-existing simple-schema path is unchanged.

    A predictions frame with no ``panel``/``k`` columns must still be
    scored via the pluggable ``calibrate`` seam applied once per panel
    (the original, pre-fix behavior), not the panel-aware IDENTITY path.
    """
    rng = np.random.default_rng(2028)
    y_true = rng.normal(size=N_GENES)
    base_pred = y_true + 0.3 * rng.normal(size=N_GENES)
    preds_for_line = pd.DataFrame(
        {"depmap_column": SLICE_GENES, "base_pred": base_pred, "y_true": y_true}
    )
    assert not ({"panel", "k"} <= set(preds_for_line.columns))
    panels_for_line = _build_panels(["LINE_SIMPLE"], SLICE_GENES, seed=2029)

    k = 10
    result = per_line_metric(preds_for_line, panels_for_line, method="dummy", k=k)

    expected_rhos = []
    for panel_id in sorted(panels_for_line["panel"].unique()):
        panel_rows = panels_for_line[panels_for_line["panel"] == panel_id]
        label_genes = set(
            panel_rows.loc[panel_rows["label_order"] <= k, "depmap_column"]
        )
        label_mask = np.array([g in label_genes for g in SLICE_GENES])
        adjusted = affine_kshot_calibrate(base_pred, y_true, label_mask)
        scored_mask = ~label_mask
        rho = spearmanr(y_true[scored_mask], adjusted[scored_mask]).statistic
        expected_rhos.append(rho)

    assert result == pytest.approx(float(np.mean(expected_rhos)), abs=1e-9)


def test_baseline_and_paired_diff_stay_finite_when_concatenated_with_panel_aware_method() -> (  # noqa: E501
    None
):
    """T2 code review regression: panel-awareness must be per-method, not
    per-column.

    ``_schema_is_panel_aware`` used to detect the Phase-E schema purely from
    column PRESENCE. When a panel-aware method (``tx1_3b_st``, emitted by
    ``make_predictions_long``) is concatenated in the same predictions table
    with a simple-schema baseline (``copy_k562``), pandas fills the
    baseline's ``panel``/``k`` cells with NaN rather than dropping the
    columns -- so the old check misclassified the baseline as panel-aware,
    every (panel, k) lookup for it came back empty, and its score (and the
    paired difference against it) became NaN. This asserts the baseline's
    per-line metric and the paired difference stay finite, and that the
    candidate's few-shot curve still reflects genuine re-ranking.
    """
    n_genes = 400
    n_features = 10
    rng = np.random.default_rng(4040)
    features = rng.normal(size=(n_genes, n_features))
    true_w = np.array([1.5, -1.0, 0.8] + [0.0] * (n_features - 3))
    y_true = features @ true_w + 0.2 * rng.normal(size=n_genes)
    base_pred = features[:, 3:].sum(axis=1) + 0.1 * features[:, 0]
    genes = [f"MGENE{i}" for i in range(n_genes)]
    model_id = "TEST_MIXED"

    manifest = pd.DataFrame({"model_id": [model_id], "role": ["test"]})
    slice_df = pd.DataFrame({"depmap_column": genes})
    panels_for_line = _build_panels(
        [model_id], genes, seed=4041, n_panels=4, n_labels=50
    )
    k_schedule = [0, 5, 25]

    tx1_long = make_predictions_long(
        model_id=model_id,
        genes=np.array(genes),
        features=features,
        base_pred=base_pred,
        y_true=y_true,
        panels_for_line=panels_for_line,
        k_schedule=k_schedule,
        method="tx1_3b_st",
    )

    copy_pred = rng.normal(size=n_genes)
    copy_simple = pd.DataFrame(
        {
            "model_id": model_id,
            "depmap_column": genes,
            "method": "copy_k562",
            "base_pred": copy_pred,
            "y_true": y_true,
        }
    )
    # Concatenating a panel-aware frame with a simple-schema frame is
    # exactly the bug trigger: the baseline's panel/k cells become NaN
    # rather than the columns being absent.
    predictions = pd.concat([tx1_long, copy_simple], ignore_index=True)
    assert predictions.loc[predictions["method"] == "copy_k562", "panel"].isna().all()
    assert predictions.loc[predictions["method"] == "copy_k562", "k"].isna().all()

    baseline_rows = predictions[predictions["method"] == "copy_k562"]
    baseline_mean = per_line_metric(
        baseline_rows, panels_for_line, method="copy_k562", k=5
    )
    assert np.isfinite(baseline_mean)

    per_line = paired_differences(
        predictions,
        manifest,
        slice_df,
        panels_for_line,
        methods=["tx1_3b_st"],
        k_schedule=k_schedule,
        baseline_method="copy_k562",
    )
    assert np.all(np.isfinite(per_line["baseline_mean_spearman"]))
    assert np.all(np.isfinite(per_line["method_mean_spearman"]))
    assert np.all(np.isfinite(per_line["paired_diff"]))

    # The few-shot curve for tx1_3b_st still reflects re-ranking: more
    # labels genuinely improves the held-out rank correlation.
    curve = per_line.set_index("k")["method_mean_spearman"]
    assert curve.loc[25] > curve.loc[0] + 0.2


def test_schema_is_panel_aware_raises_on_mixed_null_and_nonnull_rows() -> None:
    """A single method mixing panel-aware and simple-schema rows is malformed."""
    predictions = pd.DataFrame(
        {
            "model_id": ["LINE_A", "LINE_A"],
            "panel": [0, np.nan],
            "k": [5, np.nan],
            "depmap_column": ["G0", "G1"],
            "method": ["tx1_3b_st", "tx1_3b_st"],
            "base_pred": [0.1, 0.2],
            "y_true": [0.1, 0.2],
        }
    )
    with pytest.raises(EvaluationContractError, match="[Mm]ixed"):
        per_line_metric(
            predictions,
            _build_panels(["LINE_A"], ["G0", "G1"], seed=1, n_panels=1, n_labels=1),
            method="tx1_3b_st",
            k=0,
        )


# --------------------------------------------------------------------------
# line_bootstrap_ci
# --------------------------------------------------------------------------


def test_line_bootstrap_ci_is_deterministic_under_fixed_seed() -> None:
    values = np.array([0.1, 0.2, 0.15, 0.05, 0.3, 0.12, 0.18, 0.22, 0.09])
    result_a = line_bootstrap_ci(values, reps=500, seed=20260725)
    result_b = line_bootstrap_ci(values, reps=500, seed=20260725)
    assert result_a == result_b


def test_line_bootstrap_ci_matches_manual_percentile_computation() -> None:
    values = np.array([0.1, -0.05, 0.2, 0.0, 0.15])
    reps = 777
    seed = 4242
    point, lo, hi = line_bootstrap_ci(values, reps=reps, seed=seed, alpha=0.1)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    expected_means = values[idx].mean(axis=1)
    expected_lo, expected_hi = np.percentile(expected_means, [5, 95])

    assert point == pytest.approx(float(values.mean()))
    assert lo == pytest.approx(float(expected_lo))
    assert hi == pytest.approx(float(expected_hi))


def test_line_bootstrap_ci_constant_input_collapses_to_point() -> None:
    values = np.full(9, 0.5)
    point, lo, hi = line_bootstrap_ci(values, reps=200, seed=1)
    assert point == pytest.approx(0.5)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_line_bootstrap_ci_rejects_nan_input() -> None:
    with pytest.raises(ValueError):
        line_bootstrap_ci([0.1, float("nan"), 0.2])


def test_line_bootstrap_ci_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        line_bootstrap_ci(np.array([]))


# --------------------------------------------------------------------------
# evaluate: end-to-end gate behavior
# --------------------------------------------------------------------------


def test_evaluate_gate_passes_with_known_positive_advantage() -> None:
    fixture = _build_fixture()

    result = evaluate(
        predictions=fixture.predictions,
        manifest=fixture.manifest,
        slice_df=fixture.slice_df,
        panels=fixture.panels,
        methods=["tx1_3b_st", "copy_k562"],
        baseline_method="copy_k562",
        primary_method="tx1_3b_st",
    )

    gate = result["gate"]
    assert gate["k"] == 10
    assert gate["rho_min"] == pytest.approx(0.05)
    assert gate["ci_lo"] > 0.05
    assert gate["passes"] is True

    per_line = result["per_line"]
    subset = per_line[(per_line["method"] == "tx1_3b_st") & (per_line["k"] == 10)]
    assert subset.shape[0] == 9  # exactly the 9 test-role lines, train excluded

    curve = result["curve"]
    curve_row = curve[(curve["method"] == "tx1_3b_st") & (curve["k"] == 10)].iloc[0]
    assert curve_row["macro_mean"] == pytest.approx(subset["paired_diff"].mean())
    assert curve_row["n_lines"] == 9


def test_evaluate_gate_fails_for_null_case_method_equals_baseline() -> None:
    fixture = _build_fixture()

    result = evaluate(
        predictions=fixture.predictions,
        manifest=fixture.manifest,
        slice_df=fixture.slice_df,
        panels=fixture.panels,
        methods=["copy_k562"],
        baseline_method="copy_k562",
        primary_method="copy_k562",
    )

    gate = result["gate"]
    assert gate["macro_mean"] == pytest.approx(0.0)
    assert gate["ci_lo"] == pytest.approx(0.0)
    assert gate["ci_hi"] == pytest.approx(0.0)
    assert gate["passes"] is False


def test_evaluate_curve_covers_full_k_schedule() -> None:
    fixture = _build_fixture()

    result = evaluate(
        predictions=fixture.predictions,
        manifest=fixture.manifest,
        slice_df=fixture.slice_df,
        panels=fixture.panels,
        methods=["tx1_3b_st", "copy_k562"],
        baseline_method="copy_k562",
        primary_method="tx1_3b_st",
    )

    curve = result["curve"]
    observed_ks = sorted(curve.loc[curve["method"] == "tx1_3b_st", "k"].unique())
    assert observed_ks == [0, 5, 10, 25, 50]
    assert (curve["n_lines"] == 9).all()


# --------------------------------------------------------------------------
# load helpers
# --------------------------------------------------------------------------


def test_load_helpers_roundtrip_csv(tmp_path) -> None:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    slice_path = tmp_path / "differentially_essential_slice.csv"
    panels_path = tmp_path / "k_label_panels.csv"
    predictions_csv_path = tmp_path / "predictions.csv"
    predictions_parquet_path = tmp_path / "predictions.parquet"

    fixture = _build_fixture()
    fixture.manifest.to_csv(manifest_path, index=False)
    fixture.slice_df.to_csv(slice_path, index=False)
    fixture.panels.to_csv(panels_path, index=False)
    fixture.predictions.to_csv(predictions_csv_path, index=False)
    fixture.predictions.to_parquet(predictions_parquet_path, index=False)

    manifest = load_manifest(manifest_path)
    slice_df = load_slice(slice_path)
    panels = load_panels(panels_path)
    predictions_csv = load_predictions(predictions_csv_path)
    predictions_parquet = load_predictions(predictions_parquet_path)

    assert set(manifest["role"]) == {"test", "train"}
    assert len(slice_df) == N_GENES
    assert set(panels.columns) == {
        "model_id",
        "panel",
        "panel_seed",
        "label_order",
        "depmap_column",
    }
    assert len(predictions_csv) == len(fixture.predictions)
    assert len(predictions_parquet) == len(fixture.predictions)


# --------------------------------------------------------------------------
# evaluate: frozen-contract validation (the gate must refuse a wrong sample)
# --------------------------------------------------------------------------


def _evaluate_default(fixture: Fixture, **kwargs) -> dict:
    return evaluate(
        predictions=fixture.predictions,
        manifest=fixture.manifest,
        slice_df=fixture.slice_df,
        panels=fixture.panels,
        methods=["tx1_3b_st", "copy_k562"],
        baseline_method="copy_k562",
        primary_method="tx1_3b_st",
        **kwargs,
    )


def test_evaluate_rejects_missing_prediction_coverage() -> None:
    fixture = _build_fixture()
    preds = fixture.predictions
    drop = (preds["model_id"] == "TEST_3") & (preds["method"] == "tx1_3b_st")
    broken = fixture._replace(predictions=preds[~drop])
    with pytest.raises(EvaluationContractError, match="missing predictions"):
        _evaluate_default(broken)


def test_evaluate_rejects_incomplete_panel_set() -> None:
    fixture = _build_fixture()
    panels = fixture.panels
    drop = (panels["model_id"] == "TEST_2") & (panels["panel"] == 0)
    broken = fixture._replace(panels=panels[~drop])
    with pytest.raises(EvaluationContractError, match="panels per line"):
        _evaluate_default(broken)


def test_evaluate_rejects_inconsistent_y_true_across_methods() -> None:
    fixture = _build_fixture()
    preds = fixture.predictions.copy()
    mask = (
        (preds["model_id"] == "TEST_1")
        & (preds["method"] == "copy_k562")
        & (preds["depmap_column"] == SLICE_GENES[0])
    )
    preds.loc[mask, "y_true"] = preds.loc[mask, "y_true"] + 5.0
    broken = fixture._replace(predictions=preds)
    with pytest.raises(EvaluationContractError, match="y_true disagrees"):
        _evaluate_default(broken)


def test_evaluate_rejects_dropped_label_row() -> None:
    """Finding 2: a panel missing even one of its 50 frozen label rows."""
    fixture = _build_fixture()
    panels = fixture.panels
    drop = (
        (panels["model_id"] == "TEST_2")
        & (panels["panel"] == 0)
        & (panels["label_order"] == 1)
    )
    broken = fixture._replace(panels=panels[~drop])
    with pytest.raises(EvaluationContractError, match="label rows"):
        _evaluate_default(broken)


def test_evaluate_rejects_duplicate_label_order_within_panel() -> None:
    """Finding 2: two rows of the same panel sharing one label_order."""
    fixture = _build_fixture()
    panels = fixture.panels.copy()
    mask = (
        (panels["model_id"] == "TEST_2")
        & (panels["panel"] == 0)
        & (panels["label_order"] == 2)
    )
    panels.loc[mask, "label_order"] = 1  # duplicate of label_order 1
    broken = fixture._replace(panels=panels)
    with pytest.raises(EvaluationContractError, match="label_order"):
        _evaluate_default(broken)


def test_evaluate_rejects_duplicate_gene_within_panel() -> None:
    """Finding 2: two rows of the same panel sharing one depmap_column."""
    fixture = _build_fixture()
    panels = fixture.panels.copy()
    panel_mask = (panels["model_id"] == "TEST_2") & (panels["panel"] == 0)
    idx = panels.index[panel_mask]
    panels.loc[idx[1], "depmap_column"] = panels.loc[idx[0], "depmap_column"]
    broken = fixture._replace(panels=panels)
    with pytest.raises(EvaluationContractError, match="depmap_column"):
        _evaluate_default(broken)


def test_evaluate_rejects_partial_slice_coverage() -> None:
    """Finding 3: a method covering only part of the 589(here 150)-gene slice."""
    fixture = _build_fixture()
    preds = fixture.predictions
    drop = (
        (preds["model_id"] == "TEST_4")
        & (preds["method"] == "tx1_3b_st")
        & (preds["depmap_column"].isin(SLICE_GENES[:5]))
    )
    broken = fixture._replace(predictions=preds[~drop])
    with pytest.raises(EvaluationContractError, match="slice gene"):
        _evaluate_default(broken)


def test_evaluate_accepts_full_slice_coverage() -> None:
    """Finding 3: a method covering every slice gene for every line passes."""
    fixture = _build_fixture()
    result = _evaluate_default(fixture)
    assert "gate" in result


def test_validate_slice_coverage_rejects_missing_scored_gene_panel_aware() -> None:
    """Finding 3: the panel-aware branch also enforces full scored-gene coverage.

    Deliberately passes ``k_schedule=[1]`` (matching the single k value this
    fixture's predictions use): the grid-completeness check added for the
    T2 review's Finding 2 requires every (panel, k) cell in
    ``registered_panels x k_schedule`` to be present, and panel=0's only
    registered k here is 1, so this isolates the scored-gene-coverage check
    this test targets from that separate, now-mandatory grid check.
    """
    slice_genes = {"G0", "G1", "G2", "G3", "G4"}
    panels = pd.DataFrame(
        {
            "model_id": ["LINE_A", "LINE_A"],
            "panel": [0, 0],
            "label_order": [1, 2],
            "depmap_column": ["G0", "G1"],
        }
    )
    # (panel=0, k=1) scores slice_genes - {G0} = {G1, G2, G3, G4}; G4 is
    # missing from the predicted rows below.
    predictions = pd.DataFrame(
        {
            "model_id": ["LINE_A"] * 3,
            "panel": [0, 0, 0],
            "k": [1, 1, 1],
            "depmap_column": ["G1", "G2", "G3"],
            "method": ["tx1_3b_st"] * 3,
            "base_pred": [0.1, 0.2, 0.3],
            "y_true": [0.1, 0.2, 0.3],
        }
    )
    with pytest.raises(EvaluationContractError, match="scored gene"):
        _validate_slice_coverage(
            predictions, panels, slice_genes, {"LINE_A"}, {"tx1_3b_st"}, k_schedule=[1]
        )


def test_validate_slice_coverage_accepts_full_panel_aware_coverage() -> None:
    """Finding 3: full scored-gene coverage in the panel-aware schema passes.

    See the k_schedule=[1] note on the sibling rejection test above.
    """
    slice_genes = {"G0", "G1", "G2", "G3", "G4"}
    panels = pd.DataFrame(
        {
            "model_id": ["LINE_A", "LINE_A"],
            "panel": [0, 0],
            "label_order": [1, 2],
            "depmap_column": ["G0", "G1"],
        }
    )
    predictions = pd.DataFrame(
        {
            "model_id": ["LINE_A"] * 4,
            "panel": [0, 0, 0, 0],
            "k": [1, 1, 1, 1],
            "depmap_column": ["G1", "G2", "G3", "G4"],
            "method": ["tx1_3b_st"] * 4,
            "base_pred": [0.1, 0.2, 0.3, 0.4],
            "y_true": [0.1, 0.2, 0.3, 0.4],
        }
    )
    _validate_slice_coverage(  # must not raise
        predictions, panels, slice_genes, {"LINE_A"}, {"tx1_3b_st"}, k_schedule=[1]
    )


def test_validate_slice_coverage_rejects_missing_panel_k_cell() -> None:
    """T2 code review regression: an entirely absent (panel, k) cell.

    A prior version of panel-aware coverage validation only checked
    (panel, k) combinations PRESENT in the predictions; if a whole
    registered panel or k value were absent, validation passed silently and
    ``per_line_metric`` would average over fewer than the registered panel
    count. Dropping (panel=1, k=5) entirely (not just some of its genes)
    must now raise, naming the missing cell.
    """
    slice_genes = {"G0", "G1", "G2"}
    panels = pd.DataFrame(
        {
            "model_id": ["LINE_A", "LINE_A"],
            "panel": [0, 1],
            "label_order": [1, 1],
            "depmap_column": ["G0", "G0"],
        }
    )
    k_schedule = [0, 5]
    rows = [
        {
            "model_id": "LINE_A",
            "panel": panel_id,
            "k": k,
            "depmap_column": gene,
            "method": "tx1_3b_st",
            "base_pred": 0.0,
            "y_true": 0.0,
        }
        for panel_id in (0, 1)
        for k in k_schedule
        for gene in slice_genes
        if (panel_id, k) != (1, 5)  # drop this cell entirely
    ]
    predictions = pd.DataFrame(rows)
    with pytest.raises(EvaluationContractError, match=r"\(panel, k\) cell"):
        _validate_slice_coverage(
            predictions,
            panels,
            slice_genes,
            {"LINE_A"},
            {"tx1_3b_st"},
            k_schedule=k_schedule,
        )


def test_validate_slice_coverage_accepts_complete_panel_k_grid() -> None:
    """T2 code review regression: the complete panel x k_schedule grid passes."""
    slice_genes = {"G0", "G1", "G2"}
    panels = pd.DataFrame(
        {
            "model_id": ["LINE_A", "LINE_A"],
            "panel": [0, 1],
            "label_order": [1, 1],
            "depmap_column": ["G0", "G0"],
        }
    )
    k_schedule = [0, 5]
    rows = [
        {
            "model_id": "LINE_A",
            "panel": panel_id,
            "k": k,
            "depmap_column": gene,
            "method": "tx1_3b_st",
            "base_pred": 0.0,
            "y_true": 0.0,
        }
        for panel_id in (0, 1)
        for k in k_schedule
        for gene in slice_genes
    ]
    predictions = pd.DataFrame(rows)
    _validate_slice_coverage(  # must not raise
        predictions,
        panels,
        slice_genes,
        {"LINE_A"},
        {"tx1_3b_st"},
        k_schedule=k_schedule,
    )


def test_evaluate_rejects_mixed_nan_and_finite_y_true_across_methods() -> None:
    """Finding 4: one method NaN + another finite for the same (line, gene).

    The old ``nanmax - nanmin`` spread check ignores NaN entirely, so this
    case silently reported zero spread and passed; it must now raise.
    """
    fixture = _build_fixture()
    preds = fixture.predictions.copy()
    mask = (
        (preds["model_id"] == "TEST_1")
        & (preds["method"] == "copy_k562")
        & (preds["depmap_column"] == SLICE_GENES[0])
    )
    preds.loc[mask, "y_true"] = float("nan")
    broken = fixture._replace(predictions=preds)
    with pytest.raises(EvaluationContractError, match="y_true disagrees"):
        _evaluate_default(broken)


def test_evaluate_accepts_all_nan_y_true_for_a_gene() -> None:
    """Finding 4: a gene that is NaN across EVERY method is consistent."""
    fixture = _build_fixture()
    preds = fixture.predictions.copy()
    mask = preds["depmap_column"] == SLICE_GENES[0]
    preds.loc[mask, "y_true"] = float("nan")
    broken = fixture._replace(predictions=preds)
    result = _evaluate_default(broken)
    assert "gate" in result


def test_evaluate_accepts_all_equal_finite_y_true() -> None:
    """Finding 4: identical finite y_true across methods remains accepted."""
    fixture = _build_fixture()  # y_true is shared by construction per gene/line
    result = _evaluate_default(fixture)
    assert "gate" in result


def test_evaluate_rejects_duplicate_test_line() -> None:
    fixture = _build_fixture()
    dup = fixture.manifest[fixture.manifest["model_id"] == "TEST_0"]
    manifest = pd.concat([fixture.manifest, dup], ignore_index=True)
    broken = fixture._replace(manifest=manifest)
    with pytest.raises(EvaluationContractError, match="repeats"):
        _evaluate_default(broken)


def test_evaluate_rejects_manifest_with_wrong_expected_count() -> None:
    fixture = _build_fixture()
    manifest = fixture.manifest[fixture.manifest["model_id"] != "TEST_8"]
    preds = fixture.predictions[fixture.predictions["model_id"] != "TEST_8"]
    panels = fixture.panels[fixture.panels["model_id"] != "TEST_8"]
    broken = Fixture(manifest, fixture.slice_df, panels, preds)
    # A self-consistent 8-line sample validates without a pinned count...
    _evaluate_default(broken)
    # ...but a formal run pinned to the registered 9 rejects it.
    with pytest.raises(EvaluationContractError, match="Expected 9"):
        _evaluate_default(broken, expected_test_lines=9)


def test_evaluate_strict_false_bypasses_contract_validation() -> None:
    fixture = _build_fixture()
    preds = fixture.predictions.copy()
    mask = (
        (preds["model_id"] == "TEST_1")
        & (preds["method"] == "copy_k562")
        & (preds["depmap_column"] == SLICE_GENES[0])
    )
    preds.loc[mask, "y_true"] = preds.loc[mask, "y_true"] + 5.0
    broken = fixture._replace(predictions=preds)
    with pytest.raises(EvaluationContractError):
        _evaluate_default(broken)  # strict default catches it
    result = _evaluate_default(broken, strict=False)  # bypass for diagnostics
    assert "gate" in result


# --------------------------------------------------------------------------
# verify_artifact_hashes
# --------------------------------------------------------------------------


def _write_phase_a_dir(tmp_path) -> tuple:
    """Write a tiny phase_a dir (manifest/slice/panels) + matching registration.

    Returns:
        ``(phase_a_dir, manifest_path, slice_path, panels_path)``.
    """
    phase_a_dir = tmp_path / "phase_a"
    phase_a_dir.mkdir()
    manifest_path = phase_a_dir / "cell_line_manifest.csv"
    slice_path = phase_a_dir / "differentially_essential_slice.csv"
    panels_path = phase_a_dir / "k_label_panels.csv"
    manifest_path.write_text("model_id,role\nTEST_0,test\n")
    slice_path.write_text("depmap_column\nG0\nG1\n")
    panels_path.write_text("model_id,panel,label_order,depmap_column\nTEST_0,0,1,G0\n")

    registration = {
        "artifacts": {
            "cell_line_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "differential_slice_sha256": hashlib.sha256(
                slice_path.read_bytes()
            ).hexdigest(),
            "k_label_panels_sha256": hashlib.sha256(
                panels_path.read_bytes()
            ).hexdigest(),
        }
    }
    (phase_a_dir / "phase_a_registration.json").write_text(json.dumps(registration))
    return phase_a_dir, manifest_path, slice_path, panels_path


def test_verify_artifact_hashes_passes_for_matching_files(tmp_path) -> None:
    phase_a_dir, *_ = _write_phase_a_dir(tmp_path)
    verify_artifact_hashes(phase_a_dir)  # must not raise


def test_verify_artifact_hashes_rejects_modified_file(tmp_path) -> None:
    """Finding 3: a same-shaped but modified artifact must fail verification."""
    phase_a_dir, manifest_path, _, _ = _write_phase_a_dir(tmp_path)
    manifest_path.write_text("model_id,role\nTEST_0,test\nTEST_1,test\n")  # tamper

    with pytest.raises(EvaluationContractError, match="SHA-256 mismatch"):
        verify_artifact_hashes(phase_a_dir)


def test_verify_artifact_hashes_rejects_missing_registration_file(tmp_path) -> None:
    phase_a_dir = tmp_path / "empty_phase_a"
    phase_a_dir.mkdir()
    with pytest.raises(EvaluationContractError, match="registration"):
        verify_artifact_hashes(phase_a_dir)


def test_verify_artifact_hashes_rejects_missing_hash_entry(tmp_path) -> None:
    phase_a_dir, manifest_path, slice_path, panels_path = _write_phase_a_dir(tmp_path)
    registration = {
        "artifacts": {
            "cell_line_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            # differential_slice_sha256 missing entirely
            "k_label_panels_sha256": hashlib.sha256(
                panels_path.read_bytes()
            ).hexdigest(),
        }
    }
    (phase_a_dir / "phase_a_registration.json").write_text(json.dumps(registration))

    with pytest.raises(EvaluationContractError, match="differential_slice_sha256"):
        verify_artifact_hashes(phase_a_dir)


def test_verify_artifact_hashes_rejects_missing_artifact_file(tmp_path) -> None:
    phase_a_dir, manifest_path, slice_path, panels_path = _write_phase_a_dir(tmp_path)
    panels_path.unlink()

    with pytest.raises(EvaluationContractError, match="Missing frozen artifact"):
        verify_artifact_hashes(phase_a_dir)
