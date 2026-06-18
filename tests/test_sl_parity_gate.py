# tests/test_sl_parity_gate.py
"""Parity gate (A/B/C unchanged) and end-to-end smoke for selectivity mode."""

from __future__ import annotations

import pandas as pd

from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.evaluate import run_cv


def test_selectivity_mode_preserves_abc_parity(synthetic_selectivity_fixture, tmp_path):
    fx = synthetic_selectivity_fixture
    common = dict(
        input_csv=fx["benchmark_csv"], folds=(0, 1), ranking_k=(2, 5), seed=17
    )
    base = run_cv(SLBaselineConfig(output_dir=tmp_path / "base", **common))
    sel = run_cv(
        SLBaselineConfig(
            output_dir=tmp_path / "sel", depmap_dir=fx["depmap_dir"], **common
        )
    )
    # A/B/C rows on the full_universe slice must match bit-for-bit.
    keys = ["split_type", "model", "metric"]
    base_abc = (
        base[base["model"].isin(["A", "B", "C"])].set_index(keys)["mean"].sort_index()
    )
    sel_full = sel[sel["slice"] == "full_universe"] if "slice" in sel.columns else sel
    sel_abc = (
        sel_full[sel_full["model"].isin(["A", "B", "C"])]
        .set_index(keys)["mean"]
        .sort_index()
    )
    pd.testing.assert_series_equal(base_abc, sel_abc, check_exact=False, atol=1e-9)


def test_selectivity_mode_emits_xcl_and_slices(synthetic_selectivity_fixture, tmp_path):
    fx = synthetic_selectivity_fixture
    out = tmp_path / "run"
    run_cv(
        SLBaselineConfig(
            input_csv=fx["benchmark_csv"],
            output_dir=out,
            depmap_dir=fx["depmap_dir"],
            folds=(0, 1),
            ranking_k=(2, 5),
        )
    )
    fold_metrics = pd.read_csv(out / "fold_metrics.csv")
    assert {"A", "B", "C", "A_xcl", "B_xcl"}.issubset(set(fold_metrics["model"]))
    xcl = fold_metrics[fold_metrics["model"] == "A_xcl"]
    assert "full_universe" in set(xcl["slice"])
    # at least one diagnostic slice present (data permitting)
    assert {"non_pan_essential", "covered_pairs"} & set(xcl["slice"])


def test_selectivity_rejects_non_rand_negatives(
    synthetic_selectivity_fixture, tmp_path
):
    fx = synthetic_selectivity_fixture
    frame = pd.read_csv(fx["benchmark_csv"])
    frame.loc[0, "negative_sampling_method"] = "Dep"
    bad = tmp_path / "bad.csv"
    frame.to_csv(bad, index=False)
    import pytest

    with pytest.raises(ValueError, match="Rand"):
        run_cv(
            SLBaselineConfig(
                input_csv=bad,
                output_dir=tmp_path / "x",
                depmap_dir=fx["depmap_dir"],
                folds=(0,),
                ranking_k=(2,),
            )
        )
