from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.freeze_tx1_phase_a import (
    PhaseAConfig,
    _allocate_test_counts,
    define_differential_slice,
    materialize_k_label_panels,
    power_table,
    stratified_holdout,
)


def test_allocate_test_counts_preserves_training_line_per_lineage() -> None:
    counts = pd.Series(
        {
            "Lung": 12,
            "Bowel": 7,
            "Pancreas": 6,
            "CNS": 3,
            "Skin": 3,
            "Esophagus": 2,
            "Singleton": 1,
        }
    )
    allocation = _allocate_test_counts(counts, 9)
    assert sum(allocation.values()) == 9
    assert "Singleton" not in allocation
    assert all(allocation[key] < counts[key] for key in allocation)


def test_stratified_holdout_is_seeded_and_has_nine_test_lines() -> None:
    lineages = ["Lung"] * 12 + ["Bowel"] * 7 + ["Pancreas"] * 6
    lineages += ["CNS"] * 3 + ["Skin"] * 3 + ["Esophagus"] * 2
    lineages += ["Singleton"] * 5
    pool = pd.DataFrame(
        {
            "model_id": [f"ACH-{index:06d}" for index in range(38)],
            "lineage": lineages,
        }
    )
    config = PhaseAConfig()
    first, _ = stratified_holdout(pool, config)
    second, _ = stratified_holdout(pool, config)
    assert first.equals(second)
    assert (first["role"] == "test").sum() == 9


def test_differential_slice_uses_training_variance_and_prevalence() -> None:
    values = pd.DataFrame(
        {
            "DIFF (1)": [-1.2, -0.9, 0.1, 0.2, 0.4, 0.5],
            "COMMON (2)": [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7],
            "FLAT (3)": [-0.8, -0.7, 0.0, 0.0, 0.0, 0.0],
            "MISSING (4)": [-1.0, -0.8, 0.1, 0.2, 0.3, np.nan],
        },
        index=[f"ACH-{index:06d}" for index in range(6)],
    )
    config = PhaseAConfig(slice_variance_quantile=0.0)
    result = define_differential_slice(values, config)
    assert result["gene_symbol"].tolist() == ["DIFF", "FLAT"]


def test_k_label_panels_are_seeded_nested_and_disjoint_from_scoring() -> None:
    genes = [f"G{index}" for index in range(100)]
    config = PhaseAConfig(k_label_panels=2)
    first = materialize_k_label_panels(["ACH-TEST"], genes, config)
    second = materialize_k_label_panels(["ACH-TEST"], genes, config)
    assert first.equals(second)
    assert len(first) == 100
    panel = first[first["panel"] == 0]
    k10 = set(panel.loc[panel["label_order"] <= 10, "depmap_column"])
    k25 = set(panel.loc[panel["label_order"] <= 25, "depmap_column"])
    assert k10 < k25
    assert k10.isdisjoint(set(genes) - k10)


def test_power_table_fixes_rho_min_from_utility_then_checks_power() -> None:
    residuals = np.array([0.10, 0.15, 0.20, 0.25] * 10)
    config = PhaseAConfig(power_repetitions=20, bootstrap_repetitions=50)
    _, rho_min, alternative, power = power_table(residuals, config)
    assert rho_min == 0.05
    assert alternative == 0.15
    assert power >= 0.80
