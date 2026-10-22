"""PIE reproduction's comparison: released against reproduced metric tables, on
synthetic tables shaped like PIE's (the module imports without PIE installed)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.experiments.stage_one import pie_reproduction as reproduction

METRICS = ["binary_auprc", "spearman_lfc"]


def metrics_table(rows: list[tuple[str, int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["context", "n_rows", *METRICS])


def granular_table(rows: list[tuple[str, str, int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["context", "perturbation", "n_rows", *METRICS])


RELEASED = metrics_table(
    [("hepg2", 3, 0.40, 0.60), ("k562", 2, 0.30, 0.50), ("all", 5, 0.35, 0.55)]
)


def test_compare_metrics_differences_per_context_and_metric():
    reproduced = metrics_table(
        [("k562", 2, 0.25, 0.50), ("all", 5, 0.33, 0.56), ("hepg2", 3, 0.41, 0.60)]
    )
    table = reproduction.compare_metrics(RELEASED, reproduced)
    assert list(table.columns) == [
        "context",
        "metric",
        "released",
        "reproduced",
        "difference",
    ]
    got = table.set_index(["context", "metric"])["difference"]
    expected = {
        ("hepg2", "binary_auprc"): 0.01,
        ("hepg2", "spearman_lfc"): 0.0,
        ("k562", "binary_auprc"): -0.05,
        ("k562", "spearman_lfc"): 0.0,
        ("all", "binary_auprc"): -0.02,
        ("all", "spearman_lfc"): 0.01,
    }
    assert len(got) == len(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value)


def test_compare_metrics_raises_on_a_missing_context():
    reproduced = RELEASED[RELEASED["context"] != "k562"]
    with pytest.raises(ValueError, match="keys differ"):
        reproduction.compare_metrics(RELEASED, reproduced)


def test_compare_metrics_raises_on_differing_n_rows():
    reproduced = RELEASED.assign(n_rows=[3, 1, 5])
    with pytest.raises(ValueError, match="n_rows"):
        reproduction.compare_metrics(RELEASED, reproduced)


GRANULAR = granular_table(
    [
        ("hepg2", "AAR2", 1, 1.0, 0.9),
        ("hepg2", "NA", 1, 0.2, float("nan")),
        ("k562", "AAR2", 1, 0.5, 0.4),
        ("k562", "BRCA1", 1, float("nan"), float("nan")),
    ]
)


def test_compare_granular_raises_on_differing_pairs():
    reproduced = GRANULAR.assign(perturbation=["AAR2", "NA", "AAR2", "BRCA2"])
    with pytest.raises(ValueError, match="keys differ"):
        reproduction.compare_granular(GRANULAR, reproduced)


def test_compare_granular_maxima_means_and_nan_mismatch():
    reproduced = granular_table(
        [
            ("k562", "AAR2", 1, 0.5, 0.1),
            ("hepg2", "AAR2", 1, 0.9, 0.9),
            ("hepg2", "NA", 1, 0.5, 0.3),
            ("k562", "BRCA1", 1, float("nan"), float("nan")),
        ]
    )
    table = reproduction.compare_granular(GRANULAR, reproduced).set_index("metric")
    # binary_auprc: both defined on three pairs, |differences| 0.1, 0.3, 0.
    assert table.loc["binary_auprc", "n_pairs"] == 3
    assert table.loc["binary_auprc", "nan_mismatch"] == 0
    assert table.loc["binary_auprc", "max_abs_difference"] == pytest.approx(0.3)
    assert table.loc["binary_auprc", "mean_abs_difference"] == pytest.approx(0.4 / 3)
    # spearman_lfc: (hepg2, NA) is undefined only in the released table.
    assert table.loc["spearman_lfc", "n_pairs"] == 2
    assert table.loc["spearman_lfc", "nan_mismatch"] == 1
    assert table.loc["spearman_lfc", "max_abs_difference"] == pytest.approx(0.3)
    assert table.loc["spearman_lfc", "mean_abs_difference"] == pytest.approx(0.15)


def test_compare_granular_raises_on_differing_n_rows():
    with pytest.raises(ValueError, match="n_rows"):
        reproduction.compare_granular(GRANULAR, GRANULAR.assign(n_rows=[1, 2, 1, 1]))


def test_read_table_keeps_gene_names_and_reads_empty_fields_as_nan(tmp_path):
    path = tmp_path / "granular_best_auprc.csv"
    GRANULAR.to_csv(path, index=False)
    table = reproduction.read_table(path)
    assert list(table["perturbation"]) == ["AAR2", "NA", "AAR2", "BRCA1"]
    assert math.isnan(table.loc[1, "spearman_lfc"])
    assert reproduction.compare_granular(GRANULAR, table)["nan_mismatch"].sum() == 0


def test_fold_summary_is_the_mean_and_sample_sd_of_the_all_rows():
    values = {"a": (0.2, 0.6), "b": (0.4, 0.7), "c": (0.6, 0.8)}
    tables = {
        fold: metrics_table([(fold, 4, 0.0, 0.0), ("all", 4, auprc, lfc)])
        for fold, (auprc, lfc) in values.items()
    }
    summary = reproduction.fold_summary(tables).set_index("metric")
    # binary_auprc: mean 0.4, deviations ±0.2 and 0, sample variance 0.08 / 2.
    assert summary.loc["binary_auprc", "mean"] == pytest.approx(0.4)
    assert summary.loc["binary_auprc", "sd"] == pytest.approx(np.sqrt(0.04))
    assert summary.loc["spearman_lfc", "mean"] == pytest.approx(0.7)
    assert summary.loc["spearman_lfc", "sd"] == pytest.approx(np.sqrt(0.01))
    assert (summary["n_folds"] == 3).all()
