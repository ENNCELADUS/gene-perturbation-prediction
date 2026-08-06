"""Tests for train-line-only gene-mean residual targets."""

import logging

import numpy as np
import pandas as pd
import pytest

from aivc_model.residual_target import (
    ResidualTargets,
    build_residual_targets,
    fit_gene_means,
    to_matrix,
)


def _labels(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["model_id", "gene_symbol", "gene_effect"])


def test_gene_mean_is_fit_from_train_lines_only_not_the_full_dataset() -> None:
    """The leakage property: mu_hat_g must never see a held-out line's label.

    GENE1's held-out lines (L4, L5) carry values two orders of magnitude
    larger than the train lines. If the held-out labels leaked into the
    mean, gene_mean would move far off the train-only mean. It must not.
    """
    labels = _labels(
        [
            {"model_id": "L1", "gene_symbol": "GENE1", "gene_effect": 10.0},
            {"model_id": "L2", "gene_symbol": "GENE1", "gene_effect": 20.0},
            {"model_id": "L3", "gene_symbol": "GENE1", "gene_effect": 30.0},
            {"model_id": "L4", "gene_symbol": "GENE1", "gene_effect": 1000.0},
            {"model_id": "L5", "gene_symbol": "GENE1", "gene_effect": 2000.0},
        ]
    )
    train_lines = ("L1", "L2", "L3")

    hand_train_mean = (10.0 + 20.0 + 30.0) / 3.0
    hand_all_mean = (10.0 + 20.0 + 30.0 + 1000.0 + 2000.0) / 5.0

    gene_mean = fit_gene_means(labels, train_lines, min_lines=3)

    assert gene_mean["GENE1"] == hand_train_mean
    assert gene_mean["GENE1"] != hand_all_mean

    # build_residual_targets must apply this same train-fit mean to the
    # held-out rows too, not a mean recomputed over all lines.
    targets = build_residual_targets(labels, train_lines)
    assert targets.gene_mean["GENE1"] == hand_train_mean
    held_out_row = targets.long.loc[targets.long["model_id"] == "L4"].iloc[0]
    assert held_out_row["gene_mean"] == hand_train_mean
    assert held_out_row["residual"] == 1000.0 - hand_train_mean


def test_min_lines_drops_under_observed_genes_with_correct_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    labels = _labels(
        [
            {"model_id": "L1", "gene_symbol": "ENOUGH", "gene_effect": 1.0},
            {"model_id": "L2", "gene_symbol": "ENOUGH", "gene_effect": 2.0},
            {"model_id": "L3", "gene_symbol": "ENOUGH", "gene_effect": 3.0},
            {"model_id": "L4", "gene_symbol": "ENOUGH", "gene_effect": 4.0},
            {"model_id": "L1", "gene_symbol": "SPARSE", "gene_effect": 5.0},
            {"model_id": "L2", "gene_symbol": "SPARSE", "gene_effect": np.nan},
            {"model_id": "L3", "gene_symbol": "SPARSE", "gene_effect": np.nan},
            {"model_id": "L4", "gene_symbol": "SPARSE", "gene_effect": 6.0},
        ]
    )
    train_lines = ("L1", "L2", "L3")

    with caplog.at_level(logging.WARNING):
        gene_mean = fit_gene_means(labels, train_lines, min_lines=3)
    assert "ENOUGH" in gene_mean.index
    assert "SPARSE" not in gene_mean.index
    assert any(
        "dropping 1 gene" in record.getMessage() for record in caplog.records
    )

    targets = build_residual_targets(labels, train_lines)
    assert "SPARSE" not in targets.long["gene_symbol"].unique()
    assert len(targets.long) == 4
    assert targets.n_genes == 1
    assert set(targets.long["model_id"]) == {"L1", "L2", "L3", "L4"}


def test_fit_gene_means_raises_on_empty_train_lines() -> None:
    labels = _labels(
        [{"model_id": "L1", "gene_symbol": "GENE1", "gene_effect": 1.0}]
    )
    with pytest.raises(ValueError, match="empty"):
        fit_gene_means(labels, [])


def test_fit_gene_means_raises_on_unknown_train_line() -> None:
    labels = _labels(
        [{"model_id": "L1", "gene_symbol": "GENE1", "gene_effect": 1.0}]
    )
    with pytest.raises(ValueError, match="absent"):
        fit_gene_means(labels, ["L1", "L99"])


def test_fit_gene_means_raises_on_duplicate_train_lines() -> None:
    labels = _labels(
        [
            {"model_id": "L1", "gene_symbol": "GENE1", "gene_effect": 1.0},
            {"model_id": "L2", "gene_symbol": "GENE1", "gene_effect": 2.0},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        fit_gene_means(labels, ["L1", "L1", "L2"])


def test_nan_gene_effect_produces_nan_residual_and_is_not_dropped() -> None:
    labels = _labels(
        [
            {"model_id": "L1", "gene_symbol": "GENE1", "gene_effect": 1.0},
            {"model_id": "L2", "gene_symbol": "GENE1", "gene_effect": 2.0},
            {"model_id": "L3", "gene_symbol": "GENE1", "gene_effect": np.nan},
            {"model_id": "L4", "gene_symbol": "GENE1", "gene_effect": 4.0},
        ]
    )
    train_lines = ("L1", "L2", "L3", "L4")

    targets = build_residual_targets(labels, train_lines)

    assert len(targets.long) == 4
    nan_row = targets.long.loc[targets.long["model_id"] == "L3"].iloc[0]
    assert np.isnan(nan_row["gene_effect"])
    assert np.isnan(nan_row["residual"])
    # NaN residual must not have been filled with 0 or any other value.
    assert targets.long["residual"].isna().sum() == 1


def test_residual_plus_gene_mean_equals_gene_effect_invariant() -> None:
    labels = _labels(
        [
            {"model_id": "L1", "gene_symbol": "GENE1", "gene_effect": 1.5},
            {"model_id": "L2", "gene_symbol": "GENE1", "gene_effect": -2.25},
            {"model_id": "L3", "gene_symbol": "GENE1", "gene_effect": 3.0},
            {"model_id": "L4", "gene_symbol": "GENE1", "gene_effect": np.nan},
            {"model_id": "L1", "gene_symbol": "GENE2", "gene_effect": 0.1},
            {"model_id": "L2", "gene_symbol": "GENE2", "gene_effect": 0.2},
            {"model_id": "L3", "gene_symbol": "GENE2", "gene_effect": 0.3},
            {"model_id": "L4", "gene_symbol": "GENE2", "gene_effect": 0.4},
        ]
    )
    train_lines = ("L1", "L2", "L3")

    targets = build_residual_targets(labels, train_lines)

    finite = targets.long.dropna(subset=["gene_effect"])
    np.testing.assert_allclose(
        finite["residual"] + finite["gene_mean"], finite["gene_effect"]
    )


def test_build_residual_targets_returns_expected_dataclass_shape() -> None:
    labels = _labels(
        [
            {"model_id": "L1", "gene_symbol": "GENE1", "gene_effect": 1.0},
            {"model_id": "L2", "gene_symbol": "GENE1", "gene_effect": 2.0},
            {"model_id": "L3", "gene_symbol": "GENE1", "gene_effect": 3.0},
        ]
    )
    train_lines = ("L1", "L2", "L3")

    targets = build_residual_targets(labels, train_lines)

    assert isinstance(targets, ResidualTargets)
    assert list(targets.long.columns) == [
        "model_id",
        "gene_symbol",
        "gene_effect",
        "gene_mean",
        "residual",
    ]
    assert targets.train_lines == train_lines
    assert targets.n_genes == 1
    assert targets.n_lines == 3
    assert isinstance(targets.gene_mean, pd.Series)
    assert targets.gene_mean.name == "gene_mean"


def test_to_matrix_orientation_and_sorted_axes() -> None:
    labels = _labels(
        [
            {"model_id": "L2", "gene_symbol": "GENE_B", "gene_effect": 4.0},
            {"model_id": "L1", "gene_symbol": "GENE_B", "gene_effect": 3.0},
            {"model_id": "L2", "gene_symbol": "GENE_A", "gene_effect": 2.0},
            {"model_id": "L1", "gene_symbol": "GENE_A", "gene_effect": 1.0},
        ]
    )
    matrix = to_matrix(labels, "gene_effect")

    assert matrix.index.tolist() == ["GENE_A", "GENE_B"]
    assert matrix.columns.tolist() == ["L1", "L2"]
    assert matrix.loc["GENE_A", "L1"] == 1.0
    assert matrix.loc["GENE_A", "L2"] == 2.0
    assert matrix.loc["GENE_B", "L1"] == 3.0
    assert matrix.loc["GENE_B", "L2"] == 4.0


def test_to_matrix_raises_on_duplicate_gene_line_pair() -> None:
    labels = _labels(
        [
            {"model_id": "L1", "gene_symbol": "GENE_A", "gene_effect": 1.0},
            {"model_id": "L1", "gene_symbol": "GENE_A", "gene_effect": 999.0},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        to_matrix(labels, "gene_effect")
