"""Tests for HCT116 label-blind bag construction and aggregation."""

import numpy as np

from scripts.predict_exp05_hct116_smoke import (
    _aggregate_sample_gene_rows,
    _padded_group_chunks,
)


def test_short_group_is_padded_with_replacement_deterministically() -> None:
    first = _padded_group_chunks(
        10,
        18,
        chunk_size=64,
        rng=np.random.default_rng(42),
    )
    second = _padded_group_chunks(
        10,
        18,
        chunk_size=64,
        rng=np.random.default_rng(42),
    )

    assert len(first) == 1
    indices, n_real, n_padding = first[0]
    assert n_real == 8
    assert n_padding == 56
    assert len(indices) == 64
    assert set(indices[:n_real]) == set(range(10, 18))
    assert set(indices).issubset(set(range(10, 18)))
    np.testing.assert_array_equal(indices, second[0][0])


def test_long_group_uses_all_real_cells_and_pads_only_remainder() -> None:
    chunks = _padded_group_chunks(
        0,
        70,
        chunk_size=64,
        rng=np.random.default_rng(7),
    )

    assert [(n_real, n_padding) for _, n_real, n_padding in chunks] == [
        (64, 0),
        (6, 58),
    ]
    real_indices = np.concatenate([indices[:n_real] for indices, n_real, _ in chunks])
    assert sorted(real_indices.tolist()) == list(range(70))


def test_gene_aggregation_weights_sample_gene_groups_equally() -> None:
    rows = [
        {
            "sample": "batch1",
            "perturbation_gene": "TP53",
            "y_pred": 0.0,
            "n_real_cells": 8,
            "n_padding_draws": 56,
            "n_chunks": 1,
        },
        {
            "sample": "batch2",
            "perturbation_gene": "TP53",
            "y_pred": 2.0,
            "n_real_cells": 130,
            "n_padding_draws": 62,
            "n_chunks": 3,
        },
    ]

    result = _aggregate_sample_gene_rows(rows, {"TP53": "train"}).iloc[0]

    assert result["y_pred"] == 1.0
    assert result["n_real_cells"] == 138
    assert result["n_padding_draws"] == 118
    assert result["n_chunks"] == 4
    assert result["n_samples"] == 2
    assert bool(result["is_e_shared_train"]) is True
