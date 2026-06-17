"""Tests for per-gene embedding pooling, alignment, and fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sl_benchmark_baseline.embeddings import (
    align_to_universe,
    load_gene_embeddings,
)


def test_load_gene_embeddings_mean_pools_each_bag(synthetic_bags_npz):
    table = load_gene_embeddings(synthetic_bags_npz)
    assert table.dim == 2
    np.testing.assert_allclose(table.vectors_by_symbol["G0"], [2.0, 1.0])
    np.testing.assert_allclose(table.vectors_by_symbol["G1"], [5.0, 5.0])
    np.testing.assert_allclose(table.vectors_by_symbol["G2"], [2.0, 2.0])


def test_load_gene_embeddings_records_feature_set(tmp_path: Path):
    npz_path = tmp_path / "with_feature_set.npz"
    np.savez_compressed(
        npz_path,
        cell_delta_pcs=np.array([[1.0, 0.0], [3.0, 2.0]], dtype=np.float32),
        bag_offsets=np.array([0, 2], dtype=np.int64),
        perturbation_gene=np.asarray(["G0"], dtype=object),
        feature_set=np.asarray("single_cell_pc_delta", dtype=object),
    )
    table = load_gene_embeddings(npz_path)
    assert table.feature_set == "single_cell_pc_delta"
    assert table.dim == 2


def test_load_gene_embeddings_feature_set_none_when_absent(synthetic_bags_npz):
    # The legacy synthetic fixture has no feature_set key; load must not crash.
    table = load_gene_embeddings(synthetic_bags_npz)
    assert table.feature_set is None


def test_load_gene_embeddings_rejects_missing_required_key(tmp_path: Path):
    npz_path = tmp_path / "missing_key.npz"
    np.savez_compressed(
        npz_path,
        cell_delta_pcs=np.array([[1.0, 0.0]], dtype=np.float32),
        # bag_offsets and perturbation_gene intentionally omitted
    )
    try:
        load_gene_embeddings(npz_path)
    except ValueError as error:
        assert "bag_offsets" in str(error) or "perturbation_gene" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing NPZ keys")


def test_align_to_universe_marks_coverage_and_zero_fallback(synthetic_bags_npz):
    table = load_gene_embeddings(synthetic_bags_npz)
    symbols = np.asarray(["G1", "UNCOVERED", "G0"])
    emb, mask = align_to_universe(table, symbols, fallback_strategy="zero")
    assert emb.shape == (3, 2)
    np.testing.assert_allclose(mask, [1.0, 0.0, 1.0])
    np.testing.assert_allclose(emb[0], [5.0, 5.0])
    np.testing.assert_allclose(emb[1], [0.0, 0.0])
    np.testing.assert_allclose(emb[2], [2.0, 1.0])


def test_align_to_universe_global_mean_fallback(synthetic_bags_npz):
    table = load_gene_embeddings(synthetic_bags_npz)
    symbols = np.asarray(["G0", "G1", "G2", "UNCOVERED"])
    emb, mask = align_to_universe(table, symbols, fallback_strategy="global_mean")
    # global mean of covered = mean([2,1],[5,5],[2,2]) = [3, 8/3]
    np.testing.assert_allclose(emb[3], [3.0, 8.0 / 3.0])
    np.testing.assert_allclose(mask, [1.0, 1.0, 1.0, 0.0])


def test_align_to_universe_global_mean_is_coverage_stable(synthetic_bags_npz):
    # global_mean is computed over covered genes only, so adding extra uncovered
    # symbols must not change the fallback vector (label-free, fold-stable).
    table = load_gene_embeddings(synthetic_bags_npz)
    emb_small, _ = align_to_universe(
        table, np.asarray(["G0", "G1", "G2"]), fallback_strategy="global_mean"
    )
    emb_big, _ = align_to_universe(
        table,
        np.asarray(["G0", "G1", "G2", "U0", "U1"]),
        fallback_strategy="global_mean",
    )
    expected = np.vstack([emb_small[0], emb_small[1], emb_small[2]]).mean(axis=0)
    np.testing.assert_allclose(emb_big[3], expected)  # U0 fallback
    np.testing.assert_allclose(emb_big[4], expected)  # U1 fallback
