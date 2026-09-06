"""Tests for the Stage 0 Tx1 input probe (``docs/04`` §6).

The GPU forward pass is not testable here, so these cover the parts that
decide whether the probe's answer means anything:

1. the raw-count guard, because the probe's claim is vacuous if its
   reference arm is already normalized (Replogle ships ``_raw_`` and
   ``_normalized_`` h5ads one word apart);
2. the CPM transform's structural invariants -- zeros preserved, within-cell
   order preserved, per-cell sums hit the target -- since those are exactly
   the properties per-cell quantile binning is blind to;
3. the arm comparison, driven by a **mock encoder that reimplements the
   collator's binning rule**, so the invariance claim is exercised end to
   end on the real algorithm without a checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import sparse

from src.experiments.historical.stage0_tx1_input_probe import (
    CPM_TARGET_SUM,
    assert_raw_counts,
    compare,
    detected_genes,
    run_probe,
    to_cpm,
)


def _adata(matrix: sparse.csr_matrix):
    """Wrap a count matrix in the minimal AnnData the probe expects."""
    import anndata as ad
    import pandas as pd

    n_genes = matrix.shape[1]
    var = pd.DataFrame(
        {"ensembl_id": [f"ENSG{index:011d}" for index in range(n_genes)]},
        index=[f"ENSG{index:011d}" for index in range(n_genes)],
    )
    obs = pd.DataFrame(
        {"cell_type": ["probe"] * matrix.shape[0]},
        index=[f"cell{index}" for index in range(matrix.shape[0])],
    )
    return ad.AnnData(X=sparse.csr_matrix(matrix), obs=obs, var=var)


def _counts(n_cells: int = 6, n_genes: int = 12, seed: int = 0) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float64)
    dense[dense < 1] = 0.0
    dense[:, 0] = 5.0  # guarantee every cell has a positive library size
    return sparse.csr_matrix(dense)


class _BinningEncoder:
    """Mock encoder applying the collator's per-cell quantile binning.

    Mirrors ``tahoe_x1.data.collator.binning``: bucketize each cell against
    quantiles of its own nonzero values. Any embedding built from those bins
    is what the real model sees, so a transform this mock is blind to is one
    the real encoder is blind to as well.
    """

    def __init__(self, n_bins: int = 51, subsample_to: int | None = None) -> None:
        self.n_bins = n_bins
        self.subsample_to = subsample_to
        self.calls = 0

    def __call__(self, adata) -> np.ndarray:  # noqa: ANN001 - test double
        self.calls += 1
        matrix = sparse.csr_matrix(adata.X).toarray()
        grades = torch.linspace(0, 1, self.n_bins - 1, dtype=torch.float64)
        rows = []
        for row in matrix:
            values = torch.as_tensor(row, dtype=torch.float64)
            nonzero = values[values > 0]
            bins = torch.quantile(nonzero, grades)
            binned = torch.bucketize(values, bins).to(torch.float64)
            binned[values == 0] = 0.0
            expressed = int((row > 0).sum())
            if self.subsample_to is not None and expressed > self.subsample_to:
                # Same unseeded draw the real ``_sample`` makes.
                keep = torch.randperm(len(binned))[: self.subsample_to]
                masked = torch.zeros_like(binned)
                masked[keep] = binned[keep]
                binned = masked
            rows.append(binned.numpy())
        return np.asarray(rows, dtype=np.float32)


# --- raw-count guard ---------------------------------------------------------


def test_assert_raw_counts_accepts_integer_counts() -> None:
    audit = assert_raw_counts(_counts())
    assert audit["noninteger_fraction"] == 0.0
    assert audit["stored_values"] > 0


def test_assert_raw_counts_rejects_a_normalized_matrix() -> None:
    with pytest.raises(ValueError, match="not raw counts"):
        assert_raw_counts(to_cpm(_counts()))


def test_assert_raw_counts_rejects_negative_values() -> None:
    matrix = _counts().toarray()
    matrix[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        assert_raw_counts(sparse.csr_matrix(matrix))


# --- the CPM transform -------------------------------------------------------


def test_to_cpm_hits_the_target_library_size() -> None:
    cpm = to_cpm(_counts())
    sums = np.asarray(cpm.sum(axis=1)).ravel()
    assert np.allclose(sums, CPM_TARGET_SUM, rtol=1e-4)


def test_to_cpm_preserves_the_zero_pattern_and_ordering() -> None:
    counts = _counts()
    cpm = to_cpm(counts)
    assert np.array_equal(detected_genes(counts), detected_genes(cpm))
    for raw_row, cpm_row in zip(counts.toarray(), cpm.toarray()):
        assert np.array_equal(
            np.argsort(raw_row, kind="stable"), np.argsort(cpm_row, kind="stable")
        )


def test_to_cpm_rejects_an_empty_cell() -> None:
    matrix = np.zeros((2, 4))
    matrix[0, 0] = 3.0
    with pytest.raises(ValueError, match="positive library size"):
        to_cpm(sparse.csr_matrix(matrix))


# --- arm comparison ----------------------------------------------------------


def test_compare_reports_identity_for_the_same_block() -> None:
    block = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = compare(block, block)
    assert result["identical"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["cosine_mean"] == pytest.approx(1.0)


def test_binning_encoder_is_blind_to_cpm() -> None:
    """The core claim: per-cell quantile binning cannot see a CPM rescale."""
    adata = _adata(_counts())
    report = run_probe(adata, _BinningEncoder(), seed=7, max_length=1024)
    cpm_arm = report["arms"]["cpm_vs_raw"]["all_cells"]
    assert cpm_arm["identical"] is True
    assert cpm_arm["max_abs_diff"] == 0.0


def test_probe_separates_subsampling_from_cpm() -> None:
    """Unseeded gene subsampling must show up as its own arm, not as CPM.

    With the RNG pinned per arm, the CPM arm stays exact even though every
    cell is wide enough to be subsampled; only the deliberately unseeded
    repeat moves. That separation is the whole point of the probe -- read
    together, the two effects are confounded.
    """
    adata = _adata(_counts(n_cells=8, n_genes=40, seed=3))
    encoder = _BinningEncoder(subsample_to=5)
    report = run_probe(adata, encoder, seed=11, max_length=4)

    assert report["detected_genes"]["over_max_length"] == adata.n_obs
    assert report["arms"]["cpm_vs_raw"]["all_cells"]["identical"] is True
    assert report["arms"]["repeat_seeded_vs_raw"]["all_cells"]["identical"] is True
    assert report["arms"]["repeat_unseeded_vs_raw"]["all_cells"]["identical"] is False


def test_probe_splits_wide_and_narrow_cells() -> None:
    adata = _adata(_counts(n_cells=6, n_genes=12, seed=1))
    report = run_probe(adata, _BinningEncoder(), seed=5, max_length=6)
    arm = report["arms"]["cpm_vs_raw"]
    assert "wide_cells" in arm or "narrow_cells" in arm
    assert report["cells"] == adata.n_obs
