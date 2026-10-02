import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from src.data.expression import library_sizes, log_normalize, median_library_size


def _toy():
    return np.array([[1, 0, 3, 6], [0, 2, 2, 0], [5, 5, 0, 10]], dtype=np.float32)


def test_matches_scanpy_whole_library_then_hvg_slice():
    x = _toy()
    expected = ad.AnnData(x.copy())
    sc.pp.normalize_total(expected, target_sum=100.0)
    sc.pp.log1p(expected)
    got = log_normalize(x[:, [0, 2]], library_sizes(x), 100.0)
    np.testing.assert_allclose(got, expected.X[:, [0, 2]], rtol=1e-6)


def test_library_size_uses_all_genes_not_the_slice():
    x = _toy()
    np.testing.assert_array_equal(library_sizes(x), [10, 4, 20])
    assert not np.allclose(
        log_normalize(x[:, :2], library_sizes(x), 10.0),
        log_normalize(x[:, :2], library_sizes(x[:, :2]), 10.0),
    )


def test_sparse_input_matches_dense():
    x = _toy()
    np.testing.assert_allclose(library_sizes(sparse.csr_matrix(x)), library_sizes(x))
    np.testing.assert_allclose(
        log_normalize(sparse.csr_matrix(x), library_sizes(x), 50.0),
        log_normalize(x, library_sizes(x), 50.0),
    )


def test_zero_library_row_stays_zero():
    x = np.zeros((2, 3), dtype=np.float32)
    x[1] = [1, 1, 2]
    out = log_normalize(x, library_sizes(x), 10.0)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    np.testing.assert_array_equal(out[0], 0.0)


def test_median_library_size_pools_sources():
    assert median_library_size(np.array([1.0, 3.0]), np.array([10.0])) == 3.0
