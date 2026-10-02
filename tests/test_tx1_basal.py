"""Raw-count helpers and readers in ``src.data.basal``."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.sparse import csr_matrix

from src.data.basal import (
    _cap_indices_by_group,
    _group_candidate_indices_by_label,
    _materialize_rows,
    align_columns,
    assert_tx1_input_contract,
    perturbseq_control_library_sizes,
    read_perturbseq_response_cells,
    require_raw_counts,
    symbol_column,
)
from src.data.response_streaming import read_xatlas_response_cells


def _perturbseq_h5ad(path: Path, labels: list[str], *, ensembl_index=False) -> Path:
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 10, size=(len(labels), 3)).astype(np.float32)
    obs = pd.DataFrame({"gene": labels}, index=[f"cell{i}" for i in range(len(labels))])
    ids = [f"ENSG{i:011d}" for i in range(3)]
    if ensembl_index:
        var = pd.DataFrame(
            {"gene_name": ["A", "B", "C"]}, index=pd.Index(ids, name="gene_id")
        )
    else:
        var = pd.DataFrame(
            {"gene_id": ids, "gene_name": ["A", "B", "C"]}, index=["a", "b", "c"]
        )
    ad.AnnData(X=csr_matrix(counts), obs=obs, var=var).write_h5ad(path)
    return path


def test_require_raw_counts_rejects_normalised_matrix():
    require_raw_counts(csr_matrix(np.array([[0.0, 3.0], [1.0, 0.0]])), "ok")
    with pytest.raises(ValueError, match="integer"):
        require_raw_counts(np.array([[0.5, 1.0]]), "line")
    with pytest.raises(ValueError, match="integer"):
        require_raw_counts(csr_matrix(np.array([[-1.0, 1.0]])), "line")


def test_align_columns_sums_duplicates_and_zero_fills():
    matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    aligned, present = align_columns(matrix, ["B", "A", "B"], ["A", "B", "Z"])
    np.testing.assert_array_equal(aligned.toarray(), [[2, 4, 0], [5, 10, 0]])
    np.testing.assert_array_equal(present, [True, True, False])
    sparse_aligned, _ = align_columns(
        csr_matrix(matrix), ["B", "A", "B"], ["A", "B", "Z"]
    )
    np.testing.assert_array_equal(sparse_aligned.toarray(), aligned.toarray())


def test_symbol_column_auto_needs_exactly_one_candidate():
    assert (
        symbol_column(pd.DataFrame(columns=["gene_name", "x"]), "auto") == "gene_name"
    )
    with pytest.raises(ValueError):
        symbol_column(pd.DataFrame(columns=["gene_name", "gene_symbol"]), "auto")
    with pytest.raises(ValueError):
        symbol_column(pd.DataFrame(columns=["x"]), "gene_name")


def test_tx1_input_contract():
    var = pd.DataFrame({"ensembl_id": ["ENSG00000000001"]}, index=["ENSG00000000001"])
    good = ad.AnnData(
        X=csr_matrix(np.ones((2, 1), dtype=np.float32)),
        obs=pd.DataFrame({"cell_type": ["x", "x"]}, index=["a", "b"]),
        var=var,
    )
    assert_tx1_input_contract(good)
    bad = good.copy()
    bad.X = csr_matrix(-np.ones((2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="negative"):
        assert_tx1_input_contract(bad)
    bad = good.copy()
    bad.var.index = ["GENE"]
    with pytest.raises(ValueError, match="Ensembl"):
        assert_tx1_input_contract(bad)


@pytest.mark.parametrize("dense", [True, False])
def test_materialize_rows_chunks_keep_requested_order(tmp_path, dense):
    values = np.arange(60, dtype=np.float32).reshape(20, 3)
    rows = np.array([17, 2, 9, 3, 11, 0])
    if dense:
        with h5py.File(tmp_path / "x.h5", "w") as handle:
            handle["X"] = values
        with h5py.File(tmp_path / "x.h5", "r") as handle:
            out = _materialize_rows(handle["X"], rows, chunk_size=4)
            sparsified = _materialize_rows(
                handle["X"], rows, chunk_size=4, sparsify_chunks=True
            )
        np.testing.assert_array_equal(out, values[rows])
        np.testing.assert_array_equal(sparsified.toarray(), values[rows])
    else:
        out = _materialize_rows(csr_matrix(values), rows, chunk_size=4)
        np.testing.assert_array_equal(out.toarray(), values[rows])


def test_materialize_rows_holds_one_dense_chunk_at_a_time():
    import gc
    import weakref

    values = np.arange(60, dtype=np.float32).reshape(12, 5)
    alive = []

    class Backed:
        def __getitem__(self, key):
            gc.collect()
            assert sum(ref() is not None for ref in alive) == 0
            chunk = values[key[0]].copy()
            alive.append(weakref.ref(chunk))
            return chunk

    out = _materialize_rows(
        Backed(), np.arange(12)[::-1], chunk_size=3, sparsify_chunks=True
    )
    np.testing.assert_array_equal(out.toarray(), values[::-1])


@pytest.mark.parametrize("ensembl_index", [False, True])
def test_control_library_sizes_sum_every_gene(tmp_path, ensembl_index):
    labels = ["non-targeting", "TP53", "non-targeting"]
    path = _perturbseq_h5ad(tmp_path / "a.h5ad", labels, ensembl_index=ensembl_index)
    sizes = perturbseq_control_library_sizes(
        path,
        control_label="non-targeting",
        perturbation_col="gene",
        var_ensembl_col="gene_id",
        label="ACH-1 controls",
    )
    counts = ad.read_h5ad(path).X.toarray()
    np.testing.assert_array_equal(sizes, counts[[0, 2]].sum(axis=1))
    with pytest.raises(ValueError, match="neither a column nor an index"):
        perturbseq_control_library_sizes(
            path,
            control_label="non-targeting",
            perturbation_col="gene",
            var_ensembl_col="missing",
            label="ACH-1 controls",
        )


def test_perturbseq_response_reader_caps_each_gene(tmp_path):
    labels = ["non-targeting"] * 2 + ["G1"] * 5 + ["G2"] * 2
    path = _perturbseq_h5ad(tmp_path / "a.h5ad", labels, ensembl_index=True)
    kwargs = dict(
        control_label="non-targeting",
        perturbation_col="gene",
        var_ensembl_col="gene_id",
        symbol_col="gene_name",
        gene_order=("C", "A", "Z"),
        label="ACH-1",
        max_cells_per_gene=3,
        seed=4,
    )
    cells = read_perturbseq_response_cells(path, **kwargs)
    assert pd.Series(cells.labels).value_counts().to_dict() == {"G1": 3, "G2": 2}
    np.testing.assert_array_equal(cells.present, [True, True, False])
    again = read_perturbseq_response_cells(path, **kwargs)
    np.testing.assert_array_equal(again.library_sizes, cells.library_sizes)
    np.testing.assert_array_equal(
        again.hvg_counts.toarray(), cells.hvg_counts.toarray()
    )
    counts = ad.read_h5ad(path).X.toarray()
    rows = _cap_indices_by_group(
        _group_candidate_indices_by_label(np.arange(2, 9), np.array(labels[2:])), 3, 4
    )
    np.testing.assert_array_equal(cells.labels, np.array(labels)[rows])
    np.testing.assert_array_equal(cells.library_sizes, counts[rows].sum(axis=1))
    expected = np.zeros((len(rows), 3))
    expected[:, :2] = counts[rows][:, [2, 0]]
    np.testing.assert_array_equal(cells.hvg_counts.toarray(), expected)
    total = read_perturbseq_response_cells(
        path, **{**kwargs, "total_cells_per_line": 4}
    )
    assert len(total.labels) == total.hvg_counts.shape[0] == 4


def _xatlas_world(tmp_path: Path) -> tuple[Path, Path]:
    pd.DataFrame(
        {
            "ensembl_id": [f"ENSG{i:011d}" for i in range(3)],
            "gene_name": ["A", "B", "C"],
            "gene_token_id": [0, 1, 2],
        }
    ).to_parquet(tmp_path / "meta.parquet")
    rows = []
    for index, (target, passing) in enumerate(
        [("Non-Targeting", 1), ("G1", 1), ("G1", 1), ("G1", 0), ("G2", 1), ("G1", 1)]
    ):
        rows.append(
            {
                "gene_token_id": np.array([0, 2]),
                "gene_expression": np.array([float(index + 1), 0.0]),
                "cell_barcode": f"bc{index}",
                "sample": "s1",
                "gene_target": target,
                "pass_guide_filter": passing,
            }
        )
    shards = tmp_path / "shards"
    shards.mkdir()
    pd.DataFrame(rows).to_parquet(shards / "HCT116_Batch1.parquet")
    pd.DataFrame(rows[:1]).to_parquet(shards / "Other_Batch1.parquet")
    return shards, tmp_path / "meta.parquet"


def test_xatlas_response_reader_filters_and_caps(tmp_path):
    shards, meta = _xatlas_world(tmp_path)
    kwargs = dict(
        model_id="ACH-000971",
        shard_glob="HCT116_*.parquet",
        control_label="Non-Targeting",
        pass_guide_filter_value=1,
        symbol_col="gene_name",
        gene_order=("A", "C", "Z"),
    )
    cells = read_xatlas_response_cells(shards, meta, **kwargs)
    assert cells.labels.tolist() == ["G1", "G1", "G1", "G2"]
    assert sparse.issparse(cells.hvg_counts)
    assert cells.hvg_counts.toarray()[:, 0].tolist() == [2.0, 3.0, 6.0, 5.0]
    np.testing.assert_array_equal(cells.library_sizes, [2.0, 3.0, 6.0, 5.0])
    # C is in the metadata but no kept cell has a positive count for it.
    np.testing.assert_array_equal(cells.present, [True, False, False])
    capped = read_xatlas_response_cells(shards, meta, **kwargs, max_cells_per_gene=2)
    assert pd.Series(capped.labels).value_counts().to_dict() == {"G1": 2, "G2": 1}
    with pytest.raises(ValueError, match="No parquet shards"):
        read_xatlas_response_cells(shards, meta, **{**kwargs, "shard_glob": "none*"})
