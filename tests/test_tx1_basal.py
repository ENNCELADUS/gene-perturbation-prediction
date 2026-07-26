"""Tests for src/aivc_model/tx1_basal.py -- CPU-only basal AnnData assembly.

All fixtures are tiny synthetic parquet shards / h5ad files built in
``tmp_path``; the real Tahoe-100M and Perturb-seq sources live on the HPC
and are never touched here (Wave 1 Phase B, Task 1).
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.sparse import csr_matrix

from aivc_model.tx1_basal import (
    _materialize_rows,
    assert_tx1_input_contract,
    build_perturbseq_basal_adata,
    build_perturbseq_response_adata,
    build_tahoe_basal_adata,
    build_xatlas_orion_basal_adata,
    build_xatlas_orion_response_adata,
    load_line_manifest,
)
from conftest import TX1_MANIFEST_COLUMNS as _MANIFEST_COLUMNS
from conftest import tx1_manifest_row as _manifest_row
from conftest import write_tx1_gene_metadata as _write_gene_metadata
from conftest import write_tx1_line_manifest as _write_manifest
from conftest import write_tx1_perturbseq_h5ad as _write_perturbseq_h5ad
from conftest import (
    write_tx1_perturbseq_h5ad_ensembl_index as _write_perturbseq_h5ad_ensembl_index,
)
from conftest import write_tx1_shard as _write_shard
from conftest import write_tx1_xatlas_gene_metadata as _write_xatlas_gene_metadata
from conftest import write_tx1_xatlas_shard as _write_xatlas_shard


def _write_pooled_shard(shard_dir: Path, n_cells: int, cellosaurus_id: str) -> None:
    """Write one shard of ``n_cells`` distinguishable single-gene cells."""
    rows = [
        {
            "genes": np.array([3]),
            "expressions": np.array([float(index + 1)]),
            "cell_line_id": cellosaurus_id,
        }
        for index in range(n_cells)
    ]
    _write_shard(shard_dir / "part-0.parquet", rows)


def _write_multi_gene_perturbseq_h5ad(
    path: Path,
    *,
    control_label: str = "non-targeting",
    perturbation_col: str = "gene",
    control_n: int,
    gene_cells: dict[str, int],
    ensembl_col: str = "ensembl_id",
    n_genes: int = 3,
) -> Path:
    """A Perturb-seq h5ad with several distinct perturbed genes, each with its
    own cell count -- unlike conftest's ``write_tx1_perturbseq_h5ad``, which
    hardcodes every non-control cell to a single ``"TP53"`` label. Needed to
    exercise the per-gene cell cap (fix-round-2, Defect 2): a single-gene
    fixture can never show one gene capped while another is left untouched.
    """
    labels = [control_label] * control_n
    for gene, n in gene_cells.items():
        labels += [gene] * n
    n_cells = len(labels)
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 10, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {perturbation_col: labels}, index=[f"cell{index}" for index in range(n_cells)]
    )
    var = pd.DataFrame(
        {ensembl_col: [f"ENSG{index:011d}" for index in range(n_genes)]},
        index=[f"gene{index}" for index in range(n_genes)],
    )
    adata = ad.AnnData(X=csr_matrix(counts), obs=obs, var=var)
    adata.write_h5ad(path)
    return path


# --- load_line_manifest ------------------------------------------------


def test_load_line_manifest_happy_path(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _manifest_row(role="test"),
            _manifest_row(
                model_id="ACH-000002",
                role="train_response_and_head",
                basal_source="Perturb-seq non-targeting control",
            ),
        ],
    )
    manifest = load_line_manifest(path)
    assert list(manifest.columns) == list(_MANIFEST_COLUMNS)
    assert len(manifest) == 2


def test_load_line_manifest_rejects_missing_column(tmp_path: Path) -> None:
    frame = pd.DataFrame([_manifest_row()]).drop(columns=["lineage"])
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="lineage"):
        load_line_manifest(path)


def test_load_line_manifest_rejects_invalid_role(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "manifest.csv", [_manifest_row(role="bogus_role")]
    )
    with pytest.raises(ValueError, match="bogus_role"):
        load_line_manifest(path)


def test_load_line_manifest_rejects_invalid_basal_source(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "manifest.csv", [_manifest_row(basal_source="Unknown source")]
    )
    with pytest.raises(ValueError, match="Unknown source"):
        load_line_manifest(path)


def test_load_line_manifest_does_not_assert_specific_counts(tmp_path: Path) -> None:
    """A manifest with a single row (not 42) must load without complaint."""
    path = _write_manifest(tmp_path / "manifest.csv", [_manifest_row()])
    manifest = load_line_manifest(path)
    assert len(manifest) == 1


# --- build_tahoe_basal_adata --------------------------------------------


def test_build_tahoe_basal_adata_selects_only_requested_line(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_shard(
        shard_dir / "part-0.parquet",
        [
            {
                "genes": np.array([3, 4]),
                "expressions": np.array([1.0, 2.0]),
                "cell_line_id": "CVCL_A",
            },
            {
                "genes": np.array([3, 4]),
                "expressions": np.array([9.0, 9.0]),
                "cell_line_id": "CVCL_B",
            },
        ],
    )
    metadata_path = _write_gene_metadata(tmp_path / "genes.parquet", [3, 4])
    adata = build_tahoe_basal_adata(
        shard_dir,
        metadata_path,
        "CVCL_A",
        cell_line_name="LineA",
        model_id="ACH-A",
        seed=0,
    )
    assert adata.n_obs == 1
    assert adata.obs["cellosaurus_id"].tolist() == ["CVCL_A"]
    assert adata.X.toarray().tolist() == [[1.0, 2.0]]


def test_build_tahoe_basal_adata_drops_special_tokens_and_nonpositive(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_shard(
        shard_dir / "part-0.parquet",
        [
            {
                # ids 0,1,2 are special tokens (dropped regardless of value);
                # id 3 has a negative value; id 4 has a zero value; only id 5
                # (positive value, id >= 3) survives.
                "genes": np.array([0, 1, 2, 3, 4, 5]),
                "expressions": np.array([5.0, 5.0, 5.0, -1.0, 0.0, 3.0]),
                "cell_line_id": "CVCL_A",
            }
        ],
    )
    metadata_path = _write_gene_metadata(tmp_path / "genes.parquet", [3, 4, 5])
    adata = build_tahoe_basal_adata(
        shard_dir,
        metadata_path,
        "CVCL_A",
        cell_line_name="LineA",
        model_id="ACH-A",
        seed=0,
    )
    assert adata.n_vars == 1
    assert adata.var.index.tolist() == ["ENSG00000000005"]
    assert adata.X.toarray().tolist() == [[3.0]]


def test_build_tahoe_basal_adata_warns_on_missing_gene_metadata(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Otherwise-valid tokens missing from gene metadata must warn, not vanish."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_shard(
        shard_dir / "part-0.parquet",
        [
            {
                # ids 3, 4, 5 are all otherwise-valid (>= 3, positive value),
                # but gene metadata only knows about id 3, so ids 4 and 5
                # (2 of 3 valid tokens) are dropped from the panel.
                "genes": np.array([3, 4, 5]),
                "expressions": np.array([1.0, 2.0, 3.0]),
                "cell_line_id": "CVCL_A",
            }
        ],
    )
    metadata_path = _write_gene_metadata(tmp_path / "genes.parquet", [3])
    with caplog.at_level(logging.WARNING, logger="aivc_model.tx1_basal"):
        adata = build_tahoe_basal_adata(
            shard_dir,
            metadata_path,
            "CVCL_A",
            cell_line_name="LineA",
            model_id="ACH-A",
            seed=0,
        )
    assert adata.n_vars == 1
    warnings = [
        record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "missing from gene metadata index" in record.message
    ]
    assert len(warnings) == 1
    assert "2/3" in warnings[0]
    assert "66.7%" in warnings[0]


def test_build_tahoe_basal_adata_var_and_obs_schema(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_shard(
        shard_dir / "part-0.parquet",
        [
            {
                "genes": np.array([3, 4]),
                "expressions": np.array([1.0, 2.0]),
                "cell_line_id": "CVCL_A",
            }
        ],
    )
    metadata_path = _write_gene_metadata(tmp_path / "genes.parquet", [3, 4])
    adata = build_tahoe_basal_adata(
        shard_dir,
        metadata_path,
        "CVCL_A",
        cell_line_name="LineA",
        model_id="ACH-A",
        seed=0,
    )
    assert set(adata.var.index) == {"ENSG00000000003", "ENSG00000000004"}
    assert {"cell_type", "cellosaurus_id", "model_id", "basal_source"}.issubset(
        adata.obs.columns
    )
    assert adata.obs["cell_type"].tolist() == ["LineA"]
    assert adata.obs["model_id"].tolist() == ["ACH-A"]
    assert adata.obs["basal_source"].tolist() == ["Tahoe-100M DMSO"]


def test_build_tahoe_basal_adata_same_seed_reproduces_selection(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_pooled_shard(shard_dir, 30, "CVCL_A")
    metadata_path = _write_gene_metadata(tmp_path / "genes.parquet", [3])
    kwargs = {
        "cell_line_name": "LineA",
        "model_id": "ACH-A",
        "max_cells": 5,
        "seed": 7,
    }
    first = build_tahoe_basal_adata(shard_dir, metadata_path, "CVCL_A", **kwargs)
    second = build_tahoe_basal_adata(shard_dir, metadata_path, "CVCL_A", **kwargs)
    np.testing.assert_array_equal(first.X.toarray(), second.X.toarray())


def test_build_tahoe_basal_adata_different_seed_changes_selection(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_pooled_shard(shard_dir, 30, "CVCL_A")
    metadata_path = _write_gene_metadata(tmp_path / "genes.parquet", [3])
    seed7 = build_tahoe_basal_adata(
        shard_dir,
        metadata_path,
        "CVCL_A",
        cell_line_name="LineA",
        model_id="ACH-A",
        max_cells=5,
        seed=7,
    )
    seed99 = build_tahoe_basal_adata(
        shard_dir,
        metadata_path,
        "CVCL_A",
        cell_line_name="LineA",
        model_id="ACH-A",
        max_cells=5,
        seed=99,
    )
    values_seed7 = set(seed7.X.toarray().ravel().tolist())
    values_seed99 = set(seed99.X.toarray().ravel().tolist())
    assert values_seed7 != values_seed99


def test_build_tahoe_basal_adata_max_cells_none_returns_everything(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_pooled_shard(shard_dir, 12, "CVCL_A")
    metadata_path = _write_gene_metadata(tmp_path / "genes.parquet", [3])
    adata = build_tahoe_basal_adata(
        shard_dir,
        metadata_path,
        "CVCL_A",
        cell_line_name="LineA",
        model_id="ACH-A",
        max_cells=None,
        seed=0,
    )
    assert adata.n_obs == 12


def test_build_tahoe_basal_adata_max_cells_larger_than_available(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_pooled_shard(shard_dir, 5, "CVCL_A")
    metadata_path = _write_gene_metadata(tmp_path / "genes.parquet", [3])
    adata = build_tahoe_basal_adata(
        shard_dir,
        metadata_path,
        "CVCL_A",
        cell_line_name="LineA",
        model_id="ACH-A",
        max_cells=1000,
        seed=0,
    )
    assert adata.n_obs == 5


def test_build_tahoe_basal_adata_raises_when_no_shards(tmp_path: Path) -> None:
    shard_dir = tmp_path / "empty_shards"
    shard_dir.mkdir()
    metadata_path = _write_gene_metadata(tmp_path / "genes.parquet", [3])
    with pytest.raises(ValueError, match="No parquet shards"):
        build_tahoe_basal_adata(
            shard_dir,
            metadata_path,
            "CVCL_A",
            cell_line_name="LineA",
            model_id="ACH-A",
            seed=0,
        )


def test_build_tahoe_basal_adata_raises_when_line_not_found(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_shard(
        shard_dir / "part-0.parquet",
        [
            {
                "genes": np.array([3]),
                "expressions": np.array([1.0]),
                "cell_line_id": "CVCL_OTHER",
            }
        ],
    )
    metadata_path = _write_gene_metadata(tmp_path / "genes.parquet", [3])
    with pytest.raises(ValueError, match="CVCL_MISSING"):
        build_tahoe_basal_adata(
            shard_dir,
            metadata_path,
            "CVCL_MISSING",
            cell_line_name="LineA",
            model_id="ACH-A",
            seed=0,
        )


# --- assert_tx1_input_contract -------------------------------------------


def test_assert_tx1_input_contract_accepts_valid_adata() -> None:
    adata = ad.AnnData(
        X=csr_matrix(np.array([[1.0, 0.0]], dtype=np.float32)),
        obs=pd.DataFrame({"cell_type": ["LineA"]}, index=["cell0"]),
        var=pd.DataFrame({"ensembl_id": ["ENSG1", "ENSG2"]}, index=["ENSG1", "ENSG2"]),
    )
    assert_tx1_input_contract(adata)


def test_assert_tx1_input_contract_rejects_negative_x() -> None:
    adata = ad.AnnData(
        X=np.array([[-1.0, 2.0]], dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["LineA"]}, index=["cell0"]),
        var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
    )
    with pytest.raises(ValueError, match="negative"):
        assert_tx1_input_contract(adata)


def test_assert_tx1_input_contract_rejects_nan_dense_x() -> None:
    """NaN must not slip past the negativity check (NaN < 0 is False in numpy)."""
    adata = ad.AnnData(
        X=np.array([[1.0, np.nan]], dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["LineA"]}, index=["cell0"]),
        var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
    )
    with pytest.raises(ValueError, match="non-finite"):
        assert_tx1_input_contract(adata)


def test_assert_tx1_input_contract_rejects_nan_sparse_x() -> None:
    adata = ad.AnnData(
        X=csr_matrix(np.array([[1.0, np.nan]], dtype=np.float32)),
        obs=pd.DataFrame({"cell_type": ["LineA"]}, index=["cell0"]),
        var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
    )
    with pytest.raises(ValueError, match="non-finite"):
        assert_tx1_input_contract(adata)


def test_assert_tx1_input_contract_rejects_missing_cell_type() -> None:
    adata = ad.AnnData(
        X=np.array([[1.0, 2.0]], dtype=np.float32),
        obs=pd.DataFrame({"other": ["x"]}, index=["cell0"]),
        var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
    )
    with pytest.raises(ValueError, match="cell_type"):
        assert_tx1_input_contract(adata)


def test_assert_tx1_input_contract_rejects_non_ensembl_var() -> None:
    adata = ad.AnnData(
        X=np.array([[1.0, 2.0]], dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["LineA"]}, index=["cell0"]),
        var=pd.DataFrame(index=["GENE_A", "GENE_B"]),
    )
    with pytest.raises(ValueError, match="Ensembl"):
        assert_tx1_input_contract(adata)


# --- _materialize_rows: bounded chunking (fix-round-2, Defect 1) -----------


def test_materialize_rows_chunked_matches_single_call_dense(tmp_path: Path) -> None:
    """The test that matters most for Defect 1: reading in bounded chunks
    must return EXACTLY the same rows, in the same order, as one single
    (unchunked) call -- proving the chunking split introduces no reordering
    or dropped rows. Uses an unsorted, non-contiguous ``row_indices`` so a
    chunk-boundary or sort/inverse-permutation bug would show up."""
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 100, size=(23, 4)).astype(np.float32)
    h5ad_path = tmp_path / "dense.h5ad"
    ad.AnnData(X=dense).write_h5ad(h5ad_path)
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        row_indices = np.array([17, 2, 9, 0, 22, 5, 6, 11, 3], dtype=np.int64)
        chunked = _materialize_rows(backed.X, row_indices, chunk_size=3)
        single_call = _materialize_rows(backed.X, row_indices, chunk_size=1000)
        np.testing.assert_array_equal(np.asarray(chunked), np.asarray(single_call))
        np.testing.assert_array_equal(np.asarray(chunked), dense[row_indices])
    finally:
        backed.file.close()


def test_materialize_rows_chunked_matches_single_call_sparse(tmp_path: Path) -> None:
    """Same guarantee as the dense test, for a backed sparse ``X`` -- the real
    GWPS anchor files vary between the two, and the fix must hold for both."""
    rng = np.random.default_rng(1)
    dense = rng.integers(0, 5, size=(19, 4)).astype(np.float32)
    h5ad_path = tmp_path / "sparse.h5ad"
    ad.AnnData(X=csr_matrix(dense)).write_h5ad(h5ad_path)
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        row_indices = np.array([14, 1, 8, 0, 18, 4, 6, 10, 2], dtype=np.int64)
        chunked = _materialize_rows(backed.X, row_indices, chunk_size=2)
        single_call = _materialize_rows(backed.X, row_indices, chunk_size=1000)
        assert sparse.issparse(chunked)
        np.testing.assert_array_equal(chunked.toarray(), single_call.toarray())
        np.testing.assert_array_equal(chunked.toarray(), dense[row_indices])
    finally:
        backed.file.close()


def test_materialize_rows_default_chunk_size_matches_ground_truth(
    tmp_path: Path,
) -> None:
    """Regression against the real default (not just an artificially small
    ``chunk_size`` chosen to force multiple chunks)."""
    rng = np.random.default_rng(2)
    dense = rng.integers(0, 100, size=(37, 5)).astype(np.float32)
    h5ad_path = tmp_path / "dense.h5ad"
    ad.AnnData(X=dense).write_h5ad(h5ad_path)
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        row_indices = np.array([30, 1, 15, 4, 36, 0], dtype=np.int64)
        result = _materialize_rows(backed.X, row_indices)  # default chunk_size
        np.testing.assert_array_equal(np.asarray(result), dense[row_indices])
    finally:
        backed.file.close()


def test_materialize_rows_empty_selection_does_not_crash(tmp_path: Path) -> None:
    dense = np.zeros((5, 2), dtype=np.float32)
    h5ad_path = tmp_path / "dense.h5ad"
    ad.AnnData(X=dense).write_h5ad(h5ad_path)
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        result = _materialize_rows(backed.X, np.array([], dtype=np.int64))
        assert np.asarray(result).shape == (0, 2)
    finally:
        backed.file.close()


# --- build_perturbseq_basal_adata ----------------------------------------


def test_build_perturbseq_basal_adata_selects_only_controls(tmp_path: Path) -> None:
    h5ad_path = _write_perturbseq_h5ad(tmp_path / "data.h5ad", n_control=4, n_other=6)
    adata = build_perturbseq_basal_adata(
        h5ad_path,
        control_label="non-targeting",
        perturbation_col="gene",
        cell_line_name="LineP",
        model_id="ACH-P",
        cellosaurus_id="CVCL_P",
        var_ensembl_col="ensembl_id",
        seed=0,
    )
    assert adata.n_obs == 4
    assert {"cell_type", "cellosaurus_id", "model_id", "basal_source"}.issubset(
        adata.obs.columns
    )
    assert adata.obs["basal_source"].unique().tolist() == [
        "Perturb-seq non-targeting control"
    ]
    assert adata.obs["cell_type"].unique().tolist() == ["LineP"]
    assert adata.var.index.tolist() == [f"ENSG{index:011d}" for index in range(3)]


def test_build_perturbseq_basal_adata_excludes_perturbed_cells(
    tmp_path: Path,
) -> None:
    h5ad_path = _write_perturbseq_h5ad(tmp_path / "data.h5ad", n_control=3, n_other=5)
    control_only = ad.read_h5ad(h5ad_path)
    expected = control_only[control_only.obs["gene"] == "non-targeting"].X.toarray()
    adata = build_perturbseq_basal_adata(
        h5ad_path,
        control_label="non-targeting",
        perturbation_col="gene",
        cell_line_name="LineP",
        model_id="ACH-P",
        cellosaurus_id="CVCL_P",
        var_ensembl_col="ensembl_id",
        seed=0,
    )
    np.testing.assert_array_equal(adata.X.toarray(), expected)


def test_build_perturbseq_basal_adata_raises_on_missing_ensembl_column(
    tmp_path: Path,
) -> None:
    h5ad_path = _write_perturbseq_h5ad(
        tmp_path / "data.h5ad",
        n_control=2,
        n_other=2,
        ensembl_col="not_ensembl",
    )
    with pytest.raises(ValueError) as excinfo:
        build_perturbseq_basal_adata(
            h5ad_path,
            control_label="non-targeting",
            perturbation_col="gene",
            cell_line_name="LineP",
            model_id="ACH-P",
            cellosaurus_id="CVCL_P",
            var_ensembl_col="ensembl_id",
            seed=0,
        )
    assert "ensembl_id" in str(excinfo.value)
    assert "not_ensembl" in str(excinfo.value)


def test_build_perturbseq_basal_adata_max_cells_caps_deterministically(
    tmp_path: Path,
) -> None:
    h5ad_path = _write_perturbseq_h5ad(tmp_path / "data.h5ad", n_control=20, n_other=5)
    kwargs = {
        "control_label": "non-targeting",
        "perturbation_col": "gene",
        "cell_line_name": "LineP",
        "model_id": "ACH-P",
        "cellosaurus_id": "CVCL_P",
        "var_ensembl_col": "ensembl_id",
        "max_cells": 5,
        "seed": 3,
    }
    first = build_perturbseq_basal_adata(h5ad_path, **kwargs)
    second = build_perturbseq_basal_adata(h5ad_path, **kwargs)
    assert first.n_obs == 5
    assert first.obs_names.tolist() == second.obs_names.tolist()
    np.testing.assert_array_equal(first.X.toarray(), second.X.toarray())


def test_build_perturbseq_basal_adata_max_cells_none_returns_everything(
    tmp_path: Path,
) -> None:
    h5ad_path = _write_perturbseq_h5ad(tmp_path / "data.h5ad", n_control=7, n_other=2)
    adata = build_perturbseq_basal_adata(
        h5ad_path,
        control_label="non-targeting",
        perturbation_col="gene",
        cell_line_name="LineP",
        model_id="ACH-P",
        cellosaurus_id="CVCL_P",
        var_ensembl_col="ensembl_id",
        max_cells=None,
        seed=0,
    )
    assert adata.n_obs == 7


def test_build_perturbseq_basal_adata_accepts_ensembl_only_index(
    tmp_path: Path,
) -> None:
    """Some Perturb-seq h5ad sources (e.g. the local Replogle-lineage K562
    essential file) carry Ensembl ids only as ``var.index``, with no
    separate materialized column -- distinct from the column-based schema
    the other tests in this module use (audited for the local Adamson UPR
    file). Both must be accepted."""
    h5ad_path = _write_perturbseq_h5ad_ensembl_index(
        tmp_path / "data.h5ad", n_control=3, n_other=2, index_name="gene_id"
    )
    adata = build_perturbseq_basal_adata(
        h5ad_path,
        control_label="non-targeting",
        perturbation_col="gene",
        cell_line_name="LineP",
        model_id="ACH-P",
        cellosaurus_id="CVCL_P",
        var_ensembl_col="gene_id",
        seed=0,
    )
    assert adata.n_obs == 3
    assert adata.var.index.tolist() == [f"ENSG{index:011d}" for index in range(3)]


def test_build_perturbseq_basal_adata_ensembl_index_name_mismatch_raises(
    tmp_path: Path,
) -> None:
    h5ad_path = _write_perturbseq_h5ad_ensembl_index(
        tmp_path / "data.h5ad", n_control=2, n_other=2, index_name="gene_id"
    )
    with pytest.raises(ValueError) as excinfo:
        build_perturbseq_basal_adata(
            h5ad_path,
            control_label="non-targeting",
            perturbation_col="gene",
            cell_line_name="LineP",
            model_id="ACH-P",
            cellosaurus_id="CVCL_P",
            var_ensembl_col="ensembl_id",
            seed=0,
        )
    message = str(excinfo.value)
    assert "ensembl_id" in message
    assert "gene_id" in message


# --- build_xatlas_orion_basal_adata ---------------------------------------


def _xatlas_row(
    *,
    gene_token_id: list[int],
    gene_expression: list[float],
    cell_barcode: str,
    sample: str = "HCT116_Batch1",
    gene_target: str = "Non-Targeting",
    pass_guide_filter: int = 1,
) -> dict[str, object]:
    return {
        "gene_token_id": np.array(gene_token_id),
        "gene_expression": np.array(gene_expression, dtype=np.float32),
        "cell_barcode": cell_barcode,
        "sample": sample,
        "gene_target": gene_target,
        "pass_guide_filter": pass_guide_filter,
    }


def test_build_xatlas_orion_basal_adata_selects_only_non_targeting_and_passing_guide(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0, 1],
                gene_expression=[1.0, 2.0],
                cell_barcode="AAA",
            ),
            _xatlas_row(
                gene_token_id=[0, 1],
                gene_expression=[9.0, 9.0],
                cell_barcode="BBB",
                pass_guide_filter=0,  # fails guide QC -> excluded
            ),
            _xatlas_row(
                gene_token_id=[0, 1],
                gene_expression=[9.0, 9.0],
                cell_barcode="CCC",
                gene_target="TP53",  # perturbed, not control -> excluded
            ),
            _xatlas_row(
                gene_token_id=[0, 1],
                gene_expression=[3.0, 4.0],
                cell_barcode="DDD",
            ),
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0, 1])
    adata = build_xatlas_orion_basal_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        seed=0,
    )
    assert adata.n_obs == 2
    assert sorted(adata.obs_names.tolist()) == [
        "HCT116_Batch1:AAA",
        "HCT116_Batch1:DDD",
    ]


def test_build_xatlas_orion_basal_adata_low_token_ids_are_not_special(
    tmp_path: Path,
) -> None:
    """Regression: X-Atlas-Orion's gene_token_id vocabulary starts at 0 and
    does not reserve low ids as special tokens (confirmed against the real,
    locally audited gene_metadata.parquet: token 0 = DDX11L2, a real gene).
    Tokens 0/1/2 must survive, unlike Tahoe-100M's ``< 3`` special-token
    filter."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0, 1, 2],
                gene_expression=[5.0, 6.0, 7.0],
                cell_barcode="AAA",
            )
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0, 1, 2])
    adata = build_xatlas_orion_basal_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        seed=0,
    )
    assert adata.n_vars == 3
    assert set(adata.var.index) == {
        "ENSG00000000000",
        "ENSG00000000001",
        "ENSG00000000002",
    }


def test_build_xatlas_orion_basal_adata_drops_nonpositive_values(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0, 1, 2],
                gene_expression=[0.0, -1.0, 3.0],
                cell_barcode="AAA",
            )
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0, 1, 2])
    adata = build_xatlas_orion_basal_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        seed=0,
    )
    assert adata.n_vars == 1
    assert adata.var.index.tolist() == ["ENSG00000000002"]
    assert adata.X.toarray().tolist() == [[3.0]]


def test_build_xatlas_orion_basal_adata_warns_on_missing_gene_metadata(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0, 1, 2],
                gene_expression=[1.0, 2.0, 3.0],
                cell_barcode="AAA",
            )
        ],
    )
    # gene metadata only knows about token 0 -> tokens 1, 2 (2/3) dropped.
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    with caplog.at_level(logging.WARNING, logger="aivc_model.tx1_basal"):
        adata = build_xatlas_orion_basal_adata(
            shard_dir,
            metadata_path,
            cell_line_name="HCT116",
            model_id="ACH-000971",
            cellosaurus_id="CVCL_0291",
            seed=0,
        )
    assert adata.n_vars == 1
    warnings = [
        record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "missing from gene metadata index" in record.message
    ]
    assert len(warnings) == 1
    assert "2/3" in warnings[0]


def test_build_xatlas_orion_basal_adata_var_and_obs_schema(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0, 1], gene_expression=[1.0, 2.0], cell_barcode="AAA"
            )
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0, 1])
    adata = build_xatlas_orion_basal_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        seed=0,
    )
    assert {
        "cell_type",
        "cellosaurus_id",
        "model_id",
        "basal_source",
        "sample",
    }.issubset(adata.obs.columns)
    assert adata.obs["cell_type"].tolist() == ["HCT116"]
    assert adata.obs["cellosaurus_id"].tolist() == ["CVCL_0291"]
    assert adata.obs["model_id"].tolist() == ["ACH-000971"]
    assert adata.obs["basal_source"].tolist() == ["Perturb-seq non-targeting control"]
    assert adata.obs["sample"].tolist() == ["HCT116_Batch1"]
    assert {"ensembl_id", "gene_name"}.issubset(adata.var.columns)


def test_build_xatlas_orion_basal_adata_same_seed_reproduces_selection(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0],
                gene_expression=[float(index + 1)],
                cell_barcode=f"cell{index}",
            )
            for index in range(30)
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    kwargs = {
        "cell_line_name": "HCT116",
        "model_id": "ACH-000971",
        "cellosaurus_id": "CVCL_0291",
        "max_cells": 5,
        "seed": 7,
    }
    first = build_xatlas_orion_basal_adata(shard_dir, metadata_path, **kwargs)
    second = build_xatlas_orion_basal_adata(shard_dir, metadata_path, **kwargs)
    assert first.obs_names.tolist() == second.obs_names.tolist()
    np.testing.assert_array_equal(first.X.toarray(), second.X.toarray())


def test_build_xatlas_orion_basal_adata_different_seed_changes_selection(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0],
                gene_expression=[float(index + 1)],
                cell_barcode=f"cell{index}",
            )
            for index in range(30)
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    seed7 = build_xatlas_orion_basal_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        max_cells=5,
        seed=7,
    )
    seed99 = build_xatlas_orion_basal_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        max_cells=5,
        seed=99,
    )
    assert set(seed7.obs_names.tolist()) != set(seed99.obs_names.tolist())


def test_build_xatlas_orion_basal_adata_max_cells_none_returns_everything(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0],
                gene_expression=[float(index + 1)],
                cell_barcode=f"cell{index}",
            )
            for index in range(12)
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    adata = build_xatlas_orion_basal_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        max_cells=None,
        seed=0,
    )
    assert adata.n_obs == 12


def test_build_xatlas_orion_basal_adata_shard_glob_filters_other_lines(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [_xatlas_row(gene_token_id=[0], gene_expression=[1.0], cell_barcode="AAA")],
    )
    _write_xatlas_shard(
        shard_dir / "OtherLine_Batch1.parquet",
        [_xatlas_row(gene_token_id=[0], gene_expression=[9.0], cell_barcode="ZZZ")],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    adata = build_xatlas_orion_basal_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        shard_glob="HCT116_Batch*.parquet",
        seed=0,
    )
    assert adata.n_obs == 1
    assert adata.obs_names.tolist() == ["HCT116_Batch1:AAA"]


def test_build_xatlas_orion_basal_adata_raises_when_no_shard_matches_glob(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "OtherLine_Batch1.parquet",
        [_xatlas_row(gene_token_id=[0], gene_expression=[1.0], cell_barcode="AAA")],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    with pytest.raises(ValueError, match="No parquet shards"):
        build_xatlas_orion_basal_adata(
            shard_dir,
            metadata_path,
            cell_line_name="HCT116",
            model_id="ACH-000971",
            cellosaurus_id="CVCL_0291",
            shard_glob="HCT116_Batch*.parquet",
            seed=0,
        )


def test_build_xatlas_orion_basal_adata_raises_when_no_cells_survive_filters(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0],
                gene_expression=[1.0],
                cell_barcode="AAA",
                gene_target="TP53",
            )
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    with pytest.raises(ValueError, match="No cells found"):
        build_xatlas_orion_basal_adata(
            shard_dir,
            metadata_path,
            cell_line_name="HCT116",
            model_id="ACH-000971",
            cellosaurus_id="CVCL_0291",
            seed=0,
        )


def test_build_xatlas_orion_basal_adata_pass_guide_filter_value_is_configurable(
    tmp_path: Path,
) -> None:
    """``pass_guide_filter_value`` must be a real, honored parameter, not a
    hardcoded ``== 1`` literal."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0],
                gene_expression=[1.0],
                cell_barcode="AAA",
                pass_guide_filter=1,
            ),
            _xatlas_row(
                gene_token_id=[0],
                gene_expression=[2.0],
                cell_barcode="BBB",
                pass_guide_filter=2,
            ),
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    adata = build_xatlas_orion_basal_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        pass_guide_filter_value=2,
        seed=0,
    )
    assert adata.obs_names.tolist() == ["HCT116_Batch1:BBB"]


def test_build_xatlas_orion_basal_adata_logs_filter_removal_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(gene_token_id=[0], gene_expression=[1.0], cell_barcode="AAA"),
            _xatlas_row(
                gene_token_id=[0],
                gene_expression=[1.0],
                cell_barcode="BBB",
                gene_target="TP53",
            ),
            _xatlas_row(
                gene_token_id=[0],
                gene_expression=[1.0],
                cell_barcode="CCC",
                pass_guide_filter=0,
            ),
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    with caplog.at_level(logging.INFO, logger="aivc_model.tx1_basal"):
        build_xatlas_orion_basal_adata(
            shard_dir,
            metadata_path,
            cell_line_name="HCT116",
            model_id="ACH-000971",
            cellosaurus_id="CVCL_0291",
            seed=0,
        )
    filter_logs = [
        record.message
        for record in caplog.records
        if "cell filtering under" in record.message
    ]
    assert len(filter_logs) == 1
    assert "3 total rows" in filter_logs[0]
    assert "2 after" in filter_logs[0]  # after gene_target control filter
    assert "1 after" in filter_logs[0]  # after guide filter


# --- build_perturbseq_response_adata (Phase C Task 5) ----------------------


def test_build_perturbseq_response_adata_selects_only_perturbed_cells(
    tmp_path: Path,
) -> None:
    h5ad_path = _write_perturbseq_h5ad(tmp_path / "data.h5ad", n_control=4, n_other=6)
    adata = build_perturbseq_response_adata(
        h5ad_path,
        control_label="non-targeting",
        perturbation_col="gene",
        cell_line_name="LineP",
        model_id="ACH-P",
        cellosaurus_id="CVCL_P",
        var_ensembl_col="ensembl_id",
    )
    assert adata.n_obs == 6
    assert set(adata.obs["perturbation_gene"].unique()) == {"TP53"}
    assert "non-targeting" not in adata.obs["perturbation_gene"].unique()


def test_build_perturbseq_response_adata_genes_filter_restricts_selection(
    tmp_path: Path,
) -> None:
    h5ad_path = _write_perturbseq_h5ad(tmp_path / "data.h5ad", n_control=2, n_other=3)
    adata = build_perturbseq_response_adata(
        h5ad_path,
        control_label="non-targeting",
        perturbation_col="gene",
        cell_line_name="LineP",
        model_id="ACH-P",
        cellosaurus_id="CVCL_P",
        var_ensembl_col="ensembl_id",
        genes=["TP53"],
    )
    assert adata.n_obs == 3
    with pytest.raises(ValueError, match="No perturbed cells"):
        build_perturbseq_response_adata(
            h5ad_path,
            control_label="non-targeting",
            perturbation_col="gene",
            cell_line_name="LineP",
            model_id="ACH-P",
            cellosaurus_id="CVCL_P",
            var_ensembl_col="ensembl_id",
            genes=["NOT_A_REAL_GENE"],
        )


def test_build_perturbseq_response_adata_accepts_ensembl_only_index(
    tmp_path: Path,
) -> None:
    h5ad_path = _write_perturbseq_h5ad_ensembl_index(
        tmp_path / "data.h5ad", n_control=2, n_other=3, index_name="gene_id"
    )
    adata = build_perturbseq_response_adata(
        h5ad_path,
        control_label="non-targeting",
        perturbation_col="gene",
        cell_line_name="LineP",
        model_id="ACH-P",
        cellosaurus_id="CVCL_P",
        var_ensembl_col="gene_id",
    )
    assert adata.n_obs == 3
    assert adata.var.index.tolist() == [f"ENSG{index:011d}" for index in range(3)]


# --- build_perturbseq_response_adata: per-gene cap (fix-round-2 Defect 2) --


def test_build_perturbseq_response_adata_caps_per_gene_deterministically(
    tmp_path: Path,
) -> None:
    h5ad_path = _write_multi_gene_perturbseq_h5ad(
        tmp_path / "data.h5ad", control_n=2, gene_cells={"GENE_A": 20, "GENE_B": 3}
    )
    kwargs = {
        "control_label": "non-targeting",
        "perturbation_col": "gene",
        "cell_line_name": "LineP",
        "model_id": "ACH-P",
        "cellosaurus_id": "CVCL_P",
        "var_ensembl_col": "ensembl_id",
        "max_cells_per_gene": 5,
        "seed": 3,
    }
    first = build_perturbseq_response_adata(h5ad_path, **kwargs)
    second = build_perturbseq_response_adata(h5ad_path, **kwargs)
    counts = first.obs["perturbation_gene"].value_counts()
    assert counts["GENE_A"] == 5, "over-cap gene must be capped to max_cells_per_gene"
    assert counts["GENE_B"] == 3, "under-cap gene must be left untouched"
    assert first.obs_names.tolist() == second.obs_names.tolist()
    np.testing.assert_array_equal(first.X.toarray(), second.X.toarray())


def test_build_perturbseq_response_adata_max_cells_per_gene_none_returns_everything(
    tmp_path: Path,
) -> None:
    h5ad_path = _write_multi_gene_perturbseq_h5ad(
        tmp_path / "data.h5ad", control_n=2, gene_cells={"GENE_A": 7, "GENE_B": 4}
    )
    adata = build_perturbseq_response_adata(
        h5ad_path,
        control_label="non-targeting",
        perturbation_col="gene",
        cell_line_name="LineP",
        model_id="ACH-P",
        cellosaurus_id="CVCL_P",
        var_ensembl_col="ensembl_id",
        max_cells_per_gene=None,
    )
    assert adata.n_obs == 11


def test_build_perturbseq_response_adata_different_seed_changes_capped_selection(
    tmp_path: Path,
) -> None:
    h5ad_path = _write_multi_gene_perturbseq_h5ad(
        tmp_path / "data.h5ad", control_n=2, gene_cells={"GENE_A": 20}
    )
    kwargs = {
        "control_label": "non-targeting",
        "perturbation_col": "gene",
        "cell_line_name": "LineP",
        "model_id": "ACH-P",
        "cellosaurus_id": "CVCL_P",
        "var_ensembl_col": "ensembl_id",
        "max_cells_per_gene": 5,
    }
    seed7 = build_perturbseq_response_adata(h5ad_path, seed=7, **kwargs)
    seed99 = build_perturbseq_response_adata(h5ad_path, seed=99, **kwargs)
    assert set(seed7.obs_names.tolist()) != set(seed99.obs_names.tolist())


# --- build_xatlas_orion_response_adata (Phase C Task 5) ---------------------


def test_build_xatlas_orion_response_adata_selects_only_perturbed_passing_guide(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0, 1],
                gene_expression=[1.0, 2.0],
                cell_barcode="AAA",
                gene_target="TP53",
            ),
            _xatlas_row(
                gene_token_id=[0, 1],
                gene_expression=[9.0, 9.0],
                cell_barcode="BBB",
                gene_target="TP53",
                pass_guide_filter=0,  # fails guide QC -> excluded
            ),
            _xatlas_row(
                gene_token_id=[0, 1],
                gene_expression=[9.0, 9.0],
                cell_barcode="CCC",  # control -> excluded
            ),
            _xatlas_row(
                gene_token_id=[0, 1],
                gene_expression=[3.0, 4.0],
                cell_barcode="DDD",
                gene_target="MYC",
            ),
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0, 1])
    adata = build_xatlas_orion_response_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
    )
    assert adata.n_obs == 2
    assert sorted(adata.obs_names.tolist()) == [
        "HCT116_Batch1:AAA",
        "HCT116_Batch1:DDD",
    ]
    assert sorted(adata.obs["perturbation_gene"].tolist()) == ["MYC", "TP53"]


def test_build_xatlas_orion_response_adata_genes_filter_restricts_selection(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0],
                gene_expression=[1.0],
                cell_barcode="AAA",
                gene_target="TP53",
            ),
            _xatlas_row(
                gene_token_id=[0],
                gene_expression=[2.0],
                cell_barcode="BBB",
                gene_target="MYC",
            ),
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    adata = build_xatlas_orion_response_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        genes=["MYC"],
    )
    assert adata.n_obs == 1
    assert adata.obs["perturbation_gene"].tolist() == ["MYC"]


def test_build_xatlas_orion_response_adata_raises_when_nothing_matches(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            _xatlas_row(
                gene_token_id=[0],
                gene_expression=[1.0],
                cell_barcode="AAA",  # control only, no perturbed cells
            ),
        ],
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    with pytest.raises(ValueError, match="No perturbed cells"):
        build_xatlas_orion_response_adata(
            shard_dir,
            metadata_path,
            cell_line_name="HCT116",
            model_id="ACH-000971",
            cellosaurus_id="CVCL_0291",
        )


# --- build_xatlas_orion_response_adata: per-gene cap (fix-round-2 Defect 2) -


def _write_multi_gene_xatlas_response_shard(
    path: Path, gene_cells: dict[str, int]
) -> None:
    """One X-Atlas-Orion shard with several distinct perturbed genes, each
    with its own cell count -- needed to show one gene capped while another,
    smaller one is left untouched."""
    rows = [
        _xatlas_row(
            gene_token_id=[0],
            gene_expression=[float(index + 1)],
            cell_barcode=f"{gene}_{index}",
            gene_target=gene,
        )
        for gene, n in gene_cells.items()
        for index in range(n)
    ]
    _write_xatlas_shard(path, rows)


def test_build_xatlas_orion_response_adata_caps_per_gene_deterministically(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_multi_gene_xatlas_response_shard(
        shard_dir / "HCT116_Batch1.parquet", {"GENE_A": 20, "GENE_B": 3}
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    kwargs = {
        "cell_line_name": "HCT116",
        "model_id": "ACH-000971",
        "cellosaurus_id": "CVCL_0291",
        "max_cells_per_gene": 5,
        "seed": 3,
    }
    first = build_xatlas_orion_response_adata(shard_dir, metadata_path, **kwargs)
    second = build_xatlas_orion_response_adata(shard_dir, metadata_path, **kwargs)
    counts = first.obs["perturbation_gene"].value_counts()
    assert counts["GENE_A"] == 5, "over-cap gene must be capped to max_cells_per_gene"
    assert counts["GENE_B"] == 3, "under-cap gene must be left untouched"
    assert first.obs_names.tolist() == second.obs_names.tolist()
    np.testing.assert_array_equal(first.X.toarray(), second.X.toarray())


def test_build_xatlas_orion_response_adata_max_cells_per_gene_none_returns_everything(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_multi_gene_xatlas_response_shard(
        shard_dir / "HCT116_Batch1.parquet", {"GENE_A": 7, "GENE_B": 4}
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    adata = build_xatlas_orion_response_adata(
        shard_dir,
        metadata_path,
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        max_cells_per_gene=None,
    )
    assert adata.n_obs == 11


def test_build_xatlas_orion_response_adata_different_seed_changes_capped_selection(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_multi_gene_xatlas_response_shard(
        shard_dir / "HCT116_Batch1.parquet", {"GENE_A": 20}
    )
    metadata_path = _write_xatlas_gene_metadata(tmp_path / "genes.parquet", [0])
    kwargs = {
        "cell_line_name": "HCT116",
        "model_id": "ACH-000971",
        "cellosaurus_id": "CVCL_0291",
        "max_cells_per_gene": 5,
    }
    seed7 = build_xatlas_orion_response_adata(
        shard_dir, metadata_path, seed=7, **kwargs
    )
    seed99 = build_xatlas_orion_response_adata(
        shard_dir, metadata_path, seed=99, **kwargs
    )
    assert set(seed7.obs_names.tolist()) != set(seed99.obs_names.tolist())


def test_perturbseq_basal_adata_emits_ensembl_id_column(tmp_path: Path) -> None:
    """The Tx1 encoder reads var['ensembl_id']; the builder must materialize it.

    Regression: a Perturb-seq h5ad carries its Ensembl ids in ``var.index`` with
    no ``ensembl_id`` column, so the encoder died with ``KeyError: 'ensembl_id'``
    only after the 3B model had already been loaded onto the GPU.
    """
    h5ad_path = _write_perturbseq_h5ad_ensembl_index(
        tmp_path / "index_ids.h5ad", n_control=3, n_other=2
    )
    adata = build_perturbseq_basal_adata(
        h5ad_path,
        control_label="non-targeting",
        perturbation_col="gene",
        cell_line_name="LineP",
        model_id="ACH-P",
        cellosaurus_id="CVCL_P",
        var_ensembl_col="gene_id",
        seed=0,
    )
    assert "ensembl_id" in adata.var.columns
    assert adata.var["ensembl_id"].tolist() == adata.var.index.astype(str).tolist()


def test_tx1_input_contract_rejects_missing_ensembl_id_column() -> None:
    """The contract must enforce what the Tx1 encoder actually requires."""
    var = pd.DataFrame(index=[f"ENSG{index:011d}" for index in range(2)])
    obs = pd.DataFrame({"cell_type": ["LineP"]}, index=["c0"])
    adata = ad.AnnData(
        X=csr_matrix(np.array([[1.0, 2.0]], dtype=np.float32)), obs=obs, var=var
    )
    with pytest.raises(ValueError, match="missing the 'ensembl_id' column"):
        assert_tx1_input_contract(adata)
