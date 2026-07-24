"""Tests for src/aivc_model/tx1_basal.py -- CPU-only basal AnnData assembly.

All fixtures are tiny synthetic parquet shards / h5ad files built in
``tmp_path``; the real Tahoe-100M and Perturb-seq sources live on the HPC
and are never touched here (Wave 1 Phase B, Task 1).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from aivc_model.tx1_basal import (
    assert_tx1_input_contract,
    build_perturbseq_basal_adata,
    build_tahoe_basal_adata,
    load_line_manifest,
)

_MANIFEST_COLUMNS = (
    "model_id",
    "cellosaurus_id",
    "cell_line_name",
    "lineage",
    "dmso_cells",
    "basal_source",
    "tx1_pretraining_exposure",
    "role",
    "omics_expression_available",
)


def _manifest_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model_id": "ACH-000001",
        "cellosaurus_id": "CVCL_0001",
        "cell_line_name": "Line1",
        "lineage": "Lung",
        "dmso_cells": 1000,
        "basal_source": "Tahoe-100M DMSO",
        "tx1_pretraining_exposure": "known_present",
        "role": "train_head",
        "omics_expression_available": True,
    }
    row.update(overrides)
    return row


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows, columns=_MANIFEST_COLUMNS).to_csv(path, index=False)
    return path


def _write_gene_metadata(path: Path, tokens: list[int]) -> Path:
    pd.DataFrame(
        {
            "token_id": tokens,
            "ensembl_id": [f"ENSG{token:011d}" for token in tokens],
            "gene_symbol": [f"GENE{token}" for token in tokens],
        }
    ).to_parquet(path)
    return path


def _write_shard(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_parquet(path)
    return path


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
        var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
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


# --- build_perturbseq_basal_adata ----------------------------------------


def _write_perturbseq_h5ad(
    path: Path,
    *,
    n_control: int,
    n_other: int,
    ensembl_col: str = "ensembl_id",
    perturbation_col: str = "gene",
    control_label: str = "non-targeting",
    n_genes: int = 3,
) -> Path:
    n_cells = n_control + n_other
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 10, size=(n_cells, n_genes)).astype(np.float32)
    labels = [control_label] * n_control + ["TP53"] * n_other
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
