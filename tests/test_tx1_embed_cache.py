"""Tx1 basal embedding cache: existing on-disk format, encoding missing lines."""

from __future__ import annotations

import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.data.tx1_cache import (
    EMBEDDING_WIDTH,
    encode_lines,
    load_hvg_gene_order,
    load_line_cache,
    missing_lines,
    read_registry_source,
    tx1_cell_indices,
    write_line_cache,
)


def _source(path: Path, model_id: str, n_cells: int = 6, *, scale: float = 1.0) -> Path:
    counts = np.arange(n_cells * 4, dtype=np.float32).reshape(n_cells, 4) * scale
    obs = pd.DataFrame(
        {"model_id": model_id}, index=[f"{model_id}-{i}" for i in range(n_cells)]
    )
    var = pd.DataFrame(
        {
            "ensembl_id": [f"ENSG{i:011d}" for i in range(4)],
            "gene_symbol": ["A", "B", "A", "C"],
        },
        index=list("wxyz"),
    )
    ad.AnnData(X=csr_matrix(counts), obs=obs, var=var).write_h5ad(path)
    return path


def test_reader_opens_the_existing_format_without_sidecars(tmp_path):
    line = tmp_path / "ACH-1"
    line.mkdir()
    embeddings = np.random.default_rng(0).normal(size=(3, EMBEDDING_WIDTH))
    np.save(line / "embeddings.npy", embeddings.astype(np.float32))
    np.save(line / "hvg.npy", np.ones((3, 2), dtype=np.float32))
    pd.DataFrame(index=["c0", "c1", "c2"]).to_parquet(line / "obs.parquet")
    loaded, hvg, obs = load_line_cache(tmp_path, "ACH-1")
    np.testing.assert_array_equal(loaded, embeddings.astype(np.float32))
    assert hvg.shape == (3, 2)
    assert list(obs.index) == ["c0", "c1", "c2"]
    assert missing_lines(tmp_path, ["ACH-1", "ACH-2"]) == ["ACH-2"]


def test_writer_rejects_wrong_embedding_width(tmp_path):
    obs = pd.DataFrame(index=["c0"])
    with pytest.raises(ValueError, match="embeddings"):
        write_line_cache(tmp_path, "ACH-1", np.zeros((1, 5)), np.zeros((1, 2)), obs)
    assert not (tmp_path / "ACH-1").exists()


def test_hvg_gene_order_comes_from_var_dims(tmp_path):
    (tmp_path / "var_dims.pkl").write_bytes(pickle.dumps({"gene_names": ["B", "A"]}))
    assert list(load_hvg_gene_order(tmp_path)) == ["B", "A"]


def test_encode_lines_writes_selected_raw_cells(tmp_path):
    registry = pd.DataFrame(
        {"source_path": [str(_source(tmp_path / "a.h5ad", "ACH-1"))]},
        index=pd.Index(["ACH-1"], name="model_id"),
    )
    seen = []

    def encoder(adata):
        seen.append(adata)
        return np.ones((adata.n_obs, EMBEDDING_WIDTH), dtype=np.float32)

    encoded = encode_lines(
        registry,
        tmp_path / "cache",
        encoder=encoder,
        hvg_order=("A", "C", "Z"),
        var_ensembl_col="ensembl_id",
        hvg_gene_symbol_col="auto",
        max_cells_per_line=4,
        seed=0,
    )
    assert encoded == ["ACH-1"]
    cells = tx1_cell_indices([f"ACH-1-{i}" for i in range(6)], "ACH-1", 4, 0)
    assert list(seen[0].obs_names) == [f"ACH-1-{i}" for i in cells]
    np.testing.assert_array_equal(
        seen[0].X.toarray(), np.arange(24).reshape(6, 4)[cells]
    )
    _, hvg, obs = load_line_cache(tmp_path / "cache", "ACH-1")
    raw = np.arange(24, dtype=np.float32).reshape(6, 4)[cells]
    np.testing.assert_array_equal(
        hvg, np.column_stack([raw[:, 0] + raw[:, 2], raw[:, 3], np.zeros(4)])
    )
    assert list(obs.index) == list(seen[0].obs_names)


def test_registry_source_must_be_raw_counts_of_its_line(tmp_path):
    with pytest.raises(ValueError, match="integer"):
        read_registry_source(
            _source(tmp_path / "a.h5ad", "ACH-1", scale=0.5),
            model_id="ACH-1",
            var_ensembl_col="ensembl_id",
        )
    with pytest.raises(ValueError, match="model_id"):
        read_registry_source(
            _source(tmp_path / "b.h5ad", "ACH-2"),
            model_id="ACH-1",
            var_ensembl_col="ensembl_id",
        )
