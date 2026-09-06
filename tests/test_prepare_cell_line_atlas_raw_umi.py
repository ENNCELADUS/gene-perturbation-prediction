from __future__ import annotations

import json
import pickle
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

from src.data.prepare.prepare_cell_line_atlas_raw_umi import run


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    genes = ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"]
    pd.DataFrame(
        {
            "ensembl_id": genes,
            "gene_symbol": ["A", "B", "AL627309.1"],
            "token_id": [3, 4, 5],
        }
    ).to_parquet(tmp_path / "gene_metadata.parquet", index=False)
    (tmp_path / "vocab.json").write_text(
        json.dumps({gene: index for index, gene in enumerate(genes, start=3)})
    )
    state = tmp_path / "state"
    state.mkdir()
    with (state / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": ["A", "B", "MISSING_HVG"]}, handle)
    manifest = pd.DataFrame(
        [
            {
                "source": "breast",
                "model_id": "ACH-000001",
                "patient_id": "PT-1",
                "cell_line_name": "BREAST1",
                "source_cell_line_name": "BREAST1",
                "source_matrix": "breast.mtx",
                "matrix_semantics": "raw_umi_counts",
            },
            {
                "source": "ccla",
                "model_id": "ACH-000002",
                "patient_id": "PT-2",
                "cell_line_name": "CCLA1",
                "source_cell_line_name": "CCLA1",
                "source_matrix": "ccla.tsv",
                "matrix_semantics": "raw_umi_counts",
            },
        ]
    )
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    mmwrite(tmp_path / "breast.mtx", sparse.coo_matrix([[1, 0], [0, 2], [3, 4]]))
    (tmp_path / "features.tsv").write_text("\n".join(genes) + "\n")
    (tmp_path / "barcodes.tsv").write_text("b1\nb2\n")
    pd.DataFrame(
        {
            "model_id": ["ACH-000001"],
            "cell_id": ["b2"],
            "matrix_file": ["breast.mtx"],
        }
    ).to_csv(tmp_path / "breast_selected.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"ID": ["A", "AL627309.1", "NO_MATCH"], "c1": [5, 6, 7], "other": [1, 1, 1]}
    ).to_csv(tmp_path / "ccla.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "model_id": ["ACH-000002"],
            "cell_id": ["c1"],
            "matrix_file": ["ccla.tsv"],
        }
    ).to_csv(tmp_path / "ccla_selected.tsv", sep="\t", index=False)
    config = {
        "manifest_path": str(tmp_path / "manifest.csv"),
        "gene_metadata_path": str(tmp_path / "gene_metadata.parquet"),
        "gene_metadata_ensembl_column": "ensembl_id",
        "gene_metadata_symbol_column": "gene_symbol",
        "gene_metadata_token_column": "token_id",
        "minimum_token_id": 3,
        "tx1_vocab_path": str(tmp_path / "vocab.json"),
        "tx1_model_dir": str(tmp_path / "unused-model"),
        "run_tx1_loader_probe": False,
        "tx1_loader_probe_max_length": 2048,
        "state_model_dir": str(state),
        "expected_lines": 2,
        "expected_matrix_semantics": "raw_umi_counts",
        "sources": {
            "breast": {
                "kind": "matrix_market_genes_by_cells",
                "matrix_path": str(tmp_path / "breast.mtx"),
                "features_path": str(tmp_path / "features.tsv"),
                "feature_column": 0,
                "barcodes_path": str(tmp_path / "barcodes.tsv"),
                "barcode_column": 0,
                "selected_cells_path": str(tmp_path / "breast_selected.tsv"),
                "selected_model_id_column": "model_id",
                "selected_cell_id_column": "cell_id",
                "selected_matrix_column": "matrix_file",
                "gene_id_kind": "ensembl",
            },
            "ccla": {
                "kind": "wide_tsv_genes_by_cells",
                "source_root": str(tmp_path),
                "feature_id_column": "ID",
                "selected_cells_path": str(tmp_path / "ccla_selected.tsv"),
                "selected_model_id_column": "model_id",
                "selected_cell_id_column": "cell_id",
                "selected_matrix_column": "matrix_file",
                "gene_id_kind": "symbol",
            },
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, tmp_path / "output"


def test_run_writes_atomic_audited_h5ad_bundle(tmp_path: Path) -> None:
    config, output = _fixture(tmp_path)
    run(config, output, None)
    assert (output / "SOURCE_H5AD_VERIFIED").exists()
    summary = json.loads((output / "qc" / "summary.json").read_text())
    assert summary["n_lines"] == 2
    assert summary["n_cells"] == 2
    assert (output / "provenance" / "config_snapshot.json").exists()
    input_hashes = json.loads((output / "provenance" / "input_sha256.json").read_text())
    assert str(tmp_path / "manifest.csv") in input_hashes
    breast = ad.read_h5ad(output / "h5ad" / "ACH-000001.h5ad")
    assert sparse.issparse(breast.X)
    assert breast.obs["source_cell_id"].tolist() == ["b2"]
    assert breast.var["ensembl_id"].tolist() == [
        "ENSG00000000002",
        "ENSG00000000003",
    ]
    ccla = ad.read_h5ad(output / "h5ad" / "ACH-000002.h5ad")
    assert ccla.obs["source_cell_id"].tolist() == ["c1"]
    assert ccla.var["gene_symbol"].tolist() == ["A", "AL627309.1"]


def test_failure_does_not_write_success_or_output(tmp_path: Path) -> None:
    config, output = _fixture(tmp_path)
    payload = json.loads(config.read_text())
    payload["sources"]["ccla"]["feature_id_column"] = "guessed_wrong"
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing required columns"):
        run(config, output, None)
    assert not output.exists()


def test_only_model_id_is_a_one_line_smoke(tmp_path: Path) -> None:
    config, output = _fixture(tmp_path)
    run(config, output, "ACH-000001")
    assert sorted(path.name for path in (output / "h5ad").glob("*.h5ad")) == [
        "ACH-000001.h5ad"
    ]


def test_unknown_only_model_id_has_clear_error(tmp_path: Path) -> None:
    config, output = _fixture(tmp_path)
    with pytest.raises(ValueError, match="absent from the configured raw-UMI"):
        run(config, output, "ACH-DOES-NOT-EXIST")
    assert not output.exists()
