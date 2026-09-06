from __future__ import annotations

import argparse
import gzip
import json
import pickle
import hashlib
from pathlib import Path

import pandas as pd

from src.experiments.historical.prepare_kinker_cpm_h5ad import build


def test_build_marks_kinker_cpm_as_sensitivity_only(tmp_path: Path) -> None:
    matrix = tmp_path / "GSE157220_CPM_data.txt.gz"
    with gzip.open(matrix, "wt") as handle:
        handle.write("GENE\tcell-a\tcell-b\n")
        handle.write("TSPAN6\t400000.5\t200000.25\n")
        handle.write("TNMD\t599999.5\t799999.75\n")
        for index in range(255):
            handle.write(f"unmapped-{index}\t0\t0\n")
    selected = tmp_path / "selected_cells.tsv"
    pd.DataFrame(
        {
            "cell_id": ["cell-a", "cell-b"],
            "model_id": ["ACH-000001", "ACH-000002"],
            "source_cell_line_name": ["LINE_A", "LINE_B"],
            "source_cluster": ["a", "b"],
            "matrix_file": [matrix.name, matrix.name],
        }
    ).to_csv(selected, sep="\t", index=False)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "source": ["kinker_sccle", "kinker_sccle"],
            "model_id": ["ACH-000001", "ACH-000002"],
            "patient_id": ["PT-1", "PT-2"],
            "cell_line_name": ["Line A", "Line B"],
            "source_cell_line_name": ["LINE_A", "LINE_B"],
            "matrix_semantics": ["processed_cpm", "processed_cpm"],
            "published_cells": [1, 1],
        }
    ).to_csv(manifest, index=False)
    gene_metadata = tmp_path / "gene_metadata.parquet"
    pd.DataFrame(
        {
            "gene_symbol": ["<pad>", "TSPAN6", "TNMD"],
            "ensembl_id": ["not-a-gene", "ENSG00000000003", "ENSG00000000005"],
            "token_id": [0, 3, 4],
        }
    ).to_parquet(gene_metadata)
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"ENSG00000000003": 3, "ENSG00000000005": 4}))
    hvg = tmp_path / "var_dims.pkl"
    hvg.write_bytes(pickle.dumps({"gene_names": ["TSPAN6", "MISSING"]}))
    upstream_sha = tmp_path / "sha256.txt"
    upstream_sha.write_text(
        f"{hashlib.sha256(matrix.read_bytes()).hexdigest()}  {matrix.name}\n"
    )
    output = tmp_path / "output"

    qc = build(
        argparse.Namespace(
            matrix=matrix,
            selected_cells=selected,
            line_manifest=manifest,
            gene_metadata=gene_metadata,
            vocab_json=vocab,
            hvg_var_dims=hvg,
            upstream_sha256=upstream_sha,
            output_dir=output,
            expected_lines=2,
            expected_selected_cells=2,
            expected_source_cells=2,
            expected_cpm_library_sum=1_000_000.0,
            cpm_library_sum_rtol=0.05,
            cpm_median_rtol=0.05,
            max_library_sum_cv=0.05,
            min_cpm_scale_pass_fraction=1.0,
            min_noninteger_fraction=0.01,
            max_hvg_fill_rate=0.5,
            min_detected_genes=1,
        )
    )

    assert qc["status"] == "verified_processed_cpm_sensitivity_only"
    assert qc["selected_cells"] == 2
    assert qc["selected_lines"] == 2
    assert qc["hvg_fill_rate"] == 0.5
    assert qc["tx1_raw_count_contract"] == "blocked"
    assert (output / "CPM_SENSITIVITY_H5AD_VERIFIED").is_file()
    manifest_out = pd.read_csv(output / "manifest.tsv", sep="\t")
    assert set(manifest_out["tx1_raw_count_contract"]) == {"blocked"}
    mapping = pd.read_csv(output / "gene_mapping.tsv", sep="\t")
    assert mapping["sensitivity_ensembl_id"].tolist() == [
        "ENSG00000000003",
        "ENSG00000000005",
    ]


def test_build_rejects_selected_cell_absent_from_matrix(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.txt.gz"
    with gzip.open(matrix, "wt") as handle:
        handle.write("GENE\tpresent\nTSPAN6\t1.5\n")
    selected = tmp_path / "selected.tsv"
    pd.DataFrame(
        {
            "cell_id": ["absent"],
            "model_id": ["ACH-000001"],
            "source_cell_line_name": ["LINE_A"],
            "source_cluster": ["a"],
            "matrix_file": [matrix.name],
        }
    ).to_csv(selected, sep="\t", index=False)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "source": ["kinker_sccle"],
            "model_id": ["ACH-000001"],
            "patient_id": ["PT-1"],
            "cell_line_name": ["Line A"],
            "source_cell_line_name": ["LINE_A"],
            "matrix_semantics": ["processed_cpm"],
            "published_cells": [1],
        }
    ).to_csv(manifest, index=False)
    metadata = tmp_path / "metadata.parquet"
    pd.DataFrame(
        {
            "gene_symbol": ["TSPAN6"],
            "ensembl_id": ["ENSG00000000003"],
            "token_id": [3],
        }
    ).to_parquet(metadata)
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"ENSG00000000003": 3}))
    hvg = tmp_path / "var_dims.pkl"
    hvg.write_bytes(pickle.dumps({"gene_names": ["TSPAN6"]}))
    upstream_sha = tmp_path / "sha256.txt"
    upstream_sha.write_text(
        f"{hashlib.sha256(matrix.read_bytes()).hexdigest()}  {matrix.name}\n"
    )

    import pytest

    with pytest.raises(ValueError, match="selected cells absent"):
        build(
            argparse.Namespace(
                matrix=matrix,
                selected_cells=selected,
                line_manifest=manifest,
                gene_metadata=metadata,
                vocab_json=vocab,
                hvg_var_dims=hvg,
                upstream_sha256=upstream_sha,
                output_dir=tmp_path / "output",
                expected_lines=1,
                expected_selected_cells=1,
                expected_source_cells=1,
                expected_cpm_library_sum=1.5,
                cpm_library_sum_rtol=0.05,
                cpm_median_rtol=0.05,
                max_library_sum_cv=0.05,
                min_cpm_scale_pass_fraction=1.0,
                min_noninteger_fraction=0.01,
                max_hvg_fill_rate=0.0,
                min_detected_genes=1,
            )
        )
