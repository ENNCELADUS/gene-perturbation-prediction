"""Tests for the raw-UMI Kinker ingest (``docs/04`` §6 branch 1).

The CPM sibling's tests assert its output is *blocked* from the Tx1 raw-count
path. These assert the inverse, plus the two guards that are specific to this
matrix's layout: the ``Cell_line`` metadata row must reconcile with the
selection made from the CPM matrix, and a non-integer value must be refused
rather than quietly ingested as counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest

from src.data.basal import assert_tx1_input_contract
from src.data.prepare.prepare_kinker_umi_h5ad import build


def _write_matrix(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n")


def _fixture(tmp_path: Path, *, values: tuple[str, str] = ("4", "7")) -> dict:
    """Build a two-cell, two-line fixture mirroring the SCP file layout."""
    matrix = tmp_path / "UMIcount_data.txt"
    _write_matrix(
        matrix,
        [
            "\tcell-a\tcell-b",
            "Cell_line\tLINE_A\tLINE_B",
            "Pool_ID\tpool1\tpool2",
            f"TSPAN6\t{values[0]}\t{values[1]}",
            "TNMD\t3\t5",
        ]
        + [f"unmapped-{index}\t0\t0" for index in range(8)],
    )
    selected = tmp_path / "selected_cells.tsv"
    pd.DataFrame(
        {
            "cell_id": ["cell-a", "cell-b"],
            "model_id": ["ACH-000001", "ACH-000002"],
            "source_cell_line_name": ["LINE_A", "LINE_B"],
            "source_cluster": ["a", "b"],
            "matrix_file": ["GSE157220_CPM_data.txt.gz"] * 2,
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
    return {
        "matrix": matrix,
        "selected_cells": selected,
        "line_manifest": manifest,
        "gene_metadata": gene_metadata,
        "vocab_json": vocab,
        "hvg_var_dims": hvg,
    }


def _args(tmp_path: Path, files: dict, **overrides) -> argparse.Namespace:
    namespace = argparse.Namespace(
        output_dir=tmp_path / "output",
        expected_matrix_sha256=None,
        selection_matrix_name="GSE157220_CPM_data.txt.gz",
        manifest_matrix_semantics="processed_cpm",
        expected_lines=2,
        expected_selected_cells=2,
        expected_source_cells=2,
        max_hvg_fill_rate=0.6,
        min_detected_genes=1,
        chunk_rows=4,
        **files,
    )
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


def test_build_emits_tx1_contract_passing_raw_count_h5ads(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    qc = build(_args(tmp_path, files))

    assert qc["status"] == "verified_raw_umi"
    assert qc["tx1_raw_count_contract"] == "passed"
    assert qc["source_numeric_audit"]["noninteger_nonzero_values"] == 0
    assert qc["cell_line_labels_cross_checked"] == 2

    written = ad.read_h5ad(tmp_path / "output" / "h5ad" / "ACH-000001.h5ad")
    # The whole point of this ingest: the artifact the CPM path deliberately
    # fails must now pass.
    assert_tx1_input_contract(written)
    assert written.uns["expression_semantics"] == "raw_umi"
    assert written.uns["tx1_standard_raw_count_compatible"] is True
    assert list(written.var.index) == ["ENSG00000000003", "ENSG00000000005"]
    assert "ensembl_id" in written.var.columns


def test_build_refuses_a_non_integer_matrix(tmp_path: Path) -> None:
    """A normalized matrix must not be ingested as counts."""
    files = _fixture(tmp_path, values=("4.5", "7"))
    with pytest.raises(ValueError, match="not raw counts"):
        build(_args(tmp_path, files))
    assert not (tmp_path / "output").exists()


def test_build_rejects_disagreeing_cell_line_labels(tmp_path: Path) -> None:
    """The matrix's own Cell_line row must reconcile with the selection."""
    files = _fixture(tmp_path)
    _write_matrix(
        files["matrix"],
        [
            "\tcell-a\tcell-b",
            "Cell_line\tWRONG_LINE\tLINE_B",
            "Pool_ID\tpool1\tpool2",
            "TSPAN6\t4\t7",
            "TNMD\t3\t5",
        ],
    )
    with pytest.raises(ValueError, match="cell-line labels disagree"):
        build(_args(tmp_path, files))


def test_build_rejects_an_unexpected_header_layout(tmp_path: Path) -> None:
    """A ``GENE`` header means this is the CPM matrix, not the UMI one."""
    files = _fixture(tmp_path)
    _write_matrix(
        files["matrix"],
        [
            "GENE\tcell-a\tcell-b",
            "Cell_line\tLINE_A\tLINE_B",
            "Pool_ID\tpool1\tpool2",
            "TSPAN6\t4\t7",
        ],
    )
    with pytest.raises(ValueError, match="empty field"):
        build(_args(tmp_path, files))


def test_build_rejects_a_missing_metadata_row(tmp_path: Path) -> None:
    """Row offsets are computed from METADATA_ROWS; a missing one shifts genes."""
    files = _fixture(tmp_path)
    _write_matrix(
        files["matrix"],
        [
            "\tcell-a\tcell-b",
            "Cell_line\tLINE_A\tLINE_B",
            "TSPAN6\t4\t7",
            "TNMD\t3\t5",
        ],
    )
    with pytest.raises(ValueError, match="expected metadata row"):
        build(_args(tmp_path, files))


def test_build_pins_the_matrix_hash_when_asked(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    with pytest.raises(ValueError, match="expected-matrix-sha256"):
        build(_args(tmp_path, files, expected_matrix_sha256="0" * 64))

    digest = hashlib.sha256(files["matrix"].read_bytes()).hexdigest()
    qc = build(_args(tmp_path, files, expected_matrix_sha256=digest))
    assert qc["source_hashes"]["matrix_sha256"] == digest
