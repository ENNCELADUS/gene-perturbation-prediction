"""Tests for the label-blind HCT116 transport cache."""

from pathlib import Path
import json
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse

from scripts.build_exp05_hct116_cache import build_hct116_cache
from aivc_model.gene_splits import sha256_file


def _write_inputs(
    root: Path, *, include_batch_two_control: bool = True
) -> dict[str, Path]:
    metadata = root / "gene_metadata.parquet"
    pd.DataFrame(
        {
            "ensembl_id": ["ENSG1", "ENSG2", "ENSG9"],
            "gene_name": ["A", "B", "C"],
            "gene_token_id": [10, 20, 90],
        }
    ).to_parquet(metadata, index=False)
    k562 = root / "k562.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix(np.ones((1, 3), dtype=np.float32)),
        var=pd.DataFrame(
            {"gene_name": ["A", "B", "C"]},
            index=pd.Index(["ENSG1", "ENSG2", "ENSG3"], name="gene_id"),
        ),
    ).write_h5ad(k562)
    model = root / "state"
    model.mkdir()
    with (model / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": ["A", "B", "C"]}, handle)
    checkpoint = root / "frozen.ckpt"
    checkpoint.write_bytes(b"frozen-head")
    fills = root / "fills.npy"
    np.save(fills, np.asarray([2.0, 3.0, 7.0], dtype=np.float32))
    controls = root / "k562_controls.npy"
    np.save(
        controls,
        np.asarray([[1.0, 2.0, 6.0], [3.0, 4.0, 8.0]], dtype=np.float32),
    )
    contract = root / "transform.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract_id": "test-frozen-transform",
                "target_sum_policy": "median_non_targeting_library_size",
                "log1p": False,
                "relative_control_zscore": True,
                "zscore_groupby": "sample",
                "zscore_ddof": 0,
                "zero_variance_policy": "unit_denominator",
                "provenance_verified": True,
                "source": "synthetic-test-fixture",
                "k562_h5ad_sha256": sha256_file(k562),
                "k562_reference_controls_sha256": sha256_file(controls),
                "frozen_checkpoint_sha256": sha256_file(checkpoint),
                "feature_fill_values_sha256": sha256_file(fills),
                "state_var_dims_sha256": sha256_file(model / "var_dims.pkl"),
            }
        )
        + "\n"
    )
    rows = [
        ([10, 20], [1, 1], "batch1", "Non-Targeting", 1),
        ([10, 20], [3, 1], "batch1", "TP53", 1),
        ([10, 20], [9, 1], "batch1", "TP53", 1),
        ([10, 20], [1, 9], "batch1", "SKIP", 0),
        ([10, 20], [2, 2], "batch2", "KRAS", 1),
    ]
    if include_batch_two_control:
        rows.append(([10, 20], [1, 3], "batch2", "Non-Targeting", 1))
    parquet = root / "hct116.parquet"
    table = pa.table(
        {
            "gene_token_id": [row[0] for row in rows],
            "gene_expression": [row[1] for row in rows],
            "sample": [row[2] for row in rows],
            "gene_target": [row[3] for row in rows],
            "pass_guide_filter": [row[4] for row in rows],
            "depmap_gene_effect": [-99.0] * len(rows),
        }
    )
    pq.write_table(table, parquet, row_group_size=2)
    return {
        "metadata": metadata,
        "k562": k562,
        "model": model,
        "checkpoint": checkpoint,
        "controls": controls,
        "contract": contract,
        "fills": fills,
        "parquet": parquet,
    }


def _build(paths: dict[str, Path], output: Path) -> Path:
    return build_hct116_cache(
        [paths["parquet"]],
        paths["metadata"],
        paths["k562"],
        paths["model"],
        paths["checkpoint"],
        paths["controls"],
        paths["fills"],
        paths["contract"],
        output,
        min_cells_per_group=1,
        max_cells_per_group=1,
        row_batch_size=2,
    )


def test_cache_is_label_blind_context_preserving_and_deterministic(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _build(paths, first)
    _build(paths, second)

    response = np.load(first / "response_cells.npy")
    controls = np.load(first / "control_cells.npy")
    assert np.load(first / "response_genes.npy").tolist() == ["TP53", "KRAS"]
    assert np.load(first / "response_samples.npy").tolist() == ["batch1", "batch2"]
    np.testing.assert_allclose(response, [[0.75, -0.75, 0.0], [0.75, -0.75, 0.0]])
    np.testing.assert_allclose(controls, np.zeros((2, 3), dtype=np.float32))

    first_manifest = __import__("json").loads((first / "manifest.json").read_text())
    second_manifest = __import__("json").loads((second / "manifest.json").read_text())
    assert first_manifest["label_blind"] is True
    assert first_manifest["files"] == second_manifest["files"]
    assert len(first_manifest["sources"]["frozen_checkpoint"]) == 64
    assert first_manifest["parameters"]["min_cells_per_group"] == 1
    qa = __import__("json").loads((first / "qa.json").read_text())
    assert "depmap_gene_effect" not in qa["read_columns"]
    assert qa["mapped_feature_count"] == 3
    assert qa["unresolved_features"] == []
    assert qa["mapping_method_counts"] == {
        "ensembl": 2,
        "exact_symbol": 1,
        "unresolved": 0,
    }
    assert qa["control_alignment"]["method"].startswith("replogle_sample_relative")
    assert qa["control_alignment"]["hct116_to_k562_mean_std_matching"] is False
    assert qa["control_alignment"]["raw_cell_line_baseline_preserved"] is False
    assert qa["control_alignment"]["relative_perturbation_context_preserved"] is True
    assert qa["control_alignment"]["mean_difference_l2"] > 0
    assert qa["normalization"]["resolved_target_sum"] == 3.0
    assert qa["normalization"]["k562_training_transform_verified"] is True
    assert qa["state_batch_embedding"] == "not_used_observed_response_transport"
    assert qa["response_distribution"]["nonfinite_count"] == 0
    assert np.load(first / "response_group_offsets.npy").tolist() == [0, 1, 2]
    assert np.load(first / "response_group_counts.npy").tolist() == [1, 1]


def test_cache_rejects_response_batch_without_controls(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, include_batch_two_control=False)
    with pytest.raises(ValueError, match="samples lack Non-Targeting controls"):
        _build(paths, tmp_path / "cache")


def test_cache_rejects_unverified_transform_provenance(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    contract = json.loads(paths["contract"].read_text())
    contract["provenance_verified"] = False
    paths["contract"].write_text(json.dumps(contract) + "\n")
    with pytest.raises(ValueError, match="provenance must be verified"):
        _build(paths, tmp_path / "cache")


def test_cache_rejects_transform_artifact_mismatch(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    contract = json.loads(paths["contract"].read_text())
    contract["frozen_checkpoint_sha256"] = "0" * 64
    paths["contract"].write_text(json.dumps(contract) + "\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _build(paths, tmp_path / "cache")
