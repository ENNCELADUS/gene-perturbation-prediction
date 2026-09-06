"""Tests for src/data/response_cache.py (fix-round-3, Fix 2).

The one non-negotiable property this module exists for: a stale cache is
NEVER silently reused. Every "refuses to load" test below asserts a raise,
not a warning-and-continue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.response_cache import (
    load_response_targets_cache,
    response_targets_fingerprint,
    write_response_targets_cache,
)


def _write_bytes(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def _sample_bags() -> tuple[list[str], list[np.ndarray], pd.DataFrame]:
    genes = ["GENE_A@ACH-1", "GENE_B@ACH-1", "GENE_A@ACH-2"]
    target_bags = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[5.0, 6.0]], dtype=np.float32),
        np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
    ]
    metadata = pd.DataFrame(
        {
            "perturbation_gene": ["GENE_A", "GENE_B", "GENE_A"],
            "model_id": ["ACH-1", "ACH-1", "ACH-2"],
            "cell_line_name": ["LineOne", "LineOne", "LineTwo"],
            "n_cells": [2, 1, 3],
        }
    )
    return genes, target_bags, metadata


def _fingerprint_inputs(tmp_path: Path) -> dict[str, object]:
    manifest = _write_bytes(tmp_path / "manifest.csv", b"manifest-v1")
    sources = _write_bytes(tmp_path / "sources.json", b"{}")
    source_a = _write_bytes(tmp_path / "source_a.h5ad", b"source-a")
    tx1_manifest = _write_bytes(tmp_path / "tx1_manifest.json", b"{}")
    var_dims = _write_bytes(tmp_path / "var_dims.pkl", b"var-dims")
    return dict(
        cell_line_manifest_path=manifest,
        perturbseq_sources_path=sources,
        referenced_source_paths=[source_a],
        tx1_cache_manifest_path=tx1_manifest,
        checkpoint_var_dims_path=var_dims,
        max_cells_per_gene=256,
        total_cells_per_line=None,
        seed=42,
        genes=None,
    )


# --- write/load round trip --------------------------------------------------


def test_write_then_load_round_trips_exactly(tmp_path: Path) -> None:
    genes, target_bags, metadata = _sample_bags()
    fingerprint = response_targets_fingerprint(**_fingerprint_inputs(tmp_path))
    cache_dir = tmp_path / "cache"
    write_response_targets_cache(
        cache_dir, fingerprint, genes=genes, target_bags=target_bags, metadata=metadata
    )
    loaded_genes, loaded_target_bags, loaded_metadata = load_response_targets_cache(
        cache_dir, fingerprint
    )
    assert loaded_genes.tolist() == genes
    for expected, actual in zip(target_bags, loaded_target_bags, strict=True):
        np.testing.assert_array_equal(actual, expected)
    pd.testing.assert_frame_equal(
        loaded_metadata.reset_index(drop=True), metadata.reset_index(drop=True)
    )


def test_write_then_load_empty_bags_round_trips(tmp_path: Path) -> None:
    fingerprint = response_targets_fingerprint(**_fingerprint_inputs(tmp_path))
    cache_dir = tmp_path / "cache"
    metadata = pd.DataFrame(
        {"perturbation_gene": [], "model_id": [], "cell_line_name": [], "n_cells": []}
    )
    write_response_targets_cache(
        cache_dir, fingerprint, genes=[], target_bags=[], metadata=metadata
    )
    genes, target_bags, loaded_metadata = load_response_targets_cache(
        cache_dir, fingerprint
    )
    assert len(genes) == 0
    assert len(target_bags) == 0
    assert len(loaded_metadata) == 0


# --- refuse-and-rebuild discipline (never warn-and-continue) --------------


def test_load_missing_cache_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_response_targets_cache(tmp_path / "never_written", "any-fingerprint")


def test_load_stale_fingerprint_raises_not_warns(tmp_path: Path) -> None:
    genes, target_bags, metadata = _sample_bags()
    fingerprint = response_targets_fingerprint(**_fingerprint_inputs(tmp_path))
    cache_dir = tmp_path / "cache"
    write_response_targets_cache(
        cache_dir, fingerprint, genes=genes, target_bags=target_bags, metadata=metadata
    )
    with pytest.raises(ValueError, match="stale"):
        load_response_targets_cache(cache_dir, "a-completely-different-fingerprint")


def test_load_stale_schema_version_raises(tmp_path: Path) -> None:
    import json

    genes, target_bags, metadata = _sample_bags()
    fingerprint = response_targets_fingerprint(**_fingerprint_inputs(tmp_path))
    cache_dir = tmp_path / "cache"
    write_response_targets_cache(
        cache_dir, fingerprint, genes=genes, target_bags=target_bags, metadata=metadata
    )
    manifest_path = cache_dir / "response_targets" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = -1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema_version"):
        load_response_targets_cache(cache_dir, fingerprint)


def test_write_rejects_mismatched_bag_lengths(tmp_path: Path) -> None:
    genes, target_bags, metadata = _sample_bags()
    with pytest.raises(ValueError, match="same length"):
        write_response_targets_cache(
            tmp_path / "cache",
            "fingerprint",
            genes=genes[:2],  # mismatched length vs target_bags/metadata
            target_bags=target_bags,
            metadata=metadata,
        )


def test_write_rejects_mismatched_target_widths(tmp_path: Path) -> None:
    genes = ["GENE_A@ACH-1", "GENE_B@ACH-1"]
    target_bags = [
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),  # different width -- must reject
    ]
    metadata = pd.DataFrame(
        {
            "perturbation_gene": ["GENE_A", "GENE_B"],
            "model_id": ["ACH-1", "ACH-1"],
            "cell_line_name": ["LineOne", "LineOne"],
            "n_cells": [2, 2],
        }
    )
    with pytest.raises(ValueError, match="target width"):
        write_response_targets_cache(
            tmp_path / "cache",
            "fingerprint",
            genes=genes,
            target_bags=target_bags,
            metadata=metadata,
        )


# --- fingerprint sensitivity: every declared input actually changes it -----


def test_fingerprint_changes_when_max_cells_per_gene_changes(tmp_path: Path) -> None:
    inputs = _fingerprint_inputs(tmp_path)
    baseline = response_targets_fingerprint(**inputs)
    inputs["max_cells_per_gene"] = 128
    assert response_targets_fingerprint(**inputs) != baseline


def test_fingerprint_changes_when_total_cells_per_line_changes(tmp_path: Path) -> None:
    inputs = _fingerprint_inputs(tmp_path)
    baseline = response_targets_fingerprint(**inputs)
    inputs["total_cells_per_line"] = 1000
    assert response_targets_fingerprint(**inputs) != baseline


def test_fingerprint_changes_when_seed_changes(tmp_path: Path) -> None:
    inputs = _fingerprint_inputs(tmp_path)
    baseline = response_targets_fingerprint(**inputs)
    inputs["seed"] = 7
    assert response_targets_fingerprint(**inputs) != baseline


def test_fingerprint_changes_when_genes_restriction_changes(tmp_path: Path) -> None:
    inputs = _fingerprint_inputs(tmp_path)
    baseline = response_targets_fingerprint(**inputs)
    inputs["genes"] = ["GENE_A", "GENE_B"]
    assert response_targets_fingerprint(**inputs) != baseline


def test_fingerprint_changes_when_manifest_content_changes(tmp_path: Path) -> None:
    inputs = _fingerprint_inputs(tmp_path)
    baseline = response_targets_fingerprint(**inputs)
    _write_bytes(Path(inputs["cell_line_manifest_path"]), b"manifest-v2-different")
    assert response_targets_fingerprint(**inputs) != baseline


def test_fingerprint_changes_when_referenced_source_bytes_change_at_same_path(
    tmp_path: Path,
) -> None:
    """A modified raw source file at the SAME configured path must still
    invalidate the cache -- not just a changed path/glob string."""
    inputs = _fingerprint_inputs(tmp_path)
    baseline = response_targets_fingerprint(**inputs)
    (source_path,) = inputs["referenced_source_paths"]
    _write_bytes(Path(source_path), b"modified-source-bytes")
    assert response_targets_fingerprint(**inputs) != baseline


def test_fingerprint_changes_when_tx1_cache_manifest_changes(tmp_path: Path) -> None:
    inputs = _fingerprint_inputs(tmp_path)
    baseline = response_targets_fingerprint(**inputs)
    _write_bytes(Path(inputs["tx1_cache_manifest_path"]), b'{"different": true}')
    assert response_targets_fingerprint(**inputs) != baseline


def test_fingerprint_changes_when_checkpoint_var_dims_changes(tmp_path: Path) -> None:
    inputs = _fingerprint_inputs(tmp_path)
    baseline = response_targets_fingerprint(**inputs)
    _write_bytes(Path(inputs["checkpoint_var_dims_path"]), b"different-var-dims")
    assert response_targets_fingerprint(**inputs) != baseline


def test_fingerprint_stable_for_identical_inputs(tmp_path: Path) -> None:
    inputs = _fingerprint_inputs(tmp_path)
    assert response_targets_fingerprint(**inputs) == response_targets_fingerprint(
        **inputs
    )


def test_fingerprint_tx1_cache_manifest_missing_is_a_stable_value(
    tmp_path: Path,
) -> None:
    """A never-yet-built Tx1 cache manifest (unusual, but the code must not
    crash) hashes to a stable ``None`` marker rather than raising."""
    inputs = _fingerprint_inputs(tmp_path)
    inputs["tx1_cache_manifest_path"] = tmp_path / "does_not_exist.json"
    first = response_targets_fingerprint(**inputs)
    second = response_targets_fingerprint(**inputs)
    assert first == second
