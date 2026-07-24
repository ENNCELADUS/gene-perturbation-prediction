"""Tests for src/aivc_model/tx1_embed_cache.py -- Tx1 basal embedding cache.

No GPU and no real Tx1/Tahoe/Perturb-seq data exist on this machine (Wave 1
Phase B Global Constraint 6), so every test substitutes a deterministic stub
``EncoderFn`` over tiny synthetic fixtures built in ``tmp_path``, reusing the
shared parquet-shard/h5ad/manifest fixtures in ``conftest.py``.
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aivc_model.tx1_embed_cache as tx1_embed_cache_module
from aivc_model.tx1_basal import load_line_manifest
from aivc_model.tx1_embed_cache import (
    EMBEDDING_WIDTH,
    MODEL_LABEL,
    REJECTED_MODEL_LABEL,
    PerturbseqSource,
    embed_lines,
    embedding_norm_stats,
    load_hvg_gene_order,
    load_line_cache,
    verify_cache,
    write_line_cache,
    write_run_manifest,
)
from conftest import tx1_manifest_row as _manifest_row
from conftest import write_tx1_gene_metadata as _write_gene_metadata
from conftest import write_tx1_line_manifest as _write_manifest
from conftest import write_tx1_perturbseq_h5ad as _write_perturbseq_h5ad
from conftest import write_tx1_shard as _write_shard

# --- shared fixtures --------------------------------------------------------


def _write_var_dims(model_dir: Path, gene_names: list[str]) -> Path:
    """Write a minimal released-checkpoint ``var_dims.pkl``."""
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": gene_names}, handle)
    return model_dir


def _embeddings(n_cells: int) -> np.ndarray:
    """A valid-width (``EMBEDDING_WIDTH``) array with one distinct value per row."""
    values = (np.arange(n_cells, dtype=np.float32) + 1.0)[:, None]
    return np.tile(values, (1, EMBEDDING_WIDTH))


def _obs(n_cells: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"cell_type": ["LineA"] * n_cells},
        index=[f"cell{index}" for index in range(n_cells)],
    )


class _CountingEncoder:
    """Deterministic stub ``EncoderFn`` that counts its own invocations."""

    def __init__(self, width: int = EMBEDDING_WIDTH) -> None:
        self.width = width
        self.call_count = 0

    def __call__(self, adata: object) -> np.ndarray:
        self.call_count += 1
        n_cells = adata.n_obs
        values = (np.arange(n_cells, dtype=np.float32) + 1.0)[:, None]
        return np.tile(values, (1, self.width)).astype(np.float32)


class _RaisingOnceEncoder:
    """Raises on the ``fail_on_call``-th invocation (1-indexed), else succeeds."""

    def __init__(self, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.call_count = 0

    def __call__(self, adata: object) -> np.ndarray:
        self.call_count += 1
        if self.call_count == self.fail_on_call:
            raise RuntimeError("simulated encoder crash")
        return _embeddings(adata.n_obs)


def _tahoe_shard_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Two Tahoe DMSO lines (CVCL_A, CVCL_B), 3 genes (tokens 3,4,5), 2 cells each."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rows = [
        {
            "genes": np.array([3, 4, 5]),
            "expressions": np.array([1.0 + cell, 2.0 + cell, 3.0 + cell]),
            "cell_line_id": cellosaurus_id,
        }
        for cellosaurus_id in ("CVCL_A", "CVCL_B")
        for cell in range(2)
    ]
    _write_shard(shard_dir / "part-0.parquet", rows)
    gene_metadata_path = _write_gene_metadata(tmp_path / "genes.parquet", [3, 4, 5])
    return shard_dir, gene_metadata_path


def _two_line_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _manifest_row(
                model_id="ACH-A", cellosaurus_id="CVCL_A", cell_line_name="LineA"
            ),
            _manifest_row(
                model_id="ACH-B", cellosaurus_id="CVCL_B", cell_line_name="LineB"
            ),
        ]
    )


def _build_good_cache(tmp_path: Path) -> Path:
    """Write a 2-line cache with a matching run manifest.json; return cache_dir."""
    cache_dir = tmp_path / "cache"
    line_entries: dict[str, dict[str, object]] = {}
    for model_id, n_cells in (("ACH-A", 2), ("ACH-B", 3)):
        embeddings = _embeddings(n_cells)
        hvg = np.arange(n_cells * 2, dtype=np.float32).reshape(n_cells, 2)
        arrays = write_line_cache(
            cache_dir,
            model_id,
            embeddings,
            hvg,
            _obs(n_cells),
            hvg_gene_order=["G0", "G1"],
        )
        line_entries[model_id] = {
            "arrays": arrays,
            "norm_stats": embedding_norm_stats(embeddings),
            "n_cells": n_cells,
            "hvg_fill_rate": 0.0,
            "basal_source": "Tahoe-100M DMSO",
            "cellosaurus_id": f"CVCL_{model_id}",
        }
    write_run_manifest(
        cache_dir,
        model_label=MODEL_LABEL,
        source_manifest={"files": {}},
        line_entries=line_entries,
        config_snapshot={"seed": 0},
    )
    return cache_dir


# --- write_line_cache / load_line_cache -------------------------------------


def test_write_line_cache_round_trip(tmp_path: Path) -> None:
    embeddings = _embeddings(3)
    hvg = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    obs = _obs(3)
    metadata = write_line_cache(
        tmp_path, "ACH-A", embeddings, hvg, obs, hvg_gene_order=["G0", "G1"]
    )
    assert set(metadata) == {"embeddings.npy", "hvg.npy", "obs.parquet"}
    for entry in metadata.values():
        assert set(entry) == {"sha256", "shape", "dtype"}
    loaded_embeddings, loaded_hvg, loaded_obs = load_line_cache(tmp_path, "ACH-A")
    np.testing.assert_array_equal(np.asarray(loaded_embeddings), embeddings)
    np.testing.assert_array_equal(np.asarray(loaded_hvg), hvg)
    assert loaded_obs["cell_type"].tolist() == obs["cell_type"].tolist()


def test_write_line_cache_wrong_width_raises(tmp_path: Path) -> None:
    embeddings = np.zeros((2, EMBEDDING_WIDTH - 1), dtype=np.float32)
    hvg = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        write_line_cache(
            tmp_path, "ACH-A", embeddings, hvg, _obs(2), hvg_gene_order=["G0"]
        )
    message = str(excinfo.value)
    assert str(EMBEDDING_WIDTH - 1) in message
    assert str(EMBEDDING_WIDTH) in message


def test_write_line_cache_hvg_gene_order_mismatch_raises(tmp_path: Path) -> None:
    embeddings = _embeddings(2)
    hvg = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="hvg_matrix width"):
        write_line_cache(
            tmp_path, "ACH-A", embeddings, hvg, _obs(2), hvg_gene_order=["G0", "G1"]
        )


def test_write_line_cache_row_count_mismatch_raises(tmp_path: Path) -> None:
    embeddings = _embeddings(2)
    hvg = np.zeros((3, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="row count mismatch"):
        write_line_cache(
            tmp_path, "ACH-A", embeddings, hvg, _obs(2), hvg_gene_order=["G0"]
        )


def test_write_line_cache_non_finite_embeddings_raises(tmp_path: Path) -> None:
    embeddings = _embeddings(2)
    embeddings[0, 0] = np.nan
    hvg = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        write_line_cache(
            tmp_path, "ACH-A", embeddings, hvg, _obs(2), hvg_gene_order=["G0"]
        )


def test_write_line_cache_atomic_no_partial_dir_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-write must leave no partial (or temp) line directory."""
    original_save = tx1_embed_cache_module.np.save
    call_count = {"n": 0}

    def flaky_save(path: Path, array: np.ndarray) -> None:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated crash")
        return original_save(path, array)

    monkeypatch.setattr(tx1_embed_cache_module.np, "save", flaky_save)
    embeddings = _embeddings(2)
    hvg = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(RuntimeError, match="simulated crash"):
        write_line_cache(
            tmp_path, "ACH-A", embeddings, hvg, _obs(2), hvg_gene_order=["G0"]
        )
    assert not (tmp_path / "ACH-A").exists()
    assert list(tmp_path.glob(".tmp-*")) == []


# --- embedding_norm_stats ----------------------------------------------------


def test_embedding_norm_stats_hand_computed() -> None:
    embeddings = np.array([[3.0, 4.0], [0.6, 0.8], [1.0, 0.0]], dtype=np.float32)
    stats = embedding_norm_stats(embeddings)
    assert stats["mean"] == pytest.approx(7.0 / 3.0, rel=1e-5)
    assert stats["std"] == pytest.approx(1.8856180831641267, rel=1e-5)
    assert stats["min"] == pytest.approx(1.0, rel=1e-5)
    assert stats["max"] == pytest.approx(5.0, rel=1e-5)
    assert stats["fraction_unit_norm"] == pytest.approx(2.0 / 3.0, rel=1e-5)


def test_embedding_norm_stats_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        embedding_norm_stats(np.zeros((0, 4), dtype=np.float32))


# --- write_run_manifest -------------------------------------------------


def test_write_run_manifest_rejects_rejected_label(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=REJECTED_MODEL_LABEL):
        write_run_manifest(
            tmp_path,
            model_label=REJECTED_MODEL_LABEL,
            source_manifest={},
            line_entries={},
            config_snapshot={},
        )


def test_write_run_manifest_records_config_snapshot(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    path = write_run_manifest(
        cache_dir,
        model_label=MODEL_LABEL,
        source_manifest={"files": {"model.safetensors": {"sha256": "abc"}}},
        line_entries={},
        config_snapshot={"seed": 7, "max_cells_per_line": 100},
    )
    manifest = json.loads(path.read_text())
    assert manifest["model_label"] == MODEL_LABEL
    assert manifest["config_snapshot"]["seed"] == 7
    assert (
        manifest["tx1_source_manifest"]["files"]["model.safetensors"]["sha256"] == "abc"
    )


# --- verify_cache: the Phase B exit criterion -------------------------------


def test_verify_cache_verified_on_good_cache(tmp_path: Path) -> None:
    cache_dir = _build_good_cache(tmp_path)
    report = verify_cache(cache_dir)
    assert report["status"] == "verified"
    assert report["discrepancies"] == []
    assert report["lines_expected"] == 2
    assert report["lines_present"] == 2


def test_verify_cache_failed_missing_manifest(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any("manifest" in discrepancy for discrepancy in report["discrepancies"])


def test_verify_cache_failed_truncated_array(tmp_path: Path) -> None:
    cache_dir = _build_good_cache(tmp_path)
    np.save(cache_dir / "ACH-A" / "embeddings.npy", _embeddings(1))
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any(
        "ACH-A" in discrepancy
        and ("shape mismatch" in discrepancy or "sha256 mismatch" in discrepancy)
        for discrepancy in report["discrepancies"]
    )


def test_verify_cache_failed_corrupted_sha256(tmp_path: Path) -> None:
    cache_dir = _build_good_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["lines"]["ACH-A"]["arrays"]["hvg.npy"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any(
        "ACH-A" in discrepancy and "hvg.npy sha256 mismatch" in discrepancy
        for discrepancy in report["discrepancies"]
    )


def test_verify_cache_failed_row_count_disagreement(tmp_path: Path) -> None:
    cache_dir = _build_good_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Overwrite hvg.npy with a different row count and update its manifest
    # metadata to match, so the sha256/shape-vs-manifest checks pass and only
    # the cross-array row-count-agreement check is exercised.
    path = cache_dir / "ACH-A" / "hvg.npy"
    np.save(path, np.zeros((99, 2), dtype=np.float32))
    manifest["lines"]["ACH-A"]["arrays"]["hvg.npy"] = (
        tx1_embed_cache_module._npy_metadata(path)
    )
    manifest_path.write_text(json.dumps(manifest))
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any(
        "row counts disagree" in discrepancy for discrepancy in report["discrepancies"]
    )


def test_verify_cache_failed_line_missing_from_disk(tmp_path: Path) -> None:
    cache_dir = _build_good_cache(tmp_path)
    shutil.rmtree(cache_dir / "ACH-B")
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any(
        "ACH-B" in discrepancy and "missing from cache_dir" in discrepancy
        for discrepancy in report["discrepancies"]
    )


def test_verify_cache_failed_extra_untracked_directory(tmp_path: Path) -> None:
    cache_dir = _build_good_cache(tmp_path)
    (cache_dir / "ACH-STALE").mkdir()
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any("ACH-STALE" in discrepancy for discrepancy in report["discrepancies"])


def test_verify_cache_failed_wrong_model_label(tmp_path: Path) -> None:
    cache_dir = _build_good_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_label"] = "some_other_label"
    manifest_path.write_text(json.dumps(manifest))
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any("model_label" in discrepancy for discrepancy in report["discrepancies"])


# --- load_hvg_gene_order ------------------------------------------------


def test_load_hvg_gene_order_reads_var_dims(tmp_path: Path) -> None:
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["G0", "G1", "G2"])
    names = load_hvg_gene_order(hvg_dir)
    assert names.tolist() == ["G0", "G1", "G2"]


# --- embed_lines: orchestration ----------------------------------------


def test_embed_lines_builds_tahoe_lines_and_records_hvg_fill_rate(
    tmp_path: Path,
) -> None:
    shard_dir, gene_metadata_path = _tahoe_shard_fixture(tmp_path)
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _manifest_row(
                model_id="ACH-A", cellosaurus_id="CVCL_A", cell_line_name="LineA"
            ),
            _manifest_row(
                model_id="ACH-B", cellosaurus_id="CVCL_B", cell_line_name="LineB"
            ),
        ],
    )
    manifest = load_line_manifest(manifest_path)
    # GENE99 is not present in the source shard -> 1/3 of checkpoint genes
    # must be zero-filled and the fill rate recorded.
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["GENE3", "GENE4", "GENE99"])
    encoder = _CountingEncoder()
    cache_dir = tmp_path / "cache"
    entries = embed_lines(
        manifest,
        cache_dir,
        encoder=encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        seed=0,
    )
    assert set(entries) == {"ACH-A", "ACH-B"}
    assert encoder.call_count == 2
    for model_id in ("ACH-A", "ACH-B"):
        assert entries[model_id]["n_cells"] == 2
        assert entries[model_id]["hvg_fill_rate"] == pytest.approx(1.0 / 3.0)
        embeddings, hvg, _ = load_line_cache(cache_dir, model_id)
        assert embeddings.shape == (2, EMBEDDING_WIDTH)
        assert hvg.shape == (2, 3)
        np.testing.assert_array_equal(
            np.asarray(hvg)[:, 2], np.zeros(2, dtype=np.float32)
        )


def test_embed_lines_dispatches_perturbseq_source(tmp_path: Path) -> None:
    h5ad_path = _write_perturbseq_h5ad(
        tmp_path / "data.h5ad", n_control=3, n_other=2, n_genes=2
    )
    manifest = pd.DataFrame(
        [
            _manifest_row(
                model_id="ACH-P",
                cellosaurus_id="CVCL_P",
                cell_line_name="LineP",
                basal_source="Perturb-seq non-targeting control",
            )
        ]
    )
    hvg_dir = _write_var_dims(
        tmp_path / "hvg_state", ["ENSG00000000000", "ENSG00000000001"]
    )
    source = PerturbseqSource(
        h5ad_path=h5ad_path,
        control_label="non-targeting",
        perturbation_col="gene",
        var_ensembl_col="ensembl_id",
    )
    cache_dir = tmp_path / "cache"
    entries = embed_lines(
        manifest,
        cache_dir,
        encoder=_CountingEncoder(),
        shard_dir=tmp_path / "unused_shards",
        gene_metadata_path=tmp_path / "unused_genes.parquet",
        hvg_state_model_dir=hvg_dir,
        hvg_gene_symbol_col="ensembl_id",
        perturbseq_sources={"ACH-P": source},
        seed=0,
    )
    assert entries["ACH-P"]["n_cells"] == 3
    assert entries["ACH-P"]["hvg_fill_rate"] == 0.0
    embeddings, hvg, _ = load_line_cache(cache_dir, "ACH-P")
    assert embeddings.shape == (3, EMBEDDING_WIDTH)
    assert hvg.shape == (3, 2)


def test_embed_lines_perturbseq_without_source_config_raises(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            _manifest_row(
                model_id="ACH-P", basal_source="Perturb-seq non-targeting control"
            )
        ]
    )
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["G0"])
    with pytest.raises(ValueError, match="ACH-P"):
        embed_lines(
            manifest,
            tmp_path / "cache",
            encoder=_CountingEncoder(),
            shard_dir=tmp_path / "shards",
            gene_metadata_path=tmp_path / "genes.parquet",
            hvg_state_model_dir=hvg_dir,
            seed=0,
        )


def test_embed_lines_missing_hvg_symbol_column_raises(tmp_path: Path) -> None:
    shard_dir, gene_metadata_path = _tahoe_shard_fixture(tmp_path)
    manifest = pd.DataFrame(
        [
            _manifest_row(
                model_id="ACH-A", cellosaurus_id="CVCL_A", cell_line_name="LineA"
            )
        ]
    )
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["GENE3"])
    with pytest.raises(ValueError, match="not_a_real_column"):
        embed_lines(
            manifest,
            tmp_path / "cache",
            encoder=_CountingEncoder(),
            shard_dir=shard_dir,
            gene_metadata_path=gene_metadata_path,
            hvg_state_model_dir=hvg_dir,
            hvg_gene_symbol_col="not_a_real_column",
            seed=0,
        )


def test_embed_lines_only_lines_filters(tmp_path: Path) -> None:
    shard_dir, gene_metadata_path = _tahoe_shard_fixture(tmp_path)
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["GENE3"])
    encoder = _CountingEncoder()
    entries = embed_lines(
        _two_line_manifest(),
        tmp_path / "cache",
        encoder=encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        only_lines=["ACH-A"],
        seed=0,
    )
    assert set(entries) == {"ACH-A"}
    assert encoder.call_count == 1


def test_embed_lines_resumability_skips_encoder_for_cached_lines(
    tmp_path: Path,
) -> None:
    shard_dir, gene_metadata_path = _tahoe_shard_fixture(tmp_path)
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["GENE3"])
    cache_dir = tmp_path / "cache"
    manifest = _two_line_manifest()

    first_encoder = _CountingEncoder()
    embed_lines(
        manifest,
        cache_dir,
        encoder=first_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        seed=0,
    )
    assert first_encoder.call_count == 2

    second_encoder = _CountingEncoder()
    entries = embed_lines(
        manifest,
        cache_dir,
        encoder=second_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        seed=0,
    )
    assert second_encoder.call_count == 0
    assert set(entries) == {"ACH-A", "ACH-B"}


def test_embed_lines_atomicity_raising_encoder_leaves_no_partial_dir(
    tmp_path: Path,
) -> None:
    shard_dir, gene_metadata_path = _tahoe_shard_fixture(tmp_path)
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["GENE3"])
    cache_dir = tmp_path / "cache"
    encoder = _RaisingOnceEncoder(fail_on_call=2)
    with pytest.raises(RuntimeError, match="simulated encoder crash"):
        embed_lines(
            _two_line_manifest(),
            cache_dir,
            encoder=encoder,
            shard_dir=shard_dir,
            gene_metadata_path=gene_metadata_path,
            hvg_state_model_dir=hvg_dir,
            seed=0,
        )
    assert (cache_dir / "ACH-A").is_dir()
    assert not (cache_dir / "ACH-B").exists()
    assert list(cache_dir.glob(".tmp-*")) == []
    # No run manifest was written yet -- verify_cache must fail loudly rather
    # than mistake this partial, mid-run state for a good cache.
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
