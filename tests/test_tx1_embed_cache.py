"""Tests for src/aivc_model/tx1_embed_cache.py -- Tx1 basal embedding cache.

No GPU and no real Tx1/Tahoe/Perturb-seq data exist on this machine (Wave 1
Phase B Global Constraint 6), so every test substitutes a deterministic stub
``EncoderFn`` over tiny synthetic fixtures built in ``tmp_path``, reusing the
shared parquet-shard/h5ad/manifest fixtures in ``conftest.py``.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import aivc_model.tx1_embed_cache as tx1_embed_cache_module
import scripts.build_tx1_basal_embeddings as build_tx1_basal_embeddings_module
from aivc_model.gene_splits import sha256_file
from aivc_model.tx1_basal import load_line_manifest
from aivc_model.tx1_embed_cache import (
    EMBEDDING_WIDTH,
    MODEL_LABEL,
    REJECTED_MODEL_LABEL,
    PerturbseqSource,
    XatlasOrionSource,
    authenticate_tx1_registration,
    embed_lines,
    embed_registry_lines,
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
from conftest import write_tx1_xatlas_gene_metadata as _write_xatlas_gene_metadata
from conftest import write_tx1_xatlas_shard as _write_xatlas_shard
from scripts.build_tx1_basal_embeddings import (
    _load_perturbseq_source_config,
    _require_perturbseq_sources_configured,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PERTURBSEQ_SOURCE_CONFIG = (
    _REPO_ROOT
    / "configs"
    / "experiments"
    / "13_geneeffect_226"
    / "perturbseq_sources.json"
)
_FROZEN_LINE_MANIFEST = (
    _REPO_ROOT / "results" / "phase_a_tx1_20260724" / "cell_line_manifest.csv"
)
_TX1_REGISTRATION = (
    _REPO_ROOT / "results" / "phase_a_tx1_20260724" / "phase_a_registration.json"
)

# --- shared fixtures --------------------------------------------------------


def test_registered_tx1_authority_is_pinned_and_rejects_tamper(tmp_path: Path) -> None:
    source_manifest, manifest_sha256 = authenticate_tx1_registration(_TX1_REGISTRATION)
    assert source_manifest["model_label"] == MODEL_LABEL
    assert source_manifest["expected_obsm_width"] == EMBEDDING_WIDTH
    assert manifest_sha256 == (
        "10516bb0f42c2280f8b529455c1bb4a727a7e859e9c88b70aed71a37a793300c"
    )

    tampered = json.loads(_TX1_REGISTRATION.read_text())
    tampered["phase_a_gate_evidence"]["tx1_source_manifest"]["model_revision"] = (
        "0" * 40
    )
    tampered_path = tmp_path / "registration.json"
    tampered_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="registration SHA-256 mismatch"):
        authenticate_tx1_registration(tampered_path)


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


def _write_registry_h5ad(
    path: Path, *, value: int = 1, dtype: np.dtype = np.dtype(np.int32)
) -> None:
    ad.AnnData(
        X=np.asarray([[value, 0], [0, value], [value, value]], dtype=dtype),
        obs=pd.DataFrame(
            {"model_id": ["ACH-A"] * 3},
            index=["cell-c", "cell-a", "cell-b"],
        ),
        var=pd.DataFrame(
            {
                "gene_id": ["ENSG000001", "ENSG000002"],
                "gene_symbol": ["GENE1", "GENE2"],
            },
            index=["one", "two"],
        ),
    ).write_h5ad(path)


def test_registry_uint16_counts_are_promoted_before_encoding(tmp_path: Path) -> None:
    source = tmp_path / "ACH-A.h5ad"
    _write_registry_h5ad(source, dtype=np.dtype(np.uint16))
    registry = pd.DataFrame(
        {
            "source_path": [str(source)],
            "source_kind": ["h5ad"],
            "matrix_semantics": ["raw_umi_counts"],
        },
        index=pd.Index(["ACH-A"], name="model_id"),
    )
    state_dir = _write_var_dims(tmp_path / "state", ["GENE1", "GENE2"])
    observed_dtypes: list[np.dtype] = []

    def encoder(adata: ad.AnnData) -> np.ndarray:
        observed_dtypes.append(adata.X.dtype)
        return np.zeros((adata.n_obs, EMBEDDING_WIDTH), dtype=np.float32)

    embed_registry_lines(
        registry,
        tmp_path / "cache",
        encoder=encoder,
        hvg_state_model_dir=state_dir,
        var_ensembl_col="gene_id",
    )
    assert observed_dtypes == [np.dtype(np.float32)]


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


def _tahoe_shard_fixture_one_line(tmp_path: Path, *, n_cells: int) -> tuple[Path, Path]:
    """One Tahoe DMSO line (CVCL_A), 3 genes (tokens 3,4,5), ``n_cells`` cells.

    Unlike :func:`_tahoe_shard_fixture` (fixed at 2 cells/line), this lets
    cap/seed-change tests select genuinely different subsets on resume.
    """
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rows = [
        {
            "genes": np.array([3, 4, 5]),
            "expressions": np.array([1.0 + cell, 2.0 + cell, 3.0 + cell]),
            "cell_line_id": "CVCL_A",
        }
        for cell in range(n_cells)
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


def _build_good_cache(
    tmp_path: Path, *, extra_frozen_only_lines: list[dict[str, object]] | None = None
) -> Path:
    """Write a 2-line embedded cache with a matching run manifest.json.

    Also writes a frozen line manifest CSV (Critical 1's
    ``config_snapshot["line_manifest_path"]``/``["line_manifest_sha256"]``
    source of truth) recording the same two embedded lines, plus any
    ``extra_frozen_only_lines`` that are present in the frozen manifest but
    were never embedded (for Critical-1 missing-line coverage).

    Returns:
        The cache_dir path.
    """
    frozen_rows = [
        _manifest_row(
            model_id="ACH-A", cellosaurus_id="CVCL_A", cell_line_name="LineA"
        ),
        _manifest_row(
            model_id="ACH-B", cellosaurus_id="CVCL_B", cell_line_name="LineB"
        ),
        *(extra_frozen_only_lines or []),
    ]
    frozen_manifest_path = _write_manifest(
        tmp_path / "frozen_manifest.csv", frozen_rows
    )

    cache_dir = tmp_path / "cache"
    hvg_gene_order = ["G0", "G1"]
    hvg_gene_order_sha256 = tx1_embed_cache_module._hvg_gene_order_signature(
        hvg_gene_order
    )["sha256"]
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
            hvg_gene_order=hvg_gene_order,
        )
        line_entries[model_id] = {
            "arrays": arrays,
            "norm_stats": embedding_norm_stats(embeddings),
            "n_cells": n_cells,
            "hvg_fill_rate": 0.0,
            "basal_source": "Tahoe-100M DMSO",
            "cellosaurus_id": f"CVCL_{model_id}",
            "hvg_gene_order_sha256": hvg_gene_order_sha256,
        }
    write_run_manifest(
        cache_dir,
        model_label=MODEL_LABEL,
        source_manifest={"files": {}},
        line_entries=line_entries,
        config_snapshot={
            "seed": 0,
            "line_manifest_path": str(frozen_manifest_path),
            "line_manifest_sha256": sha256_file(frozen_manifest_path),
        },
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


def test_embedding_norm_stats_does_not_upcast_full_array_to_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Important 4: no full ``(n_cells, width)`` float64 copy may be made --
    that would double a ~1.4 GB float32 cache to ~2.8 GB on every resumed
    pass, for every already-cached line. Only the tiny ``(n_cells,)``-length
    norm reduction output may be upcast."""
    embeddings = np.tile(
        np.array([3.0, 4.0], dtype=np.float32), (5, EMBEDDING_WIDTH // 2)
    )
    original_asarray = np.asarray
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def spy_asarray(*args: object, **kwargs: object) -> np.ndarray:
        calls.append((args, kwargs))
        return original_asarray(*args, **kwargs)

    monkeypatch.setattr(tx1_embed_cache_module.np, "asarray", spy_asarray)
    embedding_norm_stats(embeddings)
    full_array_float64_calls = [
        (args, kwargs)
        for args, kwargs in calls
        if kwargs.get("dtype") == np.float64
        and getattr(args[0], "shape", None) == embeddings.shape
    ]
    assert full_array_float64_calls == []


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


def test_write_run_manifest_merges_with_existing_lines(tmp_path: Path) -> None:
    """Important 3: the documented 4-GPU ``--only-line`` sharding workflow is
    4 separate invocations, each writing a disjoint model_id subset -- the
    last one to finish must not erase the other three's recorded lines."""
    cache_dir = tmp_path / "cache"
    write_run_manifest(
        cache_dir,
        model_label=MODEL_LABEL,
        source_manifest={"files": {}},
        line_entries={"ACH-A": {"n_cells": 2}},
        config_snapshot={"only_line": ["ACH-A"]},
    )
    write_run_manifest(
        cache_dir,
        model_label=MODEL_LABEL,
        source_manifest={"files": {}},
        line_entries={"ACH-B": {"n_cells": 3}},
        config_snapshot={"only_line": ["ACH-B"]},
    )
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert set(manifest["lines"]) == {"ACH-A", "ACH-B"}
    assert manifest["lines"]["ACH-A"]["n_cells"] == 2
    assert manifest["lines"]["ACH-B"]["n_cells"] == 3


def test_write_run_manifest_new_entry_overrides_same_model_id(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_run_manifest(
        cache_dir,
        model_label=MODEL_LABEL,
        source_manifest={"files": {}},
        line_entries={"ACH-A": {"n_cells": 2}},
        config_snapshot={},
    )
    write_run_manifest(
        cache_dir,
        model_label=MODEL_LABEL,
        source_manifest={"files": {}},
        line_entries={"ACH-A": {"n_cells": 99}},
        config_snapshot={},
    )
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert manifest["lines"]["ACH-A"]["n_cells"] == 99


# --- verify_cache: the Phase B exit criterion -------------------------------


def test_verify_cache_verified_on_good_cache(tmp_path: Path) -> None:
    cache_dir = _build_good_cache(tmp_path)
    report = verify_cache(cache_dir)
    assert report["status"] == "verified"
    assert report["discrepancies"] == []
    assert report["lines_expected"] == 2
    assert report["lines_present"] == 2
    assert report["manifest_sha256"] == sha256_file(cache_dir / "manifest.json")
    assert set(report["line_artifact_sha256"]) == {"ACH-A", "ACH-B"}


def test_verify_cache_verified_with_explicit_expected_model_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = _build_good_cache(tmp_path)
    monkeypatch.setattr(
        tx1_embed_cache_module,
        "load_line_manifest",
        lambda path: pytest.fail("explicit membership must not load legacy schema"),
    )
    report = verify_cache(cache_dir, expected_model_ids=["ACH-A", "ACH-B"])
    assert report["status"] == "verified"
    assert report["discrepancies"] == []
    assert report["lines_expected"] == 2
    assert report["lines_present"] == 2


def test_verify_cache_explicit_expected_model_ids_rejects_missing_line(
    tmp_path: Path,
) -> None:
    cache_dir = _build_good_cache(tmp_path)
    report = verify_cache(
        cache_dir, expected_model_ids=["ACH-A", "ACH-B", "ACH-MISSING"]
    )
    assert report["status"] == "failed"
    assert any(
        "ACH-MISSING" in discrepancy and "run manifest" in discrepancy
        for discrepancy in report["discrepancies"]
    )
    assert any(
        "ACH-MISSING" in discrepancy and "directory missing" in discrepancy
        for discrepancy in report["discrepancies"]
    )


def test_verify_cache_explicit_expected_model_ids_rejects_extra_line(
    tmp_path: Path,
) -> None:
    cache_dir = _build_good_cache(tmp_path)
    report = verify_cache(cache_dir, expected_model_ids=["ACH-A"])
    assert report["status"] == "failed"
    assert any(
        "ACH-B" in discrepancy and "out-of-contract" in discrepancy
        for discrepancy in report["discrepancies"]
    )
    assert any(
        "ACH-B" in discrepancy and "directory not recorded" in discrepancy
        for discrepancy in report["discrepancies"]
    )


@pytest.mark.parametrize(
    ("expected_model_ids", "message"),
    [
        ([], "non-empty sequence"),
        (["ACH-A", "ACH-A"], "unique"),
        ([""], "non-empty strings"),
        (["ACH-A", 1], "non-empty strings"),
    ],
)
def test_verify_cache_rejects_invalid_expected_model_ids(
    tmp_path: Path, expected_model_ids: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        verify_cache(tmp_path / "cache", expected_model_ids=expected_model_ids)


def test_verify_cache_rejects_explicit_ids_with_frozen_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        verify_cache(
            tmp_path / "cache",
            frozen_manifest_path=tmp_path / "frozen.csv",
            expected_model_ids=["ACH-A"],
        )


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


def test_verify_cache_failed_n_cells_disagrees_with_actual_rows(tmp_path: Path) -> None:
    """Codex P1-b (part 2): verify_cache must cross-check a line's recorded
    manifest n_cells against actual embeddings.npy/hvg.npy/obs.parquet row
    counts. Without this, a resumed run whose --max-cells-per-line changed
    could skip the encoder, leave stale (smaller) on-disk arrays, but record
    the freshly-resolved (larger) n_cells -- and verify_cache would still
    report 'verified' because it never looked at n_cells at all."""
    cache_dir = _build_good_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["lines"]["ACH-A"]["n_cells"] = 999  # ACH-A actually has 2 rows
    manifest_path.write_text(json.dumps(manifest))
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any(
        "ACH-A" in discrepancy and "n_cells" in discrepancy
        for discrepancy in report["discrepancies"]
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


def test_verify_cache_failed_standalone_embedding_width_reassertion(
    tmp_path: Path,
) -> None:
    """Isolate ``_verify_line``'s standalone ``EMBEDDING_WIDTH`` re-assertion,
    analogous to ``test_verify_cache_failed_row_count_disagreement``: the
    on-disk array is tampered to a wrong width but its manifest metadata is
    updated to match (sha256/shape checks pass, and the row count -- 2 --
    still agrees with hvg/obs), so only the width check can catch it."""
    cache_dir = _build_good_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = cache_dir / "ACH-A" / "embeddings.npy"
    wrong_width = np.zeros((2, EMBEDDING_WIDTH - 1), dtype=np.float32)
    np.save(path, wrong_width)
    manifest["lines"]["ACH-A"]["arrays"]["embeddings.npy"] = (
        tx1_embed_cache_module._npy_metadata(path)
    )
    manifest_path.write_text(json.dumps(manifest))
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    expected_width_text = f"embeddings width {EMBEDDING_WIDTH - 1}"
    assert any(
        "ACH-A" in discrepancy and expected_width_text in discrepancy
        for discrepancy in report["discrepancies"]
    )


# --- verify_cache: frozen Phase-A manifest cross-check (Critical 1) --------


def test_verify_cache_failed_frozen_manifest_line_missing(tmp_path: Path) -> None:
    """The frozen Phase-A manifest is the single source of which lines must
    exist (Critical 1): a run that embeds only 2 of the frozen manifest's 3
    lines must fail verify_cache, even though its own manifest.json is
    internally consistent (disk matches exactly what the run itself claims
    to have written)."""
    cache_dir = _build_good_cache(
        tmp_path,
        extra_frozen_only_lines=[
            _manifest_row(
                model_id="ACH-C", cellosaurus_id="CVCL_C", cell_line_name="LineC"
            )
        ],
    )
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any(
        "ACH-C" in discrepancy and "never" in discrepancy
        for discrepancy in report["discrepancies"]
    )


def test_verify_cache_failed_extra_line_not_in_frozen_manifest(tmp_path: Path) -> None:
    """Codex P2: verify_cache must reject the reverse direction too -- an
    extra model_id present (and even fully, validly written to disk) in
    this run's manifest.json that is absent from the frozen Phase-A
    manifest must fail, not silently report 'verified'. Before this fix,
    only frozen-manifest-lines-missing-from-the-run was checked, never an
    out-of-contract line present in the run but absent from the contract."""
    cache_dir = _build_good_cache(tmp_path)
    hvg_gene_order = ["G0", "G1"]
    rogue_arrays = write_line_cache(
        cache_dir,
        "ACH-ROGUE",
        _embeddings(2),
        np.zeros((2, 2), dtype=np.float32),
        _obs(2),
        hvg_gene_order=hvg_gene_order,
    )
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["lines"]["ACH-ROGUE"] = {
        "arrays": rogue_arrays,
        "n_cells": 2,
        "hvg_gene_order_sha256": manifest["lines"]["ACH-A"]["hvg_gene_order_sha256"],
    }
    manifest_path.write_text(json.dumps(manifest))
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any(
        "ACH-ROGUE" in discrepancy and "out-of-contract" in discrepancy
        for discrepancy in report["discrepancies"]
    )


def test_verify_cache_failed_frozen_manifest_not_found(tmp_path: Path) -> None:
    cache_dir = _build_good_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_snapshot"]["line_manifest_path"] = str(
        tmp_path / "does_not_exist.csv"
    )
    manifest_path.write_text(json.dumps(manifest))
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any(
        "frozen line manifest not found" in discrepancy
        for discrepancy in report["discrepancies"]
    )


def test_verify_cache_failed_frozen_manifest_sha256_mismatch(tmp_path: Path) -> None:
    """A frozen manifest that has changed on disk since the run must fail
    loudly rather than be trusted as-is."""
    cache_dir = _build_good_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_snapshot"]["line_manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any("sha256" in discrepancy for discrepancy in report["discrepancies"])


def test_verify_cache_frozen_manifest_path_override(tmp_path: Path) -> None:
    """An explicit override lets verify_cache locate the frozen manifest even
    if the run's own recorded path is stale/unavailable on this host."""
    cache_dir = _build_good_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    real_frozen_path = Path(manifest["config_snapshot"]["line_manifest_path"])
    manifest["config_snapshot"]["line_manifest_path"] = "/nonexistent/on/this/host.csv"
    manifest_path.write_text(json.dumps(manifest))
    report = verify_cache(cache_dir, frozen_manifest_path=real_frozen_path)
    assert report["status"] == "verified"


# --- verify_cache: HVG gene-order consistency (Critical 2) -----------------


def test_verify_cache_failed_hvg_gene_order_disagreement(tmp_path: Path) -> None:
    cache_dir = _build_good_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["lines"]["ACH-B"]["hvg_gene_order_sha256"] = "deadbeef" * 8
    manifest_path.write_text(json.dumps(manifest))
    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any(
        "hvg_gene_order" in discrepancy for discrepancy in report["discrepancies"]
    )


# --- load_hvg_gene_order ------------------------------------------------


def test_load_hvg_gene_order_reads_var_dims(tmp_path: Path) -> None:
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["G0", "G1", "G2"])
    names = load_hvg_gene_order(hvg_dir)
    assert names.tolist() == ["G0", "G1", "G2"]


# --- embed_lines: orchestration ----------------------------------------


def test_registry_cache_binds_source_bytes_and_invalidates_resume(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ACH-A.h5ad"
    _write_registry_h5ad(source)
    registry = pd.DataFrame(
        {
            "source_path": [str(source)],
            "source_kind": ["h5ad"],
            "matrix_semantics": ["raw_umi_counts"],
        },
        index=pd.Index(["ACH-A"], name="model_id"),
    )
    state_dir = _write_var_dims(tmp_path / "state", ["GENE1", "GENE2"])
    cache_dir = tmp_path / "cache"
    encoder = _CountingEncoder()
    entries = embed_registry_lines(
        registry,
        cache_dir,
        encoder=encoder,
        hvg_state_model_dir=state_dir,
        var_ensembl_col="gene_id",
        max_cells_per_line=2,
        seed=7,
    )
    source_hash = sha256_file(source)
    entry = entries["ACH-A"]
    assert entry["model_id"] == "ACH-A"
    assert entry["source_sha256"] == source_hash
    assert entry["matrix_semantics"] == "raw_umi_counts"
    assert entry["sample_provenance"]["selection_algorithm"] == (
        "sha256_model_id_cell_id_v1"
    )
    assert len(entry["sample_provenance"]["selected_cell_ids"]) == 2
    write_run_manifest(
        cache_dir,
        model_label=MODEL_LABEL,
        source_manifest={"files": {"weights": {"sha256": "a" * 64}}},
        line_entries=entries,
        config_snapshot={},
    )
    report = verify_cache(
        cache_dir,
        expected_model_ids=["ACH-A"],
        expected_source_sha256={"ACH-A": source_hash},
        expected_matrix_semantics="raw_umi_counts",
    )
    assert report["status"] == "verified", report["discrepancies"]
    registered_source, _ = authenticate_tx1_registration(_TX1_REGISTRATION)
    self_attested = verify_cache(
        cache_dir,
        expected_model_ids=["ACH-A"],
        expected_source_sha256={"ACH-A": source_hash},
        expected_matrix_semantics="raw_umi_counts",
        expected_tx1_source_manifest=registered_source,
    )
    assert self_attested["status"] == "failed"
    assert any(
        "differs from pinned registration" in item
        for item in self_attested["discrepancies"]
    )

    _write_registry_h5ad(source, value=2)
    mutated_hash = sha256_file(source)
    stale = verify_cache(
        cache_dir,
        expected_model_ids=["ACH-A"],
        expected_source_sha256={"ACH-A": mutated_hash},
        expected_matrix_semantics="raw_umi_counts",
    )
    assert stale["status"] == "failed"
    assert "registry source SHA-256 mismatch" in stale["discrepancies"][0]

    updated = embed_registry_lines(
        registry,
        cache_dir,
        encoder=encoder,
        hvg_state_model_dir=state_dir,
        var_ensembl_col="gene_id",
        max_cells_per_line=2,
        seed=7,
    )
    assert encoder.call_count == 2
    assert updated["ACH-A"]["source_sha256"] == mutated_hash


def test_source_bound_verifier_rejects_same_id_legacy_cache(tmp_path: Path) -> None:
    cache_dir = _build_good_cache(tmp_path)
    report = verify_cache(
        cache_dir,
        expected_model_ids=["ACH-A", "ACH-B"],
        expected_source_sha256={"ACH-A": "a" * 64, "ACH-B": "b" * 64},
        expected_matrix_semantics="raw_umi_counts",
    )
    assert report["status"] == "failed"
    assert any("source SHA-256 mismatch" in item for item in report["discrepancies"])
    assert any(
        "sample provenance is missing" in item for item in report["discrepancies"]
    )


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


def test_embed_lines_dispatches_xatlas_orion_source(tmp_path: Path) -> None:
    """The parquet-sourced HCT116 anchor line must dispatch through the same
    ``perturbseq_sources`` mapping as the h5ad-sourced lines, distinguished
    by the configured source's type (``XatlasOrionSource`` vs.
    ``PerturbseqSource``)."""
    shard_dir = tmp_path / "xatlas_shards"
    shard_dir.mkdir()
    _write_xatlas_shard(
        shard_dir / "HCT116_Batch1.parquet",
        [
            {
                "gene_token_id": np.array([0, 1]),
                "gene_expression": np.array([1.0, 2.0], dtype=np.float32),
                "cell_barcode": "AAA",
                "sample": "HCT116_Batch1",
                "gene_target": "Non-Targeting",
                "pass_guide_filter": 1,
            },
            {
                "gene_token_id": np.array([0, 1]),
                "gene_expression": np.array([3.0, 4.0], dtype=np.float32),
                "cell_barcode": "BBB",
                "sample": "HCT116_Batch1",
                "gene_target": "Non-Targeting",
                "pass_guide_filter": 1,
            },
        ],
    )
    gene_metadata_path = _write_xatlas_gene_metadata(
        tmp_path / "xatlas_genes.parquet", [0, 1]
    )
    manifest = pd.DataFrame(
        [
            _manifest_row(
                model_id="ACH-000971",
                cellosaurus_id="CVCL_0291",
                cell_line_name="HCT116",
                basal_source="Perturb-seq non-targeting control",
            )
        ]
    )
    hvg_dir = _write_var_dims(
        tmp_path / "hvg_state", ["ENSG00000000000", "ENSG00000000001"]
    )
    source = XatlasOrionSource(
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        control_label="Non-Targeting",
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
        perturbseq_sources={"ACH-000971": source},
        seed=0,
    )
    assert entries["ACH-000971"]["n_cells"] == 2
    assert entries["ACH-000971"]["hvg_fill_rate"] == 0.0
    embeddings, hvg, obs = load_line_cache(cache_dir, "ACH-000971")
    assert embeddings.shape == (2, EMBEDDING_WIDTH)
    assert hvg.shape == (2, 2)
    assert obs["sample"].tolist() == ["HCT116_Batch1", "HCT116_Batch1"]


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


def test_resolve_hvg_matrix_auto_detects_one_recognized_symbol_column(
    tmp_path: Path,
) -> None:
    adata = ad.AnnData(
        X=np.asarray([[1.0, 2.0]], dtype=np.float32),
        var=pd.DataFrame({"gene_name": ["G0", "G1"]}),
    )
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["G0", "G1"])

    matrix, names, fill_rate = tx1_embed_cache_module._resolve_hvg_matrix(
        adata, hvg_dir, "auto"
    )

    np.testing.assert_array_equal(matrix, adata.X)
    np.testing.assert_array_equal(names, np.asarray(["G0", "G1"]))
    assert fill_rate == 0.0


def test_resolve_hvg_matrix_sums_duplicate_raw_count_symbols(tmp_path: Path) -> None:
    adata = ad.AnnData(
        X=np.asarray([[1.0, 2.0, 4.0]], dtype=np.float32),
        var=pd.DataFrame({"gene_symbols": ["G0", "G0", "G1"]}),
    )
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["G0", "G1"])

    matrix, names, fill_rate = tx1_embed_cache_module._resolve_hvg_matrix(
        adata, hvg_dir, "auto"
    )

    np.testing.assert_array_equal(matrix, np.asarray([[3.0, 4.0]]))
    np.testing.assert_array_equal(names, np.asarray(["G0", "G1"]))
    assert fill_rate == 0.0


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


def test_embed_lines_hvg_gene_order_change_forces_reembedding(tmp_path: Path) -> None:
    """Critical 2: resuming with a corrected ``--hvg-state-model-dir`` must
    re-embed every previously-cached line, not silently skip it. Without the
    fix, ``_is_cached`` only checks width/row-count self-consistency, so a
    line cached under one HVG checkpoint's gene order would be skipped
    forever after the checkpoint changes, permanently keeping the stale
    ``hvg.npy`` gene order."""
    shard_dir, gene_metadata_path = _tahoe_shard_fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    manifest = _two_line_manifest()

    hvg_dir_a = _write_var_dims(tmp_path / "hvg_state_a", ["GENE3", "GENE4"])
    first_encoder = _CountingEncoder()
    embed_lines(
        manifest,
        cache_dir,
        encoder=first_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir_a,
        seed=0,
    )
    assert first_encoder.call_count == 2
    _, hvg_before, _ = load_line_cache(cache_dir, "ACH-A")
    np.testing.assert_array_equal(np.asarray(hvg_before)[0], [1.0, 2.0])  # GENE3, GENE4

    # Resume against a corrected checkpoint with the same width but a
    # different gene order (GENE4 before GENE3).
    hvg_dir_b = _write_var_dims(tmp_path / "hvg_state_b", ["GENE4", "GENE3"])
    second_encoder = _CountingEncoder()
    entries = embed_lines(
        manifest,
        cache_dir,
        encoder=second_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir_b,
        seed=0,
    )
    assert second_encoder.call_count == 2
    assert set(entries) == {"ACH-A", "ACH-B"}
    _, hvg_after, _ = load_line_cache(cache_dir, "ACH-A")
    np.testing.assert_array_equal(np.asarray(hvg_after)[0], [2.0, 1.0])  # GENE4, GENE3


def test_embed_lines_cap_change_on_resume_forces_reembedding(tmp_path: Path) -> None:
    """Codex P1-b: resuming with a changed --max-cells-per-line must
    re-embed, not silently skip the encoder for a now-stale, smaller
    on-disk cache while recording the newly-resolved (larger) n_cells in
    the manifest. Before this fix, ``_is_cached`` only checked internal
    row-count self-consistency and the HVG-order hash -- never comparing
    against the freshly resolved selection -- so this exact scenario left
    100 cached cells on disk while the manifest claimed 200."""
    shard_dir, gene_metadata_path = _tahoe_shard_fixture_one_line(tmp_path, n_cells=4)
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["GENE3"])
    cache_dir = tmp_path / "cache"
    manifest = pd.DataFrame(
        [
            _manifest_row(
                model_id="ACH-A", cellosaurus_id="CVCL_A", cell_line_name="LineA"
            )
        ]
    )

    first_encoder = _CountingEncoder()
    embed_lines(
        manifest,
        cache_dir,
        encoder=first_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        max_cells_per_line=1,
        seed=0,
    )
    assert first_encoder.call_count == 1
    embeddings_before, _, _ = load_line_cache(cache_dir, "ACH-A")
    assert embeddings_before.shape[0] == 1

    second_encoder = _CountingEncoder()
    entries = embed_lines(
        manifest,
        cache_dir,
        encoder=second_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        max_cells_per_line=2,
        seed=0,
    )
    assert second_encoder.call_count == 1  # must re-embed, not silently skip
    embeddings_after, _, _ = load_line_cache(cache_dir, "ACH-A")
    assert embeddings_after.shape[0] == 2
    assert entries["ACH-A"]["n_cells"] == 2
    # The re-embedded cache must itself verify cleanly (recorded n_cells now
    # agrees with the actual, freshly-written row count) -- proving this is
    # a genuine fix, not just a shifted-but-still-wrong n_cells value.
    frozen_manifest_path = _write_manifest(
        tmp_path / "frozen.csv",
        [
            _manifest_row(
                model_id="ACH-A", cellosaurus_id="CVCL_A", cell_line_name="LineA"
            )
        ],
    )
    write_run_manifest(
        cache_dir,
        model_label=MODEL_LABEL,
        source_manifest={"files": {}},
        line_entries=entries,
        config_snapshot={
            "line_manifest_path": str(frozen_manifest_path),
            "line_manifest_sha256": sha256_file(frozen_manifest_path),
        },
    )
    report = verify_cache(cache_dir)
    assert report["status"] == "verified", report["discrepancies"]


def test_embed_lines_seed_change_on_resume_forces_reembedding(tmp_path: Path) -> None:
    """Codex P1-b: resuming with an unchanged cap but a changed --seed must
    also re-embed. The recorded sample-provenance signature includes the
    seed itself (not just a hash of the selected cell ids), so this is
    caught even for a Tahoe basal source whose per-cell identifiers are
    reservoir-position labels rather than real source-row barcodes."""
    shard_dir, gene_metadata_path = _tahoe_shard_fixture_one_line(tmp_path, n_cells=6)
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["GENE3"])
    cache_dir = tmp_path / "cache"
    manifest = pd.DataFrame(
        [
            _manifest_row(
                model_id="ACH-A", cellosaurus_id="CVCL_A", cell_line_name="LineA"
            )
        ]
    )

    first_encoder = _CountingEncoder()
    embed_lines(
        manifest,
        cache_dir,
        encoder=first_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        max_cells_per_line=2,
        seed=0,
    )
    assert first_encoder.call_count == 1

    second_encoder = _CountingEncoder()
    embed_lines(
        manifest,
        cache_dir,
        encoder=second_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        max_cells_per_line=2,
        seed=99,
    )
    assert second_encoder.call_count == 1  # must re-embed, not silently skip


def test_embed_lines_unchanged_cap_and_seed_still_skips_encoder(tmp_path: Path) -> None:
    """Sanity converse of the two tests above: an identical resume (same
    cap, same seed, same source) must still hit the resumability fast path
    and skip the encoder -- the new provenance check must not force
    needless re-embedding when nothing actually changed."""
    shard_dir, gene_metadata_path = _tahoe_shard_fixture_one_line(tmp_path, n_cells=4)
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["GENE3"])
    cache_dir = tmp_path / "cache"
    manifest = pd.DataFrame(
        [
            _manifest_row(
                model_id="ACH-A", cellosaurus_id="CVCL_A", cell_line_name="LineA"
            )
        ]
    )

    first_encoder = _CountingEncoder()
    embed_lines(
        manifest,
        cache_dir,
        encoder=first_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        max_cells_per_line=2,
        seed=0,
    )
    assert first_encoder.call_count == 1

    second_encoder = _CountingEncoder()
    entries = embed_lines(
        manifest,
        cache_dir,
        encoder=second_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        max_cells_per_line=2,
        seed=0,
    )
    assert second_encoder.call_count == 0
    assert entries["ACH-A"]["n_cells"] == 2


def test_embed_lines_legacy_cache_without_provenance_sidecar_is_accepted(
    tmp_path: Path,
) -> None:
    """Backward compatibility (mandatory for the in-flight HPC run): a line
    cached by code that predates this provenance signature has no
    ``sample_provenance.json`` sidecar on disk. Resuming with the *same*
    cap/seed the in-flight run always used must still skip the encoder --
    treating "no recorded signature" as a mismatch would force re-embedding
    every already-completed line and discard already-completed GPU work."""
    shard_dir, gene_metadata_path = _tahoe_shard_fixture_one_line(tmp_path, n_cells=4)
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["GENE3"])
    cache_dir = tmp_path / "cache"
    manifest = pd.DataFrame(
        [
            _manifest_row(
                model_id="ACH-A", cellosaurus_id="CVCL_A", cell_line_name="LineA"
            )
        ]
    )

    legacy_encoder = _CountingEncoder()
    embed_lines(
        manifest,
        cache_dir,
        encoder=legacy_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        max_cells_per_line=2,
        seed=20260725,
    )
    assert legacy_encoder.call_count == 1
    # Simulate "written by pre-fix code": no sample_provenance.json sidecar.
    sidecar = cache_dir / "ACH-A" / "sample_provenance.json"
    assert sidecar.is_file()
    sidecar.unlink()

    resumed_encoder = _CountingEncoder()
    entries = embed_lines(
        manifest,
        cache_dir,
        encoder=resumed_encoder,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        hvg_state_model_dir=hvg_dir,
        max_cells_per_line=2,
        seed=20260725,
    )
    assert resumed_encoder.call_count == 0  # legacy cache accepted, not re-embedded
    assert entries["ACH-A"]["n_cells"] == 2


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


def test_embed_lines_wrong_width_encoder_raises(tmp_path: Path) -> None:
    """Minor 5: drive a wrong-width stub encoder through the full
    ``embed_lines``/``_embed_one_line`` pipeline (not just
    ``write_line_cache`` directly), proving the width check is reachable
    from the actual orchestration path a real run uses."""
    shard_dir, gene_metadata_path = _tahoe_shard_fixture(tmp_path)
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["GENE3"])
    manifest = pd.DataFrame(
        [
            _manifest_row(
                model_id="ACH-A", cellosaurus_id="CVCL_A", cell_line_name="LineA"
            )
        ]
    )
    bad_encoder = _CountingEncoder(width=EMBEDDING_WIDTH - 1)
    with pytest.raises(ValueError, match=str(EMBEDDING_WIDTH)):
        embed_lines(
            manifest,
            tmp_path / "cache",
            encoder=bad_encoder,
            shard_dir=shard_dir,
            gene_metadata_path=gene_metadata_path,
            hvg_state_model_dir=hvg_dir,
            seed=0,
        )


# --- CLI: scripts/build_tx1_basal_embeddings.py Perturb-seq wiring ---------


def test_load_perturbseq_source_config_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "perturbseq_sources.json"
    config_path.write_text(
        json.dumps(
            {
                "ACH-P": {
                    "h5ad_path": "/data/k562.h5ad",
                    "perturbation_col": "gene",
                    "control_label": "non-targeting",
                    "var_ensembl_col": "ensembl_id",
                }
            }
        )
    )
    sources = _load_perturbseq_source_config(config_path)
    assert set(sources) == {"ACH-P"}
    source = sources["ACH-P"]
    assert isinstance(source, PerturbseqSource)
    assert source.h5ad_path == Path("/data/k562.h5ad")
    assert source.control_label == "non-targeting"
    assert source.perturbation_col == "gene"
    assert source.var_ensembl_col == "ensembl_id"


def test_load_perturbseq_source_config_missing_key_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "perturbseq_sources.json"
    config_path.write_text(json.dumps({"ACH-P": {"h5ad_path": "/data/k562.h5ad"}}))
    with pytest.raises(ValueError, match="ACH-P"):
        _load_perturbseq_source_config(config_path)


def test_load_perturbseq_source_config_not_an_object_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "perturbseq_sources.json"
    config_path.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(ValueError, match="JSON object"):
        _load_perturbseq_source_config(config_path)


def test_load_perturbseq_source_config_xatlas_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "perturbseq_sources.json"
    config_path.write_text(
        json.dumps(
            {
                "ACH-000971": {
                    "source_type": "xatlas_orion_parquet",
                    "shard_dir": "/data/xatlas_orion/data",
                    "shard_glob": "HCT116_Batch*.parquet",
                    "gene_metadata_path": (
                        "/data/xatlas_orion/metadata/gene_metadata.parquet"
                    ),
                    "control_label": "Non-Targeting",
                    "pass_guide_filter_value": 1,
                }
            }
        )
    )
    sources = _load_perturbseq_source_config(config_path)
    assert set(sources) == {"ACH-000971"}
    source = sources["ACH-000971"]
    assert isinstance(source, XatlasOrionSource)
    assert source.shard_dir == Path("/data/xatlas_orion/data")
    assert source.shard_glob == "HCT116_Batch*.parquet"
    assert source.gene_metadata_path == Path(
        "/data/xatlas_orion/metadata/gene_metadata.parquet"
    )
    assert source.control_label == "Non-Targeting"
    assert source.pass_guide_filter_value == 1


def test_load_perturbseq_source_config_xatlas_optional_keys_default(
    tmp_path: Path,
) -> None:
    """``shard_glob``/``pass_guide_filter_value`` are optional; omitting them
    must fall back to :class:`XatlasOrionSource`'s own defaults, not raise."""
    config_path = tmp_path / "perturbseq_sources.json"
    config_path.write_text(
        json.dumps(
            {
                "ACH-000971": {
                    "source_type": "xatlas_orion_parquet",
                    "shard_dir": "/data/xatlas_orion/data",
                    "gene_metadata_path": (
                        "/data/xatlas_orion/metadata/gene_metadata.parquet"
                    ),
                    "control_label": "Non-Targeting",
                }
            }
        )
    )
    source = _load_perturbseq_source_config(config_path)["ACH-000971"]
    assert source.shard_glob == "*.parquet"
    assert source.pass_guide_filter_value == 1


def test_load_perturbseq_source_config_xatlas_missing_key_raises(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "perturbseq_sources.json"
    config_path.write_text(
        json.dumps(
            {
                "ACH-000971": {
                    "source_type": "xatlas_orion_parquet",
                    "shard_dir": "/data/xatlas_orion/data",
                }
            }
        )
    )
    with pytest.raises(ValueError, match="ACH-000971"):
        _load_perturbseq_source_config(config_path)


def test_load_perturbseq_source_config_unknown_source_type_raises(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "perturbseq_sources.json"
    config_path.write_text(
        json.dumps({"ACH-P": {"source_type": "bogus_type", "h5ad_path": "/x.h5ad"}})
    )
    with pytest.raises(ValueError, match="ACH-P"):
        _load_perturbseq_source_config(config_path)


def test_shipped_perturbseq_source_config_names_the_4_anchor_lines() -> None:
    """The committed ``perturbseq_sources.json`` must parse and must name
    exactly the frozen manifest's 4 non-Tahoe (``basal_source ==
    "Perturb-seq non-targeting control"``) lines -- neither more nor
    fewer -- so an unconfigured or misconfigured anchor fails loudly before
    any GPU work, per the CLI's documented contract."""
    frozen_manifest = load_line_manifest(_FROZEN_LINE_MANIFEST)
    expected = set(
        frozen_manifest.loc[
            frozen_manifest["basal_source"] == "Perturb-seq non-targeting control",
            "model_id",
        ].astype(str)
    )
    sources = _load_perturbseq_source_config(_PERTURBSEQ_SOURCE_CONFIG)
    assert set(sources) == expected
    assert isinstance(sources["ACH-000971"], XatlasOrionSource)
    for model_id in expected - {"ACH-000971"}:
        assert isinstance(sources[model_id], PerturbseqSource)


def test_require_perturbseq_sources_configured_raises_for_unconfigured_line() -> None:
    """CLI gap: before this fix, this CLI could not embed any of the 4
    Perturb-seq-sourced lines at all -- ``_run_embedding`` unconditionally
    rejected any in-scope non-Tahoe ``basal_source``. This proves the
    replacement validation still fails loudly for a truly unconfigured
    line, listing its model_id."""
    working = pd.DataFrame(
        [
            _manifest_row(model_id="ACH-A", basal_source="Tahoe-100M DMSO"),
            _manifest_row(
                model_id="ACH-P", basal_source="Perturb-seq non-targeting control"
            ),
        ]
    )
    with pytest.raises(ValueError, match="ACH-P"):
        _require_perturbseq_sources_configured(working, {})


def test_require_perturbseq_sources_configured_passes_when_all_configured() -> None:
    """CLI gap: a Perturb-seq line with a configured source must not raise,
    proving the CLI can now proceed to embed it (unlike before this fix,
    which rejected every non-Tahoe line unconditionally)."""
    working = pd.DataFrame(
        [
            _manifest_row(
                model_id="ACH-P", basal_source="Perturb-seq non-targeting control"
            )
        ]
    )
    source = PerturbseqSource(
        h5ad_path=Path("/data/k562.h5ad"),
        control_label="non-targeting",
        perturbation_col="gene",
        var_ensembl_col="ensembl_id",
    )
    _require_perturbseq_sources_configured(working, {"ACH-P": source})  # no raise


# --- verify_cache: --only-line restricted scope (Codex P1-c) ---------------


def test_verify_cache_only_lines_restricted_run_verifies_own_scope(
    tmp_path: Path,
) -> None:
    """Codex P1-c: a restricted --only-line shard that has only embedded
    ACH-A (of the frozen manifest's ACH-A/ACH-B) must itself report
    'verified' when checked with only_lines=["ACH-A"] -- it must not be
    penalized for ACH-B (another shard's line) being absent, and ACH-B's
    directory must not need to exist under cache_dir at all."""
    cache_dir = _build_good_cache(tmp_path)
    shutil.rmtree(cache_dir / "ACH-B")
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["lines"]["ACH-B"]
    manifest_path.write_text(json.dumps(manifest))

    report = verify_cache(cache_dir, only_lines=["ACH-A"])
    assert report["status"] == "verified", report["discrepancies"]
    assert report["lines_expected"] == 1
    assert report["lines_present"] == 1


def test_verify_cache_unrestricted_still_fails_on_the_same_incomplete_cache(
    tmp_path: Path,
) -> None:
    """The converse of the test above: the same partial cache (only ACH-A
    embedded) must still fail the full, unrestricted check -- proving the
    restricted-scope fix does not weaken the real 'verified embedding
    caches' exit criterion (Global Constraints), it only adds a narrower
    per-shard mode alongside it."""
    cache_dir = _build_good_cache(tmp_path)
    shutil.rmtree(cache_dir / "ACH-B")
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["lines"]["ACH-B"]
    manifest_path.write_text(json.dumps(manifest))

    report = verify_cache(cache_dir)
    assert report["status"] == "failed"
    assert any("ACH-B" in discrepancy for discrepancy in report["discrepancies"])


def test_verify_cache_only_lines_still_rejects_out_of_contract_request(
    tmp_path: Path,
) -> None:
    """A restricted run's own requested lines must still be legitimate
    frozen-manifest lines (Codex P2 applied to the restricted case): asking
    to verify a model_id the frozen manifest never named is a failure, not
    a silently accepted no-op scope."""
    cache_dir = _build_good_cache(tmp_path)
    report = verify_cache(cache_dir, only_lines=["ACH-A", "ACH-NOT-IN-CONTRACT"])
    assert report["status"] == "failed"
    assert any(
        "ACH-NOT-IN-CONTRACT" in discrepancy for discrepancy in report["discrepancies"]
    )


def test_verify_cache_explicit_expected_ids_only_lines_is_shard_check(
    tmp_path: Path,
) -> None:
    cache_dir = _build_good_cache(tmp_path)
    shutil.rmtree(cache_dir / "ACH-B")
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["lines"]["ACH-B"]
    manifest_path.write_text(json.dumps(manifest))

    report = verify_cache(
        cache_dir,
        only_lines=["ACH-A"],
        expected_model_ids=["ACH-A", "ACH-B"],
    )
    assert report["status"] == "verified", report["discrepancies"]
    assert report["lines_expected"] == 1
    assert report["lines_present"] == 1


def test_verify_cache_explicit_expected_ids_rejects_invalid_only_line(
    tmp_path: Path,
) -> None:
    cache_dir = _build_good_cache(tmp_path)
    report = verify_cache(
        cache_dir,
        only_lines=["ACH-NOT-IN-CONTRACT"],
        expected_model_ids=["ACH-A", "ACH-B"],
    )
    assert report["status"] == "failed"
    assert any(
        "ACH-NOT-IN-CONTRACT" in discrepancy and "expected_model_ids" in discrepancy
        for discrepancy in report["discrepancies"]
    )


# --- CLI: entrypoint importability under direct execution (Codex P1-a) -----


def test_cli_help_succeeds_under_direct_script_execution() -> None:
    """Codex P1-a: ``python scripts/build_tx1_basal_embeddings.py --help``
    must succeed without the caller setting PYTHONPATH. Before the fix,
    Python puts scripts/ (not the repo root) on sys.path when the script is
    executed directly, so ``from scripts.verify_tx1_obsm_width import ...``
    raised ModuleNotFoundError even for --help -- confirmed in production
    (the HPC run only worked via an explicit PYTHONPATH=src:. workaround)."""
    script = _REPO_ROOT / "scripts" / "build_tx1_basal_embeddings.py"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert "usage:" in result.stdout.lower()


# --- CLI: --only-line shard exit code vs --verify-only aggregation ---------
# (Codex P1-c)


def test_cli_only_line_shard_exits_zero_but_verify_only_over_incomplete_cache_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex P1-c end to end: with --only-line, every shard used to run the
    FULL-cache verifier, which necessarily reports other shards' lines as
    missing -- so main() exited 1 for a shard that actually succeeded. A
    restricted --only-line run over a 2-line frozen manifest, having
    embedded only its own 1 requested line, must now exit 0; an explicit,
    unrestricted --verify-only pass over that same (still 1-of-2-lines)
    cache must still exit 1 -- the full criterion is not weakened."""
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
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", ["GENE3"])
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    source_manifest_path = tmp_path / "source_manifest.json"
    source_manifest_path.write_text(json.dumps({"files": {}}))
    cache_dir = tmp_path / "cache"

    monkeypatch.setattr(
        build_tx1_basal_embeddings_module,
        "_build_tx1_encoder",
        lambda model_dir, batch_size, max_length: (_CountingEncoder(), {}),
    )
    monkeypatch.setattr(
        build_tx1_basal_embeddings_module,
        "_verify_model_dir_matches_source_manifest",
        lambda model_dir, source_manifest: None,
    )

    shard_argv = [
        "build_tx1_basal_embeddings.py",
        "--cache-dir",
        str(cache_dir),
        "--line-manifest",
        str(manifest_path),
        "--model-dir",
        str(model_dir),
        "--shard-dir",
        str(shard_dir),
        "--gene-metadata",
        str(gene_metadata_path),
        "--hvg-state-model-dir",
        str(hvg_dir),
        "--tx1-source-manifest",
        str(source_manifest_path),
        "--only-line",
        "ACH-A",
    ]
    monkeypatch.setattr(sys, "argv", shard_argv)
    build_tx1_basal_embeddings_module.main()  # must not raise SystemExit(1)

    verify_only_argv = [
        "build_tx1_basal_embeddings.py",
        "--cache-dir",
        str(cache_dir),
        "--line-manifest",
        str(manifest_path),
        "--verify-only",
    ]
    monkeypatch.setattr(sys, "argv", verify_only_argv)
    with pytest.raises(SystemExit) as excinfo:
        build_tx1_basal_embeddings_module.main()
    assert excinfo.value.code == 1
