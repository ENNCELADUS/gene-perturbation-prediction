"""Tests for src/aivc_model/tx1_geneeffect_pipeline_run.py -- Phase D Task 5's
real-run orchestration library functions, exercised directly against tiny
synthetic fixtures (not through the CLI). See
tests/test_train_tx1_geneeffect_head.py for the full end-to-end CLI-level
coverage of Wave 3 Codex gate P1-1/P1-2/P1-3/P2-3.

No GPU and no real Phase C checkpoint exist on this machine, so every test
here builds tiny synthetic fixtures the same way every other Phase D test
file does (``tx1_embed_cache.write_line_cache``, the shared ``conftest``
manifest helpers). Nothing here is allowed to skip silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aivc_model.model import (
    LinearMockStateModel,
    PerturbationVectorAdapter,
    StateForwardAdapter,
)
from aivc_model.tx1_embed_cache import EMBEDDING_WIDTH, write_line_cache
from aivc_model.tx1_geneeffect_data import TRAIN_HEAD_ROLE
from aivc_model.tx1_geneeffect_pipeline_run import (
    _ensure_fresh_run_dir,
    warm_predicted_response_cache,
)
from aivc_model.tx1_predicted_response import ARM_TX1, ForwardOnlyStateModel
from aivc_model.tx1_predicted_response_cache import load_predicted_response_cache
from conftest import write_tx1_cache_run_manifest as _write_cache_run_manifest

_PERT_DIM = 3
_RESPONSE_DIM = 5
_CELL_SET_LEN = 4


def _forward_only_model(vocabulary: list[str]) -> ForwardOnlyStateModel:
    """A small, freshly initialized ST + perturbation-adapter model."""
    state_model = LinearMockStateModel(EMBEDDING_WIDTH, _RESPONSE_DIM, _PERT_DIM)
    perturbations = PerturbationVectorAdapter(vocabulary, {}, _PERT_DIM)
    return ForwardOnlyStateModel(StateForwardAdapter(state_model), perturbations)


def _write_tx1_cache(tmp_path: Path, model_id: str) -> Path:
    cache_dir = tmp_path / "tx1_cache"
    n_cells = 6
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(n_cells, EMBEDDING_WIDTH)).astype(np.float32)
    hvg = rng.normal(size=(n_cells, 2)).astype(np.float32)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_cells)])
    arrays = write_line_cache(
        cache_dir, model_id, embeddings, hvg, obs, hvg_gene_order=["H1", "H2"]
    )
    _write_cache_run_manifest(cache_dir, {model_id: arrays}, ["H1", "H2"])
    return cache_dir


# ---------------------------------------------------------------------------
# P1-4: warm_predicted_response_cache threads the full ordered vocabulary
# into the D11 cache fingerprint (predicted_response_fingerprint's own
# order-sensitivity is tested in isolation in test_tx1_predicted_response.py;
# this proves the WIRING actually forwards it, not just the primitive).
# ---------------------------------------------------------------------------


def test_warm_predicted_response_cache_reordered_vocabulary_misses_the_cache(
    tmp_path: Path,
) -> None:
    """Two calls differing ONLY in vocabulary construction order must not
    reuse each other's cache entry."""
    model_id = "ACH-1"
    tx1_cache_dir = _write_tx1_cache(tmp_path, model_id)
    checkpoint_path = tmp_path / "pytorch_model.bin"
    checkpoint_path.write_bytes(b"checkpoint-bytes")
    predicted_response_cache_dir = tmp_path / "predicted_response_cache"
    phase_b_manifest_path = tx1_cache_dir / "manifest.json"

    model = _forward_only_model(["G1", "G2", "G3"])
    fingerprint_a = warm_predicted_response_cache(
        model,
        tx1_cache_dir,
        predicted_response_cache_dir,
        model_id,
        TRAIN_HEAD_ROLE,
        ["G1"],
        ["G1", "G2", "G3"],
        ARM_TX1,
        _CELL_SET_LEN,
        0,
        checkpoint_path,
        phase_b_manifest_path,
    )
    load_predicted_response_cache(
        predicted_response_cache_dir, model_id, ARM_TX1, fingerprint_a
    )  # sanity: a real entry was written under fingerprint_a

    reordered_model = _forward_only_model(["G3", "G1", "G2"])  # same genes, reordered
    fingerprint_b = warm_predicted_response_cache(
        reordered_model,
        tx1_cache_dir,
        predicted_response_cache_dir,
        model_id,
        TRAIN_HEAD_ROLE,
        ["G1"],
        ["G3", "G1", "G2"],
        ARM_TX1,
        _CELL_SET_LEN,
        0,
        checkpoint_path,
        phase_b_manifest_path,
    )

    assert fingerprint_a != fingerprint_b
    load_predicted_response_cache(
        predicted_response_cache_dir, model_id, ARM_TX1, fingerprint_b
    )  # a SEPARATE entry exists for the reordered vocabulary too


def test_warm_predicted_response_cache_identical_vocabulary_order_reuses_cache(
    tmp_path: Path,
) -> None:
    """Sanity check: an identical vocabulary (same genes, same order, a
    fresh model instance) must hit the SAME cache entry -- P1-4's fix must
    not make every call a forced miss."""
    model_id = "ACH-1"
    tx1_cache_dir = _write_tx1_cache(tmp_path, model_id)
    checkpoint_path = tmp_path / "pytorch_model.bin"
    checkpoint_path.write_bytes(b"checkpoint-bytes")
    predicted_response_cache_dir = tmp_path / "predicted_response_cache"
    phase_b_manifest_path = tx1_cache_dir / "manifest.json"
    common_args = (
        tx1_cache_dir,
        predicted_response_cache_dir,
        model_id,
        TRAIN_HEAD_ROLE,
        ["G1"],
        ["G1", "G2", "G3"],
        ARM_TX1,
        _CELL_SET_LEN,
        0,
        checkpoint_path,
        phase_b_manifest_path,
    )

    fingerprint_a = warm_predicted_response_cache(
        _forward_only_model(["G1", "G2", "G3"]), *common_args
    )
    fingerprint_b = warm_predicted_response_cache(
        _forward_only_model(["G1", "G2", "G3"]), *common_args
    )

    assert fingerprint_a == fingerprint_b


# ---------------------------------------------------------------------------
# P2-3: a fresh run directory is required
# ---------------------------------------------------------------------------


def test_ensure_fresh_run_dir_accepts_a_nonexistent_directory(tmp_path: Path) -> None:
    _ensure_fresh_run_dir(tmp_path / "runs" / "new_run")  # must not raise


def test_ensure_fresh_run_dir_rejects_an_existing_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "existing_run"
    run_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        _ensure_fresh_run_dir(run_dir)
