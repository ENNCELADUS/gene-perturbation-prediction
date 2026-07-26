"""Tests for src/aivc_model/tx1_geneeffect_train.py and its sibling
src/aivc_model/tx1_geneeffect_train_io.py -- Phase D Task 3: training
``Tx1GeneEffectHead`` over the 28/5/9 split, D2's pooled-feature export, and
D12's validation-line-selection guard.

No GPU, no real Phase C checkpoint, no real DepMap file, and no real Tx1
embeddings exist on this machine, so every test builds tiny synthetic
fixtures directly (``LineExamples`` built from hand-generated tensors) or,
for the I/O-layer tests, via ``tmp_path`` caches written the same way
Task 1/Task 2's own tests do (``tx1_embed_cache.write_line_cache``,
``tx1_predicted_response_cache.write_predicted_response_cache``, the shared
``conftest`` manifest helpers). Nothing here is allowed to skip silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aivc_model.gene_splits import sha256_file
from aivc_model.tx1_basal import load_line_manifest
from aivc_model.tx1_embed_cache import EMBEDDING_WIDTH, write_line_cache
from aivc_model.tx1_geneeffect_data import (
    TEST_ROLE,
    TRAIN_HEAD_ROLE,
    TRAIN_RESPONSE_ROLE,
    build_line_role_split,
    generate_validation_line_registration,
)
from aivc_model.tx1_geneeffect_head import Tx1GeneEffectHead
from aivc_model.tx1_geneeffect_train import (
    SELECTION_METRIC_NAME,
    EpochMetrics,
    LineExamples,
    TrainingConfig,
    compute_line_features_and_predictions,
    train_tx1_geneeffect_head,
)
from aivc_model.tx1_geneeffect_train import _select_best_epoch
from aivc_model.tx1_geneeffect_train_io import (
    Tx1GeneEffectProvenance,
    assemble_line_examples,
    assemble_split_examples,
    build_provenance,
    load_checkpoint,
    save_checkpoint,
    write_provenance,
)
from aivc_model.tx1_predicted_response import ARM_HVG, ARM_TX1
from aivc_model.tx1_predicted_response_cache import write_predicted_response_cache
from conftest import tx1_manifest_row as _manifest_row
from conftest import write_tx1_cache_run_manifest as _write_cache_run_manifest
from conftest import write_tx1_line_manifest as _write_manifest


# --- synthetic LineExamples builder ------------------------------------------


def _make_line_examples(
    model_id: str,
    role: str,
    *,
    n_genes: int,
    response_dim: int,
    basal_dim: int,
    n_response_cells: int = 12,
    n_basal_cells: int = 16,
    seed: int = 0,
    arm: str = ARM_TX1,
) -> LineExamples:
    """Synthetic per-line examples with a learnable signal.

    Each gene's response bag is centered on that gene's own target scalar
    (broadcast across every response-dim column, with a small amount of
    per-cell noise), so the moment-pooled response mean strongly predicts
    the target -- enough signal for a small MLP to actually learn something
    in a handful of epochs, without any real ST/DepMap data.
    """
    rng = np.random.default_rng(seed)
    targets_np = rng.uniform(-2.0, 2.0, size=n_genes).astype(np.float32)
    genes = tuple(f"G{i}" for i in range(n_genes))
    response_bags = tuple(
        torch.as_tensor(
            rng.normal(
                loc=float(target), scale=0.05, size=(n_response_cells, response_dim)
            ).astype(np.float32)
        )
        for target in targets_np
    )
    basal_bag = torch.as_tensor(
        rng.normal(loc=0.0, scale=0.1, size=(n_basal_cells, basal_dim)).astype(
            np.float32
        )
    )
    return LineExamples(
        model_id=model_id,
        role=role,
        arm=arm,
        basal_bag=basal_bag,
        genes=genes,
        response_bags=response_bags,
        targets=torch.as_tensor(targets_np),
    )


# --- train_tx1_geneeffect_head: drives loss down -----------------------------


def test_train_tx1_geneeffect_head_drives_loss_down() -> None:
    train_a = _make_line_examples(
        "ACH-TRAIN-A", TRAIN_HEAD_ROLE, n_genes=8, response_dim=4, basal_dim=3, seed=1
    )
    train_b = _make_line_examples(
        "ACH-TRAIN-B", TRAIN_HEAD_ROLE, n_genes=8, response_dim=4, basal_dim=3, seed=2
    )
    validation = _make_line_examples(
        "ACH-VAL", TRAIN_HEAD_ROLE, n_genes=8, response_dim=4, basal_dim=3, seed=3
    )
    config = TrainingConfig(
        hidden=32, moments=2, lam=1.0, learning_rate=0.05, epochs=150, seed=0
    )

    _head, result = train_tx1_geneeffect_head([train_a, train_b], [validation], config)

    initial_loss = result.history[0].train_loss
    final_loss = result.history[-1].train_loss
    assert final_loss < initial_loss * 0.5
    assert min(entry.train_loss for entry in result.history) < 0.5


def test_train_tx1_geneeffect_head_requires_nonempty_lines() -> None:
    line = _make_line_examples(
        "ACH-1", TRAIN_HEAD_ROLE, n_genes=3, response_dim=2, basal_dim=2, seed=4
    )
    with pytest.raises(ValueError, match="train_lines must be non-empty"):
        train_tx1_geneeffect_head([], [line], TrainingConfig(epochs=1))
    with pytest.raises(ValueError, match="validation_lines must be non-empty"):
        train_tx1_geneeffect_head([line], [], TrainingConfig(epochs=1))


def test_train_tx1_geneeffect_head_raises_on_dim_mismatch() -> None:
    train_line = _make_line_examples(
        "ACH-1", TRAIN_HEAD_ROLE, n_genes=3, response_dim=2, basal_dim=2, seed=5
    )
    mismatched_validation = _make_line_examples(
        "ACH-2", TRAIN_HEAD_ROLE, n_genes=3, response_dim=2, basal_dim=5, seed=6
    )
    with pytest.raises(ValueError, match="basal_bag feature width disagrees"):
        train_tx1_geneeffect_head(
            [train_line], [mismatched_validation], TrainingConfig(epochs=1)
        )


# --- D2: pooled feature vector exposure --------------------------------------


def test_compute_line_features_and_predictions_matches_forward_and_width() -> None:
    line = _make_line_examples(
        "ACH-1", TRAIN_HEAD_ROLE, n_genes=5, response_dim=4, basal_dim=3, seed=7
    )
    head = Tx1GeneEffectHead(response_dim=4, basal_dim=3, hidden=16, moments=2)

    features, predictions = compute_line_features_and_predictions(head, line)

    expected_width = (4 + 3) * 2
    assert features.shape == (5, expected_width)
    assert predictions.shape == (5,)
    for index, response_bag in enumerate(line.response_bags):
        expected = head.forward(response_bag, line.basal_bag)
        assert torch.allclose(predictions[index], expected, atol=1e-5)


# --- D6: a test-role line must never enter training --------------------------


def test_train_tx1_geneeffect_head_rejects_test_role_line_in_train_set() -> None:
    bad = _make_line_examples(
        "ACH-TEST-1", TEST_ROLE, n_genes=3, response_dim=2, basal_dim=2, seed=8
    )
    validation = _make_line_examples(
        "ACH-VAL", TRAIN_HEAD_ROLE, n_genes=3, response_dim=2, basal_dim=2, seed=9
    )
    with pytest.raises(ValueError, match="refusing to admit"):
        train_tx1_geneeffect_head([bad], [validation], TrainingConfig(epochs=1))


def test_train_tx1_geneeffect_head_rejects_test_role_line_in_validation_set() -> None:
    train_line = _make_line_examples(
        "ACH-TRAIN", TRAIN_HEAD_ROLE, n_genes=3, response_dim=2, basal_dim=2, seed=10
    )
    bad = _make_line_examples(
        "ACH-TEST-1", TEST_ROLE, n_genes=3, response_dim=2, basal_dim=2, seed=11
    )
    with pytest.raises(ValueError, match="refusing to admit"):
        train_tx1_geneeffect_head([train_line], [bad], TrainingConfig(epochs=1))


# --- D12: a validation line must never enter the training set ---------------
#
# This is the one defect with no visible symptom: both lines below carry an
# admissible role (train_head), so nothing about shapes, dtypes, or the
# forward pass would look wrong -- only the explicit disjointness guard
# catches it. Removing `_assert_examples_admissible`'s intersection check
# would make this test pass silently (no exception), which is exactly the
# failure mode the guard exists to prevent.


def test_train_tx1_geneeffect_head_rejects_validation_line_in_training_set() -> None:
    train_line = _make_line_examples(
        "ACH-TRAIN", TRAIN_HEAD_ROLE, n_genes=3, response_dim=2, basal_dim=2, seed=12
    )
    shared_model_id = "ACH-SHARED-VALIDATION-LINE"
    validation_line = _make_line_examples(
        shared_model_id,
        TRAIN_HEAD_ROLE,
        n_genes=3,
        response_dim=2,
        basal_dim=2,
        seed=13,
    )
    # The same validation model_id, independently re-built (a realistic
    # caller bug: e.g. failing to subtract the validation set out of the
    # training pool before assembly), leaked into the training list too.
    leaked_into_train = _make_line_examples(
        shared_model_id,
        TRAIN_HEAD_ROLE,
        n_genes=3,
        response_dim=2,
        basal_dim=2,
        seed=13,
    )

    with pytest.raises(ValueError, match="validation line"):
        train_tx1_geneeffect_head(
            [train_line, leaked_into_train], [validation_line], TrainingConfig(epochs=1)
        )


# --- D12: selection uses the validation lines and nothing else --------------


def test_select_best_epoch_uses_validation_loss_not_train_loss() -> None:
    """Adversarial: epoch 0 has the worst train_loss but the best
    (lowest) validation_loss; epoch 1 is the reverse. Selection must pick
    epoch 0 -- this would fail if `_select_best_epoch` were changed to key
    off `train_loss` instead."""
    history = (
        EpochMetrics(epoch=0, train_loss=0.9, validation_loss=0.1),
        EpochMetrics(epoch=1, train_loss=0.1, validation_loss=0.9),
    )
    best = _select_best_epoch(history)
    assert best.epoch == 0
    assert best.validation_loss == pytest.approx(0.1)


def test_select_best_epoch_raises_on_empty_history() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _select_best_epoch(())


def test_train_tx1_geneeffect_head_selection_matches_recorded_history() -> None:
    train_a = _make_line_examples(
        "ACH-A", TRAIN_HEAD_ROLE, n_genes=6, response_dim=3, basal_dim=2, seed=21
    )
    train_b = _make_line_examples(
        "ACH-B", TRAIN_HEAD_ROLE, n_genes=6, response_dim=3, basal_dim=2, seed=22
    )
    validation = _make_line_examples(
        "ACH-V", TRAIN_HEAD_ROLE, n_genes=6, response_dim=3, basal_dim=2, seed=23
    )
    config = TrainingConfig(hidden=16, epochs=10, learning_rate=0.05, seed=0)

    _head, result = train_tx1_geneeffect_head([train_a, train_b], [validation], config)

    expected_best = min(result.history, key=lambda entry: entry.validation_loss)
    assert result.best_epoch == expected_best.epoch
    assert result.best_validation_metric == pytest.approx(expected_best.validation_loss)
    assert result.selection_metric == SELECTION_METRIC_NAME
    assert set(result.train_model_ids) == {"ACH-A", "ACH-B"}
    assert result.validation_model_ids == ("ACH-V",)


# --- assemble_line_examples (I/O) --------------------------------------------


def test_assemble_line_examples_rejects_test_role_before_touching_any_cache(
    tmp_path: Path,
) -> None:
    # Deliberately non-existent cache directories: if the role guard did not
    # fire first, this would raise FileNotFoundError instead.
    with pytest.raises(ValueError, match="refusing to admit"):
        assemble_line_examples(
            "ACH-TEST-1",
            TEST_ROLE,
            ARM_TX1,
            tmp_path / "no_such_tx1_cache",
            tmp_path / "no_such_predicted_cache",
            "fp",
            pd.Series({"G1": 0.1}),
        )


def _write_basal_cache(tx1_cache_dir: Path, model_id: str, n_cells: int = 4) -> dict:
    embeddings = (
        np.random.default_rng(0)
        .normal(size=(n_cells, EMBEDDING_WIDTH))
        .astype(np.float32)
    )
    hvg = np.random.default_rng(1).normal(size=(n_cells, 2)).astype(np.float32)
    obs = pd.DataFrame({"cell": range(n_cells)})
    return write_line_cache(
        tx1_cache_dir, model_id, embeddings, hvg, obs, hvg_gene_order=["H1", "H2"]
    )


def test_assemble_line_examples_builds_expected_bags_and_intersects_genes(
    tmp_path: Path,
) -> None:
    tx1_cache_dir = tmp_path / "tx1_cache"
    _write_basal_cache(tx1_cache_dir, "ACH-1", n_cells=4)

    predicted_cache_dir = tmp_path / "predicted_cache"
    responses = {
        "G1": np.random.default_rng(2).normal(size=(5, 3)).astype(np.float32),
        "G2": np.random.default_rng(3).normal(size=(5, 3)).astype(np.float32),
        "G3": np.random.default_rng(4).normal(size=(5, 3)).astype(np.float32),
    }
    write_predicted_response_cache(
        predicted_cache_dir, "ACH-1", ARM_TX1, "fp", responses
    )

    # G3 has a cached response but no finite target -> excluded. G4 has a
    # target but no cached response -> excluded. Only G1/G2 have both.
    gene_effect = pd.Series({"G1": 0.5, "G2": -1.0, "G3": np.nan, "G4": 2.0})

    line = assemble_line_examples(
        "ACH-1",
        TRAIN_HEAD_ROLE,
        ARM_TX1,
        tx1_cache_dir,
        predicted_cache_dir,
        "fp",
        gene_effect,
    )

    assert line.genes == ("G1", "G2")
    assert line.basal_bag.shape == (4, EMBEDDING_WIDTH)
    assert len(line.response_bags) == 2
    assert line.response_bags[0].shape == (5, 3)
    assert torch.allclose(line.targets, torch.tensor([0.5, -1.0]))


def test_assemble_line_examples_raises_when_no_gene_has_both(tmp_path: Path) -> None:
    tx1_cache_dir = tmp_path / "tx1_cache"
    _write_basal_cache(tx1_cache_dir, "ACH-1")
    predicted_cache_dir = tmp_path / "predicted_cache"
    write_predicted_response_cache(
        predicted_cache_dir,
        "ACH-1",
        ARM_TX1,
        "fp",
        {"G1": np.zeros((2, 3), dtype=np.float32)},
    )
    gene_effect = pd.Series({"G2": 1.0})  # no overlap with the cached G1

    with pytest.raises(ValueError, match="no gene has"):
        assemble_line_examples(
            "ACH-1",
            TRAIN_HEAD_ROLE,
            ARM_TX1,
            tx1_cache_dir,
            predicted_cache_dir,
            "fp",
            gene_effect,
        )


def test_assemble_line_examples_rejects_unknown_arm(tmp_path: Path) -> None:
    tx1_cache_dir = tmp_path / "tx1_cache"
    _write_basal_cache(tx1_cache_dir, "ACH-1")
    with pytest.raises(ValueError, match="unknown arm"):
        assemble_line_examples(
            "ACH-1",
            TRAIN_HEAD_ROLE,
            "bogus_arm",
            tx1_cache_dir,
            tmp_path / "predicted_cache",
            "fp",
            pd.Series({"G1": 0.1}),
        )


# --- assemble_split_examples: wiring to Task 1's own selector ----------------


def test_assemble_split_examples_wires_to_task1_selector_and_skips_test_line(
    tmp_path: Path,
) -> None:
    rows = []
    for lineage in ("LinA", "LinB"):
        for index in range(3):
            rows.append(
                _manifest_row(
                    model_id=f"ACH-{lineage}-{index}",
                    lineage=lineage,
                    role=TRAIN_HEAD_ROLE,
                )
            )
    rows.append(
        _manifest_row(
            model_id="ACH-ANCHOR",
            lineage="Anchor",
            role=TRAIN_RESPONSE_ROLE,
            basal_source="Perturb-seq non-targeting control",
        )
    )
    rows.append(
        _manifest_row(model_id="ACH-HELDOUT", lineage="TestLineage", role=TEST_ROLE)
    )
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, rows)
    manifest = load_line_manifest(manifest_path)

    registration = generate_validation_line_registration(
        manifest, manifest_path, validation_count=1, seed=0
    )
    split = build_line_role_split(manifest, registration)

    tx1_cache_dir = tmp_path / "tx1_cache"
    predicted_cache_dir = tmp_path / "predicted_cache"
    genes = ["G1", "G2"]
    gene_effect_by_line: dict[str, pd.Series] = {}
    fingerprint_by_line: dict[str, str] = {}
    line_arrays: dict[str, dict] = {}
    for row in manifest.itertuples(index=False):
        model_id = str(row.model_id)
        if str(row.role) == TEST_ROLE:
            continue  # deliberately never write a cache for the held-out line
        arrays = _write_basal_cache(tx1_cache_dir, model_id)
        line_arrays[model_id] = arrays
        responses = {
            gene: np.random.default_rng(hash((model_id, gene)) % (2**31))
            .normal(size=(5, 3))
            .astype(np.float32)
            for gene in genes
        }
        # Deliberately a DIFFERENT literal per line (not the real
        # sha256-based fingerprint, since these are hand-built synthetic
        # bytes) -- this is the property `fingerprint_by_line` exists to
        # carry: a real run's per-model_id fingerprint always differs.
        fingerprint = f"fp-{model_id}"
        fingerprint_by_line[model_id] = fingerprint
        write_predicted_response_cache(
            predicted_cache_dir, model_id, ARM_TX1, fingerprint, responses
        )
        gene_effect_by_line[model_id] = pd.Series({"G1": 0.5, "G2": -0.2})
    _write_cache_run_manifest(tx1_cache_dir, line_arrays, ["H1", "H2"])

    train_lines, validation_lines = assemble_split_examples(
        manifest,
        registration,
        arm=ARM_TX1,
        tx1_cache_dir=tx1_cache_dir,
        predicted_response_cache_dir=predicted_cache_dir,
        fingerprint_by_line=fingerprint_by_line,
        gene_effect_by_line=gene_effect_by_line,
    )

    assert {line.model_id for line in train_lines} == set(split.train_model_ids)
    assert {line.model_id for line in validation_lines} == set(
        split.validation_model_ids
    )
    all_ids = {line.model_id for line in train_lines} | {
        line.model_id for line in validation_lines
    }
    assert "ACH-HELDOUT" not in all_ids
    assert all_ids.isdisjoint(split.test_model_ids)


# --- checkpoint save/load round trip ------------------------------------------


def test_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    head = Tx1GeneEffectHead(response_dim=3, basal_dim=2, hidden=8, moments=2)
    line = _make_line_examples(
        "ACH-1", TRAIN_HEAD_ROLE, n_genes=4, response_dim=3, basal_dim=2, seed=41
    )
    _, predictions_before = compute_line_features_and_predictions(head, line)

    checkpoint_path = tmp_path / "head.pt"
    save_checkpoint(head, checkpoint_path)
    loaded = load_checkpoint(checkpoint_path)

    _, predictions_after = compute_line_features_and_predictions(loaded, line)
    assert torch.allclose(predictions_before, predictions_after)
    assert loaded.response_dim == head.response_dim
    assert loaded.basal_dim == head.basal_dim
    assert loaded.moments == head.moments


# --- provenance ---------------------------------------------------------------


def _write_phase_b_manifest(path: Path, entries: dict[str, tuple[str, str]]) -> None:
    """``entries``: ``{model_id: (embeddings_sha256, hvg_sha256)}``."""
    lines = {
        model_id: {
            "arrays": {
                "embeddings.npy": {"sha256": embeddings_sha256},
                "hvg.npy": {"sha256": hvg_sha256},
            }
        }
        for model_id, (embeddings_sha256, hvg_sha256) in entries.items()
    }
    path.write_text(json.dumps({"lines": lines}))


def test_build_provenance_records_all_hashes_and_true_selection_metric(
    tmp_path: Path,
) -> None:
    train_a = _make_line_examples(
        "ACH-A", TRAIN_HEAD_ROLE, n_genes=4, response_dim=2, basal_dim=2, seed=31
    )
    validation = _make_line_examples(
        "ACH-V", TRAIN_HEAD_ROLE, n_genes=4, response_dim=2, basal_dim=2, seed=32
    )
    config = TrainingConfig(hidden=8, epochs=3, seed=7)
    _head, result = train_tx1_geneeffect_head([train_a], [validation], config)

    checkpoint_path = tmp_path / "pytorch_model.bin"
    checkpoint_path.write_bytes(b"fake-checkpoint-bytes")

    depmap_path = tmp_path / "CRISPRGeneEffect.csv"
    depmap_path.write_text("model_id,G1 (1)\nACH-A,0.1\n")

    phase_b_manifest_path = tmp_path / "manifest.json"
    _write_phase_b_manifest(
        phase_b_manifest_path,
        {
            "ACH-A": ("hash-ACH-A-emb", "hash-ACH-A-hvg"),
            "ACH-V": ("hash-ACH-V-emb", "hash-ACH-V-hvg"),
        },
    )

    provenance = build_provenance(
        st_checkpoint_path=checkpoint_path,
        phase_b_manifest_path=phase_b_manifest_path,
        depmap_gene_effect_path=depmap_path,
        arm=ARM_TX1,
        config=config,
        result=result,
    )

    assert provenance.st_checkpoint_sha256 == sha256_file(checkpoint_path)
    assert provenance.depmap_gene_effect_sha256 == sha256_file(depmap_path)
    assert provenance.seed == config.seed
    assert provenance.validation_model_ids == result.validation_model_ids
    assert provenance.train_model_ids == result.train_model_ids
    assert (
        provenance.selection_metric == result.selection_metric == SELECTION_METRIC_NAME
    )
    assert provenance.selection_metric_value == pytest.approx(
        result.best_validation_metric
    )
    assert provenance.best_epoch == result.best_epoch
    assert provenance.phase_b_cache_manifest_hashes["ACH-A"] == {
        "embeddings_sha256": "hash-ACH-A-emb",
        "hvg_sha256": "hash-ACH-A-hvg",
    }
    assert provenance.phase_b_cache_manifest_hashes["ACH-V"] == {
        "embeddings_sha256": "hash-ACH-V-emb",
        "hvg_sha256": "hash-ACH-V-hvg",
    }


def test_build_provenance_raises_when_line_not_recorded_in_phase_b_manifest(
    tmp_path: Path,
) -> None:
    train_a = _make_line_examples(
        "ACH-A", TRAIN_HEAD_ROLE, n_genes=3, response_dim=2, basal_dim=2, seed=33
    )
    validation = _make_line_examples(
        "ACH-V", TRAIN_HEAD_ROLE, n_genes=3, response_dim=2, basal_dim=2, seed=34
    )
    config = TrainingConfig(hidden=8, epochs=1, seed=0)
    _head, result = train_tx1_geneeffect_head([train_a], [validation], config)

    checkpoint_path = tmp_path / "pytorch_model.bin"
    checkpoint_path.write_bytes(b"fake")
    depmap_path = tmp_path / "CRISPRGeneEffect.csv"
    depmap_path.write_text("model_id,G1 (1)\nACH-A,0.1\n")
    phase_b_manifest_path = tmp_path / "manifest.json"
    # ACH-V is deliberately absent from the recorded lines.
    _write_phase_b_manifest(phase_b_manifest_path, {"ACH-A": ("h1", "h2")})

    with pytest.raises(ValueError, match="not recorded in Phase B cache manifest"):
        build_provenance(
            st_checkpoint_path=checkpoint_path,
            phase_b_manifest_path=phase_b_manifest_path,
            depmap_gene_effect_path=depmap_path,
            arm=ARM_TX1,
            config=config,
            result=result,
        )


def test_write_provenance_round_trips_through_json(tmp_path: Path) -> None:
    provenance = Tx1GeneEffectProvenance(
        st_checkpoint_sha256="a" * 64,
        phase_b_cache_manifest_hashes={
            "ACH-1": {"embeddings_sha256": "e", "hvg_sha256": "h"}
        },
        depmap_gene_effect_sha256="b" * 64,
        seed=0,
        validation_model_ids=("ACH-V",),
        train_model_ids=("ACH-1",),
        selection_metric=SELECTION_METRIC_NAME,
        selection_metric_value=0.1,
        best_epoch=2,
        arm=ARM_TX1,
        config={"hidden": 8},
        history=({"epoch": 0, "train_loss": 1.0, "validation_loss": 1.0},),
    )
    path = tmp_path / "provenance.json"
    write_provenance(provenance, path)

    loaded = json.loads(path.read_text())
    assert loaded["selection_metric"] == SELECTION_METRIC_NAME
    assert loaded["validation_model_ids"] == ["ACH-V"]
    assert loaded["phase_b_cache_manifest_hashes"]["ACH-1"]["embeddings_sha256"] == "e"


# --- ARM_HVG sanity: both arms are the identical head/data shape (D7) -------


def test_assemble_line_examples_supports_hvg_arm(tmp_path: Path) -> None:
    tx1_cache_dir = tmp_path / "tx1_cache"
    _write_basal_cache(tx1_cache_dir, "ACH-1", n_cells=6)
    predicted_cache_dir = tmp_path / "predicted_cache"
    write_predicted_response_cache(
        predicted_cache_dir,
        "ACH-1",
        ARM_HVG,
        "fp",
        {"G1": np.zeros((3, 2), dtype=np.float32)},
    )
    gene_effect = pd.Series({"G1": 0.3})

    line = assemble_line_examples(
        "ACH-1",
        TRAIN_HEAD_ROLE,
        ARM_HVG,
        tx1_cache_dir,
        predicted_cache_dir,
        "fp",
        gene_effect,
    )

    assert line.arm == ARM_HVG
    # The HVG matrix's own width (2), not EMBEDDING_WIDTH.
    assert line.basal_bag.shape == (6, 2)
