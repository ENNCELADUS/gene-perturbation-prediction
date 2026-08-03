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

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import aivc_model.tx1_geneeffect_train_io as tx1_geneeffect_train_io_module
from aivc_model.gene_splits import sha256_file
from aivc_model.model import (
    LinearMockStateModel,
    PerturbationVectorAdapter,
    StateForwardAdapter,
)
from aivc_model.tx1_basal import load_line_manifest
from aivc_model.tx1_embed_cache import (
    EMBEDDING_WIDTH,
    load_line_cache,
    write_line_cache,
)
from aivc_model.tx1_geneeffect_data import (
    TEST_ROLE,
    TRAIN_HEAD_ROLE,
    TRAIN_RESPONSE_ROLE,
    build_line_role_split,
    generate_validation_line_registration,
)
from aivc_model.tx1_geneeffect_head import (
    FUSION_INTERACTION_RESIDUAL,
    Tx1GeneEffectHead,
    rank_variance_loss,
)
from aivc_model.tx1_geneeffect_train import (
    SELECTION_METRIC_NAME,
    EpochMetrics,
    LineExamples,
    PooledLineExamples,
    TrainingConfig,
    compute_line_features_and_predictions,
    train_tx1_geneeffect_head,
)
from aivc_model.tx1_geneeffect_train import (
    _line_pooled_features,
    _precompute_pooled_features,
    _select_best_epoch,
)
from aivc_model.tx1_geneeffect_train_io import (
    Tx1GeneEffectProvenance,
    assemble_line_features,
    assemble_split_features,
    build_provenance,
    load_checkpoint,
    save_checkpoint,
    write_provenance,
)
from aivc_model.tx1_predicted_response import (
    ARM_HVG,
    ARM_TX1,
    ForwardOnlyStateModel,
    generate_predicted_response,
)
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


def test_train_tx1_geneeffect_head_runs_on_requested_device() -> None:
    train_line = _make_line_examples(
        "ACH-TRAIN", TRAIN_HEAD_ROLE, n_genes=4, response_dim=2, basal_dim=2, seed=51
    )
    validation = _make_line_examples(
        "ACH-VAL", TRAIN_HEAD_ROLE, n_genes=4, response_dim=2, basal_dim=2, seed=52
    )

    head, result = train_tx1_geneeffect_head(
        [train_line],
        [validation],
        TrainingConfig(hidden=8, epochs=1),
        device=torch.device("cpu"),
    )

    assert next(head.parameters()).device == torch.device("cpu")
    assert len(result.history) == 1


def test_train_tx1_geneeffect_head_wires_interaction_architecture() -> None:
    train_line = _make_line_examples(
        "ACH-TRAIN", TRAIN_HEAD_ROLE, n_genes=8, response_dim=4, basal_dim=3, seed=61
    )
    validation = _make_line_examples(
        "ACH-VAL", TRAIN_HEAD_ROLE, n_genes=8, response_dim=4, basal_dim=3, seed=62
    )
    head, result = train_tx1_geneeffect_head(
        [train_line],
        [validation],
        TrainingConfig(
            hidden=16,
            epochs=5,
            fusion=FUSION_INTERACTION_RESIDUAL,
            projection_dim=8,
        ),
    )
    assert head.fusion == FUSION_INTERACTION_RESIDUAL
    assert head.projection_dim == 8
    assert all(np.isfinite(entry.validation_loss) for entry in result.history)


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


# --- P2-1: pooled features are precomputed once, not every epoch ------------


def test_precomputed_pooled_features_match_fresh_recomputation() -> None:
    """The precomputed feature cache must be bit-for-bit identical to what
    freshly recomputing ``_line_pooled_features`` from a line's own (frozen)
    basal/response bags produces -- proving the precompute step introduces
    no staleness, off-by-one, or cross-line mixup. This would fail if
    ``_precompute_pooled_features`` ever cached the WRONG line's tensor
    under a given ``model_id``, or pooled with the wrong ``moments``."""
    lines = [
        _make_line_examples(
            "ACH-A", TRAIN_HEAD_ROLE, n_genes=4, response_dim=3, basal_dim=2, seed=101
        ),
        _make_line_examples(
            "ACH-B", TRAIN_HEAD_ROLE, n_genes=5, response_dim=3, basal_dim=2, seed=102
        ),
    ]
    moments = 3

    precomputed = _precompute_pooled_features(lines, moments)

    assert set(precomputed) == {"ACH-A", "ACH-B"}
    for line in lines:
        fresh = _line_pooled_features(line, moments)
        assert torch.equal(precomputed[line.model_id], fresh)


def test_train_tx1_geneeffect_head_matches_a_naive_per_epoch_recomputation() -> None:
    """P2-1's deeper claim: precomputing pooled features once must not
    change training dynamics at all. Reimplements the OLD per-epoch
    recomputation training loop by hand (identical seed, data, and
    hyperparameters) and asserts the real (new, precomputing)
    ``train_tx1_geneeffect_head`` produces BIT-FOR-BIT identical final head
    weights and the same selected epoch -- proving moving the pooling
    earlier in time changed nothing about what was computed."""
    response_dim, basal_dim = 3, 2
    train_a = _make_line_examples(
        "ACH-TRAIN-A",
        TRAIN_HEAD_ROLE,
        n_genes=6,
        response_dim=response_dim,
        basal_dim=basal_dim,
        seed=201,
    )
    train_b = _make_line_examples(
        "ACH-TRAIN-B",
        TRAIN_HEAD_ROLE,
        n_genes=6,
        response_dim=response_dim,
        basal_dim=basal_dim,
        seed=202,
    )
    validation = _make_line_examples(
        "ACH-VAL",
        TRAIN_HEAD_ROLE,
        n_genes=6,
        response_dim=response_dim,
        basal_dim=basal_dim,
        seed=203,
    )
    config = TrainingConfig(
        hidden=16, moments=2, lam=1.0, learning_rate=0.05, epochs=5, seed=7
    )

    new_head, new_result = train_tx1_geneeffect_head(
        [train_a, train_b], [validation], config
    )

    # The OLD code path, reimplemented here: recompute pooled features from
    # the raw bags on every single epoch, rather than once up front.
    torch.manual_seed(int(config.seed))
    old_head = Tx1GeneEffectHead(
        response_dim=response_dim,
        basal_dim=basal_dim,
        hidden=config.hidden,
        moments=config.moments,
    )
    optimizer = torch.optim.Adam(old_head.parameters(), lr=config.learning_rate)
    ordered_train = sorted([train_a, train_b], key=lambda line: line.model_id)
    ordered_validation = [validation]
    history: list[tuple[int, float]] = []
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    for epoch in range(config.epochs):
        old_head.train()
        for line in ordered_train:
            optimizer.zero_grad()
            features = _line_pooled_features(line, old_head.moments)  # recomputed
            predictions = old_head.net(features).squeeze(-1)
            loss = rank_variance_loss(predictions, line.targets, lam=config.lam)
            loss.backward()
            optimizer.step()

        old_head.eval()
        validation_losses = []
        with torch.no_grad():
            for line in ordered_validation:
                features = _line_pooled_features(line, old_head.moments)  # recomputed
                predictions = old_head.net(features).squeeze(-1)
                loss = rank_variance_loss(predictions, line.targets, lam=config.lam)
                validation_losses.append(float(loss))
        history.append((epoch, float(np.mean(validation_losses))))
        epoch_states[epoch] = copy.deepcopy(old_head.state_dict())

    best_epoch = min(history, key=lambda entry: entry[1])[0]
    old_head.load_state_dict(epoch_states[best_epoch])

    assert new_result.best_epoch == best_epoch
    for new_param, old_param in zip(
        new_head.parameters(), old_head.parameters(), strict=True
    ):
        assert torch.equal(new_param, old_param)


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


# --- streaming assembly (I/O) -------------------------------------------------


def test_assemble_line_features_rejects_test_role_before_touching_cache(
    tmp_path: Path,
) -> None:
    # Deliberately non-existent cache directories: if the role guard did not
    # fire first, this would raise FileNotFoundError instead.
    with pytest.raises(ValueError, match="refusing to admit"):
        assemble_line_features(
            "ACH-TEST-1",
            TEST_ROLE,
            ARM_HVG,
            tmp_path / "no_such_tx1_cache",
            pd.Series({"G1": 0.1}),
            ForwardOnlyStateModel(
                StateForwardAdapter(LinearMockStateModel(2, 3, 2)),
                PerturbationVectorAdapter(["G1"], {}, 2),
            ),
            moments=2,
            cell_set_len=2,
            seed=0,
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


def test_assemble_line_features_matches_raw_pooling_and_writes_no_cache(
    tmp_path: Path,
) -> None:
    tx1_cache_dir = tmp_path / "tx1_cache"
    _write_basal_cache(tx1_cache_dir, "ACH-1", n_cells=4)

    model = ForwardOnlyStateModel(
        StateForwardAdapter(LinearMockStateModel(2, 3, 2)),
        PerturbationVectorAdapter(["G1", "G2"], {}, 2),
    )
    gene_effect = pd.Series({"G1": 0.5, "G2": -1.0})
    line = assemble_line_features(
        "ACH-1",
        TRAIN_HEAD_ROLE,
        ARM_HVG,
        tx1_cache_dir,
        gene_effect,
        model,
        moments=2,
        cell_set_len=3,
        window_macro_batch_size=2,
        seed=0,
    )

    assert isinstance(line, PooledLineExamples)
    assert line.genes == ("G1", "G2")
    assert line.features.shape == (2, (3 + 2) * 2)
    assert torch.allclose(line.targets, torch.tensor([0.5, -1.0]))
    assert not (tmp_path / "predicted_cache").exists()

    _embeddings, hvg, _obs = load_line_cache(tx1_cache_dir, "ACH-1")
    basal = np.asarray(hvg, dtype=np.float32)
    raw = LineExamples(
        model_id="ACH-1",
        role=TRAIN_HEAD_ROLE,
        arm=ARM_HVG,
        basal_bag=torch.from_numpy(basal.copy()),
        genes=("G1", "G2"),
        response_bags=tuple(
            generate_predicted_response(
                model,
                basal,
                gene,
                cell_set_len=3,
                seed=0,
                batch_labels=np.full(len(basal), "ACH-1", dtype=object),
            )
            for gene in ("G1", "G2")
        ),
        targets=line.targets,
    )
    assert torch.allclose(
        line.features, _line_pooled_features(raw, moments=2), atol=1e-6, rtol=1e-5
    )


def test_assemble_line_features_keeps_compact_features_on_forward_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tx1_cache_dir = tmp_path / "tx1_cache"
    _write_basal_cache(tx1_cache_dir, "ACH-1", n_cells=4)
    model = ForwardOnlyStateModel(
        StateForwardAdapter(LinearMockStateModel(2, 3, 2)),
        PerturbationVectorAdapter(["G1"], {}, 2),
    )
    requested_device = torch.device("meta")

    def _pooled_on_requested_device(*args: object, **kwargs: object):
        assert kwargs["device"] == requested_device
        return torch.zeros(6, device=requested_device), 3

    monkeypatch.setattr(
        tx1_geneeffect_train_io_module,
        "generate_pooled_predicted_response",
        _pooled_on_requested_device,
    )

    line = assemble_line_features(
        "ACH-1",
        TRAIN_HEAD_ROLE,
        ARM_HVG,
        tx1_cache_dir,
        pd.Series({"G1": 0.5}),
        model,
        moments=2,
        cell_set_len=2,
        seed=0,
        device=requested_device,
    )

    assert line.features.device == requested_device
    assert line.targets.device == requested_device


def test_assemble_line_features_raises_without_finite_targets(tmp_path: Path) -> None:
    tx1_cache_dir = tmp_path / "tx1_cache"
    _write_basal_cache(tx1_cache_dir, "ACH-1")
    gene_effect = pd.Series({"G1": np.nan})
    model = ForwardOnlyStateModel(
        StateForwardAdapter(LinearMockStateModel(2, 3, 2)),
        PerturbationVectorAdapter(["G1"], {}, 2),
    )

    with pytest.raises(ValueError, match="no finite"):
        assemble_line_features(
            "ACH-1",
            TRAIN_HEAD_ROLE,
            ARM_HVG,
            tx1_cache_dir,
            gene_effect,
            model,
            moments=2,
            cell_set_len=2,
            seed=0,
        )


def test_assemble_line_features_rejects_unknown_arm(tmp_path: Path) -> None:
    tx1_cache_dir = tmp_path / "tx1_cache"
    _write_basal_cache(tx1_cache_dir, "ACH-1")
    with pytest.raises(ValueError, match="unknown arm"):
        assemble_line_features(
            "ACH-1",
            TRAIN_HEAD_ROLE,
            "bogus_arm",
            tx1_cache_dir,
            pd.Series({"G1": 0.1}),
            ForwardOnlyStateModel(
                StateForwardAdapter(LinearMockStateModel(2, 3, 2)),
                PerturbationVectorAdapter(["G1"], {}, 2),
            ),
            moments=2,
            cell_set_len=2,
            seed=0,
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
    genes = ["G1", "G2"]
    gene_effect_by_line: dict[str, pd.Series] = {}
    line_arrays: dict[str, dict] = {}
    for row in manifest.itertuples(index=False):
        model_id = str(row.model_id)
        if str(row.role) == TEST_ROLE:
            continue  # deliberately never write a cache for the held-out line
        arrays = _write_basal_cache(tx1_cache_dir, model_id)
        line_arrays[model_id] = arrays
        gene_effect_by_line[model_id] = pd.Series({"G1": 0.5, "G2": -0.2})
    _write_cache_run_manifest(tx1_cache_dir, line_arrays, ["H1", "H2"])

    model = ForwardOnlyStateModel(
        StateForwardAdapter(LinearMockStateModel(EMBEDDING_WIDTH, 3, 2)),
        PerturbationVectorAdapter(genes, {}, 2),
    )
    train_lines, validation_lines = assemble_split_features(
        manifest,
        registration,
        arm=ARM_TX1,
        tx1_cache_dir=tx1_cache_dir,
        gene_effect_by_line=gene_effect_by_line,
        model=model,
        moments=2,
        cell_set_len=3,
        seed=0,
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


def test_interaction_checkpoint_round_trip_records_architecture(tmp_path: Path) -> None:
    head = Tx1GeneEffectHead(
        response_dim=3,
        basal_dim=2,
        hidden=8,
        moments=2,
        fusion=FUSION_INTERACTION_RESIDUAL,
        projection_dim=4,
    )
    checkpoint_path = save_checkpoint(head, tmp_path / "interaction.pt")
    loaded = load_checkpoint(checkpoint_path)
    pooled = torch.randn(6, 10)
    assert loaded.fusion == FUSION_INTERACTION_RESIDUAL
    assert loaded.projection_dim == 4
    assert torch.equal(head.forward_pooled(pooled), loaded.forward_pooled(pooled))


def test_legacy_concat_checkpoint_without_architecture_fields_loads(
    tmp_path: Path,
) -> None:
    head = Tx1GeneEffectHead(response_dim=3, basal_dim=2, hidden=8, moments=2)
    checkpoint_path = save_checkpoint(head, tmp_path / "legacy.pt")
    payload = torch.load(checkpoint_path, weights_only=False)
    payload.pop("fusion")
    payload.pop("projection_dim")
    torch.save(payload, checkpoint_path)
    loaded = load_checkpoint(checkpoint_path)
    pooled = torch.randn(6, 10)
    assert loaded.fusion == "concat_mlp"
    assert torch.equal(head.forward_pooled(pooled), loaded.forward_pooled(pooled))


def test_interaction_checkpoint_missing_projection_dim_is_rejected(
    tmp_path: Path,
) -> None:
    head = Tx1GeneEffectHead(
        response_dim=3,
        basal_dim=2,
        hidden=8,
        fusion=FUSION_INTERACTION_RESIDUAL,
        projection_dim=4,
    )
    checkpoint_path = save_checkpoint(head, tmp_path / "broken.pt")
    payload = torch.load(checkpoint_path, weights_only=False)
    payload.pop("projection_dim")
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="projection_dim"):
        load_checkpoint(checkpoint_path)


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
        response_generation={
            "response_generation_seed": 11,
            "cell_set_len": 64,
            "gene_vocabulary_sha256": "vocab-hash",
            "gene_panel_sha256": "panel-hash",
        },
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
    assert provenance.response_generation["response_generation_seed"] == 11
    assert provenance.response_generation["cell_set_len"] == 64
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


def test_assemble_line_features_supports_hvg_arm(tmp_path: Path) -> None:
    tx1_cache_dir = tmp_path / "tx1_cache"
    _write_basal_cache(tx1_cache_dir, "ACH-1", n_cells=6)
    gene_effect = pd.Series({"G1": 0.3})
    model = ForwardOnlyStateModel(
        StateForwardAdapter(LinearMockStateModel(2, 2, 2)),
        PerturbationVectorAdapter(["G1"], {}, 2),
    )

    line = assemble_line_features(
        "ACH-1",
        TRAIN_HEAD_ROLE,
        ARM_HVG,
        tx1_cache_dir,
        gene_effect,
        model,
        moments=2,
        cell_set_len=4,
        seed=0,
    )

    assert line.arm == ARM_HVG
    # The HVG matrix's own width (2), not EMBEDDING_WIDTH.
    assert line.basal_dim == 2
    assert line.features.shape == (1, 8)
