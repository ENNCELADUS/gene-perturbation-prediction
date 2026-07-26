"""Tests for src/aivc_model/tx1_predicted_response.py and its sibling
src/aivc_model/tx1_predicted_response_cache.py -- Phase D Task 2: the
forward-only ST loader, predicted-response generation, and the fingerprinted
predicted-response cache (Global Constraint D11).

No GPU and no real Phase C checkpoint exist on this machine, so every test
builds tiny synthetic fixtures in ``tmp_path``: a ``LinearMockStateModel``
standing in for STATE, a hand-built ``PerturbationVectorAdapter``, a fake
flat ``AivcModel``-shaped checkpoint state dict, and a fake Tx1 embedding
cache (written via ``tx1_embed_cache.write_line_cache``, never real Tx1
inference). Nothing here is allowed to skip silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from aivc_model.model import LinearMockStateModel, PerturbationVectorAdapter
from aivc_model.tx1_embed_cache import EMBEDDING_WIDTH, write_line_cache
from aivc_model.tx1_predicted_response import (
    ARM_HVG,
    ARM_TX1,
    ForwardOnlyStateModel,
    UnknownPerturbationGeneError,
    _chunk_control_cell_indices,
    construct_forward_only_model,
    generate_predicted_response,
    generate_predicted_response_for_line,
    load_forward_only_checkpoint,
    resolve_genes_against_vocabulary,
    vocabulary_genes,
)
from aivc_model.tx1_predicted_response_cache import (
    load_predicted_response_cache,
    predicted_response_fingerprint,
    write_predicted_response_cache,
)

_INPUT_DIM = 6
_OUTPUT_DIM = 4
_PERT_DIM = 3
_GENES = ["G1", "G2"]


def _forward_only_model(
    genes: list[str] | None = None, input_dim: int = _INPUT_DIM
) -> ForwardOnlyStateModel:
    """A small, freshly initialized ST + perturbation-adapter model for tests."""
    from aivc_model.model import StateForwardAdapter

    state_model = LinearMockStateModel(input_dim, _OUTPUT_DIM, _PERT_DIM)
    known_vectors = {"G1": np.ones(_PERT_DIM, dtype=np.float32)}
    perturbations = PerturbationVectorAdapter(
        list(genes or _GENES), known_vectors, _PERT_DIM
    )
    return ForwardOnlyStateModel(StateForwardAdapter(state_model), perturbations)


# --- forward-only checkpoint loading -----------------------------------------


def _fake_full_checkpoint(model: ForwardOnlyStateModel) -> dict[str, torch.Tensor]:
    """A flat state dict shaped like Phase C's real ``AivcModel.state_dict()``:
    the model's own real keys plus fake dropped-head keys."""
    checkpoint = dict(model.state_dict())
    checkpoint["response_encoder.linear.weight"] = torch.zeros(2, 2)
    checkpoint["response_pooler.means"] = torch.zeros(2, 2)
    checkpoint["c_head.net.0.weight"] = torch.zeros(2, 2)
    checkpoint["control_expression_mean"] = torch.zeros(_OUTPUT_DIM)
    return checkpoint


def test_load_forward_only_checkpoint_restores_keys_and_reports_dropped(
    tmp_path: Path,
) -> None:
    source = _forward_only_model()
    checkpoint = _fake_full_checkpoint(source)
    checkpoint_path = tmp_path / "pytorch_model.bin"
    torch.save(checkpoint, checkpoint_path)

    destination = _forward_only_model()  # freshly initialized -- different weights
    report = load_forward_only_checkpoint(destination, checkpoint_path)

    assert sorted(report.loaded_keys) == sorted(source.state_dict().keys())
    assert sorted(report.dropped_keys) == sorted(
        [
            "response_encoder.linear.weight",
            "response_pooler.means",
            "c_head.net.0.weight",
            "control_expression_mean",
        ]
    )
    for name, tensor in source.state_dict().items():
        assert torch.equal(destination.state_dict()[name], tensor)
    assert destination.training is False  # eval() was called
    assert all(not parameter.requires_grad for parameter in destination.parameters())


def test_load_forward_only_checkpoint_raises_on_missing_key(tmp_path: Path) -> None:
    source = _forward_only_model()
    checkpoint = _fake_full_checkpoint(source)
    del checkpoint["perturbations.known_matrix"]
    checkpoint_path = tmp_path / "pytorch_model.bin"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="missing"):
        load_forward_only_checkpoint(_forward_only_model(), checkpoint_path)


def test_load_forward_only_checkpoint_raises_on_disallowed_unexpected_key(
    tmp_path: Path,
) -> None:
    source = _forward_only_model()
    checkpoint = _fake_full_checkpoint(source)
    checkpoint["some_other_module.weight"] = torch.zeros(1)
    checkpoint_path = tmp_path / "pytorch_model.bin"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="disallowed_unexpected"):
        load_forward_only_checkpoint(_forward_only_model(), checkpoint_path)


def test_load_forward_only_checkpoint_raises_on_zero_loadable_keys(
    tmp_path: Path,
) -> None:
    checkpoint = {"response_encoder.linear.weight": torch.zeros(2, 2)}
    checkpoint_path = tmp_path / "pytorch_model.bin"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="loaded_count=0"):
        load_forward_only_checkpoint(_forward_only_model(), checkpoint_path)


class _FakeStateModel(nn.Module):
    """Minimal STATE-shaped module for ``construct_forward_only_model`` tests."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        **_unused_hparams: object,
    ) -> None:
        super().__init__()
        del pert_dim
        self.basal_encoder = nn.Linear(input_dim, hidden_dim)
        self.project_out = nn.Linear(hidden_dim, output_dim)


def test_construct_forward_only_model_builds_expected_architecture(
    tmp_path: Path,
) -> None:
    reference = _FakeStateModel(
        input_dim=_INPUT_DIM, hidden_dim=5, output_dim=_OUTPUT_DIM, pert_dim=_PERT_DIM
    )
    checkpoint_path = tmp_path / "reference.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                "input_dim": _INPUT_DIM,
                "hidden_dim": 5,
                "output_dim": _OUTPUT_DIM,
                "pert_dim": _PERT_DIM,
            },
            "state_dict": reference.state_dict(),
        },
        checkpoint_path,
    )

    model = construct_forward_only_model(
        model_cls=_FakeStateModel,
        hparams_checkpoint_path=checkpoint_path,
        input_dim=_INPUT_DIM,
        output_dim=_OUTPUT_DIM,
        pert_dim=_PERT_DIM,
        genes=_GENES,
        known_perturbation_vectors=None,
    )

    assert isinstance(model, ForwardOnlyStateModel)
    assert model.state_adapter.state_model.basal_encoder.in_features == _INPUT_DIM
    assert isinstance(model.perturbations, PerturbationVectorAdapter)
    assert model.perturbations.genes == _GENES


# --- gene-vocabulary pre-filtering --------------------------------------------


def test_resolve_genes_against_vocabulary_raises_named_error_for_unknown_gene() -> None:
    model = _forward_only_model()
    with pytest.raises(UnknownPerturbationGeneError) as excinfo:
        resolve_genes_against_vocabulary(["G1", "UNKNOWN"], model.perturbations)
    assert "UNKNOWN" in str(excinfo.value)
    assert not isinstance(excinfo.value, KeyError)


def test_resolve_genes_against_vocabulary_applies_alias_map() -> None:
    model = _forward_only_model()
    resolved = resolve_genes_against_vocabulary(
        ["CRIPTO"], model.perturbations, alias_map={"CRIPTO": "G1"}
    )
    assert resolved == ("G1",)


def test_vocabulary_genes_matches_construction_vocabulary() -> None:
    model = _forward_only_model()
    assert vocabulary_genes(model.perturbations) == frozenset(_GENES)


# --- predicted-response generation --------------------------------------------


def test_generate_predicted_response_shape_and_no_gradient() -> None:
    model = _forward_only_model()
    n_cells = 13  # not a multiple of cell_set_len=4 -- exercises padding
    control_input = np.random.default_rng(0).normal(size=(n_cells, _INPUT_DIM))

    with torch.enable_grad():  # the function's own no_grad must still win
        response = generate_predicted_response(
            model, control_input, "G1", cell_set_len=4, seed=0
        )

    assert response.shape == (n_cells, _OUTPUT_DIM)
    assert response.requires_grad is False


def test_generate_predicted_response_raises_named_error_not_keyerror() -> None:
    model = _forward_only_model()
    control_input = np.zeros((5, _INPUT_DIM))
    with pytest.raises(UnknownPerturbationGeneError):
        generate_predicted_response(
            model, control_input, "NOT_A_GENE", cell_set_len=4, seed=0
        )


def test_generate_predicted_response_is_deterministic_given_seed() -> None:
    model = _forward_only_model()
    control_input = np.random.default_rng(1).normal(size=(9, _INPUT_DIM))
    first = generate_predicted_response(
        model, control_input, "G1", cell_set_len=4, seed=7
    )
    second = generate_predicted_response(
        model, control_input, "G1", cell_set_len=4, seed=7
    )
    assert torch.equal(first, second)


def test_chunk_control_cell_indices_padding_is_seed_sensitive() -> None:
    # 9 cells, cell_set_len=4 -> windows [0-3], [4-7], [8, <2 resampled pads>].
    # This is exactly why `seed` belongs in the D11 cache fingerprint: which
    # cells fill the final short window changes with the seed.
    chunks_seed_1 = _chunk_control_cell_indices(9, 4, seed=1)
    chunks_seed_2 = _chunk_control_cell_indices(9, 4, seed=2)
    assert chunks_seed_1[0].tolist() == [0, 1, 2, 3]
    assert chunks_seed_1[1].tolist() == [4, 5, 6, 7]
    assert chunks_seed_1[-1][0] == 8  # the real cell always comes first
    assert not np.array_equal(chunks_seed_1[-1], chunks_seed_2[-1])


def test_chunk_control_cell_indices_is_deterministic_given_seed() -> None:
    first = _chunk_control_cell_indices(9, 4, seed=5)
    second = _chunk_control_cell_indices(9, 4, seed=5)
    for chunk_a, chunk_b in zip(first, second, strict=True):
        assert np.array_equal(chunk_a, chunk_b)


# --- per-line generation and the training-role guard --------------------------


def _write_fake_line_cache(cache_dir: Path, model_id: str, n_cells: int) -> None:
    embeddings = np.random.default_rng(3).normal(size=(n_cells, EMBEDDING_WIDTH))
    hvg = np.random.default_rng(4).normal(size=(n_cells, 3))
    obs = pd.DataFrame(index=[f"{model_id}-{i}" for i in range(n_cells)])
    write_line_cache(
        cache_dir,
        model_id,
        embeddings,
        hvg,
        obs,
        hvg_gene_order=["GENE0", "GENE1", "GENE2"],
    )


def test_generate_predicted_response_for_line_rejects_test_role_by_default(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "tx1_cache"
    _write_fake_line_cache(cache_dir, "ACH-TEST", n_cells=5)
    model = _forward_only_model()

    with pytest.raises(ValueError, match="test"):
        generate_predicted_response_for_line(
            cache_dir,
            "ACH-TEST",
            "test",
            model,
            "G1",
            arm=ARM_TX1,
            cell_set_len=4,
            seed=0,
        )


def test_generate_predicted_response_for_line_allows_explicit_opt_out(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "tx1_cache"
    _write_fake_line_cache(cache_dir, "ACH-TEST", n_cells=5)
    model = _forward_only_model(input_dim=EMBEDDING_WIDTH)

    response = generate_predicted_response_for_line(
        cache_dir,
        "ACH-TEST",
        "test",
        model,
        "G1",
        arm=ARM_TX1,
        cell_set_len=4,
        seed=0,
        require_training_role=False,
    )
    assert response.shape == (5, _OUTPUT_DIM)


def test_generate_predicted_response_for_line_admits_training_role(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "tx1_cache"
    _write_fake_line_cache(cache_dir, "ACH-TRAIN", n_cells=5)
    model = _forward_only_model(input_dim=EMBEDDING_WIDTH)

    response = generate_predicted_response_for_line(
        cache_dir,
        "ACH-TRAIN",
        "train_head",
        model,
        "G1",
        arm=ARM_TX1,
        cell_set_len=4,
        seed=0,
    )
    assert response.shape == (5, _OUTPUT_DIM)


def test_generate_predicted_response_for_line_hvg_arm_uses_hvg_matrix(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "tx1_cache"
    _write_fake_line_cache(cache_dir, "ACH-TRAIN", n_cells=5)
    hvg_model = _forward_only_model()
    # HVG arm's basal input width is the HVG matrix width (3), not
    # EMBEDDING_WIDTH -- swap in a state model matching that input width.
    from aivc_model.model import StateForwardAdapter

    hvg_model.state_adapter = StateForwardAdapter(
        LinearMockStateModel(3, _OUTPUT_DIM, _PERT_DIM)
    )

    response = generate_predicted_response_for_line(
        cache_dir,
        "ACH-TRAIN",
        "train_head",
        hvg_model,
        "G1",
        arm=ARM_HVG,
        cell_set_len=4,
        seed=0,
    )
    assert response.shape == (5, _OUTPUT_DIM)


def test_generate_predicted_response_for_line_rejects_unknown_arm(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "tx1_cache"
    _write_fake_line_cache(cache_dir, "ACH-TRAIN", n_cells=5)
    model = _forward_only_model()

    with pytest.raises(ValueError, match="unknown arm"):
        generate_predicted_response_for_line(
            cache_dir,
            "ACH-TRAIN",
            "train_head",
            model,
            "G1",
            arm="bogus_arm",
            cell_set_len=4,
            seed=0,
        )


# --- D11: the fingerprinted predicted-response cache --------------------------


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


def _baseline_fingerprint_kwargs(tmp_path: Path) -> dict[str, object]:
    checkpoint_path = tmp_path / "checkpoint.bin"
    checkpoint_path.write_bytes(b"checkpoint-a")
    manifest_path = tmp_path / "manifest.json"
    _write_phase_b_manifest(
        manifest_path,
        {
            "ACH-0001": ("hash-aaa", "hash-hvg-aaa"),
            "ACH-0002": ("hash-aaa", "hash-hvg-aaa"),  # same hashes, different id
        },
    )
    return {
        "st_checkpoint_path": checkpoint_path,
        "phase_b_manifest_path": manifest_path,
        "model_id": "ACH-0001",
        "genes": ["G1", "G2"],
        "seed": 0,
        "arm": ARM_TX1,
        "cell_set_len": 4,
    }


def test_fingerprint_changes_when_st_checkpoint_content_changes(
    tmp_path: Path,
) -> None:
    baseline = _baseline_fingerprint_kwargs(tmp_path)
    changed_checkpoint = tmp_path / "checkpoint_b.bin"
    changed_checkpoint.write_bytes(b"checkpoint-b")
    changed = {**baseline, "st_checkpoint_path": changed_checkpoint}
    assert predicted_response_fingerprint(**baseline) != predicted_response_fingerprint(
        **changed
    )


def test_fingerprint_changes_when_phase_b_hashes_change(tmp_path: Path) -> None:
    baseline = _baseline_fingerprint_kwargs(tmp_path)
    changed_manifest = tmp_path / "manifest_b.json"
    _write_phase_b_manifest(
        changed_manifest,
        {"ACH-0001": ("hash-DIFFERENT", "hash-hvg-aaa")},
    )
    changed = {**baseline, "phase_b_manifest_path": changed_manifest}
    assert predicted_response_fingerprint(**baseline) != predicted_response_fingerprint(
        **changed
    )


def test_fingerprint_changes_when_model_id_changes(tmp_path: Path) -> None:
    baseline = _baseline_fingerprint_kwargs(tmp_path)
    changed = {**baseline, "model_id": "ACH-0002"}  # same recorded hashes as ACH-0001
    assert predicted_response_fingerprint(**baseline) != predicted_response_fingerprint(
        **changed
    )


def test_fingerprint_changes_when_gene_list_changes(tmp_path: Path) -> None:
    baseline = _baseline_fingerprint_kwargs(tmp_path)
    changed = {**baseline, "genes": ["G1", "G3"]}
    assert predicted_response_fingerprint(**baseline) != predicted_response_fingerprint(
        **changed
    )


def test_fingerprint_changes_when_seed_changes(tmp_path: Path) -> None:
    baseline = _baseline_fingerprint_kwargs(tmp_path)
    changed = {**baseline, "seed": 1}
    assert predicted_response_fingerprint(**baseline) != predicted_response_fingerprint(
        **changed
    )


def test_fingerprint_changes_when_arm_changes(tmp_path: Path) -> None:
    baseline = _baseline_fingerprint_kwargs(tmp_path)
    changed = {**baseline, "arm": ARM_HVG}
    assert predicted_response_fingerprint(**baseline) != predicted_response_fingerprint(
        **changed
    )


def test_fingerprint_changes_when_cell_set_len_changes(tmp_path: Path) -> None:
    baseline = _baseline_fingerprint_kwargs(tmp_path)
    changed = {**baseline, "cell_set_len": 8}
    assert predicted_response_fingerprint(**baseline) != predicted_response_fingerprint(
        **changed
    )


def test_write_and_load_predicted_response_cache_round_trip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "predicted_response_cache"
    responses = {
        "G1": np.arange(6, dtype=np.float32).reshape(2, 3),
        "G2": np.arange(6, 12, dtype=np.float32).reshape(2, 3),
    }
    write_predicted_response_cache(
        cache_dir, "ACH-0001", ARM_TX1, "fingerprint-a", responses
    )

    loaded = load_predicted_response_cache(
        cache_dir, "ACH-0001", ARM_TX1, "fingerprint-a"
    )

    assert set(loaded) == {"G1", "G2"}
    for gene, array in responses.items():
        np.testing.assert_array_equal(np.asarray(loaded[gene]), array)


def test_load_predicted_response_cache_refuses_on_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "predicted_response_cache"
    responses = {"G1": np.zeros((2, 3), dtype=np.float32)}
    write_predicted_response_cache(
        cache_dir, "ACH-0001", ARM_TX1, "fingerprint-a", responses
    )

    with pytest.raises(ValueError, match="stale"):
        load_predicted_response_cache(
            cache_dir, "ACH-0001", ARM_TX1, "fingerprint-B-DIFFERENT"
        )


def test_load_predicted_response_cache_refuses_on_schema_version_mismatch(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "predicted_response_cache"
    responses = {"G1": np.zeros((2, 3), dtype=np.float32)}
    manifest_path = write_predicted_response_cache(
        cache_dir, "ACH-0001", ARM_TX1, "fingerprint-a", responses
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="schema_version"):
        load_predicted_response_cache(cache_dir, "ACH-0001", ARM_TX1, "fingerprint-a")


def test_load_predicted_response_cache_raises_when_missing(tmp_path: Path) -> None:
    cache_dir = tmp_path / "predicted_response_cache"
    with pytest.raises(FileNotFoundError):
        load_predicted_response_cache(cache_dir, "ACH-NOPE", ARM_TX1, "fingerprint-a")


def test_slice_symbol_aliases_resolve_by_default() -> None:
    """CRIPTO/HEMK2 must resolve without the caller knowing to pass a map.

    Regression guard for the gap that would drop slice coverage from 587/587
    to 585/587 and fail Phase F's coverage validator *after* a training run.
    The pairs are measured (task-0-coverage.md §2), not assumed.
    """
    adapter = PerturbationVectorAdapter(["TDGF1", "N6AMT1", "ACLY"], {}, 4)
    resolved = resolve_genes_against_vocabulary(["CRIPTO", "HEMK2", "ACLY"], adapter)
    assert resolved == ("TDGF1", "N6AMT1", "ACLY")


def test_explicit_empty_alias_map_disables_aliasing() -> None:
    """Passing {} means 'no aliases', distinct from passing None."""
    adapter = PerturbationVectorAdapter(["TDGF1"], {}, 4)
    with pytest.raises(UnknownPerturbationGeneError, match="CRIPTO"):
        resolve_genes_against_vocabulary(["CRIPTO"], adapter, alias_map={})
