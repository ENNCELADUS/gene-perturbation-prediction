"""Tests for src/aivc_model/tx1_predicted_response.py and its sibling
src/aivc_model/tx1_predicted_response_cache.py -- the forward-only ST loader,
predicted-response generation, and the fingerprinted predicted-response cache
(Global Constraint D11).

No GPU and no real trained checkpoint exist on this machine, so every test
builds tiny synthetic fixtures in ``tmp_path``: a ``LinearMockStateModel``
standing in for STATE, a hand-built ``Esm2PerturbationAdapter`` (``p_g =
A_phi(E_ESM2(protein(g)))``, ``01-blueprint.md`` §3 -- the retired one-hot
``state_onehot`` tokenizer is not exercised here), a fake flat
``AivcModel``-shaped checkpoint state dict, and a fake Tx1 embedding cache
(written via ``tx1_embed_cache.write_line_cache``, never real Tx1
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

from aivc_model.gene_embeddings import Esm2EmbeddingTable
from aivc_model.residual_ladder import FixedSplit
from aivc_model.state_core import (
    Esm2PerturbationAdapter,
    LinearMockStateModel,
    StateForwardAdapter,
)
from aivc_model.tx1_embed_cache import EMBEDDING_WIDTH, write_line_cache
from aivc_model.tx1_predicted_response import (
    ARM_HVG,
    ARM_TX1,
    ForwardOnlyStateModel,
    UnknownPerturbationGeneError,
    _chunk_control_cell_indices,
    construct_forward_only_model,
    generate_pooled_predicted_response,
    generate_predicted_response,
    generate_predicted_response_for_line,
    load_forward_only_checkpoint,
    resolve_device,
    resolve_genes_against_vocabulary,
    vocabulary_genes,
)

_INPUT_DIM = 6
_OUTPUT_DIM = 4
_PERT_DIM = 3
_ESM_DIM = 5
_GENES = ["G1", "G2"]


def _esm2_table(genes: list[str] | None = None) -> Esm2EmbeddingTable:
    """A tiny ESM2 embedding table resolving every gene in ``genes``."""
    rng = np.random.default_rng(0)
    resolved = list(genes or _GENES)
    return Esm2EmbeddingTable(
        dim=_ESM_DIM,
        vectors_by_symbol={
            gene: rng.normal(size=_ESM_DIM).astype(np.float32) for gene in resolved
        },
    )


def _forward_only_model(
    genes: list[str] | None = None, input_dim: int = _INPUT_DIM
) -> ForwardOnlyStateModel:
    """A small, freshly initialized ST + perturbation-adapter model for tests."""
    state_model = LinearMockStateModel(input_dim, _OUTPUT_DIM, _PERT_DIM)
    resolved_genes = list(genes or _GENES)
    perturbations = Esm2PerturbationAdapter(
        resolved_genes,
        _esm2_table(resolved_genes),
        adapter_hidden=4,
        pert_dim=_PERT_DIM,
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
    del checkpoint["perturbations.esm_matrix"]
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


def _write_esm2_npz(path: Path, genes: list[str]) -> Path:
    """Write a small resolved-for-every-gene ESM2 ``.npz`` fixture."""
    rng = np.random.default_rng(1)
    symbols = np.array(genes, dtype=object)
    vectors = rng.normal(size=(len(genes), _ESM_DIM)).astype(np.float32)
    resolved = np.ones(len(genes), dtype=bool)
    np.savez(path, symbols=symbols, vectors=vectors, resolved=resolved)
    return path


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
    esm2_path = _write_esm2_npz(tmp_path / "esm2.npz", _GENES)

    model = construct_forward_only_model(
        model_cls=_FakeStateModel,
        hparams_checkpoint_path=checkpoint_path,
        input_dim=_INPUT_DIM,
        output_dim=_OUTPUT_DIM,
        pert_dim=_PERT_DIM,
        genes=_GENES,
        esm2_embeddings_path=esm2_path,
        esm2_adapter_hidden=4,
    )

    assert isinstance(model, ForwardOnlyStateModel)
    assert model.state_adapter.state_model.basal_encoder.in_features == _INPUT_DIM
    assert isinstance(model.perturbations, Esm2PerturbationAdapter)
    assert model.perturbations.genes == _GENES


def test_construct_forward_only_model_raises_loudly_on_unresolved_gene(
    tmp_path: Path,
) -> None:
    """Gene-vocabulary resolution must fail loudly, never zero-fill or skip."""
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
    # Only resolves "G1" -- "G2" is requested but missing from the table.
    esm2_path = _write_esm2_npz(tmp_path / "esm2.npz", ["G1"])

    with pytest.raises(ValueError, match="G2"):
        construct_forward_only_model(
            model_cls=_FakeStateModel,
            hparams_checkpoint_path=checkpoint_path,
            input_dim=_INPUT_DIM,
            output_dim=_OUTPUT_DIM,
            pert_dim=_PERT_DIM,
            genes=_GENES,
            esm2_embeddings_path=esm2_path,
            esm2_adapter_hidden=4,
        )


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


# --- P1-5: device resolution ---------------------------------------------


def test_resolve_device_explicit_string_is_used_verbatim() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_passthrough_for_existing_torch_device() -> None:
    device = torch.device("cpu")
    assert resolve_device(device) is device


def test_resolve_device_auto_falls_back_to_cpu_when_no_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_auto_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a real-GPU test (none exists on this machine) -- this only pins
    the resolution RULE (P1-5: `warm_predicted_response_cache` and
    `emit_test_predictions` must not stay hardcoded to CPU when a GPU is
    actually available)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


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
    assert response.device == torch.device("cpu")


def test_generate_predicted_response_macro_batches_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _forward_only_model()
    observed_chunk_counts: list[int] = []
    original = model.predict_response_chunks

    def _record(
        control_chunks: tuple[torch.Tensor, ...],
        gene: str,
        batch_chunks: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor, ...]:
        observed_chunk_counts.append(len(control_chunks))
        return original(control_chunks, gene, batch_chunks)

    monkeypatch.setattr(model, "predict_response_chunks", _record)
    response = generate_predicted_response(
        model,
        np.zeros((13, _INPUT_DIM), dtype=np.float32),
        "G1",
        cell_set_len=4,
        window_macro_batch_size=3,
        seed=0,
    )

    assert response.shape == (13, _OUTPUT_DIM)
    assert observed_chunk_counts == [3, 1]


def test_pooled_predicted_response_matches_unbatched_reference() -> None:
    model = _forward_only_model()
    control_input = (
        np.random.default_rng(8).normal(size=(13, _INPUT_DIM)).astype(np.float32)
    )
    reference = generate_predicted_response(
        model,
        control_input,
        "G1",
        cell_set_len=4,
        window_macro_batch_size=1,
        seed=3,
    )

    pooled, response_dim = generate_pooled_predicted_response(
        model,
        control_input,
        "G1",
        cell_set_len=4,
        window_macro_batch_size=3,
        seed=3,
    )

    expected = torch.cat([reference.mean(dim=0), reference.var(dim=0, unbiased=False)])
    assert response_dim == _OUTPUT_DIM
    assert torch.allclose(pooled, expected, atol=1e-6, rtol=1e-5)


def test_pooled_predicted_response_reuses_resident_control_tensor() -> None:
    model = _forward_only_model()
    control_input = torch.from_numpy(
        np.random.default_rng(18).normal(size=(13, _INPUT_DIM)).astype(np.float32)
    )

    from_tensor, tensor_dim = generate_pooled_predicted_response(
        model,
        control_input,
        "G1",
        cell_set_len=4,
        window_macro_batch_size=3,
        seed=3,
        device=control_input.device,
    )
    from_numpy, numpy_dim = generate_pooled_predicted_response(
        model,
        control_input.numpy(),
        "G1",
        cell_set_len=4,
        window_macro_batch_size=3,
        seed=3,
        device=control_input.device,
    )

    assert tensor_dim == numpy_dim == _OUTPUT_DIM
    assert from_tensor.device == control_input.device
    assert torch.allclose(from_tensor, from_numpy, atol=1e-6, rtol=1e-5)


def test_pooled_predicted_response_rejects_invalid_macro_batch_size() -> None:
    with pytest.raises(ValueError, match="window_macro_batch_size"):
        generate_pooled_predicted_response(
            _forward_only_model(),
            np.zeros((5, _INPUT_DIM), dtype=np.float32),
            "G1",
            cell_set_len=4,
            window_macro_batch_size=0,
            seed=0,
        )


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


# --- per-line generation and the fit-eligibility guard ------------------------
#
# ``cell_line_geneeffect_226_split`` is the membership authority (replaces the
# retired Phase-A train_head/test role column): ACH-TRAIN is a labeled train
# member, ACH-TEST is a test (inference-only) member.

_LINE_SPLIT = FixedSplit(train=("ACH-TRAIN",), val=(), test=("ACH-TEST",))


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


def test_generate_predicted_response_for_line_rejects_non_train_line_by_default(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "tx1_cache"
    _write_fake_line_cache(cache_dir, "ACH-TEST", n_cells=5)
    model = _forward_only_model()

    with pytest.raises(ValueError, match="not a labeled train member"):
        generate_predicted_response_for_line(
            cache_dir,
            "ACH-TEST",
            model,
            "G1",
            arm=ARM_TX1,
            cell_set_len=4,
            seed=0,
            split=_LINE_SPLIT,
        )


def test_generate_predicted_response_for_line_requires_split_when_checked(
    tmp_path: Path,
) -> None:
    """The leakage case: omitting `split` must not silently skip the guard."""
    cache_dir = tmp_path / "tx1_cache"
    _write_fake_line_cache(cache_dir, "ACH-TRAIN", n_cells=5)
    model = _forward_only_model()

    with pytest.raises(ValueError, match="require_fit_eligible"):
        generate_predicted_response_for_line(
            cache_dir,
            "ACH-TRAIN",
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
        model,
        "G1",
        arm=ARM_TX1,
        cell_set_len=4,
        seed=0,
        require_fit_eligible=False,
    )
    assert response.shape == (5, _OUTPUT_DIM)


def test_generate_predicted_response_for_line_admits_train_line(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "tx1_cache"
    _write_fake_line_cache(cache_dir, "ACH-TRAIN", n_cells=5)
    model = _forward_only_model(input_dim=EMBEDDING_WIDTH)

    response = generate_predicted_response_for_line(
        cache_dir,
        "ACH-TRAIN",
        model,
        "G1",
        arm=ARM_TX1,
        cell_set_len=4,
        seed=0,
        split=_LINE_SPLIT,
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
    hvg_model.state_adapter = StateForwardAdapter(
        LinearMockStateModel(3, _OUTPUT_DIM, _PERT_DIM)
    )

    response = generate_predicted_response_for_line(
        cache_dir,
        "ACH-TRAIN",
        hvg_model,
        "G1",
        arm=ARM_HVG,
        cell_set_len=4,
        seed=0,
        split=_LINE_SPLIT,
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
            model,
            "G1",
            arm="bogus_arm",
            cell_set_len=4,
            seed=0,
            split=_LINE_SPLIT,
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
        "vocabulary_genes": ["G1", "G2", "G3"],
        "seed": 0,
        "arm": ARM_TX1,
        "cell_set_len": 4,
    }




























def test_slice_symbol_aliases_resolve_by_default() -> None:
    """CRIPTO/HEMK2 must resolve without the caller knowing to pass a map.

    Regression guard for the gap that would drop slice coverage from 587/587
    to 585/587 and fail Phase F's coverage validator *after* a training run.
    The pairs are measured (task-0-coverage.md §2), not assumed.
    """
    genes = ["TDGF1", "N6AMT1", "ACLY"]
    adapter = Esm2PerturbationAdapter(
        genes, _esm2_table(genes), adapter_hidden=4, pert_dim=4
    )
    resolved = resolve_genes_against_vocabulary(["CRIPTO", "HEMK2", "ACLY"], adapter)
    assert resolved == ("TDGF1", "N6AMT1", "ACLY")


def test_explicit_empty_alias_map_disables_aliasing() -> None:
    """Passing {} means 'no aliases', distinct from passing None."""
    adapter = Esm2PerturbationAdapter(
        ["TDGF1"], _esm2_table(["TDGF1"]), adapter_hidden=4, pert_dim=4
    )
    with pytest.raises(UnknownPerturbationGeneError, match="CRIPTO"):
        resolve_genes_against_vocabulary(["CRIPTO"], adapter, alias_map={})
