"""STATE construction vocabulary, direct forwarding and deterministic padding."""

import numpy as np
import pytest
import torch
from src.data.embeddings import Esm2EmbeddingTable
from src.model.perturbation import Esm2PerturbationAdapter
from src.model.state import (
    LinearMockStateModel,
    StateForwardAdapter,
    ForwardOnlyStateModel,
)
from src.model.response import _chunk_control_cell_indices
from src.model.initialization import build_joint_model, warm_start_state_dict
from test_joint_objective import TinyState, joint_setup as joint_setup

_INPUT_DIM = 6
_OUTPUT_DIM = 4
_PERT_DIM = 3
_ESM_DIM = 5
_GENES = ["G1", "G2"]


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_joint_restore_rejects_incompatible_learned_keys(joint_setup, corruption):
    model, inputs, config, _, _ = joint_setup
    state = dict(model.state_dict())
    key = next(iter(state))
    if corruption == "missing":
        state.pop(key)
    elif corruption == "unexpected":
        state["unexpected.weight"] = torch.zeros(1)
    else:
        state[key] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        build_joint_model(
            config,
            inputs,
            architecture=model.architecture,
            model_state=state,
            projection_state=model.projection.to_state(),
            normalization_state=model.standardizer.to_state(),
            model_cls=TinyState,
        )


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_released_initialization_rejects_unexpected_incompatibility(
    joint_setup, corruption
):
    _, inputs, config, _, _ = joint_setup
    path = config["paths"]["state_checkpoint"]
    checkpoint = torch.load(path, weights_only=False)
    state = checkpoint["state_dict"]
    key = next(iter(state))
    if corruption == "missing":
        state.pop(key)
    elif corruption == "unexpected":
        state["unexpected.weight"] = torch.zeros(1)
    else:
        state[key] = torch.zeros(1)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="unexpected released STATE incompatibility"):
        build_joint_model(config, inputs, model_cls=TinyState)


def test_warm_start_reports_every_key_and_preserves_unloaded_values(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    original = {name: value.clone() for name, value in model.state_dict().items()}
    path = tmp_path / "source.ckpt"
    torch.save(
        {
            "state_dict": {
                "0.weight": torch.full_like(original["0.weight"], 7.0),
                "0.bias": torch.zeros(4),
                "extra": torch.zeros(1),
            }
        },
        path,
    )
    report = warm_start_state_dict(model, path)
    assert report.loaded_keys == ("0.weight",)
    assert report.shape_skipped_keys == ("0.bias",)
    assert report.missing_keys == ("1.bias", "1.weight")
    assert report.unexpected_keys == ("extra",)
    for name in (*report.shape_skipped_keys, *report.missing_keys):
        torch.testing.assert_close(model.state_dict()[name], original[name])
    assert torch.all(model[0].weight == 7)


def test_warm_start_refuses_zero_loaded_keys(tmp_path):
    path = tmp_path / "source.ckpt"
    torch.save({"state_dict": {"unrelated": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="loaded zero keys"):
        warm_start_state_dict(torch.nn.Linear(2, 3), path)


def test_embedding_lookup_is_by_symbol_and_rejects_unresolved_gene():
    table = _esm2_table()
    adapter = Esm2PerturbationAdapter(["G2", "G1"], table, 4, _PERT_DIM)
    torch.testing.assert_close(
        adapter.esm_matrix[1], torch.from_numpy(table.vectors_by_symbol["G1"])
    )
    torch.testing.assert_close(adapter("g1"), adapter("G1"))
    with pytest.raises(ValueError, match="Unresolved ESM-2"):
        Esm2PerturbationAdapter(["UNKNOWN"], table, 4, _PERT_DIM)


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


def test_forward_only_model_batches_distinct_gene_conditions() -> None:
    model = _forward_only_model()
    outputs = model(
        (torch.zeros(2, _INPUT_DIM), torch.ones(2, _INPUT_DIM)),
        ("G1", "G2"),
        (None, None),
    )

    assert len(outputs) == 2
    assert all(output.shape == (2, _OUTPUT_DIM) for output in outputs)


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
