"""Tests for ``src.model.state``, the lifted-symbol single source of truth.

Covers construction/behavior of each moved symbol (``LinearMockStateModel``,
``StateForwardAdapter``, ``Esm2PerturbationAdapter``, ``GeneBags``,
``encode_batch_labels``, ``resolve_state_gene_order``), which this module is
now the only definition of: the exp05 ``model.py``/``prepare.py`` they were
lifted out of were deleted at ``873c99c``.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

import src.data.gene_splits as gene_splits_module
from src.data.embeddings import Esm2EmbeddingTable
from src.model.perturbation import Esm2PerturbationAdapter
from src.data.gene_bags import GeneBags
from src.model.state import LinearMockStateModel, StateForwardAdapter
from src.data.gene_order import encode_batch_labels, resolve_state_gene_order


# --- LinearMockStateModel ----------------------------------------------------


def test_linear_mock_state_model_forward_matches_batch_shapes() -> None:
    model = LinearMockStateModel(input_dim=3, output_dim=2, pert_dim=4)
    batch = {
        "ctrl_cell_emb": torch.ones(5, 3),
        "pert_emb": torch.zeros(5, 4),
    }

    output = model(batch)

    assert output.shape == (5, 2)
    assert torch.isfinite(output).all()


# --- StateForwardAdapter -----------------------------------------------------


def test_state_forward_adapter_forward_chunks_uses_predict_step_batch_schema() -> None:
    class PredictStepState(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.seen: dict[str, object] = {}

        def predict_step(
            self,
            batch: dict[str, object],
            batch_idx: int,
            padded: bool,
        ) -> dict[str, torch.Tensor]:
            self.seen = {"batch": batch, "batch_idx": batch_idx, "padded": padded}
            ctrl = batch["ctrl_cell_emb"]
            assert isinstance(ctrl, torch.Tensor)
            return {"preds": ctrl @ self.weight}

    state = PredictStepState()
    adapter = StateForwardAdapter(state)
    control = torch.eye(2, requires_grad=True)
    batch_indices = torch.tensor([1, 2])

    (output,) = adapter.forward_chunks(
        (control,),
        torch.tensor([1.0, 0.0]),
        "GENE1",
        (batch_indices,),
    )
    output.sum().backward()

    seen_batch = state.seen["batch"]
    assert isinstance(seen_batch, dict)
    assert state.seen["padded"] is True
    assert {"ctrl_cell_emb", "pert_emb", "pert_name", "batch"}.issubset(seen_batch)
    assert state.weight.grad is not None
    assert torch.isfinite(output).all()


def test_state_forward_adapter_batches_distinct_gene_conditions() -> None:
    class RecordingState(torch.nn.Module):
        cell_sentence_len = 2

        def __init__(self) -> None:
            super().__init__()
            self.seen: dict[str, object] = {}

        def predict_step(self, batch, batch_idx, padded):
            self.seen = {"batch": batch, "batch_idx": batch_idx, "padded": padded}
            return {"preds": batch["ctrl_cell_emb"] + batch["pert_emb"]}

    state = RecordingState()
    adapter = StateForwardAdapter(state)
    outputs = adapter.forward_condition_chunks(
        (torch.zeros(2, 2), torch.ones(2, 2)),
        (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])),
        ("G1", "G2"),
        (None, None),
    )

    assert len(outputs) == 2
    assert torch.equal(outputs[0], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert torch.equal(outputs[1], torch.tensor([[1.0, 3.0], [1.0, 3.0]]))
    assert state.seen["batch"]["pert_name"] == ["G1", "G1", "G2", "G2"]
    assert state.seen["padded"] is True


def test_state_forward_adapter_forward_runs_unpadded_single_condition() -> None:
    state = LinearMockStateModel(input_dim=2, output_dim=3, pert_dim=2)
    adapter = StateForwardAdapter(state)

    output = adapter.forward(
        torch.ones(4, 2),
        torch.zeros(2),
        "GENE1",
    )

    assert output.shape == (4, 3)


# --- Esm2PerturbationAdapter --------------------------------------------------


def test_esm2_perturbation_adapter_maps_all_genes_through_one_network() -> None:
    table = Esm2EmbeddingTable(
        dim=3,
        vectors_by_symbol={
            "KNOWN": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "HELDOUT": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        },
    )
    adapter = Esm2PerturbationAdapter(
        ["KNOWN", "HELDOUT"], table, adapter_hidden=4, pert_dim=2
    )

    assert adapter("KNOWN").shape == (2,)
    assert adapter("HELDOUT").shape == (2,)
    assert adapter.forward_many(["KNOWN", "HELDOUT"]).shape == (2, 2)
    assert adapter.has_embedding("known")
    assert adapter.has_known_vector("HELDOUT")
    assert "esm_matrix" not in adapter.state_dict()
    assert adapter.state_dict()["gene_vocabulary_sha256"].shape == (32,)


def test_esm2_perturbation_adapter_raises_on_unresolved_gene_no_zero_fill() -> None:
    """An unresolved gene must fail construction, never silently zero-fill."""
    table = Esm2EmbeddingTable(
        dim=3, vectors_by_symbol={"KNOWN": np.zeros(3, dtype=np.float32)}
    )

    with pytest.raises(ValueError, match="UNRESOLVED"):
        Esm2PerturbationAdapter(
            ["KNOWN", "UNRESOLVED"], table, adapter_hidden=4, pert_dim=2
        )


# --- GeneBags ----------------------------------------------------------------


def _toy_gene_bags() -> GeneBags:
    genes = ("GENE1", "GENE2", "GENE3")
    input_bags = tuple(
        np.full((2, 3), float(index + 1), dtype=np.float32)
        for index in range(len(genes))
    )
    metadata = pd.DataFrame({"perturbation_gene": genes})
    fold = gene_splits_module.FoldSpec(0, ("GENE1", "GENE3"), (), ("GENE2",))
    return GeneBags(
        genes=np.asarray(genes, dtype=object),
        y=np.asarray([-1.0, -0.5, 0.2], dtype=np.float32),
        input_bags=input_bags,
        latent_bags=tuple(bag.copy() for bag in input_bags),
        control_input=np.zeros((2, 3), dtype=np.float32),
        control_latent=np.zeros((2, 3), dtype=np.float32),
        cell_type_bags=None,
        control_cell_type=None,
        batch_bags=None,
        control_batch=None,
        feature_names=np.asarray(["I0", "I1", "I2"], dtype=object),
        metadata=metadata,
        input_dim=3,
        latent_dim=3,
        access_recorder=gene_splits_module.GeneAccessRecorder(fold),
    )


def test_gene_bags_for_genes_returns_ordered_gene_only_view() -> None:
    bags = _toy_gene_bags()

    view = bags.for_genes(("GENE3", "GENE1"), stage="adapter_fit")

    assert view.genes.tolist() == ["GENE3", "GENE1"]
    np.testing.assert_allclose(view.y, np.asarray([0.2, -1.0], dtype=np.float32))


def test_gene_bags_for_genes_rejects_unknown_gene() -> None:
    bags = _toy_gene_bags()

    with pytest.raises(ValueError, match="unknown genes"):
        bags.for_genes(("NOPE",), stage="adapter_fit")


def test_gene_bags_for_prediction_genes_empties_input_bags_by_default() -> None:
    bags = _toy_gene_bags()

    view = bags.for_prediction_genes(
        ("GENE2",),
        stage="generation_loss_outer_test",
        checkpoint_frozen=True,
    )

    assert view.genes.tolist() == ["GENE2"]
    assert view.input_bags[0].shape == (0, bags.input_dim)


# --- encode_batch_labels ------------------------------------------------------


def test_encode_batch_labels_uses_lookup_and_fallback() -> None:
    lookup = {"31": 1, "25": 0}

    encoded = encode_batch_labels(np.asarray(["31", "missing", "25"]), lookup)

    np.testing.assert_array_equal(encoded, np.asarray([1, 0, 0]))


def test_encode_batch_labels_passes_through_none() -> None:
    assert encode_batch_labels(None, {}) is None


# --- resolve_state_gene_order -------------------------------------------------


def test_resolve_state_gene_order_uses_gene_name_in_checkpoint_order(
    tmp_path: Path,
) -> None:
    adata = ad.AnnData(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    adata.var_names = ["ENSG1", "ENSG2", "ENSG3"]
    adata.var["gene_name"] = ["B", "A", "C"]
    model_dir = tmp_path / "state"
    model_dir.mkdir()
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": ["A", "B"]}, handle)

    indices, names = resolve_state_gene_order(adata, model_dir, "gene_name")

    np.testing.assert_array_equal(indices, np.asarray([1, 0]))
    np.testing.assert_array_equal(names, np.asarray(["A", "B"], dtype=object))


def test_resolve_state_gene_order_never_falls_back_when_gene_is_missing(
    tmp_path: Path,
) -> None:
    adata = ad.AnnData(np.ones((1, 1), dtype=np.float32))
    adata.var["gene_name"] = ["A"]
    model_dir = tmp_path / "state"
    model_dir.mkdir()
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": ["A", "B"]}, handle)

    with pytest.raises(ValueError, match="1/2"):
        resolve_state_gene_order(adata, model_dir, "gene_name")


def test_predict_bags_training_advances_rng_and_eval_is_isolated():
    from src.model.response import predict_bags

    class StochasticBackbone(torch.nn.Module):
        def forward(self, controls, genes, batch_indices):
            return tuple(control + torch.rand_like(control) for control in controls)

    model = StochasticBackbone()
    control = torch.zeros(5, 8)
    torch.manual_seed(0)
    before = torch.get_rng_state().clone()
    first = predict_bags(model, [control], ["A"], seed=0)[0]
    after = torch.get_rng_state().clone()
    second = predict_bags(model, [control], ["A"], seed=0)[0]
    assert not torch.equal(before, after)
    assert not torch.equal(first, second)
    model.eval()
    before = torch.get_rng_state().clone()
    first = predict_bags(model, [control], ["A"], seed=0)[0]
    assert torch.equal(torch.get_rng_state(), before)
    torch.rand(19)
    second = predict_bags(model, [control], ["A"], seed=0)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
