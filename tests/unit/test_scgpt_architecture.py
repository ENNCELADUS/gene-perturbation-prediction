from __future__ import annotations

import torch

from src.scgpt.architecture import (
    LatentResponseCycleHead,
    SlotSetDecoder,
    SparseGraphMessagePassing,
    cardinality_labels_from_targets,
    compute_cardinality_loss,
)


def test_sparse_graph_message_passing_preserves_shape_and_gradients() -> None:
    layer = SparseGraphMessagePassing(embedding_dim=4, dropout=0.0)
    gene_features = torch.randn(3, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)

    updated = layer(gene_features, edge_index=edge_index, edge_weight=edge_weight)

    assert updated.shape == gene_features.shape
    updated.sum().backward()
    assert gene_features.grad is not None
    assert torch.isfinite(gene_features.grad).all()


def test_slot_set_decoder_uses_logsumexp_aggregation() -> None:
    slot_scores = torch.tensor(
        [
            [[0.0, 1.0, -1.0], [2.0, 3.0, 0.0]],
            [[1.0, -2.0, 0.5], [0.0, 2.0, 1.5]],
        ],
        dtype=torch.float32,
    )

    aggregated = SlotSetDecoder.aggregate_slot_scores(slot_scores, "logsumexp")

    assert torch.allclose(aggregated, torch.logsumexp(slot_scores, dim=1))


def test_cardinality_labels_and_loss_clamp_large_target_sets() -> None:
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    logits = torch.tensor(
        [
            [0.0, 3.0, -1.0],
            [0.0, -1.0, 3.0],
        ],
        requires_grad=True,
    )

    labels = cardinality_labels_from_targets(targets, max_cardinality=2)
    loss = compute_cardinality_loss(logits, targets, max_cardinality=2)

    assert labels.tolist() == [1, 2]
    assert loss.item() < 0.2
    loss.backward()
    assert logits.grad is not None


def test_latent_response_cycle_head_reconstructs_perturbed_embedding() -> None:
    head = LatentResponseCycleHead(embedding_dim=4, hidden_dim=8, dropout=0.0)
    control_embedding = torch.randn(2, 4)
    target_set_embedding = torch.randn(2, 4)
    perturbed_embedding = torch.randn(2, 4)

    loss = head.loss(
        control_embedding=control_embedding,
        target_set_embedding=target_set_embedding,
        perturbed_embedding=perturbed_embedding,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
