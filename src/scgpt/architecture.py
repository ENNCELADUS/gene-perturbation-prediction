"""Architecture blocks for scGPT gene target retrievers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cardinality_labels_from_targets(
    targets: torch.Tensor,
    max_cardinality: int,
) -> torch.Tensor:
    """Return clamped target-count labels for a multi-hot target matrix."""
    counts = targets.to(dtype=torch.float32).sum(dim=1).long()
    return counts.clamp(max=max_cardinality)


def compute_cardinality_loss(
    cardinality_logits: torch.Tensor,
    targets: torch.Tensor,
    max_cardinality: int,
) -> torch.Tensor:
    """Compute cross-entropy loss for target-set cardinality prediction."""
    labels = cardinality_labels_from_targets(
        targets.to(device=cardinality_logits.device),
        max_cardinality=max_cardinality,
    )
    return F.cross_entropy(cardinality_logits, labels)


class SparseGraphMessagePassing(nn.Module):
    """Lightweight sparse message-passing layer without torch_geometric."""

    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(
        self,
        gene_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate source-node messages into target-node representations."""
        if edge_index.numel() == 0:
            return gene_features
        source = edge_index[0].to(gene_features.device)
        target = edge_index[1].to(gene_features.device)
        weights = edge_weight.to(gene_features.device, dtype=gene_features.dtype)
        messages = self.message(gene_features[source]) * weights.unsqueeze(1)
        aggregated = torch.zeros_like(gene_features)
        aggregated.index_add_(0, target, messages)
        degree = torch.zeros(
            gene_features.size(0),
            dtype=gene_features.dtype,
            device=gene_features.device,
        )
        degree.index_add_(0, target, weights.abs())
        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(1)
        hidden = self.norm(gene_features + self.dropout(aggregated))
        return self.norm(hidden + self.dropout(self.ffn(hidden)))


class SlotSetDecoder(nn.Module):
    """Decode one phenotype representation into multiple target-query slots."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        n_slots: int,
        dropout: float,
        aggregation: str = "logsumexp",
    ) -> None:
        super().__init__()
        if n_slots < 1:
            raise ValueError("n_slots must be at least 1")
        self.n_slots = n_slots
        self.aggregation = aggregation
        self.slot_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_slots * embedding_dim),
        )
        self.slot_embeddings = nn.Parameter(torch.empty(n_slots, embedding_dim))
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)

    def forward(
        self,
        phenotype_embedding: torch.Tensor,
        gene_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return aggregated gene logits from slot-wise scores."""
        slots = self.slot_projection(phenotype_embedding).view(
            phenotype_embedding.size(0),
            self.n_slots,
            gene_embeddings.size(1),
        )
        slots = slots + self.slot_embeddings.unsqueeze(0)
        slot_scores = slots @ gene_embeddings.transpose(0, 1)
        slot_scores = slot_scores / math.sqrt(gene_embeddings.size(1))
        return self.aggregate_slot_scores(slot_scores, self.aggregation)

    @staticmethod
    def aggregate_slot_scores(
        slot_scores: torch.Tensor,
        aggregation: str,
    ) -> torch.Tensor:
        """Aggregate slot-wise gene scores into one score vector per sample."""
        if aggregation == "logsumexp":
            return torch.logsumexp(slot_scores, dim=1)
        if aggregation == "max":
            return slot_scores.max(dim=1).values
        if aggregation == "mean":
            return slot_scores.mean(dim=1)
        raise ValueError("slot aggregation must be 'logsumexp', 'max', or 'mean'")


class LatentResponseCycleHead(nn.Module):
    """Predict perturbed latent state from control state and soft target set."""

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        control_embedding: torch.Tensor,
        target_set_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted perturbed latent embedding."""
        delta = self.decoder(
            torch.cat([control_embedding, target_set_embedding], dim=1)
        )
        return control_embedding + delta

    def loss(
        self,
        control_embedding: torch.Tensor,
        target_set_embedding: torch.Tensor,
        perturbed_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return latent-space reconstruction loss."""
        prediction = self.forward(control_embedding, target_set_embedding)
        return F.mse_loss(prediction, perturbed_embedding.detach())
