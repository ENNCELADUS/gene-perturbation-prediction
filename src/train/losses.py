"""
Loss functions for scGPT fine-tuning.

Implements two training objectives aligned with retrieval:
1. InfoNCE (supervised contrastive) - retrieval-native
2. Classification - multi-class condition prediction
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Supervised contrastive / InfoNCE loss for retrieval.

    Constructs positive/negative pairs based on condition labels:
    - Positives: embeddings from the same perturbation condition
    - Negatives: embeddings from different conditions in the batch

    This is the most "retrieval-faithful" approach as it directly
    optimizes for correct condition clustering.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature for similarity scaling
            normalize: Whether to L2-normalize embeddings
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            embeddings: Batch of embeddings [B, D]
            labels: Condition labels [B] (integer encoding)

        Returns:
            Scalar loss tensor
        """
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        batch_size = embeddings.size(0)
        device = embeddings.device

        # Compute pairwise cosine similarities
        similarity_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create positive mask: 1 where labels match (excluding diagonal)
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()
        positive_mask.fill_diagonal_(0)  # Exclude self-similarity

        # For numerical stability
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Mask out self-similarity in denominator
        logits_mask = torch.ones_like(similarity_matrix)
        logits_mask.fill_diagonal_(0)

        # Compute log-softmax over negatives + single positive
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs
        mask_pos_sum = positive_mask.sum(dim=1)
        mask_pos_sum = torch.clamp(mask_pos_sum, min=1)  # Avoid division by zero

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / mask_pos_sum

        # Loss is negative mean
        loss = -mean_log_prob_pos.mean()

        return loss


class ClassificationLoss(nn.Module):
    """
    Multi-class classification loss for condition prediction.

    Simple but effective: predicts condition ID from embedding.
    At inference, top-K classes become retrieval candidates.

    Note: This is a closed-set formulation - the classifier
    can only predict conditions seen during training.
    """

    def __init__(
        self,
        num_conditions: int,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        label_smoothing: float = 0.1,
    ):
        """
        Initialize classification loss with prediction head.

        Args:
            num_conditions: Number of condition classes
            embedding_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            label_smoothing: Label smoothing for cross-entropy
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_conditions),
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.num_conditions = num_conditions

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute classification loss.

        Args:
            embeddings: Batch of embeddings [B, D]
            labels: Condition labels [B] (integer encoding)

        Returns:
            Scalar loss tensor
        """
        logits = self.classifier(embeddings)
        loss = self.criterion(logits, labels)
        return loss

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.

        Args:
            embeddings: Embeddings [B, D]

        Returns:
            Predicted class indices [B]
        """
        with torch.no_grad():
            logits = self.classifier(embeddings)
            return logits.argmax(dim=1)

    def predict_top_k(
        self,
        embeddings: torch.Tensor,
        k: int = 5,
    ) -> torch.Tensor:
        """
        Get top-K class predictions.

        Args:
            embeddings: Embeddings [B, D]
            k: Number of top predictions

        Returns:
            Top-K class indices [B, K]
        """
        with torch.no_grad():
            logits = self.classifier(embeddings)
            _, top_k = logits.topk(k, dim=1)
            return top_k
