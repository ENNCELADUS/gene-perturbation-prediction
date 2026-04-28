"""Loss functions for scGPT gene target ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneScoreLoss(Protocol):
    """Callable loss interface for gene score training."""

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute a scalar loss from logits and multi-hot targets."""


@dataclass(frozen=True)
class GeneScoreLossConfig:
    """Configuration for gene score loss construction."""

    name: str = "bce"
    hard_negatives: int = 64
    random_negatives: int = 128
    temperature: float = 1.0
    margin: float = 0.0
    hard_negative_source: str = "online"
    asymmetric_gamma_positive: float = 0.0
    asymmetric_gamma_negative: float = 4.0
    asymmetric_clip: float = 0.05
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | None) -> "GeneScoreLossConfig":
        """Build a loss config from a plain config mapping."""
        if config is None:
            return cls()
        return cls(
            name=str(config.get("name", "bce")),
            hard_negatives=int(config.get("hard_negatives", 64)),
            random_negatives=int(config.get("random_negatives", 128)),
            temperature=float(config.get("temperature", 1.0)),
            margin=float(config.get("margin", 0.0)),
            hard_negative_source=str(config.get("hard_negative_source", "online")),
            asymmetric_gamma_positive=float(
                config.get("asymmetric_gamma_positive", 0.0)
            ),
            asymmetric_gamma_negative=float(
                config.get("asymmetric_gamma_negative", 4.0)
            ),
            asymmetric_clip=float(config.get("asymmetric_clip", 0.05)),
            eps=float(config.get("eps", 1.0e-8)),
        )


class BinaryCrossEntropyGeneScoreLoss(nn.Module):
    """Full multi-label BCE over all genes."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE with logits."""
        return F.binary_cross_entropy_with_logits(logits, targets)


class SampledGeneScoreLoss(nn.Module):
    """Base class for sampled losses over positive and negative gene labels."""

    def __init__(
        self,
        hard_negatives: int,
        random_negatives: int,
        temperature: float,
        margin: float,
        hard_negative_source: str,
    ) -> None:
        super().__init__()
        if hard_negatives < 0 or random_negatives < 0:
            raise ValueError("negative sample counts must be non-negative")
        if hard_negatives + random_negatives <= 0:
            raise ValueError("at least one negative sample is required")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if hard_negative_source not in {"online", "semi_hard", "random"}:
            raise ValueError(
                "hard_negative_source must be 'online', 'semi_hard', or 'random'"
            )
        self.hard_negatives = hard_negatives
        self.random_negatives = random_negatives
        self.temperature = temperature
        self.margin = margin
        self.hard_negative_source = hard_negative_source

    def _validate_inputs(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.shape != targets.shape:
            raise ValueError(
                "logits and targets must have the same shape: "
                f"{tuple(logits.shape)} != {tuple(targets.shape)}"
            )

    def _sample_negative_indices(
        self,
        row_logits: torch.Tensor,
        row_targets: torch.Tensor,
        positive_indices: torch.Tensor,
    ) -> torch.Tensor:
        negative_indices = torch.nonzero(~row_targets, as_tuple=False).flatten()
        if negative_indices.numel() == 0:
            return negative_indices
        hard_indices = self._hard_negative_indices(
            row_logits,
            negative_indices,
            positive_indices,
        )
        random_indices = self._random_negative_indices(negative_indices, hard_indices)
        if hard_indices.numel() == 0:
            return random_indices
        if random_indices.numel() == 0:
            return hard_indices
        return torch.cat([hard_indices, random_indices])

    def _hard_negative_indices(
        self,
        row_logits: torch.Tensor,
        negative_indices: torch.Tensor,
        positive_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.hard_negatives == 0:
            return negative_indices[:0]
        if self.hard_negative_source == "online":
            return self._top_false_positive_indices(row_logits, negative_indices)
        if self.hard_negative_source == "semi_hard":
            return self._semi_hard_negative_indices(
                row_logits,
                negative_indices,
                positive_indices,
            )
        return self._random_indices(negative_indices, self.hard_negatives)

    def _top_false_positive_indices(
        self,
        row_logits: torch.Tensor,
        negative_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.hard_negatives == 0:
            return negative_indices[:0]
        sample_count = min(self.hard_negatives, negative_indices.numel())
        negative_scores = row_logits[negative_indices]
        top_positions = torch.topk(negative_scores, k=sample_count).indices
        return negative_indices[top_positions]

    def _semi_hard_negative_indices(
        self,
        row_logits: torch.Tensor,
        negative_indices: torch.Tensor,
        positive_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.hard_negatives == 0:
            return negative_indices[:0]
        sample_count = min(self.hard_negatives, negative_indices.numel())
        positive_center = row_logits[positive_indices].mean()
        distances = torch.abs(row_logits[negative_indices] - positive_center)
        closest_positions = torch.topk(-distances, k=sample_count).indices
        return negative_indices[closest_positions]

    def _random_negative_indices(
        self,
        negative_indices: torch.Tensor,
        hard_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.random_negatives == 0:
            return negative_indices[:0]
        random_pool = negative_indices
        if hard_indices.numel() > 0:
            hard_mask = torch.isin(negative_indices, hard_indices)
            random_pool = negative_indices[~hard_mask]
        if random_pool.numel() == 0:
            return random_pool
        return self._random_indices(random_pool, self.random_negatives)

    def _random_indices(self, indices: torch.Tensor, sample_count: int) -> torch.Tensor:
        if sample_count == 0:
            return indices[:0]
        sample_count = min(sample_count, indices.numel())
        permutation = torch.randperm(indices.numel(), device=indices.device)
        return indices[permutation[:sample_count]]


class SampledPairwiseRankingLoss(SampledGeneScoreLoss):
    """Pairwise ranking loss over positives and sampled false-positive negatives."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute pairwise ranking loss for each positive-negative pair."""
        self._validate_inputs(logits, targets)
        targets = targets.to(device=logits.device, dtype=torch.bool)
        pair_losses = []
        for row_logits, row_targets in zip(logits, targets):
            positive_indices = torch.nonzero(row_targets, as_tuple=False).flatten()
            if positive_indices.numel() == 0:
                continue
            negative_indices = self._sample_negative_indices(
                row_logits,
                row_targets,
                positive_indices,
            )
            if negative_indices.numel() == 0:
                continue
            positive_scores = row_logits[positive_indices]
            negative_scores = row_logits[negative_indices]
            differences = (
                negative_scores.unsqueeze(0)
                - positive_scores.unsqueeze(1)
                + self.margin
            ) / self.temperature
            pair_losses.append(F.softplus(differences).flatten())
        if not pair_losses:
            return logits.sum() * 0.0
        return torch.cat(pair_losses).mean()


class SampledSoftmaxGeneScoreLoss(SampledGeneScoreLoss):
    """Sampled multi-positive softmax over positives and mined negatives."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute sampled softmax loss per row."""
        self._validate_inputs(logits, targets)
        targets = targets.to(device=logits.device, dtype=torch.bool)
        row_losses = []
        for row_logits, row_targets in zip(logits, targets):
            positive_indices = torch.nonzero(row_targets, as_tuple=False).flatten()
            if positive_indices.numel() == 0:
                continue
            negative_indices = self._sample_negative_indices(
                row_logits,
                row_targets,
                positive_indices,
            )
            candidates = torch.cat([positive_indices, negative_indices])
            candidate_scores = row_logits[candidates] / self.temperature
            positive_scores = row_logits[positive_indices] / self.temperature
            row_losses.append(
                torch.logsumexp(candidate_scores, dim=0)
                - torch.logsumexp(positive_scores, dim=0)
            )
        if not row_losses:
            return logits.sum() * 0.0
        return torch.stack(row_losses).mean()


class SampledLambdaRankGeneScoreLoss(SampledGeneScoreLoss):
    """Sampled pairwise LambdaRank-style loss weighted by DCG rank change."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute sampled LambdaRank loss."""
        self._validate_inputs(logits, targets)
        targets = targets.to(device=logits.device, dtype=torch.bool)
        pair_losses = []
        for row_logits, row_targets in zip(logits, targets):
            positive_indices = torch.nonzero(row_targets, as_tuple=False).flatten()
            if positive_indices.numel() == 0:
                continue
            negative_indices = self._sample_negative_indices(
                row_logits,
                row_targets,
                positive_indices,
            )
            if negative_indices.numel() == 0:
                continue
            weights = self._delta_dcg_weights(
                row_logits,
                positive_indices,
                negative_indices,
            )
            positive_scores = row_logits[positive_indices]
            negative_scores = row_logits[negative_indices]
            differences = (
                negative_scores.unsqueeze(0)
                - positive_scores.unsqueeze(1)
                + self.margin
            ) / self.temperature
            pair_losses.append((F.softplus(differences) * weights).flatten())
        if not pair_losses:
            return logits.sum() * 0.0
        return torch.cat(pair_losses).mean()

    def _delta_dcg_weights(
        self,
        row_logits: torch.Tensor,
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor,
    ) -> torch.Tensor:
        rank_positions = _rank_positions(row_logits)
        positive_discounts = _dcg_discounts(rank_positions[positive_indices])
        negative_discounts = _dcg_discounts(rank_positions[negative_indices])
        weights = torch.abs(
            positive_discounts.unsqueeze(1) - negative_discounts.unsqueeze(0)
        )
        ideal_dcg = _ideal_dcg(positive_indices.numel(), row_logits.device)
        return weights / ideal_dcg.clamp_min(torch.finfo(row_logits.dtype).eps)


class AsymmetricGeneScoreLoss(nn.Module):
    """Asymmetric multi-label loss that downweights easy negative genes."""

    def __init__(
        self,
        gamma_positive: float,
        gamma_negative: float,
        clip: float,
        eps: float,
    ) -> None:
        super().__init__()
        if gamma_positive < 0 or gamma_negative < 0:
            raise ValueError("asymmetric gamma values must be non-negative")
        if clip < 0:
            raise ValueError("asymmetric clip must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.gamma_positive = gamma_positive
        self.gamma_negative = gamma_negative
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric multi-label loss."""
        if logits.shape != targets.shape:
            raise ValueError(
                "logits and targets must have the same shape: "
                f"{tuple(logits.shape)} != {tuple(targets.shape)}"
            )
        targets = targets.to(device=logits.device, dtype=logits.dtype)
        probabilities = torch.sigmoid(logits)
        positive_probs = probabilities.clamp(min=self.eps, max=1.0 - self.eps)
        negative_probs = 1.0 - probabilities
        if self.clip > 0:
            negative_probs = (negative_probs + self.clip).clamp(max=1.0)
        negative_probs = negative_probs.clamp(min=self.eps, max=1.0)
        positive_loss = targets * torch.log(positive_probs)
        negative_loss = (1.0 - targets) * torch.log(negative_probs)
        positive_weight = torch.pow(1.0 - probabilities, self.gamma_positive)
        negative_weight = torch.pow(probabilities, self.gamma_negative)
        weights = targets * positive_weight + (1.0 - targets) * negative_weight
        return -((positive_loss + negative_loss) * weights).mean()


def _rank_positions(scores: torch.Tensor) -> torch.Tensor:
    sorted_indices = torch.argsort(scores, descending=True)
    positions = torch.empty_like(sorted_indices)
    positions[sorted_indices] = torch.arange(scores.numel(), device=scores.device)
    return positions


def _dcg_discounts(rank_positions: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.log2(rank_positions.to(torch.float32) + 2.0)


def _ideal_dcg(n_positives: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(n_positives, device=device)
    return _dcg_discounts(positions).sum()


def build_gene_score_loss(config: GeneScoreLossConfig) -> GeneScoreLoss:
    """Build the configured gene score loss."""
    normalized_name = config.name.lower()
    if normalized_name == "bce":
        return BinaryCrossEntropyGeneScoreLoss()
    if normalized_name == "sampled_pairwise":
        return SampledPairwiseRankingLoss(
            hard_negatives=config.hard_negatives,
            random_negatives=config.random_negatives,
            temperature=config.temperature,
            margin=config.margin,
            hard_negative_source=config.hard_negative_source,
        )
    if normalized_name == "sampled_softmax":
        return SampledSoftmaxGeneScoreLoss(
            hard_negatives=config.hard_negatives,
            random_negatives=config.random_negatives,
            temperature=config.temperature,
            margin=config.margin,
            hard_negative_source=config.hard_negative_source,
        )
    if normalized_name == "sampled_lambdarank":
        return SampledLambdaRankGeneScoreLoss(
            hard_negatives=config.hard_negatives,
            random_negatives=config.random_negatives,
            temperature=config.temperature,
            margin=config.margin,
            hard_negative_source=config.hard_negative_source,
        )
    if normalized_name == "asymmetric":
        return AsymmetricGeneScoreLoss(
            gamma_positive=config.asymmetric_gamma_positive,
            gamma_negative=config.asymmetric_gamma_negative,
            clip=config.asymmetric_clip,
            eps=config.eps,
        )
    raise ValueError(f"unsupported gene score loss: {config.name}")
