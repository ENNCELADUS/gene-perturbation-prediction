"""Helpers for scGPT model outputs."""

from __future__ import annotations

import torch

from src.scgpt.losses import GeneScoreLoss


def logits_from_output(model_output) -> torch.Tensor:
    """Extract logits from a tensor or structured model output."""
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
        if not isinstance(logits, torch.Tensor):
            raise ValueError("model output dict must contain tensor logits")
        return logits
    if not isinstance(model_output, torch.Tensor):
        raise ValueError("model output must be a tensor or a dict with logits")
    return model_output


def cardinality_logits_from_output(model_output) -> torch.Tensor | None:
    """Extract optional cardinality logits from a structured model output."""
    if not isinstance(model_output, dict):
        return None
    cardinality_logits = model_output.get("cardinality_logits")
    return cardinality_logits if isinstance(cardinality_logits, torch.Tensor) else None


def loss_from_output(
    loss_fn: GeneScoreLoss,
    model_output,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Combine main gene-ranking loss with optional auxiliary losses."""
    logits = logits_from_output(model_output)
    loss = loss_fn(logits, targets)
    if not isinstance(model_output, dict):
        return loss
    auxiliary_losses = model_output.get("auxiliary_losses", {})
    if not isinstance(auxiliary_losses, dict):
        raise ValueError("model auxiliary_losses must be a mapping")
    for auxiliary_loss in auxiliary_losses.values():
        if not isinstance(auxiliary_loss, torch.Tensor):
            raise ValueError("auxiliary loss values must be tensors")
        loss = loss + auxiliary_loss
    return loss
