"""Three-part loss for exp08: SL BCE + adapter distill + bag supervision.

The bag loss reuses :func:`aivc_model.model._energy_distance` so both the
AIVC forward model and the SL-pair DL share the same distribution-level
supervision signal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from aivc_model.model import _energy_distance


def sl_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy on SL pair logits.

    Uses the numerically stable log-sum-exp form to avoid platform-specific
    issues with ``F.binary_cross_entropy_with_logits`` on MPS backends.
    Equivalent to ``torch.nn.BCEWithLogitsLoss(reduction='mean')``.

    Args:
        logits: Raw (pre-sigmoid) pair scores, shape (B,).
        labels: Binary SL labels (0 or 1), shape (B,).

    Returns:
        Scalar BCE loss.
    """
    y = labels.float()
    # Numerically stable: max(x,0) - x*y + log(1 + exp(-|x|))
    per_sample = (
        torch.clamp(logits, min=0) - logits * y + torch.log1p(torch.exp(-logits.abs()))
    )
    return per_sample.mean()


def distill_loss(
    adapter_tokens: torch.Tensor, target_tokens: torch.Tensor
) -> torch.Tensor:
    """MSE between adapter output and STATE's original one-hot pert token.

    Args:
        adapter_tokens: Tokens produced by the trained :class:`PertAdapter`,
            shape (B, pert_dim).
        target_tokens: Tokens from the checkpoint's own ``pert_encoder``
            applied to the in-vocab one-hot, shape (B, pert_dim).

    Returns:
        Scalar MSE distillation loss.
    """
    return F.mse_loss(adapter_tokens, target_tokens)


def bag_loss(pred_bag: torch.Tensor, real_bag: torch.Tensor) -> torch.Tensor:
    """Mean-delta MSE plus energy distance between predicted and real bags.

    The mean-delta term penalises shifts in the per-feature centre of mass;
    the energy-distance term penalises distributional divergence across cells.

    Args:
        pred_bag: Predicted response bag from the STATE encoder,
            shape (n_cells, D).
        real_bag: Ground-truth gwps bag for the same perturbation gene,
            shape (m_cells, D).

    Returns:
        Non-negative scalar bag supervision loss.
    """
    mean_delta = F.mse_loss(pred_bag.mean(dim=0), real_bag.mean(dim=0))
    energy = _energy_distance(pred_bag, real_bag)
    return mean_delta + energy


def combine(parts: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    """Weighted sum of named loss parts; missing weights default to 0.

    Args:
        parts: Named scalar loss tensors (e.g. ``{"sl": ..., "distill": ...}``).
        weights: Per-name scalar weights. Any name absent from ``weights``
            contributes 0 to the total.

    Returns:
        Weighted sum as a scalar tensor.

    Raises:
        ValueError: If ``parts`` is empty.
    """
    if not parts:
        raise ValueError("no loss parts provided")
    total: torch.Tensor | None = None
    for name, value in parts.items():
        weight = float(weights.get(name, 0.0))
        term = weight * value
        total = term if total is None else total + term
    # total cannot be None here because parts is non-empty
    assert total is not None
    return total
