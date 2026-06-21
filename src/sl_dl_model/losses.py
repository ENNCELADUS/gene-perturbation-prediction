"""Three-part loss for exp08: SL BCE + adapter distill + bag supervision.

The bag loss uses a local NaN-safe energy distance (see
:func:`_safe_energy_distance`) rather than ``aivc_model.model._energy_distance``
so the distribution-level supervision has a finite forward and backward even
when pred/real bags contain coincident cells (the ``torch.cdist`` 0/0
self-distance NaN trap).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Epsilon added under the sqrt so the Euclidean-distance gradient stays finite
# at zero pairwise distance (the cdist self-distance 0/0 NaN trap, H1b).
_ENERGY_EPS = 1e-8


def _safe_energy_distance(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """NaN-safe energy distance between two cell bags.

    Equivalent to ``2*E||X-Y|| - E||X-X'|| - E||Y-Y'||`` but computed without
    ``torch.cdist``: pairwise squared distances are formed via the quadratic
    form, clamped to be non-negative (kills the float ``sqrt(negative)`` NaN,
    H1a), and an epsilon is added under the ``sqrt`` so the gradient is finite
    at zero distance (kills the self-distance ``0/0`` NaN, H1b).

    Args:
        predicted: Predicted bag, shape ``(n, D)``.
        target: Real bag, shape ``(m, D)``.

    Returns:
        Non-negative scalar energy distance.
    """
    cross = _safe_pairwise_dist(predicted, target).mean()
    pred_self = _safe_pairwise_dist(predicted, predicted).mean()
    target_self = _safe_pairwise_dist(target, target).mean()
    return (2.0 * cross - pred_self - target_self).clamp_min(0.0)


def _safe_pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean pairwise distances with a finite gradient at zero distance.

    Args:
        a: Tensor of shape ``(n, D)``.
        b: Tensor of shape ``(m, D)``.

    Returns:
        Distance matrix of shape ``(n, m)``.
    """
    a2 = (a * a).sum(dim=-1, keepdim=True)  # (n, 1)
    b2 = (b * b).sum(dim=-1, keepdim=True)  # (m, 1)
    d2 = a2 - 2.0 * (a @ b.transpose(-2, -1)) + b2.transpose(-2, -1)
    d2 = d2.clamp_min(0.0)
    return torch.sqrt(d2 + _ENERGY_EPS)


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
    energy = _safe_energy_distance(pred_bag, real_bag)
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
        if weight == 0.0:
            if not torch.isfinite(value).all():
                # Drop a zero-weighted non-finite term so it cannot poison the
                # total via 0.0 * NaN (e.g. an unused SL term during warmup).
                continue
            # Keep finite zero-weighted parts in the graph (zero contribution
            # but a live grad path to the trainable params during warmup).
            term = 0.0 * value
        else:
            term = weight * value
        total = term if total is None else total + term
    if total is None:
        # Every part was zero-weighted and non-finite; return a finite scalar
        # zero on the right device/dtype derived from an arbitrary part.
        any_value = next(iter(parts.values()))
        return torch.zeros((), dtype=any_value.dtype, device=any_value.device)
    return total
