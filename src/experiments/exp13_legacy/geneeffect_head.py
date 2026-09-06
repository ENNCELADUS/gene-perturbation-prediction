"""experiments / exp13 legacy / geneeffect head."""

from __future__ import annotations

import math
from typing import NamedTuple
import torch
from torch.nn import functional as F
from src.model.losses import _STD_EPS

_CONSTANT_TARGET_STD_EPS: float = 1e-6


class PerGeneRankVarianceLoss(NamedTuple):
    """Result of :func:`per_gene_rank_variance_loss`.

    Attributes:
        loss: 0-D scalar tensor, gradient-bearing w.r.t. ``pred``: the mean
            of ``correlation_loss + lam * variance_matching_loss`` over
            genes whose target is not (near-)constant across contexts.
        n_genes_scored: Number of gene rows that entered the mean.
        n_genes_excluded: Number of gene rows excluded for having a
            (near-)constant target across the context axis (undefined
            correlation) -- never folded into ``loss`` as 0 or NaN.
    """

    loss: torch.Tensor
    n_genes_scored: int
    n_genes_excluded: int


def per_gene_rank_variance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float = 1.0,
) -> PerGeneRankVarianceLoss:
    """Axis-aware replacement for the retired ``rank_variance_loss``.

    That loss flattened its ``[B]`` inputs to one Pearson correlation over
    whatever the batch happened to mix (genes, contexts, or both). This
    function instead requires ``pred``/``target`` shaped
    ``[n_genes, n_contexts]`` and computes the correlation and
    variance-matching terms
    independently **per row (gene), along the context axis (dim=-1)**, then
    macro-averages the per-gene combined loss over genes -- the same axis
    choice as :func:`src.eval.metrics.per_gene_spearman`, so the
    training objective matches the benchmark's scored metric.

    A gene whose target is (near-)constant across contexts has an
    undefined correlation (a constant reference vector has no direction to
    correlate against) and is excluded from the mean rather than
    contributing a manufactured 0 or a NaN that would poison the whole
    batch's gradient -- mirroring ``residual_metrics.py``'s per-unit NaN
    convention. A gene whose *prediction* row collapses to (near-)constant
    is not excluded: like ``correlation_loss``, its standard deviation is
    floored at ``_STD_EPS`` before dividing, so it scores the finite worst
    case (correlation loss ~1) instead of being dropped -- collapse must be
    punished, not hidden from the gradient.

    Args:
        pred: Predictions, shape ``[n_genes, n_contexts]``.
        target: Targets, shape ``[n_genes, n_contexts]``, same shape as
            ``pred``.
        lam: Weight on the per-gene variance-matching term.

    Returns:
        A :class:`PerGeneRankVarianceLoss`.

    Raises:
        ValueError: If ``pred``/``target`` are not both 2-D of equal shape,
            or if every gene's target is (near-)constant (the macro-average
            would be over an empty set).
    """
    if pred.dim() != 2 or tuple(pred.shape) != tuple(target.shape):
        raise ValueError(
            "pred and target must both be 2-D [n_genes, n_contexts] of equal "
            f"shape; got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    target = target.to(device=pred.device, dtype=pred.dtype)
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError(
            "per_gene_rank_variance_loss requires finite pred/target; this "
            "loss operates on a dense labeled batch -- filter unlabeled "
            "(gene, context) cells out before calling, do not pass NaN"
        )

    n_genes = pred.shape[0]
    target_std = target.std(dim=-1, unbiased=False)
    defined = target_std > _CONSTANT_TARGET_STD_EPS
    n_excluded = int((~defined).sum().item())
    if not bool(defined.any()):
        raise ValueError(
            "per_gene_rank_variance_loss: every gene's target is "
            "(near-)constant across contexts; the per-gene macro-average is "
            "undefined for this batch"
        )

    pred_defined = pred[defined]
    target_defined = target[defined]

    pred_centered = pred_defined - pred_defined.mean(dim=-1, keepdim=True)
    target_centered = target_defined - target_defined.mean(dim=-1, keepdim=True)
    covariance = (pred_centered * target_centered).mean(dim=-1)
    pred_std = (pred_centered.square().mean(dim=-1) + _STD_EPS).sqrt()
    target_std_defined = (target_centered.square().mean(dim=-1) + _STD_EPS).sqrt()
    correlation = covariance / (pred_std * target_std_defined)
    correlation_loss_per_gene = 1.0 - correlation

    pred_row_std = pred_defined.std(dim=-1, unbiased=False)
    target_row_std = target_defined.std(dim=-1, unbiased=False)
    variance_loss_per_gene = (pred_row_std - target_row_std).square()

    per_gene_loss = correlation_loss_per_gene + float(lam) * variance_loss_per_gene
    loss = per_gene_loss.mean()
    return PerGeneRankVarianceLoss(
        loss=loss,
        n_genes_scored=n_genes - n_excluded,
        n_genes_excluded=n_excluded,
    )


class MaskedGeneEffectLoss(NamedTuple):
    """Result of :func:`masked_geneeffect_residual_loss`.

    ``pearson`` is the macro-averaged Pearson loss ``1 - r``, not the
    correlation itself. Gene counts refer only to genes selected by
    ``g_var_mask``; non-selected genes still contribute valid pairs to
    ``huber``.
    """

    total: torch.Tensor
    huber: torch.Tensor
    pearson: torch.Tensor
    n_valid_pairs: int
    n_genes_scored: int
    n_genes_excluded: int


def masked_geneeffect_residual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_mask: torch.Tensor,
    g_var_mask: torch.Tensor,
    huber_delta: float = 1.0,
    beta: float = 1.0,
) -> MaskedGeneEffectLoss:
    """Compute the formal masked Stage-2 GeneEffect residual objective.

    Huber loss is averaged over every labeled ``(gene, context)`` pair.
    Pearson loss is computed independently along the context axis for each
    gene selected by ``g_var_mask``, then macro-averaged. Selected genes with
    fewer than three labeled contexts or a (near-)constant target are
    excluded from the Pearson mean. A constant prediction is retained and
    receives a finite unit Pearson penalty rather than being dropped.

    Missing targets may be non-finite only where ``label_mask`` is false.
    The function fails closed rather than returning a manufactured zero or
    NaN when either the Huber or Pearson reduction would be empty.

    Args:
        pred: Residual predictions, shape ``[n_genes, n_contexts]``.
        target: Residual targets with the same shape as ``pred``.
        label_mask: Boolean mask with the same shape; true marks a labeled
            pair that must have finite ``pred`` and ``target`` values.
        g_var_mask: Boolean mask of shape ``[n_genes]`` selecting genes for
            the macro Pearson term.
        huber_delta: Positive transition point for Huber loss.
        beta: Non-negative weight on the macro Pearson loss.

    Returns:
        A :class:`MaskedGeneEffectLoss` containing the total and both
        component losses plus valid-pair and Pearson gene counts.

    Raises:
        ValueError: On invalid shapes or masks, invalid hyperparameters,
            non-finite labeled values, no labeled pairs, or no scorable
            ``g_var_mask`` gene.
    """
    if pred.dim() != 2 or tuple(pred.shape) != tuple(target.shape):
        raise ValueError(
            "pred and target must both be 2-D [n_genes, n_contexts] of equal "
            f"shape; got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    if tuple(label_mask.shape) != tuple(pred.shape) or label_mask.dtype != torch.bool:
        raise ValueError(
            "label_mask must be boolean with the same shape as pred; got "
            f"shape={tuple(label_mask.shape)}, dtype={label_mask.dtype}"
        )
    if tuple(g_var_mask.shape) != (pred.shape[0],) or g_var_mask.dtype != torch.bool:
        raise ValueError(
            "g_var_mask must be boolean with shape [n_genes]; got "
            f"shape={tuple(g_var_mask.shape)}, dtype={g_var_mask.dtype}"
        )
    if not pred.is_floating_point():
        raise ValueError(f"pred must have a floating dtype, got {pred.dtype}")
    if not math.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError(f"huber_delta must be finite and positive, got {huber_delta}")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError(f"beta must be finite and non-negative, got {beta}")

    target = target.to(device=pred.device)
    label_mask = label_mask.to(device=pred.device)
    g_var_mask = g_var_mask.to(device=pred.device)
    compute_dtype = (
        torch.float32 if pred.dtype in (torch.float16, torch.bfloat16) else pred.dtype
    )
    work_pred = pred.to(dtype=compute_dtype)
    work_target = target.to(dtype=compute_dtype)
    n_valid_pairs = int(label_mask.sum().item())
    if n_valid_pairs == 0:
        raise ValueError("masked_geneeffect_residual_loss: no valid Huber pairs")
    if (
        not torch.isfinite(work_pred[label_mask]).all()
        or not torch.isfinite(work_target[label_mask]).all()
    ):
        raise ValueError("pred and target must be finite at every labeled pair")

    huber = F.huber_loss(
        work_pred[label_mask],
        work_target[label_mask],
        reduction="mean",
        delta=float(huber_delta),
    )

    pearson_losses: list[torch.Tensor] = []
    n_selected = int(g_var_mask.sum().item())
    for gene_index in torch.nonzero(g_var_mask, as_tuple=False).flatten().tolist():
        valid = label_mask[gene_index]
        if int(valid.sum().item()) < 3:
            continue
        pred_gene = work_pred[gene_index, valid]
        target_gene = work_target[gene_index, valid]
        pred_centered = pred_gene - pred_gene.mean()
        target_centered = target_gene - target_gene.mean()
        target_std = target_centered.square().mean().sqrt()
        if float(target_std.detach()) <= _CONSTANT_TARGET_STD_EPS:
            continue
        pred_std = (pred_centered.square().mean() + _STD_EPS).sqrt()
        covariance = (pred_centered * target_centered).mean()
        correlation = covariance / (pred_std * target_std)
        pearson_losses.append(1.0 - correlation)

    n_genes_scored = len(pearson_losses)
    n_genes_excluded = n_selected - n_genes_scored
    if n_genes_scored == 0:
        raise ValueError(
            "masked_geneeffect_residual_loss: no scorable Pearson genes "
            "(need a G_var gene with at least 3 valid, non-constant targets)"
        )
    pearson = torch.stack(pearson_losses).mean()
    total = huber + float(beta) * pearson
    return MaskedGeneEffectLoss(
        total=total,
        huber=huber,
        pearson=pearson,
        n_valid_pairs=n_valid_pairs,
        n_genes_scored=n_genes_scored,
        n_genes_excluded=n_genes_excluded,
    )
