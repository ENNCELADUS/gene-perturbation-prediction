"""GeneEffect training objectives on residual-unit predictions."""

from __future__ import annotations

import torch
from torch.nn import functional as F

# Rows a selective gene needs in a batch before its Pearson term counts.
MIN_PEARSON_ROWS = 3
# Added to each standardised variance so a constant prediction scores Pearson 0
# with a finite gradient instead of NaN.
PEARSON_EPS = 1e-8


def blocked_pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gene_index: torch.Tensor,
    selective: torch.Tensor,
) -> torch.Tensor:
    """Mean over eligible genes of ``1 - pearson(prediction_g, target_g)``.

    A gene is eligible when it is selective, has at least ``MIN_PEARSON_ROWS`` rows
    in the batch and a non-constant target there. Without an eligible gene the
    term is zero.
    """
    rows = selective.bool()
    prediction, target = prediction[rows], target[rows]
    genes, group, counts = torch.unique(
        gene_index[rows], return_inverse=True, return_counts=True
    )
    if not len(genes):
        return prediction.sum() * 0.0

    def centred(values: torch.Tensor) -> torch.Tensor:
        sums = values.new_zeros(len(genes)).index_add_(0, group, values)
        return values - (sums / counts)[group]

    def per_gene(values: torch.Tensor) -> torch.Tensor:
        return values.new_zeros(len(genes)).index_add_(0, group, values) / counts

    prediction, target = centred(prediction), centred(target)
    covariance = per_gene(prediction * target)
    prediction_var, target_var = per_gene(prediction**2), per_gene(target**2)
    eligible = (counts >= MIN_PEARSON_ROWS) & (target_var > 0)
    if not bool(eligible.any()):
        return covariance.sum() * 0.0
    pearson = covariance / torch.sqrt(
        (prediction_var + PEARSON_EPS) * (target_var + PEARSON_EPS)
    )
    return (1.0 - pearson[eligible]).mean()


def geneeffect_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    *,
    objective: str,
    gene_index: torch.Tensor,
    selective: torch.Tensor,
) -> torch.Tensor:
    """One FP32 training objective over a batch of GeneEffect residual rows.

    ``prediction`` and ``target`` are in residual units and ``scale`` is each
    row's training-line residual SD. ``huber`` is Huber (delta 1) on residuals;
    ``standardized_mse`` the mean squared error of ``(prediction - target) /
    scale``; ``pearson_blocks`` adds to it the mean over the batch's selective
    genes of ``1 - pearson`` across their rows (``blocked_pearson_loss``).
    """
    prediction, target = prediction.float(), target.float()
    if objective == "huber":
        return F.huber_loss(prediction, target, delta=1.0)
    scale = scale.float()
    prediction, target = prediction / scale, target / scale
    loss = (prediction - target).square().mean()
    if objective == "standardized_mse":
        return loss
    if objective == "pearson_blocks":
        return loss + blocked_pearson_loss(prediction, target, gene_index, selective)
    raise ValueError(f"unknown GeneEffect objective {objective!r}")
