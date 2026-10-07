"""GeneEffect training objectives on residual-unit predictions."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.data.geneeffect import DEPENDENCY_THRESHOLD

# Rows a selective gene needs in a batch before its Pearson term counts.
MIN_PEARSON_ROWS = 3
# Added to each standardised variance so a constant prediction scores Pearson 0
# with a finite gradient instead of NaN.
PEARSON_EPS = 1e-8
# Softmax temperature of the line-ranking term, in residual-SD units.
LINE_RANKING_TEMPERATURE = 1.0


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


def line_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gene_index: torch.Tensor,
    selective: torch.Tensor,
) -> torch.Tensor:
    """Mean over eligible genes of the ListNet cross-entropy across a gene's lines.

    Inputs are standardised residuals. For each gene, the target distribution is the
    softmax over its rows of ``-target / LINE_RANKING_TEMPERATURE`` (the most
    dependent lines carry the most mass) and the loss is its cross-entropy against
    the same softmax of the prediction. Eligibility as :func:`blocked_pearson_loss`;
    without an eligible gene the term is zero.
    """
    rows = selective.bool()
    prediction, target = prediction[rows], target[rows]
    genes, group, counts = torch.unique(
        gene_index[rows], return_inverse=True, return_counts=True
    )
    if not len(genes):
        return prediction.sum() * 0.0

    def log_softmax(scores: torch.Tensor) -> torch.Tensor:
        top = scores.new_full((len(genes),), -torch.inf).scatter_reduce(
            0, group, scores, reduce="amax"
        )
        shifted = scores - top[group]
        total = shifted.new_zeros(len(genes)).index_add_(0, group, shifted.exp())
        return shifted - total.log()[group]

    target_log = log_softmax(-target / LINE_RANKING_TEMPERATURE)
    predicted_log = log_softmax(-prediction / LINE_RANKING_TEMPERATURE)
    per_gene = prediction.new_zeros(len(genes)).index_add_(
        0, group, -target_log.exp() * predicted_log
    )
    means = target.new_zeros(len(genes)).index_add_(0, group, target) / counts
    spread = target.new_zeros(len(genes)).index_add_(
        0, group, (target - means[group]).square()
    )
    eligible = (counts >= MIN_PEARSON_ROWS) & (spread > 0)
    if not bool(eligible.any()):
        return per_gene.sum() * 0.0
    return per_gene[eligible].mean()


def dependency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    gene_mean: torch.Tensor,
    selective: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy of ``GeneEffect < DEPENDENCY_THRESHOLD`` on selective rows.

    ``prediction`` and ``target`` are residuals; ``gene_mean + target`` is the
    measured GeneEffect. The logit is ``(threshold - gene_mean - prediction) /
    scale``, so a more negative predicted GeneEffect means a likelier dependency.
    Without a selective row the term is zero.
    """
    rows = selective.bool()
    if not bool(rows.any()):
        return prediction.sum() * 0.0
    absolute = gene_mean[rows] + prediction[rows]
    logit = (DEPENDENCY_THRESHOLD - absolute) / scale[rows]
    label = ((gene_mean[rows] + target[rows]) < DEPENDENCY_THRESHOLD).float()
    return F.binary_cross_entropy_with_logits(logit, label)


def geneeffect_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    *,
    objective: str,
    gene_index: torch.Tensor,
    selective: torch.Tensor,
    gene_mean: torch.Tensor,
) -> torch.Tensor:
    """One FP32 training objective over a batch of GeneEffect residual rows.

    ``prediction`` and ``target`` are in residual units, ``scale`` is each row's
    training-line residual SD and ``gene_mean`` its gene's training mean. ``huber``
    is Huber (delta 1) on residuals; ``standardized_mse`` the mean squared error of
    ``(prediction - target) / scale``; ``pearson_blocks``, ``line_ranking`` and
    ``dependency_classification`` add to it, with weight 1, the gene-blocked Pearson
    term, the ListNet term across lines (both on standardised residuals) or the
    dependency cross-entropy.
    """
    prediction, target = prediction.float(), target.float()
    if objective == "huber":
        return F.huber_loss(prediction, target, delta=1.0)
    scale = scale.float()
    standardized, standardized_target = prediction / scale, target / scale
    loss = (standardized - standardized_target).square().mean()
    if objective == "standardized_mse":
        return loss
    if objective == "pearson_blocks":
        return loss + blocked_pearson_loss(
            standardized, standardized_target, gene_index, selective
        )
    if objective == "line_ranking":
        return loss + line_ranking_loss(
            standardized, standardized_target, gene_index, selective
        )
    if objective == "dependency_classification":
        return loss + dependency_loss(
            prediction, target, scale, gene_mean.float(), selective
        )
    raise ValueError(f"unknown GeneEffect objective {objective!r}")
