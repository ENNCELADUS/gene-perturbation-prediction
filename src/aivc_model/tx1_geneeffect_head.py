"""Rebuilt GeneEffect readout head for the T2 Tx1-conditioned backbone.

Frozen spec §2.2 ("Readout") replaces the K562-fit ``TrainableDiagonalGMM``
pooler (``response.py``) plus the MSE-trained ``MLPHead`` (``model.py``) with
three pieces, all in this module:

1. :func:`moment_pool` — a permutation-invariant moment pool (mean/variance,
   optionally higher moments) over a per-cell response bag. No trainable
   parameters, unlike ``TrainableDiagonalGMM``.
2. :class:`Tx1GeneEffectHead` — an MLP head conditioned on the moment-pooled
   predicted-response bag *and* a moment-pooled summary of the cell line's
   basal (control) cell embeddings. There is no line-ID input anywhere, so
   the head is structurally forced to generalize from basal-population
   statistics rather than memorizing a fixed set of training lines.
3. :func:`correlation_loss`, :func:`variance_matching_loss`, and
   :func:`rank_variance_loss` — a rank/correlation + variance-matching
   training objective that replaces MSE. MSE is minimized by predicting the
   per-line mean for every gene (zero variance, Spearman ~ 0); the loss here
   is explicitly constructed to penalize exactly that collapse.

This module is standalone: it is not imported by ``model.py`` or
``train.py``. Wiring it into the training loop is a separate follow-up task
(see ``T2_INTEGRATION_MAP.md`` §2.3 for the seam).
"""

from __future__ import annotations

import logging

import torch
from torch import nn

_LOGGER = logging.getLogger(__name__)

#: Numerical floor added to standard deviations before dividing, to keep
#: correlation/variance losses finite for near-constant inputs.
_STD_EPS: float = 1e-6


def moment_pool(
    bag: torch.Tensor,
    mask: torch.Tensor | None = None,
    moments: int = 2,
) -> torch.Tensor:
    """Pool a per-cell bag into a permutation-invariant moment summary.

    Args:
        bag: Per-cell tensor of shape ``[N, D]`` (cells, feature dim).
        mask: Optional boolean or 0/1 tensor of shape ``[N]``. Rows where
            ``mask`` is falsy are excluded from every moment (padded rows).
            If ``None``, every row is used.
        moments: Number of moments to compute, in order:
            1 -> mean only, shape ``[D]``.
            2 (default) -> concat(mean, variance), shape ``[2D]``. Variance
                is the population (biased, ``unbiased=False``) variance,
                matching ``TrainableDiagonalGMM`` (``response.py``).
            3 -> adds the standard deviation, shape ``[3D]``.
            4 -> adds the (Fisher, population) skewness, shape ``[4D]``.
            Higher moments are not implemented; callers needing kurtosis or
            beyond should extend this function rather than call it with
            ``moments > 4``. The default of 2 (mean + variance) is the
            minimal pair that can express both a per-line location shift
            and a spread — matching the frozen spec's "mean + variance,
            optionally higher moments" and mirroring the mean/variance pair
            already used inside ``TrainableDiagonalGMM.forward``.

    Returns:
        A 1-D tensor of length ``D * moments``, invariant to row order of
        ``bag`` (and, when masked, invariant to the order/placement of the
        masked-out padding rows).

    Raises:
        ValueError: If ``moments`` is not in ``{1, 2, 3, 4}``, if ``bag`` is
            not 2-D, or if every row is masked out (empty bag).
    """
    if bag.dim() != 2:
        msg = f"bag must be 2-D [N, D], got shape {tuple(bag.shape)}"
        raise ValueError(msg)
    if moments not in (1, 2, 3, 4):
        msg = f"moments must be one of {{1, 2, 3, 4}}, got {moments}"
        raise ValueError(msg)

    if mask is not None:
        keep = mask.to(dtype=torch.bool, device=bag.device)
        if not bool(keep.any()):
            msg = "moment_pool received an empty bag (mask excludes all rows)"
            raise ValueError(msg)
        bag = bag[keep]

    n_rows = bag.shape[0]
    if n_rows == 0:
        msg = "moment_pool received an empty bag (0 rows)"
        raise ValueError(msg)

    mean = bag.mean(dim=0)
    parts = [mean]
    if moments >= 2:
        centered = bag - mean.unsqueeze(0)
        variance = centered.square().mean(dim=0)
        parts.append(variance)
    if moments >= 3:
        std = (variance + _STD_EPS).sqrt()
        parts.append(std)
    if moments >= 4:
        skew = (centered.pow(3).mean(dim=0)) / (std.pow(3) + _STD_EPS)
        parts.append(skew)
    return torch.cat(parts, dim=0)


class Tx1GeneEffectHead(nn.Module):
    """Basal-conditioned, permutation-invariant GeneEffect readout head.

    Supersedes ``TrainableDiagonalGMM`` + ``MLPHead`` (``response.py`` /
    ``model.py``). The head moment-pools the predicted-response bag and the
    basal (control) bag independently, concatenates the two summaries, and
    passes them through an MLP to a scalar. There is no trainable GMM and no
    line-ID input: the only cell-line signal available to the head is the
    basal-bag moment summary, which is defined for any unseen line.

    Attributes:
        response_dim: Feature width of a single response-bag row.
        basal_dim: Feature width of a single basal-bag row.
        moments: Number of moments computed per bag (see :func:`moment_pool`).
    """

    def __init__(
        self,
        response_dim: int,
        basal_dim: int,
        hidden: int = 256,
        moments: int = 2,
    ) -> None:
        """Initialize the head.

        Args:
            response_dim: Per-cell feature width of the predicted-response
                bag (e.g. the STATE response-encoder latent dim, 128).
            basal_dim: Per-cell feature width of the basal/control bag
                (e.g. the same response-encoder latent dim, 128).
            hidden: Hidden width of the MLP trunk.
            moments: Moment count forwarded to :func:`moment_pool` for both
                bags (see that docstring for the moments -> width mapping).
        """
        super().__init__()
        self.response_dim = int(response_dim)
        self.basal_dim = int(basal_dim)
        self.moments = int(moments)
        pooled_dim = self.response_dim * self.moments + self.basal_dim * self.moments
        self.net = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        response_bag: torch.Tensor,
        basal_bag: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a scalar GeneEffect from one (gene, line) example.

        Args:
            response_bag: Predicted per-cell response latents, shape
                ``[N, response_dim]``.
            basal_bag: Basal/control per-cell latents for the same cell
                line, shape ``[M, basal_dim]``. ``M`` need not equal ``N``.

        Returns:
            A 0-D scalar tensor: the predicted GeneEffect. Invariant to the
            row order of both ``response_bag`` and ``basal_bag``.
        """
        pooled_response = moment_pool(response_bag, moments=self.moments)
        pooled_basal = moment_pool(basal_bag, moments=self.moments)
        pooled = torch.cat([pooled_response, pooled_basal], dim=0)
        return self.net(pooled).squeeze(-1)

    def forward_batch(
        self,
        response_bags: list[torch.Tensor] | torch.Tensor,
        basal_bags: list[torch.Tensor] | torch.Tensor,
        response_mask: torch.Tensor | None = None,
        basal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict GeneEffect for a batch of (gene, line) examples.

        Accepts either a Python list of variable-``N`` bags (ragged, one
        tensor per example) or a single padded tensor plus a boolean mask.

        Args:
            response_bags: Either a list of ``B`` tensors, each
                ``[N_i, response_dim]``, or a padded tensor
                ``[B, N_max, response_dim]`` (requires ``response_mask``).
            basal_bags: Same two accepted forms as ``response_bags``, with
                ``basal_dim`` in place of ``response_dim``; its batch/pad
                size may differ from ``response_bags``.
            response_mask: Required iff ``response_bags`` is a padded
                tensor; boolean/0-1 tensor of shape ``[B, N_max]``.
            basal_mask: Required iff ``basal_bags`` is a padded tensor;
                boolean/0-1 tensor of shape ``[B, M_max]``.

        Returns:
            A 1-D tensor of shape ``[B]`` with one GeneEffect prediction per
            example, in input order.
        """
        response_list = _as_bag_list(response_bags, response_mask)
        basal_list = _as_bag_list(basal_bags, basal_mask)
        if len(response_list) != len(basal_list):
            msg = (
                "response_bags and basal_bags must have the same batch size, "
                f"got {len(response_list)} and {len(basal_list)}"
            )
            raise ValueError(msg)
        predictions = [
            self.forward(response, basal)
            for response, basal in zip(response_list, basal_list)
        ]
        return torch.stack(predictions, dim=0)


def _as_bag_list(
    bags: list[torch.Tensor] | torch.Tensor,
    mask: torch.Tensor | None,
) -> list[torch.Tensor]:
    """Normalize a ragged-list-or-padded-tensor bag batch into a list.

    Args:
        bags: A list of ``[N_i, D]`` tensors, or a padded ``[B, N_max, D]``
            tensor (requires ``mask``).
        mask: Boolean/0-1 tensor of shape ``[B, N_max]``, required when
            ``bags`` is a padded tensor; ignored (must be ``None``) when
            ``bags`` is already a list.

    Returns:
        A list of ``B`` unpadded ``[N_i, D]`` tensors.

    Raises:
        ValueError: If a padded tensor is given without a mask, or a list is
            given with a mask.
    """
    if isinstance(bags, torch.Tensor):
        if mask is None:
            msg = "a padded bag tensor requires an accompanying mask"
            raise ValueError(msg)
        keep = mask.to(dtype=torch.bool, device=bags.device)
        return [bags[i][keep[i]] for i in range(bags.shape[0])]
    if mask is not None:
        msg = "mask must be None when bags is already a list of tensors"
        raise ValueError(msg)
    return list(bags)


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute ``1 - Pearson correlation`` between predictions and targets.

    Differentiable and gradient-bearing with respect to ``pred``. Guards the
    zero-variance case (e.g. every prediction collapsed to a single value)
    by flooring both standard deviations at ``_STD_EPS`` before dividing, so
    the loss stays finite instead of producing ``NaN`` from a ``0/0``.

    A soft-Spearman alternative (ranking by a differentiable soft-sort/
    soft-rank of ``pred`` before computing Pearson correlation on the ranks)
    would be less sensitive to outliers and monotonic-but-nonlinear
    relationships; it is not implemented here because it requires an extra
    dependency or a hand-rolled soft-rank operator, and the anti-collapse
    property this loss needs (punishing zero-variance ``pred``) holds
    identically for both the Pearson and soft-Spearman forms.

    Args:
        pred: Predictions, shape ``[B]``.
        target: Targets, shape ``[B]``.

    Returns:
        A 0-D scalar tensor in ``[0, 2]``: 0 for perfect positive
        correlation, 1 for zero correlation, 2 for perfect negative
        correlation.
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1).to(device=pred.device, dtype=pred.dtype)
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    covariance = (pred_centered * target_centered).mean()
    pred_std = pred_centered.square().mean().sqrt() + _STD_EPS
    target_std = target_centered.square().mean().sqrt() + _STD_EPS
    correlation = covariance / (pred_std * target_std)
    return 1.0 - correlation


def variance_matching_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the squared difference of prediction/target standard deviations.

    This is the term that directly punishes a mean-collapsed prediction:
    if ``pred`` is (near-)constant, ``std(pred) -> 0`` while ``std(target)``
    reflects the real spread of the labels, so the loss grows with the
    label spread instead of vanishing the way MSE does under a
    mean-prediction shortcut.

    A ratio-form alternative, ``(std(pred) / (std(target) + eps) - 1) ** 2``,
    is scale-normalized (penalizes relative rather than absolute spread
    mismatch) and may be preferable when target scales vary a lot across
    training regimes; it is not used here because it divides by a
    label-derived quantity, which is undesirable when ``target`` itself can
    be near-constant for a given (gene, line) mini-batch.

    Args:
        pred: Predictions, shape ``[B]``.
        target: Targets, shape ``[B]``.

    Returns:
        A 0-D scalar tensor, ``(std(pred) - std(target)) ** 2``.
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1).to(device=pred.device, dtype=pred.dtype)
    pred_std = pred.std(unbiased=False)
    target_std = target.std(unbiased=False)
    return (pred_std - target_std).square()


def rank_variance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float = 1.0,
) -> torch.Tensor:
    """Combine :func:`correlation_loss` and :func:`variance_matching_loss`.

    Replaces the MSE head loss (``model.py:714-715``, ``pred_c``/``obs_c``).
    MSE is minimized by the per-line-mean shortcut (constant prediction,
    zero variance, Spearman ~ 0); this loss is not: the correlation term is
    undefined-toward-1 (worst) for a constant ``pred`` (guarded to exactly 1
    via the ``_STD_EPS`` floor), and the variance term is large whenever
    ``pred`` collapses relative to a spread-having ``target``.

    Args:
        pred: Predictions, shape ``[B]``.
        target: Targets, shape ``[B]``.
        lam: Weight on the variance-matching term.

    Returns:
        A 0-D scalar tensor: ``correlation_loss(pred, target) + lam *
        variance_matching_loss(pred, target)``.
    """
    return correlation_loss(pred, target) + float(lam) * variance_matching_loss(
        pred, target
    )
