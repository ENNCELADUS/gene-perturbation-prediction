"""model / losses."""

from __future__ import annotations

import torch

_STD_EPS: float = 1e-6


def moment_pool(
    bag: torch.Tensor,
    mask: torch.Tensor | None = None,
    moments: int = 2,
) -> torch.Tensor:
    """Pool a per-cell bag into a permutation-invariant moment summary.

    ``Pi = [mean, var]`` (``01-blueprint.md`` §3, the Exp13 plan
    Target architecture): the default ``moments=2`` is exactly the pair used
    to build ``z_c = Pi({x_c^(i)})`` and ``Delta_{g,c} = Pi(B_hat_{c,g}) -
    Pi({b_c^(i)})``.

    Args:
        bag: Per-cell tensor of shape ``[N, D]`` (cells, feature dim).
        mask: Optional boolean or 0/1 tensor of shape ``[N]``. Rows where
            ``mask`` is falsy are excluded from every moment (padded rows).
            If ``None``, every row is used.
        moments: Number of moments to compute, in order:
            1 -> mean only, shape ``[D]``.
            2 (default) -> concat(mean, variance), shape ``[2D]``. Variance
                is the population (biased, ``unbiased=False``) variance.
            3 -> adds the standard deviation, shape ``[3D]``.
            4 -> adds the (Fisher, population) skewness, shape ``[4D]``.
            Higher moments are not implemented; callers needing kurtosis or
            beyond should extend this function rather than call it with
            ``moments > 4``.

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
