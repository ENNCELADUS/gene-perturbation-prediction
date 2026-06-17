"""Permutation-invariant pooling of a predicted response bag into e_g."""

from __future__ import annotations

import torch
from torch import nn


class MeanStdPool(nn.Module):
    """Concatenate per-feature mean and std over the cell dimension.

    Args:
        bag: Tensor of shape ``(n_cells, D)`` representing a response bag.

    Returns:
        Tensor of shape ``(2D,)`` — concatenation of per-feature mean and
        population std (``unbiased=False``) over ``dim=0``.
    """

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        """Pool a cell bag into a fixed-size gene embedding.

        Args:
            bag: Shape ``(n_cells, D)``.

        Returns:
            Shape ``(2D,)``.
        """
        mean = bag.mean(dim=0)
        std = bag.std(dim=0, unbiased=False)
        return torch.cat([mean, std], dim=0)


def build_pool(name: str, dim: int) -> nn.Module:
    """Return a pooling module by name.

    Args:
        name: Pooling strategy name. Currently only ``"mean_std"`` is supported.
        dim: Input feature dimension (used for future pooling variants).

    Returns:
        An ``nn.Module`` that maps ``(n_cells, dim)`` to a fixed-size vector.

    Raises:
        ValueError: If ``name`` is not a recognised pooling strategy.
    """
    if name == "mean_std":
        return MeanStdPool()
    raise ValueError(f"unknown pooling: {name!r}")


def output_dim(name: str, dim: int) -> int:
    """Return the pooled embedding dimension for a given pooling strategy.

    Args:
        name: Pooling strategy name.
        dim: Input feature dimension.

    Returns:
        Output dimension after pooling.

    Raises:
        ValueError: If ``name`` is not a recognised pooling strategy.
    """
    if name == "mean_std":
        return 2 * dim
    raise ValueError(f"unknown pooling: {name!r}")
