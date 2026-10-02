"""Response prediction over STATE sentences and the response losses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def mean_delta_mse(
    predicted: torch.Tensor, observed: torch.Tensor, control_mean: torch.Tensor
) -> torch.Tensor:
    """MSE between predicted and observed mean shift from the control mean.

    The control mean cancels algebraically but stays explicit: the matched
    quantity is a perturbation effect, not absolute expression.
    """
    delta_pred = predicted.mean(dim=0) - control_mean
    delta_obs = observed.mean(dim=0) - control_mean
    return torch.mean((delta_pred - delta_obs) ** 2)


def energy_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Energy distance between two cell bags, ``2 E|x-y| - E|x-x'| - E|y-y'|``.

    Unlike the mean term it responds to a change in spread, the case where a
    model predicts the right average shift while collapsing cell-to-cell variation.
    """
    if left.numel() == 0 or right.numel() == 0:
        raise ValueError("energy distance needs at least one cell per bag")
    cross = torch.cdist(left, right).mean()
    within_left = torch.cdist(left, left).mean()
    within_right = torch.cdist(right, right).mean()
    return 2.0 * cross - within_left - within_right


def response_loss(
    predicted: torch.Tensor, observed: torch.Tensor, control: torch.Tensor
) -> torch.Tensor:
    """Mean-shift MSE from the control mean plus energy distance, for one condition."""
    return mean_delta_mse(predicted, observed, control.mean(dim=0)) + energy_distance(
        predicted, observed
    )


def _padding(n_cells: int, window: int, seed: int) -> np.ndarray:
    """Seeded resample of a bag's own cells completing its last STATE sentence."""
    return np.random.default_rng(seed).choice(
        n_cells, size=window - n_cells % window, replace=True
    )


def predict_bags(
    model: nn.Module,
    bags: Sequence[torch.Tensor],
    genes: Sequence[str],
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """Predict every basal bag under its gene in one call of a ``StateResponse``.

    Each bag is cut into ``model.cell_set_len`` sentences; an incomplete last
    sentence is padded with the bag's own cells (seeded) and the output trimmed
    back to the bag's cell count.
    """
    window = model.cell_set_len
    chunks: list[torch.Tensor] = []
    chunk_genes: list[str] = []
    counts: list[tuple[int, int]] = []
    for bag, gene in zip(bags, genes, strict=True):
        n_cells = int(bag.shape[0])
        parts = list(bag.split(window, dim=0))
        if n_cells % window:
            index = torch.as_tensor(
                _padding(n_cells, window, seed), dtype=torch.long, device=bag.device
            )
            parts[-1] = torch.cat((parts[-1], bag[index]), dim=0)
        chunks.extend(parts)
        chunk_genes.extend(str(gene) for _ in parts)
        counts.append((len(parts), n_cells))
    predicted = model(tuple(chunks), tuple(chunk_genes))
    result, offset = [], 0
    for n_chunks, n_cells in counts:
        result.append(torch.cat(predicted[offset : offset + n_chunks])[:n_cells])
        offset += n_chunks
    return tuple(result)
