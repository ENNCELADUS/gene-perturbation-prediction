"""STATE-free response model: per-cell expression shift from cell and gene."""

import torch
from torch import nn


class ResponseMLP(nn.Module):
    """``basal + f([cell ; adapter(gene)])``.

    The last layer starts at zero, so the untrained model is exactly no-change.
    """

    def __init__(self, cell_dim: int, gene_dim: int, hidden: int, out_dim: int = 2000):
        super().__init__()
        self.gene = nn.Sequential(nn.Linear(gene_dim, hidden), nn.GELU())
        self.shift = nn.Sequential(
            nn.Linear(cell_dim + hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )
        nn.init.zeros_(self.shift[-1].weight)
        nn.init.zeros_(self.shift[-1].bias)

    def forward(
        self, cells: torch.Tensor, basal_hvg: torch.Tensor, gene: torch.Tensor
    ) -> torch.Tensor:
        token = self.gene(gene).expand(cells.shape[0], -1)
        return basal_hvg + self.shift(torch.cat((cells, token), dim=1))
