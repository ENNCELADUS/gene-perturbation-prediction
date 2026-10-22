"""Fixed per-gene ESM2 vectors mapped to STATE perturbation vectors by an adapter."""

from __future__ import annotations
import numpy as np
import torch
from torch import nn
from src.data.embeddings import Esm2EmbeddingTable


class PertAdapter(nn.Module):
    """ESM2 embedding ``(B, esm_dim)`` -> STATE's raw ``(B, pert_dim)`` pert vector,
    which enters STATE's perturbation encoder; ``net`` keeps the checkpoint keys."""

    def __init__(self, esm_dim: int, hidden: int, pert_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(esm_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, pert_dim),
        )

    def forward(self, esm: torch.Tensor) -> torch.Tensor:
        return self.net(esm)


class Esm2PerturbationAdapter(nn.Module):
    """Fixed per-gene ESM-2 vectors -> STATE perturbation vectors, one adapter call
    for a batch of gene symbols (case-insensitive)."""

    def __init__(
        self,
        genes: list[str],
        table: Esm2EmbeddingTable,
        adapter_hidden: int,
        pert_dim: int,
    ) -> None:
        super().__init__()
        self.genes = [str(gene).upper() for gene in genes]
        missing = [gene for gene in self.genes if gene not in table.vectors_by_symbol]
        if missing:
            raise ValueError(f"Unresolved ESM-2 genes: {missing[:10]}")
        matrix = np.vstack([table.vectors_by_symbol[gene] for gene in self.genes])
        self._gene_to_index = {gene: index for index, gene in enumerate(self.genes)}
        self.register_buffer(
            "esm_matrix",
            torch.as_tensor(matrix, dtype=torch.float32),
            persistent=False,
        )
        self.adapter = PertAdapter(table.dim, int(adapter_hidden), int(pert_dim))

    def forward_many(self, genes: list[str] | tuple[str, ...]) -> torch.Tensor:
        indices = torch.as_tensor(
            [self._gene_to_index[str(gene).upper()] for gene in genes],
            dtype=torch.long,
            device=self.esm_matrix.device,
        )
        return self.adapter(self.esm_matrix.index_select(0, indices))
