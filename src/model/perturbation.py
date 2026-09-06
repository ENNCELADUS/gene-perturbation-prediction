"""model / perturbation."""

from __future__ import annotations
import numpy as np
import torch
from torch import nn
from src.data.embeddings import Esm2EmbeddingTable


class PertAdapter(nn.Module):
    """Map an ESM2 gene embedding to a STATE raw pert vector.

    The output enters the trainable STATE perturbation encoder in its raw
    ``pert_dim`` space. The MLP's ``net`` parameter names stay stable across
    ordinary joint checkpoints.

    Args:
        esm_dim: Dimensionality of the input ESM2 embedding.
        hidden: Hidden layer width.
        pert_dim: Output dimensionality matching the STATE pert vector size.
    """

    def __init__(self, esm_dim: int, hidden: int, pert_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(esm_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, pert_dim),
        )

    def forward(self, esm: torch.Tensor) -> torch.Tensor:
        """Map ESM2 embeddings to raw pert vectors.

        Args:
            esm: Input tensor of shape ``(B, esm_dim)``.

        Returns:
            Raw pert vector tensor of shape ``(B, pert_dim)``.
        """
        return self.net(esm)


class Esm2PerturbationAdapter(nn.Module):
    """Map fixed per-gene ESM-2 vectors to STATE perturbation vectors."""

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

    def forward(self, gene: str) -> torch.Tensor:
        return self.forward_many([gene])[0]

    def forward_many(self, genes: list[str] | tuple[str, ...]) -> torch.Tensor:
        """Map multiple genes through one adapter call."""
        indices = torch.as_tensor(
            [self._gene_to_index[str(gene).upper()] for gene in genes],
            dtype=torch.long,
            device=self.esm_matrix.device,
        )
        return self.adapter(self.esm_matrix.index_select(0, indices))

    def has_embedding(self, gene: str) -> bool:
        """Return whether the adapter contains an ESM-2 vector for ``gene``."""
        return str(gene).upper() in self._gene_to_index
