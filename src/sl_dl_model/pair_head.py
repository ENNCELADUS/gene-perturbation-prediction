"""Swap-invariant pair scorer over per-gene embeddings + GeneEffect features."""

from __future__ import annotations

import torch
from torch import nn


class SymmetricPairHead(nn.Module):
    """Score a gene pair from swap-invariant transcript + GeneEffect features.

    The transcript block is constructed as [e_a+e_b, |e_a-e_b|, e_a*e_b],
    which is invariant under swapping (e_a, e_b). When include_coverage_flag
    is True, swap-invariant coverage columns (min, max) are appended.

    Args:
        emb_dim: Dimensionality of each per-gene embedding vector.
        geneeffect_dim: Width of the GeneEffect feature block (default 5).
        hidden: Hidden layer widths for the MLP scorer.
        include_coverage_flag: If True, expect cov_a and cov_b in forward.
    """

    def __init__(
        self,
        emb_dim: int,
        geneeffect_dim: int = 5,
        hidden: tuple[int, ...] = (256, 64),
        include_coverage_flag: bool = False,
    ) -> None:
        super().__init__()
        self.include_coverage_flag = include_coverage_flag
        transcript_dim = 3 * emb_dim
        cov_dim = 2 if include_coverage_flag else 0
        in_dim = transcript_dim + geneeffect_dim + cov_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.GELU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        e_a: torch.Tensor,
        e_b: torch.Tensor,
        ge_features: torch.Tensor,
        cov_a: torch.Tensor | None = None,
        cov_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute a scalar SL logit for each pair in the batch.

        Args:
            e_a: Per-gene embeddings for gene A, shape (B, emb_dim).
            e_b: Per-gene embeddings for gene B, shape (B, emb_dim).
            ge_features: GeneEffect features, shape (B, geneeffect_dim).
                Must already be swap-invariant (caller's responsibility).
            cov_a: Coverage flag for gene A, shape (B,). Required when
                include_coverage_flag is True.
            cov_b: Coverage flag for gene B, shape (B,). Required when
                include_coverage_flag is True.

        Returns:
            Logits of shape (B,).

        Raises:
            ValueError: If include_coverage_flag is True but cov_a or cov_b
                is None.
        """
        blocks = [e_a + e_b, (e_a - e_b).abs(), e_a * e_b, ge_features]
        if self.include_coverage_flag:
            if cov_a is None or cov_b is None:
                raise ValueError(
                    "cov_a and cov_b are required when include_coverage_flag=True"
                )
            cov_min = torch.minimum(cov_a, cov_b).unsqueeze(-1)
            cov_max = torch.maximum(cov_a, cov_b).unsqueeze(-1)
            blocks += [cov_min, cov_max]
        features = torch.cat(blocks, dim=-1)
        return self.net(features).squeeze(-1)
