"""Load and validate precomputed ESM2 per-gene embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Esm2EmbeddingTable:
    """Per-gene ESM2 embedding vectors keyed by upper-case symbol."""

    dim: int
    vectors_by_symbol: dict[str, np.ndarray]


class PertAdapter(nn.Module):
    """Map an ESM2 gene embedding to a STATE raw pert vector.

    The adapter output is a synthetic *raw* perturbation vector in STATE's
    ``pert_dim`` space; it is fed into the frozen checkpoint's own
    ``pert_encoder`` (it does not replace that encoder). Its width must match
    the loaded checkpoint's ``pert_dim``.

    The layers deliberately live under the ``net`` submodule rather than on this
    module directly. Existing ESM2 AIVC checkpoints — including the frozen exp05
    checkpoint that ``load_bridge_a_context`` reads — store their parameters as
    ``...adapter.net.{0,2}.{weight,bias}``, and the load is strict. Inlining an
    ``nn.Sequential`` here would drop the ``net`` level and break every one of
    them with missing/unexpected keys.

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


def require_complete_esm_coverage(
    canonical_genes: list[str], table: Esm2EmbeddingTable
) -> None:
    """Require complete canonical coverage in exact relative manifest order."""
    canonical = [str(gene).upper() for gene in canonical_genes]
    available = [str(gene).upper() for gene in table.vectors_by_symbol]
    canonical_set = set(canonical)
    available_set = set(available)
    matched = sum(gene in available_set for gene in canonical)
    available_canonical = [gene for gene in available if gene in canonical_set]
    if available_canonical != canonical:
        raise ValueError(
            "ESM-2 canonical coverage/order mismatch: "
            f"{matched}/{len(canonical)} genes resolved in exact order"
        )


def load_esm2_embeddings(npz: Path) -> Esm2EmbeddingTable:
    """Load a precomputed ESM2 ``.npz``; drop unresolved genes.

    Args:
        npz: Path to the ``.npz`` file with keys ``symbols``, ``vectors``,
            and ``resolved``.

    Returns:
        An :class:`Esm2EmbeddingTable` containing only resolved genes.
    """
    with np.load(npz, allow_pickle=True) as payload:
        symbols = np.asarray(payload["symbols"], dtype=object)
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
        resolved = np.asarray(payload["resolved"], dtype=bool)
    table: dict[str, np.ndarray] = {}
    for symbol, vector, ok in zip(symbols, vectors, resolved, strict=True):
        if bool(ok):
            table[str(symbol).upper()] = vector
    return Esm2EmbeddingTable(dim=int(vectors.shape[1]), vectors_by_symbol=table)
