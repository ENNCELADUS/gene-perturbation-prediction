"""data / embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Esm2EmbeddingTable:
    """Per-gene ESM2 embedding vectors keyed by upper-case symbol."""

    dim: int
    vectors_by_symbol: dict[str, np.ndarray]


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
