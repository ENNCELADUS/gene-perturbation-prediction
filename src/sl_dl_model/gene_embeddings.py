"""Load precomputed ESM2 per-gene embeddings and align to a gene universe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Esm2EmbeddingTable:
    """Per-gene ESM2 embedding vectors keyed by upper-case symbol."""

    dim: int
    vectors_by_symbol: dict[str, np.ndarray]


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


def align_esm2_to_universe(
    table: Esm2EmbeddingTable,
    symbols: np.ndarray,
    fallback_strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align ESM2 vectors to universe order with a coverage mask.

    Mirrors :func:`sl_benchmark_baseline.embeddings.align_to_universe`.

    Args:
        table: Precomputed ESM2 embedding table.
        symbols: Gene symbols in universe order (shape ``(n_gene,)``).
        fallback_strategy: ``"zero"`` or ``"global_mean"``.

    Returns:
        A tuple ``(embeddings, mask)`` where ``embeddings`` has shape
        ``(n_gene, dim)`` and ``mask`` is an integer array of shape
        ``(n_gene,)`` with 1 for covered genes and 0 otherwise.

    Raises:
        ValueError: If ``fallback_strategy`` is not recognized.
    """
    if fallback_strategy not in {"zero", "global_mean"}:
        raise ValueError(f"unknown fallback_strategy: {fallback_strategy}")
    covered = [
        table.vectors_by_symbol[str(s).upper()]
        for s in symbols
        if str(s).upper() in table.vectors_by_symbol
    ]
    if fallback_strategy == "global_mean" and covered:
        fallback = np.mean(np.vstack(covered), axis=0)
    else:
        fallback = np.zeros(table.dim, dtype=float)
    embeddings = np.zeros((len(symbols), table.dim), dtype=float)
    mask = np.zeros(len(symbols), dtype=int)
    for row, symbol in enumerate(symbols):
        key = str(symbol).upper()
        if key in table.vectors_by_symbol:
            embeddings[row] = table.vectors_by_symbol[key]
            mask[row] = 1
        else:
            embeddings[row] = fallback
    return embeddings, mask
