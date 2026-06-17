"""Per-gene transcript embeddings pooled from an exp03 cell-bags NPZ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GeneEmbeddingTable:
    """Mean-pooled per-gene embedding vectors keyed by upper-case symbol.

    Attributes:
        dim: Observed embedding dimension (``cell_delta_pcs.shape[1]``).
        vectors_by_symbol: Pooled vector per covered gene symbol.
        feature_set: The exp03 ``feature_set`` tag recorded in the source NPZ
            (e.g. ``"single_cell_pc_delta"`` or ``"single_cell_scvi_delta"``),
            or ``None`` if the NPZ predates that key. This is the actual data
            contract — the CLI ``--embedding-method`` label is user-supplied and
            unvalidated, so callers should reconcile the two.
    """

    dim: int
    vectors_by_symbol: dict[str, np.ndarray]
    feature_set: str | None = None


_REQUIRED_NPZ_KEYS: tuple[str, ...] = (
    "cell_delta_pcs",
    "bag_offsets",
    "perturbation_gene",
)


def load_gene_embeddings(bags_npz: Path) -> GeneEmbeddingTable:
    """Mean-pool each gene's delta-cell bag into one per-gene vector.

    Security note: the NPZ is loaded with ``allow_pickle=True`` because exp03
    writes ``perturbation_gene`` as a NumPy object array, which cannot be
    deserialized otherwise. The ``bags_npz`` path is therefore trusted-local
    only: it must be an artifact produced by this project's exp03
    ``build-cell-bags`` step, not an externally supplied or network-fetched
    file. Required keys are validated before use so a malformed or wrong-format
    NPZ fails loudly rather than silently.

    Args:
        bags_npz: Path to a trusted exp03 cell-bags NPZ with ``cell_delta_pcs``,
            ``bag_offsets``, and ``perturbation_gene`` keys (and an optional
            ``feature_set`` tag).

    Returns:
        A :class:`GeneEmbeddingTable` over covered gene symbols.

    Raises:
        ValueError: If any required key is missing from the NPZ.
    """
    with np.load(bags_npz, allow_pickle=True) as payload:
        missing = [key for key in _REQUIRED_NPZ_KEYS if key not in payload.files]
        if missing:
            raise ValueError(
                f"bags NPZ {bags_npz} is missing required keys: {missing}; "
                "expected an exp03 build-cell-bags artifact"
            )
        cells = np.asarray(payload["cell_delta_pcs"], dtype=float)
        offsets = np.asarray(payload["bag_offsets"], dtype=int)
        genes = np.asarray(payload["perturbation_gene"], dtype=object)
        feature_set = (
            str(payload["feature_set"]) if "feature_set" in payload.files else None
        )
    vectors: dict[str, np.ndarray] = {}
    for index, symbol in enumerate(genes):
        start, stop = offsets[index], offsets[index + 1]
        if stop <= start:
            continue
        vectors[str(symbol).upper()] = cells[start:stop].mean(axis=0)
    return GeneEmbeddingTable(
        dim=cells.shape[1],
        vectors_by_symbol=vectors,
        feature_set=feature_set,
    )


def align_to_universe(
    table: GeneEmbeddingTable,
    symbols: np.ndarray,
    fallback_strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align pooled embeddings to a universe order with a coverage mask.

    The ``global_mean`` fallback is computed over all covered genes present in
    ``symbols``. Because ``symbols`` is always the full candidate universe (built
    before any fold split) and gwps coverage is fixed (not fold-dependent), this
    fallback is stable across folds and is label-free: no SL ``D`` label touches
    the embedding or the mean.

    Args:
        table: Pooled per-gene embeddings.
        symbols: Universe gene symbols in canonical order, shape ``(n_gene,)``.
        fallback_strategy: ``"zero"`` or ``"global_mean"`` for uncovered genes.

    Returns:
        ``(embeddings (n_gene, dim), coverage_mask (n_gene,))``.

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
