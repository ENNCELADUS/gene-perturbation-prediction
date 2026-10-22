"""Provenance for generated ESM-2 embedding tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np


SCHEMA_VERSION = "esm2-embedding-provenance-v1"


def build_embedding_artifact_record(path: str | Path) -> dict[str, object]:
    """Describe the arrays stored in an ESM-2 NPZ."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=True) as payload:
            if set(payload.files) != {"symbols", "vectors", "resolved"}:
                raise ValueError("ESM2 NPZ must contain symbols, vectors, and resolved")
            raw_symbols = payload["symbols"]
            vectors = payload["vectors"]
            resolved = payload["resolved"]
    except Exception as exc:
        raise ValueError(f"invalid ESM2 NPZ {path}: {exc}") from exc
    if raw_symbols.ndim != 1:
        raise ValueError("ESM2 symbols must be one-dimensional")
    symbols = tuple(str(value) for value in raw_symbols.tolist())
    if any(not symbol or symbol != symbol.upper() for symbol in symbols):
        raise ValueError("ESM2 symbols must be nonempty uppercase strings")
    if len(set(symbols)) != len(symbols):
        raise ValueError("ESM2 symbols must be unique")
    if vectors.ndim != 2 or vectors.shape[0] != len(symbols):
        raise ValueError("ESM2 vectors must be rows aligned to symbols")
    if resolved.dtype != np.dtype(bool) or resolved.shape != (len(symbols),):
        raise ValueError("ESM2 resolved must be a bool vector aligned to symbols")
    if not np.isfinite(vectors).all():
        raise ValueError("ESM2 vectors must be finite")
    resolved_symbols = [
        symbol
        for symbol, is_resolved in zip(symbols, resolved, strict=True)
        if bool(is_resolved)
    ]
    return {
        "path": str(path),
        "symbols": list(symbols),
        "row_count": len(symbols),
        "vector_width": int(vectors.shape[1]),
        "vectors_dtype": vectors.dtype.str,
        "resolved_dtype": resolved.dtype.str,
        "resolved_count": int(resolved.sum()),
        "resolved_symbols": resolved_symbols,
    }
