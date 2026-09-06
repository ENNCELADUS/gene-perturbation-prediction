"""data / gene order."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
import anndata as ad
import numpy as np


def sha256_strings(values: np.ndarray) -> str:
    """Hash an ordered string array without ambiguous concatenation."""
    digest = hashlib.sha256()
    for value in np.asarray(values).astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def encode_batch_labels(
    labels: np.ndarray | None,
    lookup: dict[str, int],
    fallback_index: int = 0,
) -> np.ndarray | None:
    """Encode batch labels for STATE checkpoints."""
    if labels is None:
        return None
    return np.asarray(
        [lookup.get(str(label), int(fallback_index)) for label in labels],
        dtype=np.int64,
    )


def resolve_state_gene_order(
    adata: ad.AnnData,
    model_dir: Path,
    symbol_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve expression columns to the exact STATE checkpoint gene order."""
    with (model_dir / "var_dims.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    checkpoint_names = np.asarray(payload["gene_names"], dtype=object).astype(str)
    source_names = adata.var[symbol_col].astype(str).to_numpy()
    positions: dict[str, int] = {}
    duplicates: set[str] = set()
    for index, symbol in enumerate(source_names):
        if symbol in positions:
            duplicates.add(symbol)
        else:
            positions[symbol] = index
    duplicate_matches = sorted(set(checkpoint_names).intersection(duplicates))
    missing = [name for name in checkpoint_names if name not in positions]
    if missing or duplicate_matches:
        matched = len(checkpoint_names) - len(missing) - len(duplicate_matches)
        raise ValueError(
            f"STATE expression alignment matched {matched}/{len(checkpoint_names)}; "
            f"missing={missing[:10]}, duplicate_matches={duplicate_matches[:10]}"
        )
    indices = np.asarray([positions[name] for name in checkpoint_names], dtype=np.int64)
    return indices, checkpoint_names.astype(object)
