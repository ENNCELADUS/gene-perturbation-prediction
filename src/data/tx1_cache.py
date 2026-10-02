"""Tx1-3B basal embedding cache: ``<ModelID>/{embeddings,hvg}.npy, obs.parquet``.

The on-disk format is fixed so cached lines are never re-encoded. Tx1 reads
raw UMI counts; ``hvg.npy`` holds raw counts in STATE's HVG order and exists
only as part of that format (prepared inputs recompute log-space HVG
expression from the raw source). The Tx1-3B forward pass is injected as an
``EncoderFn`` so encoding is testable without a GPU.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import shutil
import uuid
from pathlib import Path
from typing import Callable, Final, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.basal import (
    align_columns,
    assert_tx1_input_contract,
    require_raw_counts,
    symbol_column,
)

_LOGGER = logging.getLogger(__name__)

EncoderFn = Callable[[ad.AnnData], np.ndarray]

#: Tx1-3B cell-embedding width.
EMBEDDING_WIDTH: Final[int] = 2560
_FILES: Final[tuple[str, ...]] = ("embeddings.npy", "hvg.npy", "obs.parquet")


def load_hvg_gene_order(state_model_dir: Path) -> np.ndarray:
    """STATE's HVG gene order from the released checkpoint's ``var_dims.pkl``."""
    with (Path(state_model_dir) / "var_dims.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    return np.asarray(payload["gene_names"], dtype=object).astype(str)


def load_line_cache(
    cache_dir: Path, model_id: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Memory-mapped ``(embeddings, hvg, obs)`` of one cached line."""
    line_dir = Path(cache_dir) / model_id
    embeddings = np.load(line_dir / "embeddings.npy", mmap_mode="r")
    hvg_matrix = np.load(line_dir / "hvg.npy", mmap_mode="r")
    obs = pd.read_parquet(line_dir / "obs.parquet")
    return embeddings, hvg_matrix, obs


def missing_lines(cache_dir: Path, model_ids: Sequence[str]) -> list[str]:
    """ModelIDs whose cache directory lacks any of the three files."""
    return [
        str(model_id)
        for model_id in model_ids
        if not all(
            (Path(cache_dir) / str(model_id) / name).is_file() for name in _FILES
        )
    ]


def write_line_cache(
    cache_dir: Path,
    model_id: str,
    embeddings: np.ndarray,
    hvg_matrix: np.ndarray,
    obs: pd.DataFrame,
) -> Path:
    """Atomically write one line's three cache files; returns the line dir."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (len(obs), EMBEDDING_WIDTH):
        raise ValueError(
            f"{model_id}: embeddings {embeddings.shape} != ({len(obs)}, "
            f"{EMBEDDING_WIDTH})"
        )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_dir / f".tmp-{model_id}-{uuid.uuid4().hex}"
    tmp_dir.mkdir()
    final_dir = cache_dir / model_id
    try:
        np.save(tmp_dir / "embeddings.npy", embeddings)
        np.save(tmp_dir / "hvg.npy", np.asarray(hvg_matrix, dtype=np.float32))
        obs.to_parquet(tmp_dir / "obs.parquet")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return final_dir


def read_registry_source(
    source_path: Path, *, model_id: str, var_ensembl_col: str
) -> ad.AnnData:
    """Every basal cell of one registered raw-UMI h5ad, Tx1-ready, in memory."""
    adata = ad.read_h5ad(source_path)
    observed = set(adata.obs["model_id"].astype(str))
    if observed != {model_id}:
        raise ValueError(f"{source_path}: obs model_id values {sorted(observed)}")
    if var_ensembl_col in adata.var.columns:
        ensembl_ids = adata.var[var_ensembl_col].astype(str).to_numpy()
    elif adata.var.index.name == var_ensembl_col:
        ensembl_ids = adata.var.index.astype(str).to_numpy()
    else:
        raise ValueError(f"{source_path}: var has no {var_ensembl_col!r} column/index")
    require_raw_counts(adata.X, f"{model_id} basal source {source_path}")
    adata.var.index = ensembl_ids
    adata.var["ensembl_id"] = ensembl_ids
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = model_id
    adata.X = (
        adata.X.astype(np.float32)
        if sparse.issparse(adata.X)
        else np.asarray(adata.X, dtype=np.float32)
    )
    assert_tx1_input_contract(adata)
    return adata


def tx1_cell_indices(
    cell_ids: Sequence[str], model_id: str, max_cells: int | None, seed: int
) -> np.ndarray:
    """Cells Tx1 encodes: lowest ``sha256(seed, ModelID, cell)``, in source order."""
    if max_cells is None or max_cells >= len(cell_ids):
        return np.arange(len(cell_ids), dtype=np.int64)
    ranked = sorted(
        range(len(cell_ids)),
        key=lambda index: hashlib.sha256(
            f"{seed}\0{model_id}\0{cell_ids[index]}".encode()
        ).digest(),
    )[:max_cells]
    return np.asarray(sorted(ranked), dtype=np.int64)


def encode_lines(
    registry: pd.DataFrame,
    cache_dir: Path,
    *,
    encoder: EncoderFn,
    hvg_order: Sequence[str],
    var_ensembl_col: str,
    hvg_gene_symbol_col: str,
    max_cells_per_line: int | None,
    seed: int = 0,
) -> list[str]:
    """Encode and cache every registry line (index ModelID, ``source_path``)."""
    encoded = []
    for model_id, row in registry.iterrows():
        model_id = str(model_id)
        source = read_registry_source(
            Path(str(row["source_path"])),
            model_id=model_id,
            var_ensembl_col=var_ensembl_col,
        )
        cells = tx1_cell_indices(
            source.obs_names.astype(str), model_id, max_cells_per_line, seed
        )
        adata = source[cells].copy()
        symbols = adata.var[symbol_column(adata.var, hvg_gene_symbol_col)]
        hvg, _ = align_columns(adata.X, symbols.astype(str), hvg_order)
        _LOGGER.info("%s: encoding %d basal cells with Tx1-3B", model_id, adata.n_obs)
        write_line_cache(cache_dir, model_id, encoder(adata), hvg.toarray(), adata.obs)
        encoded.append(model_id)
    return encoded


__all__ = [
    "EMBEDDING_WIDTH",
    "EncoderFn",
    "encode_lines",
    "load_hvg_gene_order",
    "load_line_cache",
    "missing_lines",
    "read_registry_source",
    "tx1_cell_indices",
    "write_line_cache",
]
