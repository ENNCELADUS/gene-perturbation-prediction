"""Per-line pseudo-bulk: raw UMI summed over every basal cell, CPM over the whole
library, log1p. The single-cell side of the context prior's shared space."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.basal import align_columns, require_raw_counts

PSEUDOBULK_DIR = "pseudobulk"
TRANSFORM = "sum_umi_cpm_log1p"


def pseudobulk(matrix, symbols: Sequence[str], genes: Sequence[str]) -> np.ndarray:
    """``log1p(1e6 * gene total / library total)`` per gene of ``genes``; NaN where
    the source lacks the gene. Columns sharing a symbol are summed."""
    require_raw_counts(matrix, "pseudo-bulk source")
    library = float(matrix.sum())
    if not library > 0:
        raise ValueError("pseudo-bulk source has no counts")
    counts, available = align_columns(matrix, symbols, genes)
    totals = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel()
    values = np.log1p(totals * 1e6 / library)
    values[~available] = np.nan
    return values.astype(np.float32)


def read_pseudobulk(prepared_root: Path) -> pd.DataFrame:
    root = Path(prepared_root) / PSEUDOBULK_DIR
    manifest = json.loads((root / "manifest.json").read_text())
    frame = pd.read_parquet(root / "pseudobulk.parquet")
    if (
        list(frame.columns) != manifest["genes"]
        or list(frame.index) != manifest["lines"]
    ):
        raise ValueError(f"{root}: pseudo-bulk does not match its manifest")
    return frame
