"""Prepared observed response targets: one array of cells, one offset per condition."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResponseTargetsCache:
    """Memory-mapped log-space target cells keyed by ``(ModelID, gene)``."""

    model_ids: tuple[str, ...]
    genes: tuple[str, ...]
    target_cells: np.ndarray
    offsets: np.ndarray

    @property
    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.model_ids, self.genes, strict=True))

    def target_bag(self, index: int) -> np.ndarray:
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return np.asarray(self.target_cells[start:stop])


def write_response_targets(
    directory: Path, keys: Sequence[tuple[str, str]], bags: Sequence[np.ndarray]
) -> Path:
    """Atomically write ``bags`` (one ``[cells, genes]`` array per key)."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    lengths = [len(bag) for bag in bags]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tmp_dir = directory.parent / f".tmp-{directory.name}-{uuid.uuid4().hex}"
    tmp_dir.mkdir()
    try:
        cells = np.lib.format.open_memmap(
            tmp_dir / "target_cells.npy",
            mode="w+",
            dtype=np.float32,
            shape=(int(offsets[-1]), int(bags[0].shape[1])),
        )
        for start, bag in zip(offsets[:-1], bags, strict=True):
            cells[start : start + len(bag)] = bag
        cells.flush()
        del cells
        np.save(tmp_dir / "offsets.npy", offsets)
        pd.DataFrame(
            {
                "model_id": [model_id for model_id, _ in keys],
                "gene": [gene for _, gene in keys],
                "n_cells": lengths,
            }
        ).to_parquet(tmp_dir / "conditions.parquet")
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(tmp_dir, directory)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return directory


def open_response_targets(directory: Path) -> ResponseTargetsCache:
    """Open a prepared response cache read-only."""
    directory = Path(directory)
    conditions = pd.read_parquet(directory / "conditions.parquet")
    return ResponseTargetsCache(
        model_ids=tuple(conditions["model_id"].astype(str)),
        genes=tuple(conditions["gene"].astype(str)),
        target_cells=np.load(directory / "target_cells.npy", mmap_mode="r"),
        offsets=np.load(directory / "offsets.npy"),
    )
