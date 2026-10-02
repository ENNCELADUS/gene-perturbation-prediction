"""Prepared observed response targets: one array of cells, one offset per condition."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
    directory: Path,
    keys: Sequence[tuple[str, str]],
    lengths: Sequence[int],
    width: int,
    bags: Iterable[np.ndarray],
) -> Path:
    """Atomically write one ``[lengths[i], width]`` bag per key.

    ``bags`` is consumed in key order and written straight to disk, so only
    one bag needs to be resident at a time.
    """
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    lengths = [int(length) for length in lengths]
    if len(lengths) != len(keys):
        raise ValueError(f"{len(keys)} keys but {len(lengths)} bag lengths")
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tmp_dir = directory.parent / f".tmp-{directory.name}-{uuid.uuid4().hex}"
    tmp_dir.mkdir()
    try:
        cells = np.lib.format.open_memmap(
            tmp_dir / "target_cells.npy",
            mode="w+",
            dtype=np.float32,
            shape=(int(offsets[-1]), int(width)),
        )
        for key, start, length, bag in zip(
            keys, offsets[:-1], lengths, bags, strict=True
        ):
            if bag.shape != (length, width):
                raise ValueError(f"{key}: bag shape {bag.shape} != {(length, width)}")
            cells[start : start + length] = bag
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
