"""Prepared observed response targets: explicit ordered writes and pure reads."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd


# Array layout version; exact HVG order is required separately.
_SCHEMA_VERSION: Final[int] = 1

_ARRAY_FILENAMES: Final[tuple[str, ...]] = (
    "target_cells.npy",
    "offsets.npy",
    "genes.npy",
)


@dataclass(frozen=True)
class ResponseTargetsCache:
    """Memory-mapped prepared response targets with canonical condition IDs."""

    model_ids: tuple[str, ...]
    genes: tuple[str, ...]
    target_cells: np.ndarray
    offsets: np.ndarray
    metadata: pd.DataFrame

    @property
    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.model_ids, self.genes, strict=True))

    def target_bag(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return np.asarray(self.target_cells[start:stop])


def open_response_targets_cache(
    cache_dir: Path, *, expected_hvg_order: Sequence[str]
) -> ResponseTargetsCache:
    """Open a prepared response cache without fingerprinting or rebuilding it."""
    final_dir = Path(cache_dir) / "response_targets"
    required = tuple(
        final_dir / filename
        for filename in (*_ARRAY_FILENAMES, "metadata.parquet", "manifest.json")
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing prepared response cache file {missing[0]}; run "
            "`hpc/run.sh prepare <config>`"
        )
    expected_order = tuple(str(gene).strip().upper() for gene in expected_hvg_order)
    if not expected_order or len(set(expected_order)) != len(expected_order):
        raise ValueError(
            "expected response HVG order must be non-empty and unique"
            "; run `hpc/run.sh prepare <config>`"
        )
    try:
        manifest = json.loads((final_dir / "manifest.json").read_text())
        legacy_genes = np.load(final_dir / "genes.npy", allow_pickle=True)
        offsets = np.load(final_dir / "offsets.npy", mmap_mode="r")
        target_cells = np.load(final_dir / "target_cells.npy", mmap_mode="r")
        metadata = pd.read_parquet(final_dir / "metadata.parquet").reset_index(
            drop=True
        )
    except (OSError, EOFError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unable to read prepared response cache {final_dir}: {exc}"
            "; run `hpc/run.sh prepare <config>`"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError(
            f"response cache {final_dir / 'manifest.json'} has unsupported schema"
            "; run `hpc/run.sh prepare <config>`"
        )
    n_bags = manifest.get("n_bags")
    if not isinstance(n_bags, int) or isinstance(n_bags, bool) or n_bags <= 0:
        raise ValueError(
            f"response cache {final_dir} must contain at least one bag"
            "; run `hpc/run.sh prepare <config>`"
        )
    if manifest.get("target_dim") != len(expected_order):
        raise ValueError(
            f"response cache {final_dir} target width does not match "
            "prepared HVG order"
            "; run `hpc/run.sh prepare <config>`"
        )
    recorded_order = manifest.get("hvg_order")
    if not isinstance(recorded_order, list) or tuple(recorded_order) != expected_order:
        raise ValueError(
            f"response cache {final_dir / 'manifest.json'} requires matching ordered "
            "hvg_order; run `hpc/run.sh prepare <config>`"
        )
    if (
        target_cells.ndim != 2
        or target_cells.shape[1] != len(expected_order)
        or target_cells.dtype != np.dtype(np.float32)
    ):
        raise ValueError(
            f"response targets {final_dir / 'target_cells.npy'} must be float32 "
            f"[cells, {len(expected_order)}]"
            "; run `hpc/run.sh prepare <config>`"
        )
    if offsets.shape != (n_bags + 1,) or offsets.dtype != np.dtype(np.int64):
        raise ValueError(
            f"response offsets {final_dir / 'offsets.npy'} must be "
            f"int64 [{n_bags + 1}]"
            "; run `hpc/run.sh prepare <config>`"
        )
    offsets_array = np.asarray(offsets)
    if (
        int(offsets_array[0]) != 0
        or int(offsets_array[-1]) != target_cells.shape[0]
        or bool((np.diff(offsets_array) <= 0).any())
    ):
        raise ValueError(
            f"response offsets {final_dir / 'offsets.npy'} are invalid"
            "; run `hpc/run.sh prepare <config>`"
        )
    required_columns = {"perturbation_gene", "model_id", "n_cells"}
    missing_columns = sorted(required_columns - set(metadata.columns))
    if missing_columns:
        raise ValueError(
            f"response metadata {final_dir / 'metadata.parquet'} is missing "
            f"columns {missing_columns}"
            "; run `hpc/run.sh prepare <config>`"
        )
    if len(metadata) != n_bags or legacy_genes.shape != (n_bags,):
        raise ValueError(
            f"response cache {final_dir} bag counts disagree"
            "; run `hpc/run.sh prepare <config>`"
        )
    model_ids = tuple(metadata["model_id"].astype(str))
    genes = tuple(metadata["perturbation_gene"].astype(str).str.strip().str.upper())
    if any(not value for value in (*model_ids, *genes)):
        raise ValueError(
            f"response metadata {final_dir} contains empty identifiers"
            "; run `hpc/run.sh prepare <config>`"
        )
    keys = tuple(zip(model_ids, genes, strict=True))
    if len(set(keys)) != len(keys):
        raise ValueError(
            f"response metadata {final_dir} contains duplicate conditions"
            "; run `hpc/run.sh prepare <config>`"
        )
    expected_legacy = tuple(
        f"{gene}@{model_id}" for model_id, gene in zip(model_ids, genes, strict=True)
    )
    if tuple(legacy_genes.astype(str)) != expected_legacy:
        raise ValueError(
            f"legacy response keys {final_dir / 'genes.npy'} disagree with metadata"
            "; run `hpc/run.sh prepare <config>`"
        )
    lengths = np.diff(offsets_array)
    try:
        recorded_lengths = metadata["n_cells"].to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"response metadata {final_dir} has invalid n_cells"
            "; run `hpc/run.sh prepare <config>`"
        ) from exc
    if not np.array_equal(recorded_lengths, lengths):
        raise ValueError(
            f"response cache {final_dir} n_cells disagrees with offsets"
            "; run `hpc/run.sh prepare <config>`"
        )
    return ResponseTargetsCache(
        model_ids=model_ids,
        genes=genes,
        target_cells=target_cells,
        offsets=offsets,
        metadata=metadata,
    )


def write_response_targets_cache(
    cache_dir: Path,
    *,
    genes: Sequence[str],
    target_bags: Sequence[np.ndarray],
    metadata: pd.DataFrame,
    hvg_order: Sequence[str],
) -> Path:
    """Write aligned target arrays and their actual assembled gene order."""
    hvg_order = tuple(str(gene).strip().upper() for gene in hvg_order)
    if not hvg_order or len(set(hvg_order)) != len(hvg_order):
        raise ValueError("response target hvg_order must contain unique genes")
    n_bags = len(genes)
    if len(target_bags) != n_bags or len(metadata) != n_bags:
        raise ValueError(
            f"genes ({n_bags}), target_bags ({len(target_bags)}), and metadata "
            f"({len(metadata)}) must all have the same length"
        )
    if not n_bags or any(
        bag.ndim != 2 or len(bag) == 0 or not np.isfinite(bag).all()
        for bag in target_bags
    ):
        raise ValueError("response target bags must be nonempty finite matrices")
    required = {"model_id", "perturbation_gene", "n_cells"}
    if not required.issubset(metadata.columns):
        raise ValueError("response target metadata lacks identifiers or n_cells")
    if metadata.duplicated(["model_id", "perturbation_gene"]).any():
        raise ValueError("response target metadata contains duplicate conditions")
    for gene, bag, row in zip(genes, target_bags, metadata.itertuples(), strict=True):
        if gene != f"{row.perturbation_gene}@{row.model_id}" or row.n_cells != len(bag):
            raise ValueError("response target metadata differs from assembled bags")
    widths = {int(bag.shape[1]) for bag in target_bags} if target_bags else set()
    if len(widths) > 1:
        raise ValueError(f"every bag must share one target width; got {widths}")
    target_dim = widths.pop() if widths else 0
    if target_dim != len(hvg_order):
        raise ValueError("response target width differs from assembled HVG order")
    concatenated = (
        np.concatenate(
            [np.asarray(bag, dtype=np.float32) for bag in target_bags], axis=0
        )
        if target_bags
        else np.empty((0, target_dim), dtype=np.float32)
    )
    lengths = [int(bag.shape[0]) for bag in target_bags]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_dir / f".tmp-response-targets-{uuid.uuid4().hex}"
    tmp_dir.mkdir()
    final_dir = cache_dir / "response_targets"
    try:
        np.save(tmp_dir / "target_cells.npy", concatenated)
        np.save(tmp_dir / "offsets.npy", offsets)
        np.save(
            tmp_dir / "genes.npy", np.asarray(genes, dtype=object), allow_pickle=True
        )
        metadata.to_parquet(tmp_dir / "metadata.parquet")
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "hvg_order": list(hvg_order),
            "n_bags": n_bags,
            "target_dim": target_dim,
        }
        (tmp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return final_dir / "manifest.json"
