"""Artifact paths and IO helpers for exp08b."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from sl_dl_model.exp08b_config import Exp08bConfig


def fold_artifact_dir(config: Exp08bConfig, split_type: str, fold_id: int) -> Path:
    """Return the Step 1 artifact directory for one fold."""
    return (
        Path(config.output_dir)
        / config.step1_artifacts_subdir
        / f"{split_type}_fold{fold_id}"
    )


def embedding_cache_path(config: Exp08bConfig, split_type: str, fold_id: int) -> Path:
    """Return the cached e_hat NPZ path for one fold."""
    return (
        fold_artifact_dir(config, split_type, fold_id)
        / config.generator_embedding_filename
    )


def generator_manifest_path(
    config: Exp08bConfig, split_type: str, fold_id: int
) -> Path:
    """Return the Step 1 generator manifest path for one fold."""
    return (
        fold_artifact_dir(config, split_type, fold_id)
        / config.generator_manifest_filename
    )


def generator_weights_path(config: Exp08bConfig, split_type: str, fold_id: int) -> Path:
    """Return the frozen generator weights path for one fold."""
    return (
        fold_artifact_dir(config, split_type, fold_id)
        / config.generator_weights_filename
    )


def generator_monitor_path(config: Exp08bConfig, split_type: str, fold_id: int) -> Path:
    """Return the Step 1 monitor CSV path for one fold."""
    return (
        fold_artifact_dir(config, split_type, fold_id)
        / config.generator_monitor_filename
    )


def step2_output_dir(config: Exp08bConfig) -> Path:
    """Return the Step 2 official-metric output directory."""
    return Path(config.output_dir) / config.step2_results_subdir


def save_embedding_cache(
    path: Path,
    *,
    symbols: np.ndarray,
    embeddings: np.ndarray,
    coverage_mask: np.ndarray,
    embedding_method: str,
) -> None:
    """Write a fold-local cached embedding table atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            symbols=np.asarray(symbols, dtype=object),
            embeddings=np.asarray(embeddings, dtype=np.float32),
            coverage_mask=np.asarray(coverage_mask, dtype=np.int64),
            embedding_method=np.asarray(embedding_method, dtype=object),
        )
    os.replace(tmp, path)


def load_embedding_cache(path: Path) -> dict[str, Any]:
    """Load a fold-local cached embedding table."""
    with np.load(path, allow_pickle=True) as data:
        method = data["embedding_method"]
        return {
            "symbols": data["symbols"].astype(object),
            "embeddings": data["embeddings"].astype(np.float32),
            "coverage_mask": data["coverage_mask"].astype(np.int64),
            "embedding_method": str(method.item() if method.shape == () else method[0]),
        }


def write_generator_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write a generator manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def load_generator_manifest(path: Path) -> dict[str, Any]:
    """Read a generator manifest."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    return payload
