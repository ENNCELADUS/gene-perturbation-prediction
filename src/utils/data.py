"""Shared AnnData and pseudobulk helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import anndata as ad
import numpy as np
from scipy import sparse

from src.utils.metrics import normalize_condition


def load_adata(path: str | Path) -> ad.AnnData:
    """Load an AnnData object from disk."""
    return ad.read_h5ad(path)


def get_gene_names(adata: ad.AnnData) -> list[str]:
    """Return gene symbols from `var.gene_name` when present."""
    if "gene_name" in adata.var.columns:
        return [str(value) for value in adata.var["gene_name"].tolist()]
    return [str(value) for value in adata.var_names.tolist()]


def get_condition_splits(config: Mapping[str, object]) -> dict[str, list[str]]:
    """Read train/validation/test condition lists from config."""
    data_config = config.get("data_config", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("data_config must be a mapping")
    split = data_config.get("condition_split", {})
    if not isinstance(split, Mapping):
        raise ValueError("data_config.condition_split must be a mapping")
    return {
        "train": _normalize_conditions(split.get("train", [])),
        "validation": _normalize_conditions(
            split.get("validation", split.get("val", []))
        ),
        "test": _normalize_conditions(split.get("test", [])),
    }


def build_pseudobulk_matrices(config: Mapping[str, object]) -> dict[str, object]:
    """Create condition-level expression profiles from an AnnData config."""
    data_config = config.get("data_config", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("data_config must be a mapping")
    h5ad_path = data_config.get("h5ad_path")
    if not h5ad_path:
        raise ValueError("data_config.h5ad_path is required")

    adata = load_adata(str(h5ad_path))
    condition_key = str(data_config.get("condition_key", "condition"))
    splits = get_condition_splits(config)
    gene_names = get_gene_names(adata)
    matrices: dict[str, np.ndarray] = {}
    conditions: dict[str, list[str]] = {}
    for split_name, split_conditions in splits.items():
        matrix, valid_conditions = condition_profiles(
            adata=adata,
            conditions=split_conditions,
            condition_key=condition_key,
        )
        matrices[split_name] = matrix
        conditions[split_name] = valid_conditions
    return {
        "gene_names": gene_names,
        "gene_name_to_idx": {gene: idx for idx, gene in enumerate(gene_names)},
        "matrices": matrices,
        "conditions": conditions,
    }


def condition_profiles(
    adata: ad.AnnData,
    conditions: Sequence[str],
    condition_key: str = "condition",
) -> tuple[np.ndarray, list[str]]:
    """Compute mean expression profile per requested condition."""
    profiles = []
    valid_conditions = []
    obs_conditions = [
        normalize_condition(str(value)) for value in adata.obs[condition_key].tolist()
    ]
    obs_conditions = np.asarray(obs_conditions)
    for condition in conditions:
        normalized = normalize_condition(condition)
        mask = obs_conditions == normalized
        if not np.any(mask):
            continue
        matrix = adata.X[mask]
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        profiles.append(np.asarray(matrix).mean(axis=0))
        valid_conditions.append(normalized)

    if not profiles:
        return np.empty((0, adata.n_vars), dtype=np.float32), []
    return np.vstack(profiles).astype(np.float32), valid_conditions


def _normalize_conditions(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("condition split entries must be lists of condition strings")
    return [normalize_condition(str(condition)) for condition in value]
