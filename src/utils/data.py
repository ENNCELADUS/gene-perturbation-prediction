"""Shared AnnData and pseudobulk helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import anndata as ad
import numpy as np
import yaml
from scipy import sparse

from src.utils.metrics import normalize_condition, parse_condition_genes


def load_adata(path: str | Path) -> ad.AnnData:
    """Load an AnnData object from disk."""
    return ad.read_h5ad(path)


def get_gene_names(adata: ad.AnnData) -> list[str]:
    """Return gene symbols from `var.gene_name` when present."""
    if "gene_name" in adata.var.columns:
        return [str(value) for value in adata.var["gene_name"].tolist()]
    return [str(value) for value in adata.var_names.tolist()]


def get_condition_splits(config: Mapping[str, object]) -> dict[str, list[str]]:
    """Read train/validation/test condition lists from config or split artifact."""
    data_config = config.get("data_config", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("data_config must be a mapping")
    split = data_config.get("condition_split", {})
    if not isinstance(split, Mapping):
        raise ValueError("data_config.condition_split must be a mapping")
    if not _has_split_values(split):
        split_path = data_config.get("condition_split_path")
        if split_path:
            path = Path(str(split_path))
            if not path.exists():
                raise ValueError(
                    "condition split artifact does not exist. "
                    "Run the prepare stage first: "
                    f"{path}"
                )
            split = load_condition_split(path)
    return {
        "train": _normalize_conditions(split.get("train", [])),
        "validation": _normalize_conditions(
            split.get("validation", split.get("val", []))
        ),
        "test": _normalize_conditions(split.get("test", [])),
    }


def load_condition_split(path: str | Path) -> dict[str, list[str]]:
    """Load a condition split artifact."""
    with Path(path).open() as handle:
        split = yaml.safe_load(handle)
    if not isinstance(split, Mapping):
        raise ValueError(f"Condition split artifact must be a mapping: {path}")
    return {
        "train": _normalize_conditions(split.get("train", [])),
        "validation": _normalize_conditions(
            split.get("validation", split.get("val", []))
        ),
        "test": _normalize_conditions(split.get("test", [])),
    }


def save_condition_split(split: Mapping[str, Sequence[str]], path: str | Path) -> None:
    """Persist a condition split artifact as YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": _normalize_conditions(split.get("train", [])),
        "validation": _normalize_conditions(
            split.get("validation", split.get("val", []))
        ),
        "test": _normalize_conditions(split.get("test", [])),
    }
    with output_path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def infer_condition_splits(
    adata: ad.AnnData,
    condition_key: str = "condition",
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[str]]:
    """Create a deterministic condition-level split from AnnData labels."""
    conditions = sorted(
        {
            normalize_condition(str(condition))
            for condition in adata.obs[condition_key].tolist()
        }
    )
    perturbed_conditions = [
        condition for condition in conditions if parse_condition_genes(condition)
    ]
    if not perturbed_conditions:
        raise ValueError(
            "Cannot infer split without non-control perturbation conditions"
        )

    rng = np.random.RandomState(seed)
    shuffled = [
        perturbed_conditions[index]
        for index in rng.permutation(len(perturbed_conditions))
    ]
    n_conditions = len(shuffled)
    validation_count = _fraction_count(n_conditions, validation_fraction)
    test_count = _fraction_count(n_conditions, test_fraction)
    while validation_count + test_count >= n_conditions and validation_count > 0:
        validation_count -= 1
    while validation_count + test_count >= n_conditions and test_count > 0:
        test_count -= 1
    train_count = n_conditions - validation_count - test_count
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count :],
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


def _has_split_values(split: Mapping[str, object]) -> bool:
    return any(bool(split.get(key)) for key in ("train", "validation", "val", "test"))


def _fraction_count(n_items: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    return max(1, int(round(n_items * fraction)))
