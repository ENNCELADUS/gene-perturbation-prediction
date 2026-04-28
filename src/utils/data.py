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
    if isinstance(split.get("conditions"), Mapping):
        split = split["conditions"]
    return _normalize_split(split)


def save_condition_split(split: Mapping[str, object], path: str | Path) -> None:
    """Persist a condition split artifact as YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _normalize_split_artifact(split)
    with output_path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def infer_gene_heldout_condition_split(
    adata: ad.AnnData,
    condition_key: str = "condition",
    train_gene_fraction: float = 0.7,
    validation_gene_fraction: float = 0.1,
    test_gene_fraction: float = 0.2,
    min_cells_per_condition: int = 1,
) -> dict[str, object]:
    """Create a deterministic gene-held-out Norman condition split artifact."""
    condition_counts = _condition_counts(adata, condition_key)
    conditions = sorted(
        condition
        for condition, count in condition_counts.items()
        if count >= min_cells_per_condition and parse_condition_genes(condition)
    )
    if not conditions:
        raise ValueError(
            "Cannot infer split without non-control perturbation conditions"
        )

    genes = sorted(
        {
            gene
            for condition in conditions
            for gene in parse_condition_genes(condition)
        }
    )
    gene_split = _split_genes_by_fraction(
        genes,
        train_gene_fraction=train_gene_fraction,
        validation_gene_fraction=validation_gene_fraction,
        test_gene_fraction=test_gene_fraction,
    )
    condition_split = _assign_conditions_by_gene_split(conditions, gene_split)
    return {
        "strategy": "gene_heldout",
        "genes": gene_split,
        "conditions": condition_split,
        "stats": _split_stats(
            gene_split=gene_split,
            condition_split=condition_split,
            target_fractions={
                "train": train_gene_fraction,
                "validation": validation_gene_fraction,
                "test": test_gene_fraction,
            },
        ),
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


def _normalize_split(split: Mapping[str, object]) -> dict[str, list[str]]:
    return {
        "train": _normalize_conditions(split.get("train", [])),
        "validation": _normalize_conditions(
            split.get("validation", split.get("val", []))
        ),
        "test": _normalize_conditions(split.get("test", [])),
    }


def _normalize_split_artifact(split: Mapping[str, object]) -> dict[str, object]:
    if isinstance(split.get("conditions"), Mapping):
        payload = dict(split)
        payload["conditions"] = _normalize_split(split["conditions"])
        if isinstance(split.get("genes"), Mapping):
            payload["genes"] = {
                "train": _string_list(split["genes"].get("train", [])),
                "validation": _string_list(
                    split["genes"].get("validation", split["genes"].get("val", []))
                ),
                "test": _string_list(split["genes"].get("test", [])),
            }
        return payload
    return _normalize_split(split)


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("split gene entries must be lists of strings")
    return [str(item) for item in value]


def _has_split_values(split: Mapping[str, object]) -> bool:
    return any(bool(split.get(key)) for key in ("train", "validation", "val", "test"))


def _fraction_count(n_items: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    return max(1, int(round(n_items * fraction)))


def _condition_counts(adata: ad.AnnData, condition_key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for condition in adata.obs[condition_key].tolist():
        normalized = normalize_condition(str(condition))
        counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def _split_genes_by_fraction(
    genes: Sequence[str],
    train_gene_fraction: float,
    validation_gene_fraction: float,
    test_gene_fraction: float,
) -> dict[str, list[str]]:
    if not genes:
        raise ValueError("Cannot split empty gene list")
    n_genes = len(genes)
    validation_count = _fraction_count(n_genes, validation_gene_fraction)
    test_count = _fraction_count(n_genes, test_gene_fraction)
    while validation_count + test_count >= n_genes and validation_count > 0:
        validation_count -= 1
    while validation_count + test_count >= n_genes and test_count > 0:
        test_count -= 1
    train_count = n_genes - validation_count - test_count
    return {
        "train": list(genes[:train_count]),
        "validation": list(genes[train_count : train_count + validation_count]),
        "test": list(genes[train_count + validation_count :]),
    }


def _assign_conditions_by_gene_split(
    conditions: Sequence[str],
    gene_split: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    validation_genes = set(gene_split["validation"])
    test_genes = set(gene_split["test"])
    split = {"train": [], "validation": [], "test": []}
    for condition in conditions:
        genes = parse_condition_genes(condition)
        if genes & test_genes:
            split["test"].append(condition)
        elif genes & validation_genes:
            split["validation"].append(condition)
        else:
            split["train"].append(condition)
    return split


def _split_stats(
    gene_split: Mapping[str, Sequence[str]],
    condition_split: Mapping[str, Sequence[str]],
    target_fractions: Mapping[str, float],
) -> dict[str, object]:
    n_conditions = sum(len(condition_split[split]) for split in condition_split)
    condition_fractions = {
        split: _safe_fraction(len(condition_split[split]), n_conditions)
        for split in ("train", "validation", "test")
    }
    return {
        "n_train_genes": len(gene_split["train"]),
        "n_validation_genes": len(gene_split["validation"]),
        "n_test_genes": len(gene_split["test"]),
        "n_train_conditions": len(condition_split["train"]),
        "n_validation_conditions": len(condition_split["validation"]),
        "n_test_conditions": len(condition_split["test"]),
        "condition_fractions": condition_fractions,
        "condition_fraction_deltas": {
            split: condition_fractions[split] - float(target_fractions[split])
            for split in ("train", "validation", "test")
        },
    }


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
