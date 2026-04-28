"""Prepare pseudobulk artifacts for PCA+kNN."""

from __future__ import annotations

from src.utils.data import build_pseudobulk_matrices


def run(config: dict) -> dict:
    """Build condition-level matrices and return a summary."""
    data = build_pseudobulk_matrices(config)
    conditions = data["conditions"]
    return {
        "n_train": len(conditions["train"]),
        "n_validation": len(conditions["validation"]),
        "n_test": len(conditions["test"]),
        "n_genes": len(data["gene_names"]),
    }
