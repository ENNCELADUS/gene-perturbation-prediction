"""
Data splitting utilities for perturbation prediction training.
"""

import csv
from typing import List, Tuple

import numpy as np


def load_test_genes(test_genes_file: str) -> List[str]:
    """
    Load test gene names from CSV file.

    Args:
        test_genes_file: Path to CSV file with test genes (first column is gene name)

    Returns:
        List of test gene names
    """
    test_genes = []
    with open(test_genes_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # First column is "Group" which contains gene name
            gene = row.get("Group", row.get(list(row.keys())[0]))
            if gene:
                test_genes.append(gene)
    return test_genes


def get_perturbation_genes(pert_data) -> List[str]:
    """
    Extract unique perturbation conditions (excluding control).

    Args:
        pert_data: GEARS PertData object

    Returns:
        List of perturbation conditions (e.g., ["BRCA1+ctrl", "TP53+ctrl", ...])
    """
    conditions = pert_data.adata.obs["condition"].unique().tolist()
    # Exclude control condition
    pert_genes = [c for c in conditions if c != "ctrl"]
    return sorted(pert_genes)


def create_train_val_split(
    pert_genes: List[str],
    test_genes: List[str],
    train_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Create train/val split from perturbation genes, excluding test genes.

    Args:
        pert_genes: All perturbation conditions (e.g., ["GENE+ctrl", ...])
        test_genes: List of test gene names to exclude
        train_ratio: Fraction for training (e.g., 0.833 for 100/120)
        seed: Random seed

    Returns:
        Tuple of (train_perts, val_perts)
    """
    # Convert test genes to condition format
    test_conditions = {f"{g}+ctrl" for g in test_genes}

    # Filter out test genes
    trainval_perts = [p for p in pert_genes if p not in test_conditions]

    # Random split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(trainval_perts))

    n_train = int(len(trainval_perts) * train_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_perts = [trainval_perts[i] for i in train_indices]
    val_perts = [trainval_perts[i] for i in val_indices]

    return train_perts, val_perts


def filter_dataset_by_perts(
    dataset,
    perts: List[str],
    include_ctrl: bool = True,
) -> List[int]:
    """
    Get indices of cells belonging to specified perturbations.

    Args:
        dataset: PyG dataset (list) from GEARS
        perts: List of perturbation conditions to include
        include_ctrl: Whether to include control cells

    Returns:
        List of cell indices
    """
    perts_set = set(perts)
    indices = []
    for i, data in enumerate(dataset):
        cond = data.pert  # Each PyG Data object has .pert attribute
        if cond in perts_set:
            indices.append(i)
        elif include_ctrl and cond == "ctrl":
            indices.append(i)

    return indices
