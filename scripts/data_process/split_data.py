"""
Split VCC data into train and test sets using fixed test genes from CSV.

The test set contains 30 fixed genes from data/raw/test_set.csv.
The train set contains the remaining 120 genes + control cells.

Usage:
    python scripts/data_process/split_data.py
"""

import anndata as ad
import csv
import os
import sys

# Configure paths
RAW_DATA_PATH = "data/raw/adata_Training.h5ad"
TEST_GENES_CSV = "data/raw/test_set.csv"
OUTPUT_DIR = "data/processed"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.h5ad")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.h5ad")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_test_genes(csv_path: str) -> list:
    """Load test gene names from CSV file."""
    test_genes = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # First column is "Group" which contains gene name
            gene = row.get("Group", row.get(list(row.keys())[0]))
            if gene:
                test_genes.append(gene)
    return test_genes


def main():
    # Load test genes from CSV
    print(f"Loading test genes from {TEST_GENES_CSV}...")
    try:
        test_genes = load_test_genes(TEST_GENES_CSV)
    except FileNotFoundError:
        print(f"Error: Test genes file not found at {TEST_GENES_CSV}")
        sys.exit(1)

    print(f"Loaded {len(test_genes)} test genes: {test_genes[:5]}...")

    # Load raw data
    print(f"\nLoading data from {RAW_DATA_PATH}...")
    try:
        adata = ad.read_h5ad(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {RAW_DATA_PATH}")
        sys.exit(1)

    print(f"Original data shape: {adata.shape}")

    # Get unique target genes
    all_targets = adata.obs["target_gene"].unique().tolist()

    # Separate Control ('non-targeting') from Perturbations
    control_label = "non-targeting"
    if control_label in all_targets:
        perturbations = [g for g in all_targets if g != control_label]
        print(
            f"Found {len(perturbations)} perturbation genes + '{control_label}' control group."
        )
    else:
        print(
            f"Warning: '{control_label}' not found in target_gene. "
            f"Treating all {len(all_targets)} as perturbations."
        )
        perturbations = all_targets

    # Validate test genes exist in data
    test_genes_set = set(test_genes)
    perturbations_set = set(perturbations)
    missing_genes = test_genes_set - perturbations_set
    if missing_genes:
        print(
            f"Warning: {len(missing_genes)} test genes not found in data: {missing_genes}"
        )
        test_genes = [g for g in test_genes if g in perturbations_set]

    # Train genes = all perturbations except test genes
    train_genes = [g for g in perturbations if g not in test_genes_set]

    print(f"\nSplit summary:")
    print(f"  Test genes:  {len(test_genes)} (fixed from CSV)")
    print(f"  Train genes: {len(train_genes)} (remaining perturbations)")

    # Create masks
    # Train set: All control cells + cells from train_genes
    train_mask = adata.obs["target_gene"].isin(train_genes) | (
        adata.obs["target_gene"] == control_label
    )

    # Test set: Only cells from test_genes (no control)
    test_mask = adata.obs["target_gene"].isin(test_genes)

    # Subset data
    adata_train = adata[train_mask].copy()
    adata_test = adata[test_mask].copy()

    print(f"\nProcessing complete:")
    print(f"  Train set: {adata_train.shape} (includes controls)")
    print(f"  Test set:  {adata_test.shape} (perturbations only)")

    # Save files
    print(f"\nSaving to {TRAIN_FILE}...")
    adata_train.write_h5ad(TRAIN_FILE)

    print(f"Saving to {TEST_FILE}...")
    adata_test.write_h5ad(TEST_FILE)

    print("Done!")


if __name__ == "__main__":
    main()
