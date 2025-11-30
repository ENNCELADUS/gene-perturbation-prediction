import anndata as ad
import numpy as np
import os
import sys

# Configure paths
RAW_DATA_PATH = "data/raw/adata_Training.h5ad"
OUTPUT_DIR = "data/processed"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.h5ad")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.h5ad")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print(f"Loading data from {RAW_DATA_PATH}...")
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
            f"Warning: '{control_label}' not found in target_gene. Treating all {len(all_targets)} as perturbations."
        )
        perturbations = all_targets

    # Shuffle perturbations to ensure random selection
    # Using a fixed seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(perturbations)

    # Determine split size (10% for test)
    n_test = int(len(perturbations) * 0.10)
    # Ensure at least 1 gene if list is small, but max 15 for 150 genes
    n_test = max(1, n_test)

    test_genes = perturbations[:n_test]
    train_genes = perturbations[n_test:]

    print(f"\nSelected {len(test_genes)} genes for TESTING (OOD):")
    print(f"{test_genes}")
    print(f"\nRemaining {len(train_genes)} genes for TRAINING.")

    # Create masks
    # Train set: All control cells + cells from train_genes
    train_mask = adata.obs["target_gene"].isin(train_genes) | (
        adata.obs["target_gene"] == control_label
    )

    # Test set: Only cells from test_genes
    test_mask = adata.obs["target_gene"].isin(test_genes)

    # Subset data
    adata_train = adata[train_mask].copy()
    adata_test = adata[test_mask].copy()

    print(f"\nProcessing complete:")
    print(f"Train set: {adata_train.shape} (Includes controls)")
    print(f"Test set:  {adata_test.shape} (Perturbations only)")

    # Save files
    print(f"\nSaving to {TRAIN_FILE}...")
    adata_train.write_h5ad(TRAIN_FILE)

    print(f"Saving to {TEST_FILE}...")
    adata_test.write_h5ad(TEST_FILE)

    print("Done!")


if __name__ == "__main__":
    main()
