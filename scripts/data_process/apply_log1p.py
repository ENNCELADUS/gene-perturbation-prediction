"""
Apply log1p normalization to train and test h5ad files.

This script:
- Loads train.h5ad and test.h5ad
- Stores raw counts in adata.layers['counts'] for reference
- Applies log1p transformation (no other normalization)
- Records transformation metadata in adata.uns['log1p']
- Overwrites the h5ad files in place

Usage:
    python scripts/data_process/apply_log1p.py
"""

import anndata as ad
import os
import sys
import scanpy as sc

# Configure paths
OUTPUT_DIR = "data/processed"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.h5ad")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.h5ad")


def apply_log1p_to_adata(adata: ad.AnnData, file_name: str) -> ad.AnnData:
    """
    Apply log1p normalization to an AnnData object.

    Args:
        adata: AnnData object to transform
        file_name: Name of the file (for logging)

    Returns:
        Transformed AnnData object
    """
    print(f"\nProcessing {file_name}...")
    print(f"  Shape: {adata.shape}")
    print(f"  X dtype: {adata.X.dtype}")
    print(f"  X min/max: {adata.X.min():.2f} / {adata.X.max():.2f}")

    # Check if already log1p transformed
    if "log1p" in adata.uns:
        print(f"  Warning: {file_name} already has log1p metadata. Skipping transformation.")
        return adata

    # Store raw counts in layers['counts'] if not already present
    if "counts" not in adata.layers:
        print("  Storing raw counts in layers['counts']...")
        # Handle sparse matrices
        if hasattr(adata.X, "toarray"):
            adata.layers["counts"] = adata.X.copy()
        else:
            import numpy as np
            adata.layers["counts"] = np.array(adata.X).copy()
    else:
        print("  layers['counts'] already exists, preserving existing counts...")

    # Apply log1p transformation (no other normalization)
    print("  Applying log1p transformation...")
    sc.pp.log1p(adata)

    # Record transformation metadata
    adata.uns["log1p"] = {"base": None}  # natural log (base=None means natural log)

    print(f"  After log1p - X min/max: {adata.X.min():.2f} / {adata.X.max():.2f}")
    print(f"  Transformation metadata saved to uns['log1p']")

    return adata


def main():
    """Main function to apply log1p normalization to train and test files."""
    # Check if files exist
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: Train file not found at {TRAIN_FILE}")
        sys.exit(1)

    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file not found at {TEST_FILE}")
        sys.exit(1)

    # Process train file
    print("=" * 50)
    print("Applying log1p normalization to train and test data")
    print("=" * 50)

    adata_train = ad.read_h5ad(TRAIN_FILE)
    adata_train = apply_log1p_to_adata(adata_train, "train.h5ad")
    print(f"\nSaving transformed train data to {TRAIN_FILE}...")
    adata_train.write_h5ad(TRAIN_FILE)

    # Process test file
    adata_test = ad.read_h5ad(TEST_FILE)
    adata_test = apply_log1p_to_adata(adata_test, "test.h5ad")
    print(f"\nSaving transformed test data to {TEST_FILE}...")
    adata_test.write_h5ad(TEST_FILE)

    print("\n" + "=" * 50)
    print("log1p normalization complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

