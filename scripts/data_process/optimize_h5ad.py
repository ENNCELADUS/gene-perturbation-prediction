#!/usr/bin/env python
"""
Optimize h5ad files for memory efficiency.

Optimizations:
1. Remove counts layer (not needed for training)
2. Convert X from float32 to float16 (sufficient precision for log1p values)
3. Ensure sparse format is used

Usage:
    python scripts/data_process/optimize_h5ad.py
"""

import argparse
import os
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


def optimize_adata(adata: ad.AnnData, name: str) -> ad.AnnData:
    """
    Optimize AnnData for memory efficiency.

    Args:
        adata: AnnData object to optimize
        name: File name for logging

    Returns:
        Optimized AnnData object
    """
    print(f"\n{'=' * 50}")
    print(f"Optimizing {name}")
    print(f"{'=' * 50}")

    # Original stats
    print(f"Original shape: {adata.shape}")
    print(f"Original X dtype: {adata.X.dtype}")
    print(f"Original layers: {list(adata.layers.keys())}")

    original_size = 0
    if hasattr(adata.X, "data"):
        original_size += (
            adata.X.data.nbytes + adata.X.indices.nbytes + adata.X.indptr.nbytes
        )
    else:
        original_size += adata.X.nbytes
    for layer in adata.layers.values():
        if hasattr(layer, "data"):
            original_size += (
                layer.data.nbytes + layer.indices.nbytes + layer.indptr.nbytes
            )
        else:
            original_size += layer.nbytes
    print(f"Original data size: {original_size / (1024**3):.2f} GB")

    # 1. Remove counts layer if present (not needed for training)
    if "counts" in adata.layers:
        print("\n[1] Removing 'counts' layer (not needed for training)...")
        del adata.layers["counts"]
        print("    Done.")

    # 2. Convert X to float16 for memory efficiency
    print("\n[2] Converting X from float32 to float16...")
    if hasattr(adata.X, "data"):
        # Sparse matrix - convert data array
        adata.X.data = adata.X.data.astype(np.float16)
        print(f"    Sparse matrix: {adata.X.nnz} non-zero elements")
    else:
        # Dense matrix
        adata.X = adata.X.astype(np.float16)
    print(f"    New dtype: {adata.X.dtype}")

    # 3. Ensure sparse format (CSR is most efficient for row operations)
    if not sp.issparse(adata.X):
        print("\n[3] Converting X to sparse CSR format...")
        adata.X = sp.csr_matrix(adata.X)
        print("    Done.")
    elif not isinstance(adata.X, sp.csr_matrix):
        print(f"\n[3] Converting X from {type(adata.X).__name__} to CSR...")
        adata.X = adata.X.tocsr()
        print("    Done.")
    else:
        print("\n[3] X already in CSR format.")

    # Final stats
    optimized_size = 0
    if hasattr(adata.X, "data"):
        optimized_size += (
            adata.X.data.nbytes + adata.X.indices.nbytes + adata.X.indptr.nbytes
        )
    else:
        optimized_size += adata.X.nbytes
    for layer in adata.layers.values():
        if hasattr(layer, "data"):
            optimized_size += (
                layer.data.nbytes + layer.indices.nbytes + layer.indptr.nbytes
            )
        else:
            optimized_size += layer.nbytes

    print(f"\nOptimized data size: {optimized_size / (1024**3):.2f} GB")
    print(f"Memory reduction: {(1 - optimized_size / original_size) * 100:.1f}%")

    return adata


def main():
    parser = argparse.ArgumentParser(description="Optimize h5ad files for memory")
    parser.add_argument(
        "--data_dir",
        default="data/processed",
        help="Directory containing h5ad files",
    )
    parser.add_argument(
        "--gears_dir",
        default="data/processed/gears",
        help="Directory containing GEARS data",
    )
    parser.add_argument(
        "--skip_backup",
        action="store_true",
        help="Skip creating backup files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    gears_dir = Path(args.gears_dir)

    # Files to optimize
    files_to_optimize = [
        data_dir / "train.h5ad",
        data_dir / "test.h5ad",
        gears_dir / "vcc" / "perturb_processed.h5ad",
    ]

    for filepath in files_to_optimize:
        if not filepath.exists():
            print(f"Skipping {filepath} (not found)")
            continue

        # Load
        print(f"\nLoading {filepath}...")
        adata = ad.read_h5ad(filepath)

        # Optimize
        adata = optimize_adata(adata, filepath.name)

        # Backup original (optional)
        if not args.skip_backup:
            backup_path = filepath.with_suffix(".h5ad.bak")
            if not backup_path.exists():
                print(f"\nCreating backup at {backup_path}...")
                os.rename(filepath, backup_path)

        # Save optimized version
        print(f"\nSaving optimized data to {filepath}...")
        adata.write_h5ad(filepath)
        print("Done.")

    # Remove intermediate files
    intermediate_file = gears_dir / "vcc_converted.h5ad"
    if intermediate_file.exists():
        print(f"\n{'=' * 50}")
        print(f"Removing intermediate file: {intermediate_file}")
        print(f"{'=' * 50}")
        size_gb = intermediate_file.stat().st_size / (1024**3)
        os.remove(intermediate_file)
        print(f"Freed {size_gb:.2f} GB")

    print("\n" + "=" * 50)
    print("Optimization complete!")
    print("=" * 50)
    print("\nNote: Backup files (.h5ad.bak) can be deleted after verification.")


if __name__ == "__main__":
    main()
