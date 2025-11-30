#!/usr/bin/env python3
"""
Script to preview VCC data files
"""

import pandas as pd
import anndata as ad
import numpy as np


def preview_h5ad(filepath):
    """Preview AnnData H5AD file"""
    print("=" * 80)
    print(f"FILE: {filepath}")
    print("=" * 80)

    adata = ad.read_h5ad(filepath)

    print(f"\n📊 Shape: {adata.n_obs} cells × {adata.n_vars} genes")
    print(
        f"Size in memory: ~{adata.X.data.nbytes / (1024**3):.2f} GB"
        if hasattr(adata.X, "data")
        else ""
    )

    # Observations (cells)
    print("\n🔬 Observations (cells):")
    print(f"  Keys: {list(adata.obs.columns)}")
    print(f"\n  Sample obs data:")
    print(adata.obs.head(10))

    # Variables (genes)
    print("\n🧬 Variables (genes):")
    print(f"  Keys: {list(adata.var.columns)}")
    print(f"\n  Sample var data:")
    print(adata.var.head(10))

    # Check for target genes
    if "target_gene" in adata.obs.columns:
        print("\n🎯 Target Gene Distribution:")
        target_counts = adata.obs["target_gene"].value_counts()
        print(f"  Unique targets: {len(target_counts)}")
        print(
            f"  Non-targeting cells: {(adata.obs['target_gene'] == 'non-targeting').sum()}"
        )
        print(f"\n  Top 10 targets:")
        print(target_counts.head(10))

    # Check for batches
    if "batch" in adata.obs.columns:
        print("\n🔢 Batch Distribution:")
        print(adata.obs["batch"].value_counts().head(10))

    # Expression matrix
    print("\n📈 Expression Matrix:")
    print(f"  Type: {type(adata.X)}")
    print(f"  Dtype: {adata.X.dtype}")
    if hasattr(adata.X, "nnz"):
        print(
            f"  Sparsity: {(1 - adata.X.nnz / (adata.n_obs * adata.n_vars)) * 100:.2f}% zeros"
        )

    print("\n")
    return adata


def preview_csv(filepath):
    """Preview CSV file"""
    print("=" * 80)
    print(f"FILE: {filepath}")
    print("=" * 80)

    df = pd.read_csv(filepath)

    print(f"\n📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    print(f"\n📋 First 10 rows:")
    print(df.head(10))

    print(f"\n📋 Last 10 rows:")
    print(df.tail(10))

    print(f"\n📊 Data types:")
    print(df.dtypes)

    print(f"\n📊 Summary statistics:")
    print(df.describe())

    print("\n")
    return df


if __name__ == "__main__":
    data_dir = "/home/richard/projects/VCC/vcc_data"

    print("\n🔍 PREVIEWING VCC DATA FILES\n")

    # Preview H5AD file
    try:
        print("\n[1/3] Loading AnnData Training Set...")
        adata = preview_h5ad(f"{data_dir}/adata_Training.h5ad")
    except Exception as e:
        print(f"❌ Error loading adata_Training.h5ad: {e}\n")

    # Preview gene names CSV
    try:
        print("\n[2/3] Loading Gene Names CSV...")
        gene_names = preview_csv(f"{data_dir}/gene_names.csv")
    except Exception as e:
        print(f"❌ Error loading gene_names.csv: {e}\n")

    # Preview validation perturbation counts CSV
    try:
        print("\n[3/3] Loading Validation Perturbation Counts CSV...")
        pert_counts = preview_csv(f"{data_dir}/pert_counts_Validation.csv")
    except Exception as e:
        print(f"❌ Error loading pert_counts_Validation.csv: {e}\n")

    print("✅ Preview complete!")
