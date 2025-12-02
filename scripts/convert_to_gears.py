#!/usr/bin/env python
"""
Convert VCC h5ad data to GEARS-compatible format.

GEARS requires:
- obs['condition']: perturbation condition (e.g., "GENE+ctrl" or "ctrl")
- obs['cell_type']: cell type annotation
- var['gene_name']: gene names

Usage:
    python scripts/convert_to_gears.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_vcc_to_gears(
    train_path: str,
    output_dir: str,
    cell_type: str = "K562",
    control_key: str = "non-targeting",
) -> sc.AnnData:
    """
    Convert VCC h5ad to GEARS format.

    Args:
        train_path: Path to VCC training h5ad file
        output_dir: Output directory for GEARS data
        cell_type: Cell type label (VCC uses K562 cells)
        control_key: Control condition key in VCC data

    Returns:
        Converted AnnData object
    """
    print(f"Loading data from {train_path}...")
    adata = sc.read_h5ad(train_path)
    print(f"Loaded {adata.shape[0]} cells x {adata.shape[1]} genes")

    # === Convert obs columns ===

    # 1. Convert target_gene to GEARS condition format
    # VCC: "GENE" or "non-targeting"
    # GEARS: "GENE+ctrl" or "ctrl"
    def convert_condition(target_gene: str) -> str:
        if target_gene == control_key:
            return "ctrl"
        else:
            return f"{target_gene}+ctrl"

    adata.obs["condition"] = adata.obs["target_gene"].apply(convert_condition)

    # 2. Create condition_name (same as condition for single perturbations)
    adata.obs["condition_name"] = adata.obs["condition"]

    # 3. Add cell_type column
    adata.obs["cell_type"] = cell_type

    # === Convert var columns ===

    # Gene names should be in var['gene_name']
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index.tolist()

    # === Ensure X is sparse float16 (for memory efficiency) ===
    from scipy.sparse import csr_matrix

    if not hasattr(adata.X, "toarray"):
        adata.X = csr_matrix(adata.X)

    # Convert to float16 for memory efficiency
    if adata.X.dtype != np.float16:
        adata.X.data = adata.X.data.astype(np.float16)

    # Remove counts layer if present to save memory
    if "counts" in adata.layers:
        del adata.layers["counts"]

    # === Print conversion summary ===
    print("\n=== Conversion Summary ===")
    print(f"Total cells: {adata.shape[0]}")
    print(f"Total genes: {adata.shape[1]}")
    print(f"Control cells: {(adata.obs['condition'] == 'ctrl').sum()}")

    unique_perts = adata.obs["condition"].unique()
    n_perts = len([p for p in unique_perts if p != "ctrl"])
    print(f"Unique perturbations: {n_perts}")

    # === Save intermediate h5ad ===
    os.makedirs(output_dir, exist_ok=True)
    intermediate_path = os.path.join(output_dir, "vcc_converted.h5ad")
    adata.write_h5ad(intermediate_path)
    print(f"\nSaved intermediate file to {intermediate_path}")

    return adata


def process_with_gears(adata: sc.AnnData, output_dir: str, dataset_name: str = "vcc"):
    """
    Process data with GEARS to compute DE genes and create cell graphs.

    Args:
        adata: Converted AnnData object
        output_dir: Output directory for GEARS data
        dataset_name: Name for the dataset
    """
    try:
        from gears import PertData
    except ImportError:
        print("ERROR: gears package not installed or torch_geometric missing.")
        print("Install with: pip install gears torch_geometric")
        sys.exit(1)

    print("\n=== Processing with GEARS ===")

    # Initialize PertData with output directory
    pert_data = PertData(output_dir)

    # Process the data
    # This computes DE genes and creates PyG cell graphs
    pert_data.new_data_process(dataset_name=dataset_name, adata=adata)

    print(f"\nGEARS processing complete. Data saved to {output_dir}/{dataset_name}/")

    return pert_data


def main():
    parser = argparse.ArgumentParser(description="Convert VCC data to GEARS format")
    parser.add_argument(
        "--train_path",
        default="data/processed/train.h5ad",
        help="Path to VCC training h5ad file",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/gears",
        help="Output directory for GEARS data",
    )
    parser.add_argument(
        "--cell_type",
        default="K562",
        help="Cell type label (VCC uses K562 cells)",
    )
    parser.add_argument(
        "--control_key",
        default="non-targeting",
        help="Control condition key in VCC data",
    )
    parser.add_argument(
        "--dataset_name",
        default="vcc",
        help="Dataset name for GEARS",
    )
    parser.add_argument(
        "--skip_gears",
        action="store_true",
        help="Skip GEARS processing (only convert format)",
    )
    args = parser.parse_args()

    # Convert VCC format to GEARS format
    adata = convert_vcc_to_gears(
        train_path=args.train_path,
        output_dir=args.output_dir,
        cell_type=args.cell_type,
        control_key=args.control_key,
    )

    # Process with GEARS (compute DE genes, create cell graphs)
    if not args.skip_gears:
        process_with_gears(
            adata=adata,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
        )
    else:
        print("\nSkipping GEARS processing (--skip_gears flag set)")

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
