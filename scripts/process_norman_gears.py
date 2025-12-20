#!/usr/bin/env python
"""
Process Norman dataset using GEARS library.

This script loads the Norman Perturb-seq dataset and prepares train/val/test splits
for the reverse perturbation prediction task.

Reference:
- Norman et al. 2019: Original Perturb-seq data
- GEARS (v0.0.2): Data processing pipeline
"""

import os
import argparse
from pathlib import Path

from gears import PertData


def main():
    parser = argparse.ArgumentParser(description="Process Norman dataset with GEARS")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Base directory for data storage",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="simulation",
        help="Split type: 'simulation' or 'combo_seen0/1/2'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for split",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for dataloaders",
    )
    args = parser.parse_args()

    # Ensure data directory exists
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Processing Norman Dataset with GEARS")
    print("=" * 60)

    # Initialize PertData
    print(f"\n[1/4] Initializing PertData with data_dir={data_dir}")
    pert_data = PertData(str(data_dir))

    # Load Norman dataset
    print("\n[2/4] Loading Norman dataset...")
    pert_data.load(data_name="norman")

    # Print dataset info
    print("\n[3/4] Dataset Info:")
    print(f"  - AnnData shape: {pert_data.adata.shape}")
    print(f"  - Number of genes: {pert_data.adata.n_vars}")
    print(f"  - Number of cells: {pert_data.adata.n_obs}")

    # Count perturbation types
    if hasattr(pert_data, "adata") and "condition" in pert_data.adata.obs.columns:
        conditions = pert_data.adata.obs["condition"].unique()
        ctrl = sum(1 for c in conditions if c == "ctrl")
        single = sum(1 for c in conditions if "+ctrl" in c or "ctrl+" in c)
        double = len(conditions) - ctrl - single
        print(f"  - Control conditions: {ctrl}")
        print(f"  - Single-gene perturbations: {single}")
        print(f"  - Double-gene perturbations: {double}")

    # Prepare split
    print(f"\n[4/4] Preparing split: {args.split} (seed={args.seed})")
    pert_data.prepare_split(split=args.split, seed=args.seed)
    pert_data.get_dataloader(
        batch_size=args.batch_size, test_batch_size=args.batch_size
    )

    # Print split info
    print("\nSplit Statistics:")
    if hasattr(pert_data, "set2conditions"):
        for split_name, conditions in pert_data.set2conditions.items():
            print(f"  - {split_name}: {len(conditions)} conditions")

    # Get data loader info
    if hasattr(pert_data, "dataloader"):
        for split_name in ["train_loader", "val_loader", "test_loader"]:
            loader = getattr(pert_data, split_name, None)
            if loader:
                print(f"  - {split_name}: {len(loader)} batches")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Data stored in: {data_dir / 'norman'}")
    print("=" * 60)

    # Return pert_data for further use
    return pert_data


if __name__ == "__main__":
    main()
