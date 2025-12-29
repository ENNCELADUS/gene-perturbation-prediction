#!/usr/bin/env python3
"""
Detailed inspection of data/tahoe/tahoe.h5ad for:
1. Data normalization status
2. Condition/drug-level statistics for splitting
3. Data structure and quality metrics

Uses backed mode for memory efficiency with large files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import h5py

DATA_PATH = Path("data/tahoe/tahoe.h5ad")


def inspect_with_h5py():
    """Inspect using h5py for memory-efficient access."""
    print("=" * 70)
    print("TAHOE H5AD DETAILED INSPECTION (Memory-efficient)")
    print("=" * 70)

    with h5py.File(DATA_PATH, "r") as f:
        print("\n--- H5 STRUCTURE ---")

        def print_attrs(name, obj):
            print(f"  {name}")

        f.visititems(print_attrs)

        # Basic shape
        print("\n--- BASIC INFO ---")
        X = f["X"]
        if "data" in X:
            # CSR sparse
            n_elements = X["data"].shape[0]
            n_rows = X["indptr"].shape[0] - 1
            # Try to get ncols from shape attr or var
            print(f"X storage: sparse (CSR)")
            print(f"X.data shape: {X['data'].shape}")
            print(f"n_obs (rows): {n_rows}")

            # Get n_vars from var
            if "var" in f:
                if "_index" in f["var"]:
                    n_cols = f["var"]["_index"].shape[0]
                    print(f"n_vars (cols): {n_cols}")

        # OBS columns
        print(f"\nobs columns:")
        obs = f["obs"]
        obs_cols = list(obs.keys())
        print(f"  {obs_cols}")

        # VAR columns
        print(f"\nvar columns:")
        var = f["var"]
        var_cols = list(var.keys())
        print(f"  {var_cols}")

        # Layers
        print(f"\nlayers:")
        if "layers" in f:
            layers = list(f["layers"].keys())
            print(f"  {layers}")

        # Read obs as dataframe
        print("\n--- OBS DATAFRAME STATS ---")
        obs_dict = {}
        for col in obs_cols:
            if col.startswith("_"):
                continue
            try:
                data = obs[col][:]
                if hasattr(data, "astype"):
                    # Check if categorical
                    if "categories" in obs[col]:
                        cats = obs[col]["categories"][:]
                        codes = obs[col]["codes"][:]
                        if hasattr(cats[0], "decode"):
                            cats = [c.decode() for c in cats]
                        data = pd.Categorical.from_codes(codes, categories=cats)
                    elif data.dtype.kind == "S" or data.dtype.kind == "O":
                        data = [d.decode() if hasattr(d, "decode") else d for d in data]
                obs_dict[col] = data
                print(f"  Loaded: {col}")
            except Exception as e:
                print(f"  Skip {col}: {e}")

        obs_df = pd.DataFrame(obs_dict)
        print(f"\nobs shape: {obs_df.shape}")

        return f, obs_df


def analyze_normalization(f):
    """Check normalization by sampling X matrix."""
    print("\n" + "=" * 70)
    print("NORMALIZATION ANALYSIS")
    print("=" * 70)

    X = f["X"]

    # Sample from sparse data
    data = X["data"]
    n_elements = data.shape[0]
    print(f"\nTotal non-zero elements in X: {n_elements:,}")

    # Sample random non-zero values
    sample_size = min(100000, n_elements)
    np.random.seed(42)
    indices = np.random.choice(n_elements, sample_size, replace=False)
    indices.sort()

    # Read sampled values
    sampled_values = data[indices]

    print(f"\nSampled {sample_size:,} non-zero values:")
    print(f"  Min: {sampled_values.min():.6f}")
    print(f"  Max: {sampled_values.max():.6f}")
    print(f"  Mean: {sampled_values.mean():.6f}")
    print(f"  Std: {sampled_values.std():.6f}")
    print(f"  Median: {np.median(sampled_values):.6f}")

    # Check for integer values
    is_integer = np.allclose(sampled_values, np.round(sampled_values))
    print(f"\n  Values appear integer: {is_integer}")

    # Check value distribution
    print(f"\n  Value percentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    {p}th: {np.percentile(sampled_values, p):.6f}")

    # Normalization inference
    print(f"\n  Normalization indicators:")
    if sampled_values.max() > 100:
        print("    - Max value > 100: suggests RAW COUNTS")
    elif sampled_values.max() < 20:
        print("    - Max value < 20: suggests LOG TRANSFORMED")

    if is_integer:
        print("    - Integer values: suggests RAW COUNTS")
    else:
        print("    - Float values: suggests NORMALIZED/TRANSFORMED")

    # Check ctrl layer
    if "layers" in f and "ctrl" in f["layers"]:
        print("\n--- CTRL LAYER ANALYSIS ---")
        ctrl = f["layers"]["ctrl"]
        if "data" in ctrl:
            ctrl_data = ctrl["data"]
            ctrl_sample = ctrl_data[: min(100000, ctrl_data.shape[0])]
            print(
                f"  ctrl layer samples: min={ctrl_sample.min():.6f}, max={ctrl_sample.max():.6f}, mean={ctrl_sample.mean():.6f}"
            )


def analyze_conditions(obs_df):
    """Analyze condition/drug structure for splitting."""
    print("\n" + "=" * 70)
    print("CONDITION ANALYSIS FOR DRUG-BASED SPLITTING")
    print("=" * 70)

    # Drug analysis
    print("\n--- DRUG DISTRIBUTION ---")
    drug_counts = obs_df["drug"].value_counts()
    print(f"Total unique drugs: {len(drug_counts)}")
    print(f"\nAll {len(drug_counts)} drugs with cell counts:")
    for drug, count in drug_counts.items():
        print(f"  {drug:30s}: {count:>10,}")

    print(f"\nDrug cell count stats:")
    print(f"  Min cells per drug: {drug_counts.min():,}")
    print(f"  Max cells per drug: {drug_counts.max():,}")
    print(f"  Mean cells per drug: {drug_counts.mean():,.0f}")
    print(f"  Median cells per drug: {drug_counts.median():,.0f}")
    print(f"  Std cells per drug: {drug_counts.std():,.0f}")

    # Target gene analysis
    print("\n--- TARGET GENE DISTRIBUTION ---")
    gene_counts = obs_df["target_gene"].value_counts()
    print(f"Total unique target genes: {len(gene_counts)}")
    print(f"\nTarget gene cell count stats:")
    print(f"  Min: {gene_counts.min():,}")
    print(f"  Max: {gene_counts.max():,}")
    print(f"  Mean: {gene_counts.mean():,.0f}")
    print(f"  Median: {gene_counts.median():,.0f}")

    print(f"\nTop 20 target genes:")
    for gene, count in gene_counts.head(20).items():
        print(f"  {gene:20s}: {count:>10,}")

    # Cell line analysis
    print("\n--- CELL LINE DISTRIBUTION ---")
    cell_line_counts = obs_df["cell_line_id"].value_counts()
    print(f"Total unique cell lines: {len(cell_line_counts)}")
    print(f"\nTop 20 cell lines:")
    for cl, count in cell_line_counts.head(20).items():
        print(f"  {cl:15s}: {count:>10,}")

    # Drug x Target gene combinations
    print("\n--- DRUG x TARGET GENE COMBINATIONS ---")
    drug_gene_combos = obs_df.groupby(["drug", "target_gene"]).size()
    print(f"Total unique drug x target_gene combinations: {len(drug_gene_combos):,}")
    print(
        f"  Average combinations per drug: {len(drug_gene_combos) / len(drug_counts):.1f}"
    )

    # Check coverage matrix
    drug_gene_matrix = pd.crosstab(obs_df["drug"], obs_df["target_gene"])
    coverage = (drug_gene_matrix > 0).sum(axis=1)
    print(f"\nTarget gene coverage per drug:")
    print(
        f"  Drugs targeting all {len(gene_counts)} genes: {(coverage == len(gene_counts)).sum()}"
    )
    print(f"  Average genes per drug: {coverage.mean():.1f}")
    print(f"  Min genes per drug: {coverage.min()}")
    print(f"  Max genes per drug: {coverage.max()}")

    # Drug x Cell line combinations
    print("\n--- DRUG x CELL LINE COMBINATIONS ---")
    drug_cell_combos = obs_df.groupby(["drug", "cell_line_id"]).size()
    print(f"Total unique drug x cell_line combinations: {len(drug_cell_combos):,}")

    drug_cell_matrix = pd.crosstab(obs_df["drug"], obs_df["cell_line_id"])
    cell_coverage = (drug_cell_matrix > 0).sum(axis=1)
    print(f"\nCell line coverage per drug:")
    print(
        f"  Drugs in all {len(cell_line_counts)} cell lines: {(cell_coverage == len(cell_line_counts)).sum()}"
    )
    print(f"  Average cell lines per drug: {cell_coverage.mean():.1f}")

    # Condition and label
    print("\n--- CONDITION ANALYSIS ---")
    cond_counts = obs_df["condition"].value_counts()
    print(f"Unique 'condition' values: {len(cond_counts)}")
    print(f"\nTop 20 conditions:")
    for cond, count in cond_counts.head(20).items():
        print(f"  {cond:40s}: {count:>10,}")

    print("\n--- CONDITION_NAME ANALYSIS ---")
    condname_counts = obs_df["condition_name"].value_counts()
    print(f"Unique 'condition_name' values: {len(condname_counts)}")

    # Label analysis
    print("\n--- LABEL DISTRIBUTION ---")
    label_counts = obs_df["label"].value_counts()
    print(f"Unique labels: {len(label_counts)}")
    print(f"Label value range: {obs_df['label'].min()} - {obs_df['label'].max()}")

    # Control analysis
    print("\n--- CONTROL ANALYSIS ---")
    ctrl_counts = obs_df["control"].value_counts()
    print(f"Control column:")
    for val, count in ctrl_counts.items():
        print(f"  {val}: {count:,}")

    if "ctrl_drug" in obs_df.columns:
        ctrl_drug_counts = obs_df["ctrl_drug"].value_counts()
        print(f"\nctrl_drug values:")
        for val, count in ctrl_drug_counts.items():
            print(f"  {val}: {count:,}")

    # Overlap ratio
    print("\n--- OVERLAP RATIO ---")
    overlap = obs_df["overlap_ratio"]
    print(f"Overlap ratio stats:")
    print(f"  Min: {overlap.min():.4f}")
    print(f"  Max: {overlap.max():.4f}")
    print(f"  Mean: {overlap.mean():.4f}")
    print(f"  Median: {overlap.median():.4f}")
    print(f"  Std: {overlap.std():.4f}")

    # Plate analysis
    print("\n--- PLATE DISTRIBUTION ---")
    plate_counts = obs_df["plate"].value_counts()
    print(f"Unique plates: {len(plate_counts)}")
    for plate, count in plate_counts.items():
        print(f"  {plate}: {count:,}")

    return drug_counts, gene_counts


def suggest_split_strategy(obs_df, drug_counts, gene_counts):
    """Suggest splitting strategies for drug-based evaluation."""
    print("\n" + "=" * 70)
    print("SUGGESTED DRUG-BASED SPLIT STRATEGIES")
    print("=" * 70)

    total_cells = len(obs_df)
    n_drugs = len(drug_counts)

    print(f"\nDataset overview:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Total drugs: {n_drugs}")
    print(f"  Total target genes: {len(gene_counts)}")

    print("\n--- STRATEGY 1: Random Drug Holdout ---")
    print("Split drugs randomly into train/val/test")
    train_n = int(n_drugs * 0.7)
    val_n = int(n_drugs * 0.1)
    test_n = n_drugs - train_n - val_n
    print(f"  Train: {train_n} drugs")
    print(f"  Val: {val_n} drugs")
    print(f"  Test: {test_n} drugs")

    # Estimate cell counts
    avg_cells_per_drug = drug_counts.mean()
    print(
        f"  Estimated cells: ~{train_n * avg_cells_per_drug:.0f} train, ~{val_n * avg_cells_per_drug:.0f} val, ~{test_n * avg_cells_per_drug:.0f} test"
    )

    print("\n--- STRATEGY 2: Leave-One-Drug-Out ---")
    print("Hold out each drug for testing in turn")
    print(f"  {n_drugs} test folds")
    print(f"  Each test fold: 1 drug (~{avg_cells_per_drug:,.0f} cells avg)")

    print("\n--- STRATEGY 3: Stratified by Drug Properties ---")
    print("Group drugs by size/coverage, sample from each group")
    # Quartile analysis
    q1, q2, q3 = drug_counts.quantile([0.25, 0.5, 0.75])
    print(f"  Drug count quartiles: Q1={q1:,.0f}, Q2={q2:,.0f}, Q3={q3:,.0f}")
    small = (drug_counts < q1).sum()
    medium = ((drug_counts >= q1) & (drug_counts < q3)).sum()
    large = (drug_counts >= q3).sum()
    print(f"  Small drugs (<Q1): {small}")
    print(f"  Medium drugs (Q1-Q3): {medium}")
    print(f"  Large drugs (>=Q3): {large}")

    print("\n--- CONDITION STRUCTURE FOR REFERENCE ---")
    print("Each condition = target_gene + drug combination")
    print("Similar to Norman's gene perturbation structure, but:")
    print("  - Norman: Gene knockouts → condition = gene")
    print("  - Tahoe: Drug + Target gene → condition = target_gene+drug")
    print("\nFor drug-based splits:")
    print("  - Unseen drug: No cells with this drug in training")
    print("  - Can evaluate generalization to new drugs")


def main():
    print("Loading tahoe.h5ad metadata using h5py (memory-efficient)...")

    with h5py.File(DATA_PATH, "r") as f:
        # Basic structure
        print("\n--- H5 FILE STRUCTURE ---")

        # Get n_obs and n_vars
        X = f["X"]
        n_obs = X["indptr"].shape[0] - 1
        n_vars = (
            f["var"]["_index"].shape[0] if "var" in f and "_index" in f["var"] else 0
        )
        print(f"Shape: {n_obs:,} cells x {n_vars:,} genes")

        # Read obs columns
        obs = f["obs"]
        obs_cols = [k for k in obs.keys() if not k.startswith("_")]
        print(f"obs columns: {obs_cols}")

        # Layers
        layers = list(f["layers"].keys()) if "layers" in f else []
        print(f"layers: {layers}")

        # Read obs data
        print("\nLoading obs dataframe...")
        obs_dict = {}
        for col in obs_cols:
            try:
                if "categories" in obs[col]:
                    # Categorical
                    cats = obs[col]["categories"][:]
                    codes = obs[col]["codes"][:]
                    if len(cats) > 0 and hasattr(cats[0], "decode"):
                        cats = [c.decode() for c in cats]
                    obs_dict[col] = pd.Categorical.from_codes(codes, categories=cats)
                else:
                    data = obs[col][:]
                    if len(data) > 0 and hasattr(data[0], "decode"):
                        data = [d.decode() for d in data]
                    obs_dict[col] = data
            except Exception as e:
                print(f"  Warning: could not load {col}: {e}")

        obs_df = pd.DataFrame(obs_dict)
        print(f"Loaded obs: {obs_df.shape}")

        # Show sample
        print("\n--- SAMPLE OBS ROWS ---")
        print(obs_df.head(10).to_string())

        # Normalization check
        analyze_normalization(f)

    # Condition analysis (outside h5py context)
    drug_counts, gene_counts = analyze_conditions(obs_df)

    # Split suggestions
    suggest_split_strategy(obs_df, drug_counts, gene_counts)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
