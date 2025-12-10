import scanpy as sc
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def sample_data():
    input_path = "data/processed/test.h5ad"
    output_path = "tests/data/metrics_test_data.h5ad"

    print(f"Reading {input_path}...")
    adata = sc.read_h5ad(input_path)

    # 1. Select NTC and 2 specific perturbations
    # Assuming 'condition' or similar column holds perturbation info.
    # Let's check available keys if we were exploring, but I'll assume standard naming from previous context.
    # Typically 'condition' stores perturbation names, 'control' is often 'ctrl' or 'non-targeting'.

    # Let's inspect the obs columns to be safe about the column name
    print("Obs columns:", adata.obs.columns)

    pert_col = "target_gene"
    print(f"Using perturbation column: {pert_col}")

    # Check for NTC
    ntc_candidates = ["ctrl", "control", "non-targeting", "NTC", "mock"]
    ntc_val = None
    all_vals = adata.obs[pert_col].unique()

    for cand in ntc_candidates:
        if cand in all_vals:
            ntc_val = cand
            break

    if ntc_val is None:
        print("Warning: Could not identify NTC value in test.h5ad.")
        # Mock strategy: use the first perturbation as "control" for testing purposes
        ntc_val = all_vals[0]
        print(f"Mocking NTC using: {ntc_val}")

    print(f"Identified NTC (or mock): {ntc_val}")

    # Pick 2 other perturbations
    all_perts = [p for p in all_vals if p != ntc_val]
    selected_perts = all_perts[:2]
    print(f"Selected perturbations: {selected_perts}")

    # Filter cells
    subset_obs = adata.obs[adata.obs[pert_col].isin([ntc_val] + selected_perts)]
    adata_subset = adata[subset_obs.index].copy()

    # 2. Downsample genes to ~100-500
    print("Selecting top 500 highly variable genes...")
    # Use seurat flavor which works on count data typically
    try:
        sc.pp.highly_variable_genes(
            adata_subset,
            n_top_genes=500,
            flavor="seurat_v3" if "counts" in adata_subset.layers else "seurat",
        )
    except Exception as e:
        print(f"HVG selection failed: {e}. Selecting random 500 genes.")
        # Fallback to random genes if HVG fails (e.g. if precomputed pca/etc missing or data format issue)
        genes = np.random.choice(adata_subset.var_names, 500, replace=False)
        adata_subset = adata_subset[:, genes].copy()

    if "highly_variable" in adata_subset.var.columns:
        adata_subset = adata_subset[:, adata_subset.var["highly_variable"]].copy()

    # 3. Downsample cells per condition
    sampled_indices = []
    for p in [ntc_val] + selected_perts:
        indices = adata_subset.obs[adata_subset.obs[pert_col] == p].index
        if len(indices) > 50:
            sampled_indices.extend(np.random.choice(indices, 50, replace=False))
        else:
            sampled_indices.extend(indices)

    adata_final = adata_subset[sampled_indices].copy()

    # Update the NTC name in the mock control cells to 'control' to make tests clearer?
    # Or just keep it as is and let the test know.
    # Let's keep it as is but print a mapping.
    print(f"Final shape: {adata_final.shape}")
    print(f"Saving to {output_path}...")
    adata_final.write_h5ad(output_path)
    print("Done.")


if __name__ == "__main__":
    sample_data()
