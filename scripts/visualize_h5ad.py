#!/usr/bin/env python3
"""
Script to visualize the structure and components of the H5AD file
"""

import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def print_tree_structure(adata):
    """Print tree structure of AnnData object"""
    print("\n" + "=" * 80)
    print("📁 ANNDATA OBJECT STRUCTURE")
    print("=" * 80)

    print("\n🔷 adata")
    print(f"   ├── shape: ({adata.n_obs} cells, {adata.n_vars} genes)")
    print(f"   │")

    # X - Main expression matrix
    print(f"   ├── 🧬 X (Expression Matrix)")
    print(f"   │   ├── type: {type(adata.X).__name__}")
    print(f"   │   ├── shape: {adata.X.shape}")
    print(f"   │   ├── dtype: {adata.X.dtype}")
    if hasattr(adata.X, "nnz"):
        sparsity = (1 - adata.X.nnz / (adata.n_obs * adata.n_vars)) * 100
        print(f"   │   ├── sparsity: {sparsity:.2f}% zeros")
        print(f"   │   └── non-zero elements: {adata.X.nnz:,}")
    else:
        print(f"   │   └── storage: dense")

    # obs - Cell metadata
    print(f"   │")
    print(f"   ├── 🔬 obs (Cell/Observation Annotations)")
    print(f"   │   ├── shape: ({adata.n_obs} rows, {len(adata.obs.columns)} columns)")
    if len(adata.obs.columns) > 0:
        for i, col in enumerate(adata.obs.columns):
            prefix = "└──" if i == len(adata.obs.columns) - 1 else "├──"
            dtype = adata.obs[col].dtype
            n_unique = adata.obs[col].nunique()
            print(f"   │   {prefix} '{col}' ({dtype}, {n_unique} unique)")
    else:
        print(f"   │   └── (empty)")

    # var - Gene metadata
    print(f"   │")
    print(f"   ├── 🧬 var (Gene/Variable Annotations)")
    print(f"   │   ├── shape: ({adata.n_vars} rows, {len(adata.var.columns)} columns)")
    if len(adata.var.columns) > 0:
        for i, col in enumerate(adata.var.columns):
            prefix = "└──" if i == len(adata.var.columns) - 1 else "├──"
            dtype = adata.var[col].dtype
            n_unique = adata.var[col].nunique()
            print(f"   │   {prefix} '{col}' ({dtype}, {n_unique} unique)")
    else:
        print(f"   │   └── (empty)")

    # uns - Unstructured annotations
    print(f"   │")
    print(f"   ├── 📦 uns (Unstructured Annotations)")
    if len(adata.uns) > 0:
        for i, key in enumerate(adata.uns.keys()):
            prefix = "└──" if i == len(adata.uns) - 1 else "├──"
            value = adata.uns[key]
            print(f"   │   {prefix} '{key}': {type(value).__name__}")
    else:
        print(f"   │   └── (empty)")

    # obsm - Multi-dimensional obs annotations
    print(f"   │")
    print(f"   ├── 🗺️  obsm (Multi-dimensional Cell Annotations)")
    if len(adata.obsm) > 0:
        for i, key in enumerate(adata.obsm.keys()):
            prefix = "└──" if i == len(adata.obsm) - 1 else "├──"
            shape = adata.obsm[key].shape
            print(f"   │   {prefix} '{key}': shape {shape}")
    else:
        print(f"   │   └── (empty)")

    # varm - Multi-dimensional var annotations
    print(f"   │")
    print(f"   ├── 🗺️  varm (Multi-dimensional Gene Annotations)")
    if len(adata.varm) > 0:
        for i, key in enumerate(adata.varm.keys()):
            prefix = "└──" if i == len(adata.varm) - 1 else "├──"
            shape = adata.varm[key].shape
            print(f"   │   {prefix} '{key}': shape {shape}")
    else:
        print(f"   │   └── (empty)")

    # obsp - Pairwise obs annotations
    print(f"   │")
    print(f"   ├── 🔗 obsp (Pairwise Cell Annotations)")
    if len(adata.obsp) > 0:
        for i, key in enumerate(adata.obsp.keys()):
            prefix = "└──" if i == len(adata.obsp) - 1 else "├──"
            shape = adata.obsp[key].shape
            print(f"   │   {prefix} '{key}': shape {shape}")
    else:
        print(f"   │   └── (empty)")

    # varp - Pairwise var annotations
    print(f"   │")
    print(f"   └── 🔗 varp (Pairwise Gene Annotations)")
    if len(adata.varp) > 0:
        for i, key in enumerate(adata.varp.keys()):
            prefix = "└──" if i == len(adata.varp) - 1 else "├──"
            shape = adata.varp[key].shape
            print(f"       {prefix} '{key}': shape {shape}")
    else:
        print(f"       └── (empty)")

    print("\n")


def create_visualizations(adata, output_dir="/home/richard/projects/VCC"):
    """Create visualization plots"""
    print("=" * 80)
    print("📊 CREATING VISUALIZATIONS")
    print("=" * 80)

    # Set style
    sns.set_style("whitegrid")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Target gene distribution (top 20)
    ax1 = plt.subplot(2, 3, 1)
    target_counts = adata.obs["target_gene"].value_counts().head(20)
    colors = [
        "#e74c3c" if x == "non-targeting" else "#3498db" for x in target_counts.index
    ]
    target_counts.plot(kind="barh", ax=ax1, color=colors)
    ax1.set_xlabel("Number of Cells")
    ax1.set_title("Top 20 Target Genes (by cell count)", fontsize=12, fontweight="bold")
    ax1.invert_yaxis()

    # 2. Batch distribution
    ax2 = plt.subplot(2, 3, 2)
    batch_counts = adata.obs["batch"].value_counts().head(20)
    batch_counts.plot(kind="barh", ax=ax2, color="#2ecc71")
    ax2.set_xlabel("Number of Cells")
    ax2.set_title("Top 20 Batches (by cell count)", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()

    # 3. Expression sparsity per gene
    ax3 = plt.subplot(2, 3, 3)
    if hasattr(adata.X, "toarray"):
        # Sample genes for efficiency
        n_genes_sample = min(1000, adata.n_vars)
        gene_indices = np.random.choice(adata.n_vars, n_genes_sample, replace=False)
        sparsity_per_gene = []
        for i in gene_indices:
            col = adata.X[:, i].toarray().flatten()
            sparsity = (col == 0).sum() / len(col) * 100
            sparsity_per_gene.append(sparsity)

        ax3.hist(
            sparsity_per_gene, bins=50, color="#9b59b6", edgecolor="black", alpha=0.7
        )
        ax3.set_xlabel("Sparsity (%)")
        ax3.set_ylabel("Number of Genes")
        ax3.set_title(
            f"Gene Expression Sparsity Distribution\n(sampled {n_genes_sample} genes)",
            fontsize=12,
            fontweight="bold",
        )
        ax3.axvline(
            np.mean(sparsity_per_gene),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(sparsity_per_gene):.1f}%",
        )
        ax3.legend()

    # 4. Guide ID types distribution
    ax4 = plt.subplot(2, 3, 4)
    # Extract guide types
    guide_types = []
    for guide in adata.obs["guide_id"]:
        if "non-targeting" in guide:
            guide_types.append("non-targeting")
        elif "P1P2" in guide:
            guide_types.append("P1P2 (dual)")
        elif "_P1_" in guide or "_P2_" in guide:
            guide_types.append("P1 or P2 (single)")
        else:
            guide_types.append("other")

    guide_counter = Counter(guide_types)
    guide_df = pd.DataFrame.from_dict(guide_counter, orient="index", columns=["count"])
    guide_df.plot(
        kind="bar",
        ax=ax4,
        color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"],
        legend=False,
    )
    ax4.set_xlabel("Guide Type")
    ax4.set_ylabel("Number of Cells")
    ax4.set_title("Guide RNA Types Distribution", fontsize=12, fontweight="bold")
    ax4.tick_params(axis="x", rotation=45)

    # 5. Cells per batch heatmap (target gene vs batch)
    ax5 = plt.subplot(2, 3, 5)
    # Get top targets and batches
    top_targets = adata.obs["target_gene"].value_counts().head(15).index
    top_batches = adata.obs["batch"].value_counts().head(15).index

    # Create contingency table
    subset = adata.obs[
        adata.obs["target_gene"].isin(top_targets)
        & adata.obs["batch"].isin(top_batches)
    ]
    contingency = pd.crosstab(subset["target_gene"], subset["batch"])

    sns.heatmap(
        contingency,
        cmap="YlOrRd",
        ax=ax5,
        cbar_kws={"label": "Cell Count"},
        fmt="d",
        linewidths=0.5,
        square=False,
    )
    ax5.set_title(
        "Cell Distribution: Top 15 Targets × Top 15 Batches",
        fontsize=12,
        fontweight="bold",
    )
    ax5.set_xlabel("Batch")
    ax5.set_ylabel("Target Gene")

    # 6. Mean expression per cell
    ax6 = plt.subplot(2, 3, 6)
    # Sample cells for efficiency
    n_cells_sample = min(10000, adata.n_obs)
    cell_indices = np.random.choice(adata.n_obs, n_cells_sample, replace=False)

    if hasattr(adata.X, "toarray"):
        mean_expr = adata.X[cell_indices, :].mean(axis=1).A1
    else:
        mean_expr = adata.X[cell_indices, :].mean(axis=1)

    ax6.hist(mean_expr, bins=50, color="#16a085", edgecolor="black", alpha=0.7)
    ax6.set_xlabel("Mean Expression per Cell")
    ax6.set_ylabel("Number of Cells")
    ax6.set_title(
        f"Mean Expression Distribution\n(sampled {n_cells_sample} cells)",
        fontsize=12,
        fontweight="bold",
    )
    ax6.axvline(
        np.mean(mean_expr),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(mean_expr):.2f}",
    )
    ax6.legend()

    plt.tight_layout()
    output_path = f"{output_dir}/h5ad_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Visualization saved to: {output_path}")
    plt.close()

    # Create additional detailed plot for expression matrix
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Expression matrix sample heatmap
    ax_heat = axes[0]
    # Sample a small subset for visualization
    n_cells_viz = min(100, adata.n_obs)
    n_genes_viz = min(100, adata.n_vars)
    cell_idx = np.random.choice(adata.n_obs, n_cells_viz, replace=False)
    gene_idx = np.random.choice(adata.n_vars, n_genes_viz, replace=False)

    if hasattr(adata.X, "toarray"):
        expr_subset = adata.X[cell_idx, :][:, gene_idx].toarray()
    else:
        expr_subset = adata.X[cell_idx, :][:, gene_idx]

    sns.heatmap(
        expr_subset.T,
        cmap="viridis",
        ax=ax_heat,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Expression Level"},
    )
    ax_heat.set_title(
        f"Expression Matrix Sample\n({n_cells_viz} cells × {n_genes_viz} genes)",
        fontsize=12,
        fontweight="bold",
    )
    ax_heat.set_xlabel("Cells")
    ax_heat.set_ylabel("Genes")

    # Non-zero expression distribution
    ax_hist = axes[1]
    if hasattr(adata.X, "data"):
        # For sparse matrix, get non-zero values
        nonzero_vals = adata.X.data
    else:
        nonzero_vals = adata.X[adata.X > 0]

    # Sample if too many values
    if len(nonzero_vals) > 100000:
        nonzero_vals = np.random.choice(nonzero_vals, 100000, replace=False)

    ax_hist.hist(nonzero_vals, bins=100, color="#e67e22", edgecolor="black", alpha=0.7)
    ax_hist.set_xlabel("Expression Value (non-zero)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title(
        "Non-Zero Expression Value Distribution", fontsize=12, fontweight="bold"
    )
    ax_hist.set_yscale("log")

    plt.tight_layout()
    output_path2 = f"{output_dir}/h5ad_expression_details.png"
    plt.savefig(output_path2, dpi=300, bbox_inches="tight")
    print(f"✅ Expression details saved to: {output_path2}")
    plt.close()


def print_summary_stats(adata):
    """Print detailed summary statistics"""
    print("\n" + "=" * 80)
    print("📈 DETAILED STATISTICS")
    print("=" * 80)

    print("\n🔬 Cell (obs) Statistics:")
    print(f"  Total cells: {adata.n_obs:,}")
    print(f"  Target genes:")
    print(f"    - Unique targets: {adata.obs['target_gene'].nunique()}")
    print(
        f"    - Non-targeting (control): {(adata.obs['target_gene'] == 'non-targeting').sum():,}"
    )
    print(f"    - Perturbed: {(adata.obs['target_gene'] != 'non-targeting').sum():,}")
    print(f"  Batches:")
    print(f"    - Unique batches: {adata.obs['batch'].nunique()}")
    print(
        f"    - Cells per batch (mean): {adata.obs['batch'].value_counts().mean():.1f}"
    )
    print(f"    - Cells per batch (std): {adata.obs['batch'].value_counts().std():.1f}")

    print("\n🧬 Gene (var) Statistics:")
    print(f"  Total genes: {adata.n_vars:,}")
    if "gene_id" in adata.var.columns:
        print(f"  Unique Ensembl IDs: {adata.var['gene_id'].nunique()}")

    print("\n📊 Expression Matrix Statistics:")
    if hasattr(adata.X, "nnz"):
        total_elements = adata.n_obs * adata.n_vars
        print(f"  Total possible values: {total_elements:,}")
        print(f"  Non-zero values: {adata.X.nnz:,}")
        print(f"  Zero values: {total_elements - adata.X.nnz:,}")
        print(f"  Sparsity: {(1 - adata.X.nnz / total_elements) * 100:.2f}%")
        print(
            f"  Memory saved by sparse format: ~{(1 - adata.X.nnz / total_elements) * 100:.1f}%"
        )

        # Sample statistics
        sample_size = min(10000, adata.X.nnz)
        sample_indices = np.random.choice(adata.X.nnz, sample_size, replace=False)
        sample_values = adata.X.data[sample_indices]

        print(f"\n  Non-zero expression values (sampled {sample_size:,}):")
        print(f"    - Min: {sample_values.min():.4f}")
        print(f"    - Max: {sample_values.max():.4f}")
        print(f"    - Mean: {sample_values.mean():.4f}")
        print(f"    - Median: {np.median(sample_values):.4f}")
        print(f"    - Std: {sample_values.std():.4f}")


if __name__ == "__main__":
    data_path = "/home/richard/projects/VCC/vcc_data/adata_Training.h5ad"

    print("\n🔍 LOADING AND ANALYZING H5AD FILE\n")
    print(f"Loading: {data_path}")

    # Load data
    adata = ad.read_h5ad(data_path)

    # Print tree structure
    print_tree_structure(adata)

    # Print summary statistics
    print_summary_stats(adata)

    # Create visualizations
    create_visualizations(adata)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
