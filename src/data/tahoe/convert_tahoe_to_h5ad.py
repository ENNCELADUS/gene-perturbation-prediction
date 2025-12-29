#!/usr/bin/env python
"""Convert Tahoe parquet files to per-file H5AD outputs aligned with Norman structure."""

import glob
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm


def build_gene_vocabulary(files: list[str]) -> dict[int, int]:
    """Scan all parquets to build a unified gene vocabulary.

    Returns:
        Mapping from original gene index -> new vocabulary index
    """
    all_genes = set()
    for f in tqdm(files, desc="Building gene vocabulary"):
        df = pd.read_parquet(f, columns=["genes"])
        for arr in df["genes"]:
            all_genes.update(arr.tolist())

    sorted_genes = sorted(all_genes)
    return {g: i for i, g in enumerate(sorted_genes)}


def process_parquet(
    filepath: str,
    gene_to_idx: dict[int, int],
    n_genes: int,
) -> tuple[csr_matrix, csr_matrix, pd.DataFrame]:
    """Process a single parquet file.

    Returns:
        - X: expression matrix (n_cells x n_genes)
        - ctrl: control expression matrix (n_cells x n_genes)
        - obs: cell metadata DataFrame
    """
    df = pd.read_parquet(filepath)
    n_cells = len(df)

    # Build sparse matrices using lil for efficient row-by-row construction
    X = lil_matrix((n_cells, n_genes), dtype=np.float32)
    ctrl = lil_matrix((n_cells, n_genes), dtype=np.float32)

    for i in range(n_cells):
        row = df.iloc[i]

        # Treated expression
        genes = row["genes"]
        exprs = row["expressions"]
        for g, e in zip(genes, exprs):
            if g in gene_to_idx:
                X[i, gene_to_idx[g]] = e

        # Control expression
        ctrl_genes = row["ctrl_genes"]
        ctrl_exprs = row["ctrl_expressions"]
        for g, e in zip(ctrl_genes, ctrl_exprs):
            if g in gene_to_idx:
                ctrl[i, gene_to_idx[g]] = e

    # Build obs DataFrame
    obs = pd.DataFrame(
        {
            "condition": df["target_gene"] + "+" + df["drug"],
            "cell_line_id": df["cell_line_id"],
            "plate": df["plate"],
            "sample": df["sample"],
            "label": df["label"],
            "overlap_ratio": df["overlap_ratio"],
            "control": 0,
            "target_gene": df["target_gene"],
            "drug": df["drug"],
            "ctrl_drug": df["ctrl_drug"],
        }
    )
    obs["condition_name"] = obs["cell_line_id"] + "_" + obs["condition"]

    return X.tocsr(), ctrl.tocsr(), obs


def main():
    data_dir = Path("data/tahoe_mulGene_52drug")
    output_dir = data_dir / "h5ad"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(data_dir / "single_target_*.parquet")))
    print(f"Found {len(files)} parquet files")

    # Step 1: Build gene vocabulary
    print("\n=== Step 1: Building gene vocabulary ===")
    gene_to_idx = build_gene_vocabulary(files)
    n_genes = len(gene_to_idx)
    print(f"Vocabulary size: {n_genes} genes")

    # Step 2: Build var DataFrame once
    idx_to_gene = {v: k for k, v in gene_to_idx.items()}
    var = pd.DataFrame({"gene_idx": [idx_to_gene[i] for i in range(n_genes)]})
    var.index = var.index.astype(str)

    # Step 3: Process each parquet and save immediately
    print("\n=== Step 2: Processing parquet files ===")
    for f in tqdm(files, desc="Processing parquets"):
        X, ctrl, obs = process_parquet(f, gene_to_idx, n_genes)

        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=var,
            layers={"ctrl": ctrl},
        )

        # Convert categorical columns
        for col in [
            "condition",
            "cell_line_id",
            "plate",
            "sample",
            "target_gene",
            "drug",
            "ctrl_drug",
            "condition_name",
        ]:
            adata.obs[col] = adata.obs[col].astype("category")

        output_path = output_dir / f"{Path(f).stem}.h5ad"
        adata.write_h5ad(output_path)

        # Free per-file memory before moving to the next file
        del adata, X, ctrl, obs

    print("Done!")


if __name__ == "__main__":
    main()
