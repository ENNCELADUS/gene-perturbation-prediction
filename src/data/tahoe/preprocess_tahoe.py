#!/usr/bin/env python
"""
Tahoe preprocessing: normalize raw counts to match Norman pipeline.

Follows docs/roadmap/10_tahoe_data_preprocessing.md specification.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, spmatrix


def _normalize_log1p_matrix(
    matrix: np.ndarray | spmatrix,
    target_sum: float,
) -> np.ndarray | spmatrix:
    """Normalize counts per row to target_sum, then log1p (sparse-safe)."""
    if issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        matrix.data = matrix.data.astype(np.float32, copy=False)
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        scale = (target_sum / row_sums).astype(np.float32)
        matrix.data *= np.repeat(scale, np.diff(matrix.indptr))
        np.log1p(matrix.data, out=matrix.data)
        return matrix

    dense = np.array(matrix, dtype=np.float32, copy=True)
    row_sums = dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    dense = dense / row_sums * target_sum
    np.log1p(dense, out=dense)
    return dense


def _normalize_gene_combo(value: str) -> str:
    """Normalize gene combo strings (e.g., 'B+A' -> 'A+B')."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return text
    if text == "ctrl":
        return "ctrl"
    genes = [g.strip() for g in text.split("+") if g.strip() and g.strip() != "ctrl"]
    if not genes:
        return "ctrl"
    genes = sorted(genes)
    if len(genes) == 1:
        return genes[0]
    return "+".join(genes)


def _normalize_condition_series(series: pd.Series) -> pd.Series:
    """Normalize condition-like series to canonical gene order."""
    normalized = series.astype(str).map(_normalize_gene_combo)
    return normalized.astype("category")


def _split_genes(condition: str) -> list[str]:
    normalized = _normalize_gene_combo(condition)
    if normalized in {"", "ctrl"}:
        return []
    return [g for g in normalized.split("+") if g]


def preprocess_tahoe(
    input_path: str | Path,
    output_path: str | Path,
    target_sum: float = 1e4,
    n_hvg: Optional[int] = None,
    batch_size: int = 10000,
) -> None:
    """
    Preprocess Tahoe H5AD: normalize + log1p.

    Steps:
        1. Load H5AD in backed mode
        2. Normalize obs["condition"] by drug target genes
        3. Preserve raw counts in layers["counts"]
        4. Normalize + log1p on X
        5. Normalize ctrl layer → layers["ctrl_norm"]
        6. Optional HVG selection
        7. Write output

    Args:
        input_path: Path to raw Tahoe H5AD
        output_path: Path to write preprocessed H5AD
        target_sum: Target sum for library-size normalization
        n_hvg: Number of HVGs to select (None = keep all)
        batch_size: Cells to process at a time (memory control)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} in backed mode...")
    adata = sc.read_h5ad(input_path, backed="r")
    print(f"  Shape: {adata.shape}")

    # Load into memory for processing
    print("Loading data into memory...")
    adata = adata.to_memory()

    # Normalize condition strings to canonical gene order and merge by drug targets
    if "drug" in adata.obs and "target_gene" in adata.obs:
        print("Normalizing obs['condition'] based on drug target genes...")
        adata.obs["target_gene"] = _normalize_condition_series(adata.obs["target_gene"])

        drug_targets = (
            adata.obs[["drug", "target_gene"]]
            .dropna()
            .groupby("drug")["target_gene"]
            .apply(lambda values: sorted({g for v in values for g in _split_genes(v)}))
        )
        drug_conditions = {
            drug: "+".join(genes) if genes else "ctrl"
            for drug, genes in drug_targets.items()
        }
        adata.obs["condition"] = adata.obs["drug"].map(drug_conditions)
        adata.obs["condition"] = adata.obs["condition"].astype("category")

        if "condition_name" in adata.obs and "cell_line_id" in adata.obs:
            adata.obs["condition_name"] = (
                adata.obs["cell_line_id"].astype(str)
                + "_"
                + adata.obs["condition"].astype(str)
            )
            adata.obs["condition_name"] = adata.obs["condition_name"].astype("category")
    elif "condition" in adata.obs:
        print("Normalizing obs['condition']...")
        adata.obs["condition"] = _normalize_condition_series(adata.obs["condition"])

    # Step 1: Preserve raw counts
    print("Preserving raw counts in layers['counts']...")
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Step 2: Normalize + log1p on X
    print(f"Normalizing X (target_sum={target_sum})...")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Step 3: Normalize ctrl layer
    if "ctrl" in adata.layers:
        print("Normalizing ctrl layer → layers['ctrl_norm']...")
        # Store normalized control
        ctrl_data = adata.layers["ctrl"]
        ctrl_log = _normalize_log1p_matrix(ctrl_data, target_sum=target_sum)
        adata.layers["ctrl_norm"] = ctrl_log

    # Step 4: Optional HVG selection
    if n_hvg is not None and n_hvg < adata.n_vars:
        print(f"Selecting top {n_hvg} HVGs...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"  New shape: {adata.shape}")

    # Step 5: Write output
    print(f"Writing to {output_path}...")
    adata.write_h5ad(output_path)

    # Print summary
    print("\n=== Preprocessing Complete ===")
    print(f"Output: {output_path}")
    print(f"Shape: {adata.shape}")
    print(f"X range: {adata.X.min():.2f} - {adata.X.max():.2f}")
    if "counts" in adata.layers:
        counts = adata.layers["counts"]
        if issparse(counts):
            counts_max = counts.max()
        else:
            counts_max = counts.max()
        print(f"layers['counts'] max: {counts_max:.0f}")
    if "ctrl_norm" in adata.layers:
        ctrl_norm = adata.layers["ctrl_norm"]
        if issparse(ctrl_norm):
            ctrl_max = ctrl_norm.max()
        else:
            ctrl_max = ctrl_norm.max()
        print(f"layers['ctrl_norm'] range: 0 - {ctrl_max:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Tahoe H5AD: normalize + log1p"
    )
    parser.add_argument(
        "--input",
        default="data/tahoe/tahoe.h5ad",
        help="Input H5AD path",
    )
    parser.add_argument(
        "--output",
        default="data/processed/tahoe/tahoe_log1p.h5ad",
        help="Output H5AD path",
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
        help="Target sum for normalization",
    )
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=None,
        help="Number of HVGs to select (None = keep all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_tahoe(
        input_path=args.input,
        output_path=args.output,
        target_sum=args.target_sum,
        n_hvg=args.n_hvg,
    )


if __name__ == "__main__":
    main()
