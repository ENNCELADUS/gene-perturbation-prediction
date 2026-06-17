"""Prepare compact gwps PCA mean-delta bags for Exp07 SL-pair augmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from dependency_baseline.cell_bags import (
    PCA_FEATURE_SET,
    _PcaProjector,
    _chunked_gene_moments,
    _fit_incremental_pca,
    _scaled_selected_block,
    _select_hvg_indices,
    _var_symbols,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one mean pooled PCA-delta row per Replogle gwps gene for "
            "Exp07. This is a compact equivalent of mean-pooling full cell bags."
        )
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=Path(
            "data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad"
        ),
    )
    parser.add_argument(
        "--overlap-csv",
        type=Path,
        default=Path("data/sl_dependency_v0/interim/k562_gwps_sl_overlap_for_bags.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/experiments/07_k562_sl_pair_perturbseq_augmented/"
            "gwps_pca_mean_bags"
        ),
    )
    parser.add_argument("--obs-perturbation-col", default="gene")
    parser.add_argument("--var-gene-symbol-col", default="gene_name")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bags_npz = args.output_dir / "bags.npz"
    metadata_csv = args.output_dir / "feature_metadata.csv"
    summary_json = args.output_dir / "feature_summary.json"

    metadata = pd.read_csv(args.overlap_csv).copy()
    genes = metadata["perturbation_gene"].astype(str).tolist()
    gene_to_index = {gene: index for index, gene in enumerate(genes)}

    adata = ad.read_h5ad(args.h5ad, backed="r")
    try:
        obs_labels = adata.obs[args.obs_perturbation_col].astype(str).to_numpy()
        var_symbols = _var_symbols(adata, args.var_gene_symbol_col)
        mean, std, variance = _chunked_gene_moments(adata.X, args.chunk_size)
        selected_gene_indices = _select_hvg_indices(variance, args.n_hvg)
        selected_symbols = [var_symbols[index] for index in selected_gene_indices]
        selected_mean = mean[selected_gene_indices]
        selected_std = std[selected_gene_indices]
        n_components = min(args.n_pcs, adata.n_obs, len(selected_gene_indices))
        pca = _fit_incremental_pca(
            adata.X,
            selected_gene_indices,
            selected_mean,
            selected_std,
            n_components=n_components,
            chunk_size=args.chunk_size,
        )
        projector = _PcaProjector(
            pca.components_.astype(np.float32),
            pca.mean_.astype(np.float32),
        )

        gene_sums = np.zeros((len(genes), n_components), dtype=np.float64)
        gene_counts = np.zeros(len(genes), dtype=np.int64)
        control_sum = np.zeros(n_components, dtype=np.float64)
        control_count = 0
        for start in range(0, adata.n_obs, args.chunk_size):
            stop = min(start + args.chunk_size, adata.n_obs)
            labels = obs_labels[start:stop]
            scaled = _scaled_selected_block(
                adata.X,
                start,
                stop,
                selected_gene_indices,
                selected_mean,
                selected_std,
            )
            embeddings = projector.transform(scaled).astype(np.float32)
            for label in np.unique(labels):
                mask = labels == label
                if label == args.control_label:
                    control_sum += embeddings[mask].sum(axis=0, dtype=np.float64)
                    control_count += int(mask.sum())
                    continue
                gene_index = gene_to_index.get(str(label))
                if gene_index is None:
                    continue
                gene_sums[gene_index] += embeddings[mask].sum(axis=0, dtype=np.float64)
                gene_counts[gene_index] += int(mask.sum())
        if control_count == 0:
            raise ValueError(f"Control label {args.control_label!r} has no cells")
        missing = [
            gene for gene, count in zip(genes, gene_counts, strict=True) if count == 0
        ]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(
                f"{len(missing)} gwps genes have no cells in h5ad: {preview}"
            )

        control_centroid = control_sum / float(control_count)
        gene_means = gene_sums / gene_counts[:, None]
        cell_delta_pcs = (gene_means - control_centroid[None, :]).astype(np.float32)
        metadata["feature_row"] = np.arange(len(metadata))
        metadata["observed_n_cells"] = gene_counts.astype(int)
        metadata.to_csv(metadata_csv, index=False)
        np.savez_compressed(
            bags_npz,
            cell_delta_pcs=cell_delta_pcs,
            feature_set=np.asarray(PCA_FEATURE_SET, dtype=object),
            bag_offsets=np.arange(len(genes) + 1, dtype=np.int64),
            perturbation_gene=np.asarray(genes, dtype=object),
            n_cells=gene_counts.astype(np.float32),
            selected_gene_symbol=np.asarray(selected_symbols, dtype=object),
            hvg_mean=selected_mean.astype(np.float32),
            hvg_std=selected_std.astype(np.float32),
            control_embedding_centroid=control_centroid.astype(np.float32),
            pca_components=pca.components_.astype(np.float32),
            pca_mean=pca.mean_.astype(np.float32),
            pca_explained_variance_ratio=np.asarray(
                pca.explained_variance_ratio_, dtype=np.float32
            ),
        )
    finally:
        adata.file.close()

    summary = {
        "feature_set": PCA_FEATURE_SET,
        "n_bags": int(len(genes)),
        "n_cells": int(gene_counts.sum()),
        "n_hvg": int(len(selected_gene_indices)),
        "embedding_dim": int(cell_delta_pcs.shape[1]),
        "control_label": args.control_label,
        "control_cells": int(control_count),
        "min_cells_per_bag": int(gene_counts.min()),
        "median_cells_per_bag": float(np.median(gene_counts)),
        "max_cells_per_bag": int(gene_counts.max()),
        "compact_mean_pooled": True,
        "bags_npz": str(bags_npz),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"bags: {bags_npz}")
    print(f"metadata: {metadata_csv}")
    print(f"summary: {summary_json}")


if __name__ == "__main__":
    main()
