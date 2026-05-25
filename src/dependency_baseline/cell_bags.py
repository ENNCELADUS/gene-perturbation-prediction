"""Single-cell bag construction for observed Deep Sets baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from dependency_baseline.config import BaselineConfig
from dependency_baseline.features import _numeric_training_rows


@dataclass(frozen=True)
class CellBagPaths:
    bags_npz: Path
    metadata_path: Path
    summary_json: Path


def build_cell_bags(config: BaselineConfig) -> CellBagPaths:
    """Build cell-level PCA-delta bags aligned to GeneEffect labels."""
    output_dir = config.data.output_dir / "features" / "single_cell_bags"
    output_dir.mkdir(parents=True, exist_ok=True)
    bags_npz = output_dir / "replogle_k562_single_cell_bags.npz"
    metadata_path = output_dir / "feature_metadata.parquet"
    summary_json = output_dir / "feature_summary.json"

    overlap = pd.read_csv(config.data.overlap_csv)
    metadata = _numeric_training_rows(overlap, config).copy()
    genes = metadata["perturbation_gene"].astype(str).tolist()

    adata = ad.read_h5ad(config.data.h5ad_path)
    try:
        obs_labels = adata.obs[config.data.obs_perturbation_col].astype(str).to_numpy()
        matrix = _dense_float32(adata.X)
        var_symbols = _var_symbols(adata, config.data.var_gene_symbol_col)

        control_mask = obs_labels == config.data.control_label
        if not np.any(control_mask):
            msg = f"Control label {config.data.control_label!r} has no cells"
            raise ValueError(msg)

        selected_gene_indices = _select_hvg_indices(matrix, config.single_cell.n_hvg)
        selected_symbols = [var_symbols[index] for index in selected_gene_indices]
        selected = matrix[:, selected_gene_indices]
        mean = selected.mean(axis=0, dtype=np.float64)
        std = selected.std(axis=0, dtype=np.float64)
        std = np.where(std > 0, std, 1.0)
        scaled = ((selected - mean) / std).astype(np.float32)

        n_components = min(
            int(config.single_cell.n_pcs),
            scaled.shape[0],
            scaled.shape[1],
        )
        pca = PCA(n_components=n_components, random_state=config.cv.random_state)
        cell_pcs = pca.fit_transform(scaled).astype(np.float32)
        control_centroid = cell_pcs[control_mask].mean(axis=0, dtype=np.float64)

        bag_arrays: list[np.ndarray] = []
        bag_offsets = [0]
        observed_counts = []
        for gene in genes:
            mask = obs_labels == gene
            count = int(mask.sum())
            if count == 0:
                msg = f"Numeric label {gene!r} has no cells in h5ad"
                raise ValueError(msg)
            delta_pcs = cell_pcs[mask] - control_centroid[None, :]
            bag_arrays.append(delta_pcs.astype(np.float32))
            observed_counts.append(count)
            bag_offsets.append(bag_offsets[-1] + count)

        cell_delta_pcs = np.vstack(bag_arrays).astype(np.float32)
        metadata["feature_row"] = np.arange(len(metadata))
        metadata["observed_n_cells"] = observed_counts
        metadata.to_parquet(metadata_path, index=False)

        np.savez_compressed(
            bags_npz,
            cell_delta_pcs=cell_delta_pcs,
            bag_offsets=np.asarray(bag_offsets, dtype=np.int64),
            y=metadata[config.data.depmap_label_col].to_numpy(dtype=np.float32),
            n_cells=np.asarray(observed_counts, dtype=np.float32),
            perturbation_gene=np.asarray(genes, dtype=object),
            selected_gene_symbol=np.asarray(selected_symbols, dtype=object),
            pca_components=pca.components_.astype(np.float32),
            pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(
                np.float32
            ),
            hvg_mean=mean.astype(np.float32),
            hvg_std=std.astype(np.float32),
            control_pc_centroid=control_centroid.astype(np.float32),
        )
    finally:
        adata.file.close()

    summary = {
        "n_bags": int(len(metadata)),
        "n_cells": int(cell_delta_pcs.shape[0]),
        "n_hvg": int(len(selected_gene_indices)),
        "n_pcs": int(cell_delta_pcs.shape[1]),
        "control_label": config.data.control_label,
        "control_cells": int(control_mask.sum()),
        "min_cells_per_bag": int(min(observed_counts)),
        "median_cells_per_bag": float(np.median(observed_counts)),
        "max_cells_per_bag": int(max(observed_counts)),
        "single_cell_config": asdict(config.single_cell),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return CellBagPaths(
        bags_npz=bags_npz,
        metadata_path=metadata_path,
        summary_json=summary_json,
    )


def _dense_float32(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _var_symbols(adata: ad.AnnData, column: str) -> list[str]:
    if column in adata.var.columns:
        return adata.var[column].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _select_hvg_indices(matrix: np.ndarray, requested: int) -> np.ndarray:
    n_genes = matrix.shape[1]
    n_keep = min(max(1, int(requested)), n_genes)
    variance = matrix.var(axis=0)
    order = np.argsort(variance)[::-1]
    return np.sort(order[:n_keep])
