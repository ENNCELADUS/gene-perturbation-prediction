"""Single-cell bag construction for observed Deep Sets baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import IncrementalPCA

from dependency_baseline.config import BaselineConfig, ExternalFeatureSourceConfig
from dependency_baseline.features import (
    _detect_control_label,
    _external_source_overlap,
    _numeric_training_rows,
    _source_perturbation_label_col,
)


@dataclass(frozen=True)
class CellBagPaths:
    bags_npz: Path
    metadata_path: Path
    summary_json: Path


PCA_FEATURE_SET = "single_cell_pc_delta"
SCVI_FEATURE_SET = "single_cell_scvi_delta"
HVG_FEATURE_SET = "single_cell_hvg_delta"
SUPPORTED_SINGLE_CELL_FEATURE_SETS = (
    PCA_FEATURE_SET,
    SCVI_FEATURE_SET,
    HVG_FEATURE_SET,
)


def build_cell_bags(
    config: BaselineConfig,
    feature_set: str = PCA_FEATURE_SET,
) -> CellBagPaths:
    """Build cell-level embedding-delta bags aligned to GeneEffect labels."""
    _validate_feature_set(feature_set)
    output_dir = _cell_bag_output_dir(config.data.output_dir, feature_set)
    output_dir.mkdir(parents=True, exist_ok=True)
    bags_npz = output_dir / _cell_bag_filename(feature_set)
    metadata_path = output_dir / "feature_metadata.parquet"
    summary_json = output_dir / "feature_summary.json"

    overlap = pd.read_csv(config.data.overlap_csv)
    metadata = _numeric_training_rows(overlap, config).copy()
    genes = metadata["perturbation_gene"].astype(str).tolist()

    adata = ad.read_h5ad(config.data.h5ad_path, backed="r")
    try:
        obs_labels = adata.obs[config.data.obs_perturbation_col].astype(str).to_numpy()
        var_symbols = _var_symbols(adata, config.data.var_gene_symbol_col)

        control_mask = obs_labels == config.data.control_label
        if not np.any(control_mask):
            msg = f"Control label {config.data.control_label!r} has no cells"
            raise ValueError(msg)

        mean, std, variance = _chunked_gene_moments(
            adata.X,
            chunk_size=config.features.chunk_size,
        )
        selected_gene_indices = _select_hvg_indices(
            variance,
            config.single_cell.n_hvg,
        )
        selected_symbols = [var_symbols[index] for index in selected_gene_indices]
        selected_mean = mean[selected_gene_indices]
        selected_std = std[selected_gene_indices]
        if feature_set == PCA_FEATURE_SET:
            bag_arrays, observed_counts, control_centroid, extra_payload = (
                _build_replogle_pca_bags(
                    adata,
                    obs_labels,
                    genes,
                    selected_gene_indices,
                    selected_mean,
                    selected_std,
                    config,
                )
            )
        elif feature_set == HVG_FEATURE_SET:
            bag_arrays, observed_counts, control_centroid, extra_payload = (
                _build_replogle_hvg_bags(
                    adata,
                    obs_labels,
                    genes,
                    selected_gene_indices,
                    selected_mean,
                    selected_std,
                    config,
                )
            )
        else:
            bag_arrays, observed_counts, control_centroid, extra_payload = (
                _build_replogle_scvi_bags(
                    adata,
                    obs_labels,
                    genes,
                    selected_gene_indices,
                    selected_symbols,
                    config,
                    output_dir,
                )
            )
        bag_offsets = [0]
        for count in observed_counts:
            bag_offsets.append(bag_offsets[-1] + count)

        cell_embeddings = np.vstack(bag_arrays).astype(np.float32)
        metadata["feature_row"] = np.arange(len(metadata))
        metadata["observed_n_cells"] = observed_counts
        metadata.to_parquet(metadata_path, index=False)

        np.savez_compressed(
            bags_npz,
            cell_delta_pcs=cell_embeddings,
            feature_set=np.asarray(feature_set, dtype=object),
            bag_offsets=np.asarray(bag_offsets, dtype=np.int64),
            y=metadata[config.data.depmap_label_col].to_numpy(dtype=np.float32),
            n_cells=np.asarray(observed_counts, dtype=np.float32),
            perturbation_gene=np.asarray(genes, dtype=object),
            selected_gene_symbol=np.asarray(selected_symbols, dtype=object),
            hvg_mean=selected_mean.astype(np.float32),
            hvg_std=selected_std.astype(np.float32),
            control_embedding_centroid=control_centroid.astype(np.float32),
            **extra_payload,
        )
    finally:
        adata.file.close()

    summary = {
        "feature_set": feature_set,
        "n_bags": int(len(metadata)),
        "n_cells": int(cell_embeddings.shape[0]),
        "n_hvg": int(len(selected_gene_indices)),
        "embedding_dim": int(cell_embeddings.shape[1]),
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


def _build_replogle_pca_bags(
    adata: ad.AnnData,
    obs_labels: np.ndarray,
    genes: list[str],
    selected_gene_indices: np.ndarray,
    selected_mean: np.ndarray,
    selected_std: np.ndarray,
    config: BaselineConfig,
) -> tuple[list[np.ndarray], list[int], np.ndarray, dict[str, object]]:
    n_components = min(
        int(config.single_cell.n_pcs),
        int(adata.n_obs),
        len(selected_gene_indices),
    )
    pca = _fit_incremental_pca(
        adata.X,
        selected_gene_indices,
        selected_mean,
        selected_std,
        n_components=n_components,
        chunk_size=config.features.chunk_size,
    )
    bag_arrays, observed_counts, control_centroid = _collect_projected_bags(
        matrix=adata.X,
        obs_labels=obs_labels,
        genes=genes,
        selected_gene_indices=selected_gene_indices,
        reference_mean=selected_mean,
        reference_std=selected_std,
        projector=_PcaProjector(
            pca.components_.astype(np.float32),
            pca.mean_.astype(np.float32),
        ),
        control_label=config.data.control_label,
        chunk_size=config.features.chunk_size,
    )
    payload = {
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_explained_variance_ratio": np.asarray(
            pca.explained_variance_ratio_,
            dtype=np.float32,
        ),
    }
    return bag_arrays, observed_counts, control_centroid, payload


def _build_replogle_hvg_bags(
    adata: ad.AnnData,
    obs_labels: np.ndarray,
    genes: list[str],
    selected_gene_indices: np.ndarray,
    selected_mean: np.ndarray,
    selected_std: np.ndarray,
    config: BaselineConfig,
) -> tuple[list[np.ndarray], list[int], np.ndarray, dict[str, object]]:
    bag_arrays, observed_counts, control_centroid = _collect_projected_bags(
        matrix=adata.X,
        obs_labels=obs_labels,
        genes=genes,
        selected_gene_indices=selected_gene_indices,
        reference_mean=selected_mean,
        reference_std=selected_std,
        projector=_IdentityProjector(),
        control_label=config.data.control_label,
        chunk_size=config.features.chunk_size,
    )
    return bag_arrays, observed_counts, control_centroid, {}


def _build_replogle_scvi_bags(
    adata: ad.AnnData,
    obs_labels: np.ndarray,
    genes: list[str],
    selected_gene_indices: np.ndarray,
    selected_symbols: list[str],
    config: BaselineConfig,
    output_dir: Path,
) -> tuple[list[np.ndarray], list[int], np.ndarray, dict[str, object]]:
    scvi = _import_scvi()

    selected = _dense_float32(adata.X[:, selected_gene_indices])
    scvi_adata = ad.AnnData(selected)
    scvi_adata.obs[config.data.obs_perturbation_col] = obs_labels
    scvi_adata.var_names = selected_symbols
    scvi.model.SCVI.setup_anndata(scvi_adata)
    model = scvi.model.SCVI(
        scvi_adata,
        n_latent=int(config.single_cell.scvi_latent_dim),
        n_hidden=int(config.single_cell.scvi_hidden_units),
        n_layers=int(config.single_cell.scvi_layers),
        dropout_rate=float(config.single_cell.dropout),
    )
    model.train(
        max_epochs=int(config.single_cell.scvi_max_epochs),
        batch_size=int(config.single_cell.scvi_batch_size),
        early_stopping=True,
    )
    latent = np.asarray(model.get_latent_representation(), dtype=np.float32)
    model_dir = output_dir / "scvi_model"
    model.save(str(model_dir), overwrite=True, save_anndata=False)
    bag_arrays, observed_counts, control_centroid = _collect_latent_bags(
        latent,
        obs_labels,
        genes,
        control_label=config.data.control_label,
    )
    payload = {
        "scvi_model_dir": np.asarray(str(model_dir), dtype=object),
        "scvi_latent_dim": np.asarray(int(config.single_cell.scvi_latent_dim)),
        "scvi_tools_version": np.asarray(getattr(scvi, "__version__", "unknown")),
    }
    return bag_arrays, observed_counts, control_centroid, payload


def _import_scvi() -> object:
    try:
        import sys
        import types

        import jax
        import numpy as _np
        import scipy.linalg as _scipy_linalg

        if not hasattr(_scipy_linalg, "tril"):
            _scipy_linalg.tril = _np.tril
        if not hasattr(_scipy_linalg, "triu"):
            _scipy_linalg.triu = _np.triu
        if "jaxlib.xla_extension" not in sys.modules:
            xla_extension = types.ModuleType("jaxlib.xla_extension")
            xla_extension.Device = jax.Device
            sys.modules["jaxlib.xla_extension"] = xla_extension
        import scvi
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        msg = "scvi-tools is required to build single_cell_scvi_delta bags"
        raise ImportError(msg) from error
    return scvi


def build_external_cell_bags(
    config: BaselineConfig,
    reference_bags_npz: Path,
    external_name: str = "adamson_k562",
    feature_set: str | None = None,
) -> CellBagPaths:
    """Build gene-level external single-cell bags in a reference embedding space."""
    reference = np.load(reference_bags_npz, allow_pickle=True)
    resolved_feature_set = feature_set or _feature_set_from_payload(reference)
    _validate_feature_set(resolved_feature_set)
    output_dir = (
        config.data.output_dir
        / "features"
        / "external"
        / _external_cell_bag_dirname(external_name, resolved_feature_set)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bags_npz = output_dir / _external_cell_bag_filename(
        external_name,
        resolved_feature_set,
    )
    metadata_path = output_dir / "feature_metadata.parquet"
    summary_json = output_dir / "feature_summary.json"

    reference_symbols = reference["selected_gene_symbol"].astype(str).tolist()
    reference_mean = reference["hvg_mean"].astype(np.float32)
    reference_std = reference["hvg_std"].astype(np.float32)
    projector = _external_projector_from_reference(reference, resolved_feature_set)

    overlap_path = _external_overlap_path(config, external_name)
    overlap = pd.read_csv(overlap_path)
    numeric_overlap = _numeric_training_rows(overlap, config)
    source_rows: list[pd.DataFrame] = []
    source_bags: list[list[np.ndarray]] = []
    source_qa = []
    for source in _external_sources(config, external_name):
        source_overlap = _external_source_overlap(numeric_overlap, source.name)
        rows, bags, qa = _build_external_cell_source_rows(
            source=source,
            overlap=source_overlap,
            depmap_label_col=config.data.depmap_label_col,
            reference_symbols=reference_symbols,
            reference_mean=reference_mean,
            reference_std=reference_std,
            projector=projector,
            chunk_size=config.features.chunk_size,
        )
        source_qa.append(qa)
        if rows.empty:
            continue
        source_rows.append(rows)
        source_bags.extend([[bag] for bag in bags])

    if not source_rows:
        msg = "No configured external source produced numeric single-cell bags"
        raise ValueError(msg)
    row_metadata = pd.concat(source_rows, ignore_index=True)
    flat_bags = [bags[0] for bags in source_bags]
    metadata, gene_bags = _aggregate_external_cell_gene_rows(row_metadata, flat_bags)
    bag_offsets = [0]
    for bag in gene_bags:
        bag_offsets.append(bag_offsets[-1] + bag.shape[0])
    cell_embeddings = np.vstack(gene_bags).astype(np.float32)

    metadata["feature_row"] = np.arange(len(metadata))
    metadata.to_parquet(metadata_path, index=False)
    np.savez_compressed(
        bags_npz,
        cell_delta_pcs=cell_embeddings,
        feature_set=np.asarray(resolved_feature_set, dtype=object),
        bag_offsets=np.asarray(bag_offsets, dtype=np.int64),
        y=metadata[config.data.depmap_label_col].to_numpy(dtype=np.float32),
        n_cells=metadata["observed_n_cells"].to_numpy(dtype=np.float32),
        perturbation_gene=metadata["perturbation_gene"]
        .astype(str)
        .to_numpy(dtype=object),
        source_dataset=metadata["source_dataset"].astype(str).to_numpy(dtype=object),
        external_row_count=metadata["external_row_count"].to_numpy(dtype=np.int32),
        reference_bags_npz=str(reference_bags_npz),
        selected_gene_symbol=np.asarray(reference_symbols, dtype=object),
    )
    summary = {
        "external_name": external_name,
        "feature_set": resolved_feature_set,
        "n_source_rows": int(len(row_metadata)),
        "n_gene_rows": int(len(metadata)),
        "n_cells": int(cell_embeddings.shape[0]),
        "embedding_dim": int(cell_embeddings.shape[1]),
        "reference_bags_npz": str(reference_bags_npz),
        "external_overlap_csv": str(overlap_path),
        "min_cells_per_gene": int(metadata["observed_n_cells"].min()),
        "median_cells_per_gene": float(np.median(metadata["observed_n_cells"])),
        "max_cells_per_gene": int(metadata["observed_n_cells"].max()),
        "sources": source_qa,
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


def _select_hvg_indices(variance: np.ndarray, requested: int) -> np.ndarray:
    n_genes = variance.shape[0]
    n_keep = min(max(1, int(requested)), n_genes)
    order = np.argsort(variance)[::-1]
    return np.sort(order[:n_keep])


def _validate_feature_set(feature_set: str) -> None:
    if feature_set not in SUPPORTED_SINGLE_CELL_FEATURE_SETS:
        allowed = ", ".join(SUPPORTED_SINGLE_CELL_FEATURE_SETS)
        msg = (
            f"Unsupported single-cell feature set {feature_set!r}; "
            f"expected one of {allowed}"
        )
        raise ValueError(msg)


def _feature_set_from_payload(payload: object) -> str:
    if "feature_set" not in payload:
        return PCA_FEATURE_SET
    value = payload["feature_set"]
    if np.asarray(value).shape == ():
        return str(value.item())
    return str(np.asarray(value).astype(str).reshape(-1)[0])


def _cell_bag_output_dir(output_dir: Path, feature_set: str) -> Path:
    root = output_dir / "features" / "single_cell_bags"
    if feature_set == PCA_FEATURE_SET:
        return root
    return root / feature_set


def _cell_bag_filename(feature_set: str) -> str:
    if feature_set == PCA_FEATURE_SET:
        return "replogle_k562_single_cell_bags.npz"
    return f"replogle_k562_{feature_set}_bags.npz"


def _external_cell_bag_dirname(external_name: str, feature_set: str) -> str:
    if feature_set == PCA_FEATURE_SET:
        return f"{external_name}_single_cell_bags"
    return f"{external_name}_{feature_set}_bags"


def _external_cell_bag_filename(external_name: str, feature_set: str) -> str:
    if feature_set == PCA_FEATURE_SET:
        return f"{external_name}_single_cell_bags.npz"
    return f"{external_name}_{feature_set}_bags.npz"


class _IdentityProjector:
    def transform(self, scaled: np.ndarray) -> np.ndarray:
        return scaled.astype(np.float32)


@dataclass(frozen=True)
class _PcaProjector:
    components: np.ndarray
    mean: np.ndarray

    def transform(self, scaled: np.ndarray) -> np.ndarray:
        return ((scaled - self.mean[None, :]) @ self.components.T).astype(np.float32)


@dataclass(frozen=True)
class _ScviProjector:
    model_dir: Path
    reference_symbols: list[str]

    def transform_source(
        self,
        source: ExternalFeatureSourceConfig,
        matrix: object,
        source_indices: np.ndarray,
        reference_mean: np.ndarray,
    ) -> np.ndarray:
        scvi = _import_scvi()

        raw = np.tile(reference_mean[None, :], (matrix.shape[0], 1)).astype(np.float32)
        valid = source_indices >= 0
        if valid.any():
            raw[:, valid] = _dense_float32(matrix[:, source_indices[valid]])
        query = ad.AnnData(raw)
        query.var_names = self.reference_symbols
        try:
            query_model = scvi.model.SCVI.load(str(self.model_dir), adata=query)
        except Exception as error:
            msg = (
                "Frozen scVI external encoding is unsupported for this reference "
                "model/runtime. Refusing Adamson query fine-tuning; rebuild or "
                "evaluate a non-scVI feature set instead."
            )
            raise RuntimeError(msg) from error
        del source
        return np.asarray(query_model.get_latent_representation(), dtype=np.float32)


def _external_projector_from_reference(
    reference: object,
    feature_set: str,
) -> object:
    if feature_set == HVG_FEATURE_SET:
        return _IdentityProjector()
    if feature_set == SCVI_FEATURE_SET:
        if "scvi_model_dir" not in reference:
            msg = "Reference scVI bags are missing scvi_model_dir"
            raise ValueError(msg)
        return _ScviProjector(
            model_dir=Path(str(reference["scvi_model_dir"].item())),
            reference_symbols=reference["selected_gene_symbol"].astype(str).tolist(),
        )
    pca_components = reference["pca_components"].astype(np.float32)
    pca_mean = (
        reference["pca_mean"].astype(np.float32)
        if "pca_mean" in reference
        else np.zeros(len(reference["selected_gene_symbol"]), dtype=np.float32)
    )
    return _PcaProjector(pca_components, pca_mean)


def _chunked_gene_moments(
    matrix: object,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_obs, n_vars = matrix.shape
    sums = np.zeros(n_vars, dtype=np.float64)
    sums_sq = np.zeros(n_vars, dtype=np.float64)
    for start in range(0, n_obs, chunk_size):
        stop = min(start + chunk_size, n_obs)
        block = _dense_float32(matrix[start:stop])
        block64 = block.astype(np.float64, copy=False)
        sums += block64.sum(axis=0)
        sums_sq += np.square(block64).sum(axis=0)
    mean = sums / float(n_obs)
    variance = np.maximum(sums_sq / float(n_obs) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std > 0, std, 1.0)
    return mean, std, variance


def _fit_incremental_pca(
    matrix: object,
    selected_gene_indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    n_components: int,
    chunk_size: int,
) -> IncrementalPCA:
    pca = IncrementalPCA(n_components=n_components)
    n_obs = matrix.shape[0]
    fit_chunk_size = max(int(chunk_size), int(n_components))
    for start in range(0, n_obs, fit_chunk_size):
        stop = min(start + fit_chunk_size, n_obs)
        scaled = _scaled_selected_block(
            matrix,
            start,
            stop,
            selected_gene_indices,
            mean,
            std,
        )
        if scaled.shape[0] < n_components:
            continue
        pca.partial_fit(scaled)
    if not hasattr(pca, "components_"):
        msg = "Not enough cells to fit single-cell PCA"
        raise ValueError(msg)
    return pca


def _collect_projected_bags(
    *,
    matrix: object,
    obs_labels: np.ndarray,
    genes: list[str],
    selected_gene_indices: np.ndarray,
    reference_mean: np.ndarray,
    reference_std: np.ndarray,
    projector: object,
    control_label: str,
    chunk_size: int,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    group_to_index = {gene: index for index, gene in enumerate(genes)}
    grouped_embeddings: list[list[np.ndarray]] = [[] for _gene in genes]
    control_sum: np.ndarray | None = None
    control_count = 0
    for start in range(0, matrix.shape[0], chunk_size):
        stop = min(start + chunk_size, matrix.shape[0])
        labels = obs_labels[start:stop]
        scaled = _scaled_selected_block(
            matrix,
            start,
            stop,
            selected_gene_indices,
            reference_mean,
            reference_std,
        )
        embeddings = projector.transform(scaled).astype(np.float32)
        if control_sum is None:
            control_sum = np.zeros(embeddings.shape[1], dtype=np.float64)
        control_mask = labels == control_label
        if np.any(control_mask):
            control_sum += embeddings[control_mask].sum(axis=0, dtype=np.float64)
            control_count += int(control_mask.sum())
        for gene, group_index in group_to_index.items():
            mask = labels == gene
            if np.any(mask):
                grouped_embeddings[group_index].append(embeddings[mask])
    if control_count == 0 or control_sum is None:
        msg = f"Control label {control_label!r} has no cells"
        raise ValueError(msg)
    control_centroid = control_sum / float(control_count)
    bag_arrays = []
    observed_counts = []
    for gene, chunks in zip(genes, grouped_embeddings, strict=True):
        if not chunks:
            msg = f"Numeric label {gene!r} has no cells in h5ad"
            raise ValueError(msg)
        embeddings = np.vstack(chunks).astype(np.float32)
        observed_counts.append(int(embeddings.shape[0]))
        bag_arrays.append((embeddings - control_centroid[None, :]).astype(np.float32))
    return bag_arrays, observed_counts, control_centroid


def _collect_latent_bags(
    latent: np.ndarray,
    obs_labels: np.ndarray,
    genes: list[str],
    *,
    control_label: str,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    control_mask = obs_labels == control_label
    if not np.any(control_mask):
        msg = f"Control label {control_label!r} has no cells"
        raise ValueError(msg)
    control_centroid = latent[control_mask].mean(axis=0, dtype=np.float64)
    bag_arrays = []
    observed_counts = []
    for gene in genes:
        mask = obs_labels == gene
        if not np.any(mask):
            msg = f"Numeric label {gene!r} has no cells in h5ad"
            raise ValueError(msg)
        values = latent[mask].astype(np.float32)
        observed_counts.append(int(values.shape[0]))
        bag_arrays.append((values - control_centroid[None, :]).astype(np.float32))
    return bag_arrays, observed_counts, control_centroid


def _collect_pc_bags(
    matrix: object,
    obs_labels: np.ndarray,
    genes: list[str],
    selected_gene_indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    pca: IncrementalPCA,
    *,
    control_label: str,
    chunk_size: int,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    group_to_index = {gene: index for index, gene in enumerate(genes)}
    grouped_pcs: list[list[np.ndarray]] = [[] for _gene in genes]
    control_sum = np.zeros(pca.n_components_, dtype=np.float64)
    control_count = 0
    for start in range(0, matrix.shape[0], chunk_size):
        stop = min(start + chunk_size, matrix.shape[0])
        labels = obs_labels[start:stop]
        scaled = _scaled_selected_block(
            matrix,
            start,
            stop,
            selected_gene_indices,
            mean,
            std,
        )
        pcs = pca.transform(scaled).astype(np.float32)
        control_mask = labels == control_label
        if np.any(control_mask):
            control_sum += pcs[control_mask].sum(axis=0, dtype=np.float64)
            control_count += int(control_mask.sum())
        for gene, group_index in group_to_index.items():
            mask = labels == gene
            if np.any(mask):
                grouped_pcs[group_index].append(pcs[mask])
    if control_count == 0:
        msg = f"Control label {control_label!r} has no cells"
        raise ValueError(msg)
    control_centroid = control_sum / float(control_count)
    bag_arrays = []
    observed_counts = []
    for gene, chunks in zip(genes, grouped_pcs, strict=True):
        if not chunks:
            msg = f"Numeric label {gene!r} has no cells in h5ad"
            raise ValueError(msg)
        pcs = np.vstack(chunks).astype(np.float32)
        observed_counts.append(int(pcs.shape[0]))
        bag_arrays.append((pcs - control_centroid[None, :]).astype(np.float32))
    return bag_arrays, observed_counts, control_centroid


def _scaled_selected_block(
    matrix: object,
    start: int,
    stop: int,
    selected_gene_indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    block = _dense_float32(matrix[start:stop, selected_gene_indices])
    return ((block - mean[None, :]) / std[None, :]).astype(np.float32)


def _external_overlap_path(config: BaselineConfig, external_name: str) -> Path:
    if config.data.external_overlap_csvs:
        for path in config.data.external_overlap_csvs:
            if external_name.split("_")[0] in path.name:
                return path
        return config.data.external_overlap_csvs[0]
    if external_name == "adamson_k562":
        return (
            _sl_dependency_root(config) / "interim" / "k562_adamson_depmap_overlap.csv"
        )
    return config.data.overlap_csv


def _external_sources(
    config: BaselineConfig,
    external_name: str,
) -> tuple[ExternalFeatureSourceConfig, ...]:
    if config.data.external_feature_sources:
        return config.data.external_feature_sources
    if external_name != "adamson_k562":
        msg = "data.external_feature_sources must be configured"
        raise ValueError(msg)
    root = _sl_dependency_root(config) / "raw" / "adamson"
    return (
        ExternalFeatureSourceConfig(
            name="adamson_pilot",
            h5ad_path=root / "adamson_2016_pilot.h5ad",
            obs_perturbation_col="perturbation",
            var_gene_symbol_col="gene_name",
        ),
        ExternalFeatureSourceConfig(
            name="adamson_upr_epistasis",
            h5ad_path=root / "adamson_2016_upr_epistasis.h5ad",
            obs_perturbation_col="perturbation",
            var_gene_symbol_col="gene_name",
        ),
        ExternalFeatureSourceConfig(
            name="adamson_upr_perturb_seq",
            h5ad_path=root / "adamson_2016_upr_perturb_seq.h5ad",
            obs_perturbation_col="perturbation",
            var_gene_symbol_col="gene_name",
        ),
    )


def _sl_dependency_root(config: BaselineConfig) -> Path:
    h5ad_path = config.data.h5ad_path
    for parent in h5ad_path.parents:
        if parent.name == "sl_dependency_v0":
            return parent
    return Path("data/sl_dependency_v0")


def _build_external_cell_source_rows(
    *,
    source: ExternalFeatureSourceConfig,
    overlap: pd.DataFrame,
    depmap_label_col: str,
    reference_symbols: list[str],
    reference_mean: np.ndarray,
    reference_std: np.ndarray,
    projector: object,
    chunk_size: int,
) -> tuple[pd.DataFrame, list[np.ndarray], dict[str, object]]:
    adata = ad.read_h5ad(source.h5ad_path, backed="r")
    try:
        label_col = _source_perturbation_label_col(overlap)
        source_overlap = (
            overlap.assign(source_perturbation_label=overlap[label_col].astype(str))
            .drop_duplicates(["source_perturbation_label", "perturbation_gene"])
            .sort_values("source_perturbation_label")
            .reset_index(drop=True)
        )
        labels = source_overlap["source_perturbation_label"].astype(str).tolist()
        label_metadata = source_overlap.set_index("source_perturbation_label")
        obs_labels = adata.obs[source.obs_perturbation_col].astype(str).to_numpy()
        control_label = _detect_control_label(obs_labels, source.control_label)
        source_genes = _var_symbols(adata, source.var_gene_symbol_col)
        source_indices = _source_indices_for_reference(source_genes, reference_symbols)
        if isinstance(projector, _ScviProjector):
            latent = projector.transform_source(
                source,
                adata.X,
                source_indices,
                reference_mean,
            )
            rows, bags, control_count = _external_rows_from_embeddings(
                embeddings=latent,
                obs_labels=obs_labels,
                control_label=control_label,
                labels=labels,
                label_metadata=label_metadata,
                source_name=source.name,
                depmap_label_col=depmap_label_col,
            )
            qa = {
                "source_dataset": source.name,
                "h5ad_path": str(source.h5ad_path),
                "control_label": control_label,
                "control_cells": int(control_count),
                "numeric_labels_with_cells": int(len(rows)),
                "expression_genes": int(len(source_genes)),
                "matched_reference_hvgs": int(
                    sum(index >= 0 for index in source_indices)
                ),
                "missing_reference_hvgs": int(
                    sum(index < 0 for index in source_indices)
                ),
            }
            return pd.DataFrame(rows), bags, qa

        first_projected = projector.transform(
            np.zeros((1, len(reference_symbols)), dtype=np.float32)
        )
        control_sum = np.zeros(first_projected.shape[1], dtype=np.float64)
        control_count = 0
        grouped: list[list[np.ndarray]] = [[] for _label in labels]
        label_to_index = {label: index for index, label in enumerate(labels)}
        for start in range(0, adata.n_obs, chunk_size):
            stop = min(start + chunk_size, adata.n_obs)
            block_labels = obs_labels[start:stop]
            pcs = _project_external_block(
                adata.X,
                start,
                stop,
                source_indices,
                reference_mean,
                reference_std,
                projector,
            )
            control_mask = block_labels == control_label
            if np.any(control_mask):
                control_sum += pcs[control_mask].sum(axis=0, dtype=np.float64)
                control_count += int(control_mask.sum())
            for label, index in label_to_index.items():
                mask = block_labels == label
                if np.any(mask):
                    grouped[index].append(pcs[mask])
        if control_count == 0:
            msg = f"Control label {control_label!r} has no cells in {source.name}"
            raise ValueError(msg)
        control_centroid = control_sum / float(control_count)
        rows = []
        bags = []
        for label, chunks in zip(labels, grouped, strict=True):
            if not chunks:
                continue
            pcs = np.vstack(chunks).astype(np.float32)
            bags.append((pcs - control_centroid[None, :]).astype(np.float32))
            rows.append(
                {
                    "source_dataset": source.name,
                    "source_perturbation_label": label,
                    "perturbation_gene": str(
                        label_metadata.loc[label, "perturbation_gene"]
                    ),
                    "observed_n_cells": int(pcs.shape[0]),
                    depmap_label_col: float(
                        label_metadata.loc[label, depmap_label_col]
                    ),
                }
            )
        qa = {
            "source_dataset": source.name,
            "h5ad_path": str(source.h5ad_path),
            "control_label": control_label,
            "control_cells": int(control_count),
            "numeric_labels_with_cells": int(len(rows)),
            "expression_genes": int(len(source_genes)),
            "matched_reference_hvgs": int(sum(index >= 0 for index in source_indices)),
            "missing_reference_hvgs": int(sum(index < 0 for index in source_indices)),
        }
        return pd.DataFrame(rows), bags, qa
    finally:
        adata.file.close()


def _source_indices_for_reference(
    source_genes: list[str],
    reference_symbols: list[str],
) -> np.ndarray:
    source_index = {}
    for index, gene in enumerate(source_genes):
        source_index.setdefault(gene, index)
    return np.asarray(
        [source_index.get(symbol, -1) for symbol in reference_symbols],
        dtype=np.int64,
    )


def _project_external_block(
    matrix: object,
    start: int,
    stop: int,
    source_indices: np.ndarray,
    reference_mean: np.ndarray,
    reference_std: np.ndarray,
    projector: object,
) -> np.ndarray:
    raw = np.tile(reference_mean[None, :], (stop - start, 1)).astype(np.float32)
    valid = source_indices >= 0
    if valid.any():
        block = _dense_float32(matrix[start:stop, source_indices[valid]])
        raw[:, valid] = block
    scaled = ((raw - reference_mean[None, :]) / reference_std[None, :]).astype(
        np.float32
    )
    return projector.transform(scaled).astype(np.float32)


def _external_rows_from_embeddings(
    *,
    embeddings: np.ndarray,
    obs_labels: np.ndarray,
    control_label: str,
    labels: list[str],
    label_metadata: pd.DataFrame,
    source_name: str,
    depmap_label_col: str,
) -> tuple[list[dict[str, object]], list[np.ndarray], int]:
    control_mask = obs_labels == control_label
    if not np.any(control_mask):
        msg = f"Control label {control_label!r} has no cells in {source_name}"
        raise ValueError(msg)
    control_centroid = embeddings[control_mask].mean(axis=0, dtype=np.float64)
    rows = []
    bags = []
    for label in labels:
        mask = obs_labels == label
        if not np.any(mask):
            continue
        values = embeddings[mask].astype(np.float32)
        bags.append((values - control_centroid[None, :]).astype(np.float32))
        rows.append(
                {
                    "source_dataset": source_name,
                    "source_perturbation_label": label,
                    "perturbation_gene": str(
                        label_metadata.loc[label, "perturbation_gene"]
                    ),
                "observed_n_cells": int(values.shape[0]),
                depmap_label_col: float(label_metadata.loc[label, depmap_label_col]),
            }
        )
    return rows, bags, int(control_mask.sum())


def _aggregate_external_cell_gene_rows(
    row_metadata: pd.DataFrame,
    bags: list[np.ndarray],
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    rows = []
    gene_bags = []
    for gene, group in row_metadata.groupby("perturbation_gene", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        combined = np.vstack([bags[index] for index in indices]).astype(np.float32)
        rows.append(
            {
                "perturbation_gene": str(gene),
                "depmap_gene_effect": float(group["depmap_gene_effect"].mean()),
                "observed_n_cells": int(group["observed_n_cells"].sum()),
                "source_dataset": ";".join(
                    sorted(group["source_dataset"].astype(str).unique())
                ),
                "external_row_count": int(len(group)),
            }
        )
        gene_bags.append(combined)
    return pd.DataFrame(rows), gene_bags
