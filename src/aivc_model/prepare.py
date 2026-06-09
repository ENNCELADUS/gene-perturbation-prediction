"""Data preparation for the STATE-ready AIVC A->B->C pipeline."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import os
import pickle
from pathlib import Path
from typing import Any
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import yaml


@dataclass(frozen=True)
class DataConfig:
    h5ad_path: Path
    overlap_csv: Path
    output_dir: Path
    obs_perturbation_col: str = "gene"
    control_label: str = "non-targeting"
    obs_cell_type_col: str | None = None
    obs_batch_col: str | None = None
    state_embed_key: str | None = None
    state_hvg_n_top_genes: int | None = None
    scvi_obsm_key: str | None = "X_scVI"
    depmap_label_col: str = "depmap_gene_effect"
    matched_label_col: str = "has_depmap_label"
    min_cells_per_gene: int = 2


@dataclass(frozen=True)
class ExternalSourceConfig:
    name: str
    h5ad_path: Path
    obs_perturbation_col: str = "perturbation"
    control_label: str | None = None
    var_gene_symbol_col: str = "gene_name"
    obs_batch_col: str | None = None


@dataclass(frozen=True)
class ExternalTestConfig:
    name: str
    overlap_csv: Path
    sources: tuple[ExternalSourceConfig, ...]
    control_label: str | None = None
    obs_batch_col: str | None = None


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int = 42
    stratify_bins: int = 5
    train_genes: tuple[str, ...] | None = None
    val_genes: tuple[str, ...] | None = None
    test_genes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class StateConfig:
    backend: str = "state_checkpoint"
    checkpoint_path: Path | None = None
    model_dir: Path | None = None
    embed_key: str = "X_hvg"
    input_dim: int | None = None
    output_dim: int | None = None
    pert_dim: int | None = None
    hidden_dim: int = 128
    cell_set_len: int = 128
    allow_mock: bool = False
    known_perturbation_vectors: Path | None = None


@dataclass(frozen=True)
class ProjectorConfig:
    teacher: str = "obsm"
    latent_dim: int = 128
    ridge_alpha: float = 1.0
    trainable: bool = True
    scvi_max_epochs: int = 100
    scvi_batch_size: int = 256
    scvi_num_workers: int | None = None
    scvi_hidden_units: int = 128
    scvi_layers: int = 2
    scvi_dropout: float = 0.1
    scvi_disable_lightning_logger: bool = True
    scvi_suppress_slurm_warning: bool = True


@dataclass(frozen=True)
class GmmConfig:
    n_components: int = 32
    covariance_floor: float = 1e-4
    max_fit_cells: int | None = 20000


@dataclass(frozen=True)
class ModelConfig:
    c_hidden_units: tuple[int, ...] = (64, 32)
    dropout: float = 0.1


@dataclass(frozen=True)
class LossConfig:
    latent_mean_delta_weight: float = 1.0
    latent_energy_weight: float = 1.0
    hvg_mean_delta_weight: float = 0.1
    hvg_energy_weight: float = 0.1
    pred_c_weight: float = 1.0
    obs_c_weight: float = 0.25
    occupancy_weight: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    run_id: str | None = None
    seed: int = 42
    max_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    cell_set_len: int = 128
    device: str = "auto"
    float32_matmul_precision: str | None = "high"


@dataclass(frozen=True)
class AivcConfig:
    data: DataConfig
    external_test: ExternalTestConfig | None
    split: SplitConfig
    state: StateConfig
    projector: ProjectorConfig
    gmm: GmmConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig


@dataclass(frozen=True)
class GeneBags:
    genes: np.ndarray
    y: np.ndarray
    input_bags: tuple[np.ndarray, ...]
    latent_bags: tuple[np.ndarray, ...]
    control_input: np.ndarray
    control_latent: np.ndarray
    cell_type_bags: tuple[np.ndarray, ...] | None
    control_cell_type: np.ndarray | None
    batch_bags: tuple[np.ndarray, ...] | None
    control_batch: np.ndarray | None
    feature_names: np.ndarray | None
    metadata: pd.DataFrame
    input_dim: int
    latent_dim: int


@dataclass(frozen=True)
class GeneSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class CellSetChunk:
    target_indices: np.ndarray
    control_indices: np.ndarray
    target_batch: np.ndarray | None
    control_fallback_count: int


@dataclass(frozen=True)
class ExternalGeneBags:
    data: GeneBags
    qa: dict[str, object]


def load_config(path: Path) -> AivcConfig:
    """Load an AIVC YAML config."""
    raw = yaml.safe_load(path.read_text()) or {}
    return AivcConfig(
        data=_data_config(raw.get("data", {})),
        external_test=_external_test_config(raw.get("external_test")),
        split=_split_config(raw.get("split", {})),
        state=_state_config(raw.get("state", {})),
        projector=_projector_config(raw.get("projector", {})),
        gmm=_gmm_config(raw.get("gmm", {})),
        model=_model_config(raw.get("model", {})),
        loss=_loss_config(raw.get("loss", {})),
        train=_train_config(raw.get("train", {})),
    )


def load_external_gene_bags(
    config: AivcConfig,
    reference: GeneBags,
    artifacts_dir: Path,
    *,
    project_scvi: bool = True,
) -> ExternalGeneBags | None:
    """Load and merge a configured external test set in the reference feature space."""
    if config.external_test is None:
        return None
    if reference.feature_names is None:
        msg = "External test loading requires reference feature_names"
        raise ValueError(msg)
    metadata = _load_external_metadata(config)
    source_rows: list[pd.DataFrame] = []
    source_input_bags: list[np.ndarray] = []
    source_latent_bags: list[np.ndarray] = []
    source_batch_bags: list[np.ndarray] = []
    control_input_bags: list[np.ndarray] = []
    control_latent_bags: list[np.ndarray] = []
    control_batch_bags: list[np.ndarray] = []
    source_qa = []

    for source in config.external_test.sources:
        rows, bags, control_input, batch_bags, control_batch, qa = (
            _load_external_source(
                config=config,
                source=source,
                metadata=_external_source_metadata(metadata, source.name),
                reference=reference,
            )
        )
        source_qa.append(qa)
        if rows.empty:
            continue
        source_rows.append(rows)
        source_input_bags.extend(bags)
        source_latent_bags.extend(bag.copy() for bag in bags)
        control_input_bags.append(control_input)
        control_latent_bags.append(control_input.copy())
        if batch_bags is not None:
            source_batch_bags.extend(batch_bags)
        if control_batch is not None:
            control_batch_bags.append(control_batch)

    if not source_rows:
        msg = (
            "No external test source produced matched genes for "
            f"{config.external_test.name}"
        )
        raise ValueError(msg)
    if not control_input_bags:
        msg = f"No external control cells found for {config.external_test.name}"
        raise ValueError(msg)

    row_metadata = pd.concat(source_rows, ignore_index=True)
    merged_metadata, merged_input_bags, merged_latent_bags, merged_batch_bags = (
        _merge_external_gene_rows(
            row_metadata,
            source_input_bags,
            source_latent_bags,
            tuple(source_batch_bags) if source_batch_bags else None,
            config.data.depmap_label_col,
        )
    )
    control_input = np.vstack(control_input_bags).astype(np.float32)
    control_latent = np.vstack(control_latent_bags).astype(np.float32)
    control_batch = (
        np.concatenate(control_batch_bags).astype(object)
        if control_batch_bags
        else None
    )
    if config.projector.teacher == "scvi" and project_scvi:
        scvi = _import_scvi()
        model_dir = artifacts_dir / "scvi_teacher_model"
        control_latent, merged_latent_bags = _project_scvi_latent_groups(
            scvi,
            model_dir,
            control_input,
            merged_input_bags,
            reference.feature_names,
            progress_label=f"external:{config.external_test.name}",
        )

    qa = {
        "external_name": config.external_test.name,
        "n_sources": int(len(config.external_test.sources)),
        "n_gene_rows": int(len(merged_metadata)),
        "n_cells": int(sum(bag.shape[0] for bag in merged_input_bags)),
        "n_control_cells": int(control_input.shape[0]),
        "input_dim": int(control_input.shape[1]),
        "sources": source_qa,
    }
    data = GeneBags(
        genes=merged_metadata["perturbation_gene"].astype(str).to_numpy(dtype=object),
        y=merged_metadata[config.data.depmap_label_col].to_numpy(dtype=np.float32),
        input_bags=tuple(merged_input_bags),
        latent_bags=tuple(merged_latent_bags),
        control_input=control_input,
        control_latent=control_latent,
        cell_type_bags=None,
        control_cell_type=None,
        batch_bags=tuple(merged_batch_bags) if merged_batch_bags else None,
        control_batch=control_batch,
        feature_names=reference.feature_names,
        metadata=merged_metadata,
        input_dim=int(control_input.shape[1]),
        latent_dim=int(control_latent.shape[1]),
    )
    return ExternalGeneBags(data=data, qa=qa)


def load_gene_bags(config: AivcConfig) -> GeneBags:
    """Load matched perturbation-gene bags and labels."""
    metadata = _load_metadata(config.data)
    adata = ad.read_h5ad(config.data.h5ad_path)
    obs_labels = adata.obs[config.data.obs_perturbation_col].astype(str).to_numpy()
    input_matrix, feature_names = _state_input_view(adata, config)
    latent_matrix = (
        input_matrix
        if config.projector.teacher == "scvi"
        else _latent_view(adata, config.data.scvi_obsm_key)
    )
    cell_type_labels = _cell_type_labels(adata, config.data)
    batch_labels = _batch_labels(adata, config.data)
    genes: list[str] = []
    y_values: list[float] = []
    input_bags: list[np.ndarray] = []
    latent_bags: list[np.ndarray] = []
    cell_type_bags: list[np.ndarray] = []
    batch_bags: list[np.ndarray] = []
    rows: list[pd.Series] = []

    control_mask = obs_labels == config.data.control_label
    if not np.any(control_mask):
        msg = f"No control cells found for {config.data.control_label!r}"
        raise ValueError(msg)
    control_input = input_matrix[control_mask].astype(np.float32)
    control_latent = latent_matrix[control_mask].astype(np.float32)
    control_cell_type = (
        cell_type_labels[control_mask] if cell_type_labels is not None else None
    )
    control_batch = batch_labels[control_mask] if batch_labels is not None else None

    for row in metadata.itertuples(index=False):
        gene = str(row.perturbation_gene)
        mask = obs_labels == gene
        n_cells = int(mask.sum())
        if n_cells < int(config.data.min_cells_per_gene):
            continue
        genes.append(gene)
        y_values.append(float(getattr(row, config.data.depmap_label_col)))
        input_bags.append(input_matrix[mask].astype(np.float32))
        latent_bags.append(latent_matrix[mask].astype(np.float32))
        if cell_type_labels is not None:
            cell_type_bags.append(cell_type_labels[mask])
        if batch_labels is not None:
            batch_bags.append(batch_labels[mask])
        rows.append(pd.Series(row._asdict()))

    if not genes:
        msg = "No perturbation genes with matched labels and enough cells"
        raise ValueError(msg)
    kept = pd.DataFrame(rows).reset_index(drop=True)
    return GeneBags(
        genes=np.asarray(genes, dtype=object),
        y=np.asarray(y_values, dtype=np.float32),
        input_bags=tuple(input_bags),
        latent_bags=tuple(latent_bags),
        control_input=control_input,
        control_latent=control_latent,
        cell_type_bags=tuple(cell_type_bags) if cell_type_labels is not None else None,
        control_cell_type=control_cell_type,
        batch_bags=tuple(batch_bags) if batch_labels is not None else None,
        control_batch=control_batch,
        feature_names=feature_names,
        metadata=kept,
        input_dim=int(control_input.shape[1]),
        latent_dim=int(control_latent.shape[1]),
    )


def make_gene_split(genes: np.ndarray, y: np.ndarray, config: SplitConfig) -> GeneSplit:
    """Create a disjoint train/validation/test split over perturbation genes."""
    explicit = config.train_genes or config.val_genes or config.test_genes
    if explicit:
        return _explicit_split(genes, config)
    if not np.isclose(
        config.train_fraction + config.val_fraction + config.test_fraction,
        1.0,
    ):
        msg = "train_fraction + val_fraction + test_fraction must equal 1.0"
        raise ValueError(msg)
    indices = np.arange(len(genes), dtype=np.int64)
    if np.isclose(config.train_fraction, 1.0):
        return GeneSplit(
            train=indices,
            val=np.asarray([], dtype=np.int64),
            test=np.asarray([], dtype=np.int64),
        )
    stratify = _stratify_labels(y, config.stratify_bins)
    train_idx, heldout_idx = _safe_train_test_split(
        indices,
        train_size=config.train_fraction,
        random_state=config.random_state,
        stratify=stratify,
    )
    if np.isclose(config.test_fraction, 0.0):
        return GeneSplit(
            train=np.sort(train_idx),
            val=np.sort(heldout_idx),
            test=np.asarray([], dtype=np.int64),
        )
    if np.isclose(config.val_fraction, 0.0):
        return GeneSplit(
            train=np.sort(train_idx),
            val=np.asarray([], dtype=np.int64),
            test=np.sort(heldout_idx),
        )
    heldout_y = y[heldout_idx]
    heldout_stratify = _stratify_labels(heldout_y, config.stratify_bins)
    val_share = config.val_fraction / (config.val_fraction + config.test_fraction)
    val_idx, test_idx = _safe_train_test_split(
        heldout_idx,
        train_size=val_share,
        random_state=config.random_state + 1,
        stratify=heldout_stratify,
    )
    return GeneSplit(
        train=np.sort(train_idx),
        val=np.sort(val_idx),
        test=np.sort(test_idx),
    )


def sample_rows(
    matrix: np.ndarray,
    n_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a fixed-size cell panel from a bag."""
    return matrix[sample_indices(matrix.shape[0], n_rows, rng)].astype(np.float32)


def sample_indices(
    n_available: int,
    n_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample row indices, using replacement for small bags."""
    replace = n_available < n_rows
    return rng.choice(n_available, size=n_rows, replace=replace)


def sample_rows_with_labels(
    matrix: np.ndarray,
    n_rows: int,
    rng: np.random.Generator,
    labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Sample rows and aligned optional labels."""
    indices = sample_indices(matrix.shape[0], n_rows, rng)
    sampled_labels = labels[indices] if labels is not None else None
    return matrix[indices].astype(np.float32), sampled_labels, indices


def make_cell_set_chunks(
    data: GeneBags,
    gene_index: int,
    *,
    cell_set_len: int,
    rng: np.random.Generator,
    pad_short: bool,
    shuffle: bool,
) -> tuple[CellSetChunk, ...]:
    """Build STATE-style cell-set chunks for one perturbation gene."""
    if cell_set_len < 1:
        msg = "cell_set_len must be at least 1"
        raise ValueError(msg)
    groups = _condition_groups(data, gene_index)
    chunks: list[CellSetChunk] = []
    for indices in groups:
        group_indices = indices.copy()
        if shuffle:
            rng.shuffle(group_indices)
        starts = range(0, len(group_indices), int(cell_set_len))
        for start in starts:
            target_indices = group_indices[start : start + int(cell_set_len)]
            if pad_short and len(target_indices) < int(cell_set_len):
                padding = rng.choice(
                    group_indices,
                    size=int(cell_set_len) - len(target_indices),
                    replace=True,
                )
                target_indices = np.concatenate([target_indices, padding])
            target_batch = (
                data.batch_bags[gene_index][target_indices]
                if data.batch_bags is not None
                else None
            )
            control_indices, fallback_count = _sample_control_indices(
                data,
                n_rows=len(target_indices),
                rng=rng,
                target_batch=target_batch,
            )
            chunks.append(
                CellSetChunk(
                    target_indices=target_indices.astype(np.int64),
                    control_indices=control_indices,
                    target_batch=target_batch,
                    control_fallback_count=fallback_count,
                )
            )
    return tuple(chunks)


def fit_linear_projector(
    expression: np.ndarray,
    latent: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a ridge linear map from STATE output space to scVI latent space."""
    x = np.asarray(expression, dtype=np.float64)
    y = np.asarray(latent, dtype=np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)
    return weights[:-1].astype(np.float32), weights[-1].astype(np.float32)


def load_perturbation_vectors(path: Path | None) -> dict[str, np.ndarray]:
    """Load known perturbation vectors from STATE or tabular vector files."""
    if path is None:
        return {}
    if path.suffix in {".pt", ".torch"}:
        mapping = _torch_load_mapping(path)
        return {str(gene): _as_vector(vector) for gene, vector in mapping.items()}
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
        if "genes" in payload and "vectors" in payload:
            genes = payload["genes"].astype(str)
            vectors = payload["vectors"].astype(np.float32)
            return {gene: vectors[index] for index, gene in enumerate(genes)}
        return {
            str(key): np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
        }
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    gene_col = "perturbation_gene" if "perturbation_gene" in frame else "gene"
    value_cols = [col for col in frame.columns if col != gene_col]
    return {
        str(row[gene_col]): row[value_cols].to_numpy(dtype=np.float32)
        for _, row in frame.iterrows()
    }


def load_state_batch_lookup(model_dir: Path | None) -> dict[str, int]:
    """Load a STATE batch one-hot map as label -> integer index."""
    mapping = _load_state_onehot_map(model_dir, "batch_onehot_map")
    if mapping is None:
        return {}
    lookup: dict[str, int] = {}
    for label, value in mapping.items():
        index = _argmax_index(value)
        if index is not None:
            lookup[str(label)] = index
    return lookup


def encode_batch_labels(
    labels: np.ndarray | None,
    lookup: dict[str, int],
    fallback_index: int = 0,
) -> np.ndarray | None:
    """Encode batch labels for STATE checkpoints."""
    if labels is None:
        return None
    return np.asarray(
        [lookup.get(str(label), int(fallback_index)) for label in labels],
        dtype=np.int64,
    )


def with_scvi_teacher_latents(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    artifacts_dir: Path,
    *,
    fit_teacher: bool = True,
) -> GeneBags:
    """Fit a train-only scVI teacher and replace latent bags with scVI latents."""
    data, _external = with_cached_scvi_teacher_latents(
        config,
        data,
        split,
        artifacts_dir,
        external=None,
        fit_teacher=fit_teacher,
    )
    return data


def with_cached_scvi_teacher_latents(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    artifacts_dir: Path,
    *,
    external: ExternalGeneBags | None = None,
    fit_teacher: bool = True,
    progress_interval: int = 50,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[GeneBags, ExternalGeneBags | None]:
    """Fit or reuse train-only scVI teacher latents from a run-local cache."""
    if config.projector.teacher != "scvi":
        return data, external
    cached, cache_reason = _load_scvi_latent_cache_with_reason(
        config,
        data,
        split,
        artifacts_dir,
        external,
    )
    if cached is not None:
        _log(log_fn, "Reusing cached scVI teacher latents")
        return cached
    if not fit_teacher:
        msg = (
            f"Validated scVI latent cache is missing in {artifacts_dir}: {cache_reason}"
        )
        raise FileNotFoundError(msg)
    scvi = _import_scvi()
    model_dir = artifacts_dir / "scvi_teacher_model"
    if fit_teacher:
        _log(log_fn, "Fitting rank0 scVI teacher model")
        train_matrix = np.vstack(
            [data.control_input, *(data.input_bags[index] for index in split.train)]
        ).astype(np.float32)
        train_adata = ad.AnnData(train_matrix)
        if data.feature_names is not None:
            train_adata.var_names = data.feature_names.astype(str)
        with _suppress_scvi_lightning_warnings(config.projector):
            scvi.model.SCVI.setup_anndata(train_adata)
            model = scvi.model.SCVI(
                train_adata,
                n_latent=int(config.projector.latent_dim),
                n_hidden=int(config.projector.scvi_hidden_units),
                n_layers=int(config.projector.scvi_layers),
                dropout_rate=float(config.projector.scvi_dropout),
            )
            model.train(
                max_epochs=int(config.projector.scvi_max_epochs),
                batch_size=int(config.projector.scvi_batch_size),
                early_stopping=True,
                datasplitter_kwargs=_scvi_datasplitter_kwargs(config.projector),
                **_scvi_trainer_kwargs(config.projector),
            )
        model.save(str(model_dir), overwrite=True, save_anndata=False)
    elif not model_dir.exists():
        msg = f"scVI teacher model is missing at {model_dir}"
        raise FileNotFoundError(msg)
    _log(log_fn, "Projecting scVI teacher latents")
    data, external = _materialize_scvi_latents(
        scvi,
        model_dir,
        data,
        external,
        progress_interval=progress_interval,
        log_fn=log_fn,
    )
    _write_scvi_latent_cache(config, data, split, artifacts_dir, external)
    _log(log_fn, "Wrote scVI teacher latent cache")
    return data, external


def _materialize_scvi_latents(
    scvi: object,
    model_dir: Path,
    data: GeneBags,
    external: ExternalGeneBags | None,
    *,
    progress_interval: int,
    log_fn: Callable[[str], None] | None,
) -> tuple[GeneBags, ExternalGeneBags | None]:
    external_name = str(external.qa["external_name"]) if external is not None else None
    datasets = [
        ("primary", data.control_input, data.input_bags, data.feature_names),
    ]
    if external is not None and external_name is not None:
        datasets.append(
            (
                f"external:{external_name}",
                external.data.control_input,
                external.data.input_bags,
                external.data.feature_names,
            )
        )
    projected = _project_scvi_latent_collections(
        scvi,
        model_dir,
        tuple(datasets),
        progress_interval=progress_interval,
        log_fn=log_fn,
    )
    control_latent, latent_bags = projected[0]
    data = replace(
        data,
        control_latent=control_latent,
        latent_bags=latent_bags,
        latent_dim=int(control_latent.shape[1]),
    )
    if external is None:
        return data, None
    external_control_latent, external_latent_bags = projected[1]
    external = replace(
        external,
        data=replace(
            external.data,
            control_latent=external_control_latent,
            latent_bags=external_latent_bags,
            latent_dim=int(external_control_latent.shape[1]),
        ),
    )
    return data, external


def _load_scvi_latent_cache(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    artifacts_dir: Path,
    external: ExternalGeneBags | None,
) -> tuple[GeneBags, ExternalGeneBags | None] | None:
    cached, _reason = _load_scvi_latent_cache_with_reason(
        config,
        data,
        split,
        artifacts_dir,
        external,
    )
    return cached


def _load_scvi_latent_cache_with_reason(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    artifacts_dir: Path,
    external: ExternalGeneBags | None,
) -> tuple[tuple[GeneBags, ExternalGeneBags | None] | None, str | None]:
    cache_dir = _scvi_latent_cache_dir(artifacts_dir)
    if not (cache_dir / "COMPLETE").exists():
        return None, f"{cache_dir / 'COMPLETE'} does not exist"
    metadata_path = cache_dir / "metadata.json"
    primary_path = cache_dir / "primary.npz"
    if not metadata_path.exists():
        return None, f"{metadata_path} does not exist"
    if not primary_path.exists():
        return None, f"{primary_path} does not exist"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"{metadata_path} is not valid JSON: {exc}"
    expected = _scvi_cache_metadata(config, data, split, external)
    if metadata != expected:
        return None, f"{metadata_path} does not match current config/data"
    try:
        primary = np.load(primary_path, allow_pickle=False)
    except OSError as exc:
        return None, f"{primary_path} could not be loaded: {exc}"
    try:
        primary_control = np.asarray(primary["control"], dtype=np.float32)
        primary_latent_bags = tuple(
            np.asarray(primary[f"bag_{index}"], dtype=np.float32)
            for index in range(len(data.input_bags))
        )
    except (KeyError, ValueError) as exc:
        return None, f"{primary_path} is incomplete or incompatible: {exc}"
    data = replace(
        data,
        control_latent=primary_control,
        latent_bags=primary_latent_bags,
        latent_dim=int(primary_control.shape[1]),
    )
    if external is None:
        return (data, None), None
    external_path = cache_dir / "external.npz"
    if not external_path.exists():
        return None, f"{external_path} does not exist"
    try:
        external_payload = np.load(external_path, allow_pickle=False)
    except OSError as exc:
        return None, f"{external_path} could not be loaded: {exc}"
    try:
        external_control = np.asarray(external_payload["control"], dtype=np.float32)
        external_latent_bags = tuple(
            np.asarray(external_payload[f"bag_{index}"], dtype=np.float32)
            for index in range(len(external.data.input_bags))
        )
    except (KeyError, ValueError) as exc:
        return None, f"{external_path} is incomplete or incompatible: {exc}"
    external = replace(
        external,
        data=replace(
            external.data,
            control_latent=external_control,
            latent_bags=external_latent_bags,
            latent_dim=int(external_control.shape[1]),
        ),
    )
    return (data, external), None


def _write_scvi_latent_cache(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    artifacts_dir: Path,
    external: ExternalGeneBags | None,
) -> None:
    cache_dir = _scvi_latent_cache_dir(artifacts_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    complete_path = cache_dir / "COMPLETE"
    complete_path.unlink(missing_ok=True)
    _write_npz_atomic(
        cache_dir / "primary.npz",
        _latent_payload(data.control_latent, data.latent_bags),
    )
    external_path = cache_dir / "external.npz"
    if external is None:
        external_path.unlink(missing_ok=True)
    else:
        _write_npz_atomic(
            external_path,
            _latent_payload(external.data.control_latent, external.data.latent_bags),
        )
    _write_json_atomic(
        cache_dir / "metadata.json",
        _scvi_cache_metadata(config, data, split, external),
    )
    _write_text_atomic(complete_path, "ok\n")


def _project_scvi_latent_groups(
    scvi: object,
    model_dir: Path,
    control: np.ndarray,
    bags: tuple[np.ndarray, ...],
    feature_names: np.ndarray | None,
    *,
    progress_label: str,
    progress_interval: int = 50,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    return _project_scvi_latent_collections(
        scvi,
        model_dir,
        ((progress_label, control, bags, feature_names),),
        progress_interval=progress_interval,
        log_fn=log_fn,
    )[0]


def _project_scvi_latent_collections(
    scvi: object,
    model_dir: Path,
    datasets: tuple[
        tuple[str, np.ndarray, tuple[np.ndarray, ...], np.ndarray | None],
        ...,
    ],
    *,
    progress_interval: int,
    log_fn: Callable[[str], None] | None,
) -> tuple[tuple[np.ndarray, tuple[np.ndarray, ...]], ...]:
    feature_names = _shared_feature_names(datasets)
    matrices: list[np.ndarray] = []
    ranges: list[tuple[np.ndarray, tuple[np.ndarray, ...]]] = []
    offset = 0
    for _label, control, bags, _names in datasets:
        control_matrix = np.asarray(control, dtype=np.float32)
        matrices.append(control_matrix)
        control_range = np.arange(offset, offset + control_matrix.shape[0])
        offset += control_matrix.shape[0]
        bag_ranges = []
        for bag in bags:
            bag_matrix = np.asarray(bag, dtype=np.float32)
            matrices.append(bag_matrix)
            bag_ranges.append(np.arange(offset, offset + bag_matrix.shape[0]))
            offset += bag_matrix.shape[0]
        ranges.append((control_range, tuple(bag_ranges)))
    query = ad.AnnData(np.vstack(matrices).astype(np.float32))
    if feature_names is not None:
        query.var_names = feature_names.astype(str)
    projected: list[tuple[np.ndarray, tuple[np.ndarray, ...]]] = []
    with _suppress_normalized_x_warning():
        model = scvi.model.SCVI.load(str(model_dir), adata=query)
        for label, _control, bags, _names in datasets:
            dataset_index = len(projected)
            control_range, bag_ranges = ranges[dataset_index]
            control_latent = np.asarray(
                model.get_latent_representation(indices=control_range),
                dtype=np.float32,
            )
            latent_bags: list[np.ndarray] = []
            step = max(1, int(progress_interval))
            for start in range(0, len(bags), step):
                end = min(start + step, len(bags))
                batch_ranges = bag_ranges[start:end]
                batch_indices = np.concatenate(batch_ranges)
                batch_latent = np.asarray(
                    model.get_latent_representation(indices=batch_indices),
                    dtype=np.float32,
                )
                batch_offsets = np.cumsum(
                    [0, *(bags[index].shape[0] for index in range(start, end))]
                )
                latent_bags.extend(
                    batch_latent[batch_offsets[index] : batch_offsets[index + 1]]
                    for index in range(end - start)
                )
                _log(
                    log_fn,
                    f"Projected {label} scVI latents for {end}/{len(bags)} genes",
                )
            projected.append((control_latent, tuple(latent_bags)))
    return tuple(projected)


def _shared_feature_names(
    datasets: tuple[
        tuple[str, np.ndarray, tuple[np.ndarray, ...], np.ndarray | None],
        ...,
    ],
) -> np.ndarray | None:
    names = datasets[0][3]
    for _label, _control, _bags, other in datasets[1:]:
        if names is None and other is None:
            continue
        if (
            names is None
            or other is None
            or not np.array_equal(
                names.astype(str),
                other.astype(str),
            )
        ):
            msg = "scVI latent cache requires matching feature names across datasets"
            raise ValueError(msg)
    return names


def _scvi_cache_metadata(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    external: ExternalGeneBags | None,
) -> dict[str, object]:
    return {
        "version": 1,
        "teacher": "scvi",
        "latent_dim": int(config.projector.latent_dim),
        "seed": int(config.train.seed),
        "teacher_config": {
            "scvi_max_epochs": int(config.projector.scvi_max_epochs),
            "scvi_batch_size": int(config.projector.scvi_batch_size),
            "scvi_hidden_units": int(config.projector.scvi_hidden_units),
            "scvi_layers": int(config.projector.scvi_layers),
            "scvi_dropout": float(config.projector.scvi_dropout),
        },
        "train_genes": [str(data.genes[index]) for index in split.train],
        "primary": _cache_dataset_metadata(data),
        "external": (
            {
                "name": str(external.qa["external_name"]),
                **_cache_dataset_metadata(external.data),
            }
            if external is not None
            else None
        ),
    }


def _cache_dataset_metadata(data: GeneBags) -> dict[str, object]:
    return {
        "genes": [str(gene) for gene in data.genes],
        "control_shape": [int(value) for value in data.control_input.shape],
        "input_shapes": [
            [int(value) for value in bag.shape] for bag in data.input_bags
        ],
        "feature_names": (
            data.feature_names.astype(str).tolist()
            if data.feature_names is not None
            else None
        ),
    }


def _latent_payload(
    control_latent: np.ndarray,
    latent_bags: tuple[np.ndarray, ...],
) -> dict[str, np.ndarray]:
    payload = {"control": np.asarray(control_latent, dtype=np.float32)}
    payload.update(
        {
            f"bag_{index}": np.asarray(bag, dtype=np.float32)
            for index, bag in enumerate(latent_bags)
        }
    )
    return payload


def _scvi_latent_cache_dir(artifacts_dir: Path) -> Path:
    return artifacts_dir / "scvi_teacher_latents"


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp_path.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(value, encoding="utf-8")
    tmp_path.replace(path)


def _log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _scvi_datasplitter_kwargs(config: ProjectorConfig) -> dict[str, object]:
    num_workers = (
        max(0, int(config.scvi_num_workers))
        if config.scvi_num_workers is not None
        else _auto_scvi_num_workers()
    )
    return {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }


def _auto_scvi_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    candidates = [max(1, cpu_count - 1)]
    slurm_cpus = _positive_env_int("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        candidates.append(max(1, slurm_cpus - 1))
    return min(8, *candidates)


def _positive_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _scvi_trainer_kwargs(config: ProjectorConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    if config.scvi_disable_lightning_logger:
        kwargs["logger"] = False
    return kwargs


@contextmanager
def _suppress_scvi_lightning_warnings(config: ProjectorConfig) -> Any:
    info_patchers = (
        _patch_lightning_fit_stop_info() if config.scvi_disable_lightning_logger else ()
    )
    if not config.scvi_suppress_slurm_warning:
        try:
            yield
        finally:
            _restore_lightning_info(info_patchers)
        return
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r".*The `srun` command is available on your system but is not "
                    r"used.*"
                ),
                category=Warning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"adata\.X does not contain unnormalized count data.*",
                category=UserWarning,
            )
            yield
    finally:
        _restore_lightning_info(info_patchers)


def _patch_lightning_fit_stop_info() -> tuple[tuple[object, object], ...]:
    originals = []
    for module_name in (
        "lightning.pytorch.loops.fit_loop",
        "pytorch_lightning.loops.fit_loop",
    ):
        try:
            module = __import__(module_name, fromlist=["rank_zero_info"])
        except ImportError:
            continue
        original = getattr(module, "rank_zero_info", None)
        if original is None:
            continue

        def filtered_info(
            *args: Any, _original: object = original, **kwargs: Any
        ) -> Any:
            message = str(args[0]) if args else ""
            if message.startswith("`Trainer.fit` stopped: `max_epochs="):
                return None
            return _original(*args, **kwargs)

        originals.append((module, original))
        setattr(module, "rank_zero_info", filtered_info)
    return tuple(originals)


def _restore_lightning_info(originals: tuple[tuple[object, object], ...]) -> None:
    for module, original in originals:
        setattr(module, "rank_zero_info", original)


def _load_metadata(config: DataConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.overlap_csv)
    if config.matched_label_col in frame.columns:
        frame = frame.loc[frame[config.matched_label_col].astype(bool)]
    frame = frame.loc[frame[config.depmap_label_col].notna()].copy()
    if "perturbation_gene" not in frame.columns:
        msg = "overlap_csv must include perturbation_gene"
        raise ValueError(msg)
    return frame.reset_index(drop=True)


def _load_external_metadata(config: AivcConfig) -> pd.DataFrame:
    if config.external_test is None:
        msg = "external_test is not configured"
        raise ValueError(msg)
    frame = pd.read_csv(config.external_test.overlap_csv)
    if config.data.matched_label_col in frame.columns:
        frame = frame.loc[frame[config.data.matched_label_col].astype(bool)]
    frame = frame.loc[frame[config.data.depmap_label_col].notna()].copy()
    if "perturbation_gene" not in frame.columns:
        msg = "external_test.overlap_csv must include perturbation_gene"
        raise ValueError(msg)
    return frame.reset_index(drop=True)


def _load_external_source(
    *,
    config: AivcConfig,
    source: ExternalSourceConfig,
    metadata: pd.DataFrame,
    reference: GeneBags,
) -> tuple[
    pd.DataFrame,
    list[np.ndarray],
    np.ndarray,
    list[np.ndarray] | None,
    np.ndarray | None,
    dict[str, object],
]:
    adata = ad.read_h5ad(source.h5ad_path)
    obs_labels = adata.obs[source.obs_perturbation_col].astype(str).to_numpy()
    control_label = _detect_control_label(
        obs_labels,
        source.control_label
        or (config.external_test.control_label if config.external_test else None),
    )
    input_matrix, alignment_qa = _external_state_input_view(
        adata,
        source,
        config,
        reference,
    )
    batch_col = source.obs_batch_col or (
        config.external_test.obs_batch_col if config.external_test else None
    )
    batch_labels = _optional_obs_labels(adata, batch_col)
    label_col = _source_perturbation_label_col(metadata)
    source_metadata = (
        metadata.assign(source_perturbation_label=metadata[label_col].astype(str))
        .drop_duplicates(["source_perturbation_label", "perturbation_gene"])
        .sort_values("source_perturbation_label")
        .reset_index(drop=True)
    )
    control_mask = obs_labels == control_label
    if not np.any(control_mask):
        msg = f"Control label {control_label!r} has no cells in {source.name}"
        raise ValueError(msg)
    control_input = input_matrix[control_mask].astype(np.float32)
    control_batch = batch_labels[control_mask] if batch_labels is not None else None
    rows = []
    bags: list[np.ndarray] = []
    batch_bags: list[np.ndarray] = []
    for row in source_metadata.itertuples(index=False):
        label = str(row.source_perturbation_label)
        mask = obs_labels == label
        n_cells = int(mask.sum())
        if n_cells < int(config.data.min_cells_per_gene):
            continue
        bags.append(input_matrix[mask].astype(np.float32))
        if batch_labels is not None:
            batch_bags.append(batch_labels[mask])
        row_dict = row._asdict()
        row_dict["source_dataset"] = source.name
        row_dict["source_perturbation_label"] = label
        row_dict["observed_n_cells"] = n_cells
        rows.append(row_dict)
    qa = {
        "source_dataset": source.name,
        "h5ad_path": str(source.h5ad_path),
        "control_label": control_label,
        "control_cells": int(control_input.shape[0]),
        "numeric_labels_with_cells": int(len(rows)),
        **alignment_qa,
    }
    return (
        pd.DataFrame(rows),
        bags,
        control_input,
        batch_bags if batch_labels is not None else None,
        control_batch,
        qa,
    )


def _external_state_input_view(
    adata: ad.AnnData,
    source: ExternalSourceConfig,
    config: AivcConfig,
    reference: GeneBags,
) -> tuple[np.ndarray, dict[str, object]]:
    key = config.data.state_embed_key
    if key and key in adata.obsm:
        matrix = np.asarray(adata.obsm[key], dtype=np.float32)
        if matrix.shape[1] != reference.input_dim:
            msg = (
                f"External obsm[{key!r}] has dim {matrix.shape[1]}, "
                f"expected {reference.input_dim}"
            )
            raise ValueError(msg)
        return matrix, {
            "input_source": f"obsm:{key}",
            "matched_input_features": int(matrix.shape[1]),
            "missing_input_features": 0,
            "reference_input_features": int(reference.input_dim),
        }
    if reference.feature_names is None:
        msg = "Cannot align external expression without reference feature names"
        raise ValueError(msg)
    reference_names = reference.feature_names.astype(str).tolist()
    source_names = _var_symbols(adata, source.var_gene_symbol_col)
    source_to_index = {name: index for index, name in enumerate(source_names)}
    fill_values = reference.control_input.mean(axis=0).astype(np.float32)
    matrix = np.tile(fill_values[None, :], (adata.n_obs, 1)).astype(np.float32)
    matched_reference_indices = []
    matched_source_indices = []
    for ref_index, name in enumerate(reference_names):
        source_index = source_to_index.get(str(name))
        if source_index is None:
            continue
        matched_reference_indices.append(ref_index)
        matched_source_indices.append(source_index)
    if matched_reference_indices:
        matrix[:, np.asarray(matched_reference_indices, dtype=np.int64)] = _dense_slice(
            adata.X,
            np.asarray(matched_source_indices, dtype=np.int64),
        )
    matched = int(len(matched_reference_indices))
    return matrix, {
        "input_source": "X_aligned_to_reference_features",
        "source_expression_features": int(len(source_names)),
        "reference_input_features": int(len(reference_names)),
        "matched_input_features": matched,
        "missing_input_features": int(len(reference_names) - matched),
    }


def _merge_external_gene_rows(
    row_metadata: pd.DataFrame,
    input_bags: list[np.ndarray],
    latent_bags: list[np.ndarray],
    batch_bags: tuple[np.ndarray, ...] | None,
    depmap_label_col: str,
) -> tuple[
    pd.DataFrame,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    rows = row_metadata.reset_index(drop=True).copy()
    rows["_source_row"] = np.arange(len(rows), dtype=np.int64)
    merged_rows = []
    merged_input_bags = []
    merged_latent_bags = []
    merged_batch_bags = []
    for gene, group in rows.groupby("perturbation_gene", sort=True):
        source_indices = group["_source_row"].to_numpy(dtype=np.int64)
        merged_input_bags.append(
            np.vstack([input_bags[index] for index in source_indices]).astype(
                np.float32
            )
        )
        merged_latent_bags.append(
            np.vstack([latent_bags[index] for index in source_indices]).astype(
                np.float32
            )
        )
        if batch_bags is not None:
            merged_batch_bags.append(
                np.concatenate([batch_bags[index] for index in source_indices])
            )
        first = group.iloc[0].to_dict()
        source_names = group["source_dataset"].astype(str).tolist()
        first["perturbation_gene"] = str(gene)
        first["source_dataset"] = ";".join(sorted(set(source_names)))
        first["external_row_count"] = int(len(group))
        first["observed_n_cells"] = int(group["observed_n_cells"].sum())
        first[depmap_label_col] = float(group[depmap_label_col].iloc[0])
        first.pop("_source_row", None)
        merged_rows.append(first)
    metadata = pd.DataFrame(merged_rows).reset_index(drop=True)
    return (
        metadata,
        tuple(merged_input_bags),
        tuple(merged_latent_bags),
        tuple(merged_batch_bags),
    )


def _external_source_metadata(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "source_dataset" not in frame.columns:
        return frame.copy()
    source_rows = frame.loc[frame["source_dataset"].astype(str) == source_name]
    if source_rows.empty:
        return frame.copy()
    return source_rows.copy()


def _source_perturbation_label_col(frame: pd.DataFrame) -> str:
    if "source_perturbation_label" in frame.columns:
        return "source_perturbation_label"
    return "perturbation_gene"


def _detect_control_label(labels: np.ndarray, configured: str | None) -> str:
    unique = pd.Index(labels.astype(str)).unique().tolist()
    if configured:
        if configured not in set(unique):
            msg = f"Configured control label {configured!r} not found"
            raise ValueError(msg)
        return configured
    lower_to_label = {label.lower(): label for label in unique}
    for candidate in (
        "*",
        "control",
        "non-targeting",
        "non-targeting control",
        "unperturbed",
        "mock",
    ):
        if candidate in lower_to_label:
            return str(lower_to_label[candidate])
    control_like = [
        label
        for label in unique
        if "control" in label.lower() or "non-target" in label.lower()
    ]
    if len(control_like) == 1:
        return str(control_like[0])
    msg = "Could not auto-detect a unique control label"
    raise ValueError(msg)


def _optional_obs_labels(adata: ad.AnnData, column: str | None) -> np.ndarray | None:
    if column is None or column not in adata.obs:
        return None
    return adata.obs[column].astype(str).to_numpy()


def _var_symbols(adata: ad.AnnData, column: str) -> list[str]:
    if column in adata.var.columns:
        return adata.var[column].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _matrix_view(adata: ad.AnnData, key: str | None) -> np.ndarray:
    if key:
        if key not in adata.obsm:
            msg = f"AnnData is missing obsm[{key!r}]"
            raise ValueError(msg)
        return np.asarray(adata.obsm[key], dtype=np.float32)
    x = adata.X
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _state_input_view(
    adata: ad.AnnData,
    config: AivcConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    key = config.data.state_embed_key
    if key:
        if key in adata.obsm:
            matrix = np.asarray(adata.obsm[key], dtype=np.float32)
            return matrix, _feature_names(key, matrix.shape[1])
        if key == "X_hvg":
            indices = _state_hvg_indices(adata, config)
            matrix = _dense_slice(adata.X, indices)
            return matrix, np.asarray(adata.var_names[indices].astype(str))
        msg = f"AnnData is missing obsm[{key!r}]"
        raise ValueError(msg)
    matrix = _matrix_view(adata, None)
    return matrix, np.asarray(adata.var_names.astype(str))


def _batch_labels(adata: ad.AnnData, config: DataConfig) -> np.ndarray | None:
    if config.obs_batch_col is None:
        return None
    if config.obs_batch_col not in adata.obs:
        msg = f"AnnData is missing obs[{config.obs_batch_col!r}]"
        raise ValueError(msg)
    return adata.obs[config.obs_batch_col].astype(str).to_numpy()


def _cell_type_labels(adata: ad.AnnData, config: DataConfig) -> np.ndarray | None:
    if config.obs_cell_type_col is None:
        return None
    if config.obs_cell_type_col not in adata.obs:
        msg = f"AnnData is missing obs[{config.obs_cell_type_col!r}]"
        raise ValueError(msg)
    return adata.obs[config.obs_cell_type_col].astype(str).to_numpy()


def _latent_view(adata: ad.AnnData, key: str | None) -> np.ndarray:
    if key is None:
        return _matrix_view(adata, None)
    if key not in adata.obsm:
        msg = f"AnnData is missing scVI latent obsm[{key!r}]"
        raise ValueError(msg)
    return np.asarray(adata.obsm[key], dtype=np.float32)


def _explicit_split(genes: np.ndarray, config: SplitConfig) -> GeneSplit:
    gene_to_index = {str(gene): index for index, gene in enumerate(genes)}

    def indices(selected: tuple[str, ...] | None, split_name: str) -> np.ndarray:
        if selected is None:
            msg = f"Explicit split requires {split_name}_genes"
            raise ValueError(msg)
        missing = [gene for gene in selected if gene not in gene_to_index]
        if missing:
            msg = f"{split_name}_genes missing from loaded genes: {missing}"
            raise ValueError(msg)
        return np.asarray([gene_to_index[gene] for gene in selected], dtype=np.int64)

    split = GeneSplit(
        train=np.sort(indices(config.train_genes, "train")),
        val=np.sort(indices(config.val_genes, "val")),
        test=np.sort(indices(config.test_genes, "test")),
    )
    _validate_disjoint(split)
    return split


def _validate_disjoint(split: GeneSplit) -> None:
    sets = [set(split.train), set(split.val), set(split.test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        msg = "train, val, and test genes must be disjoint"
        raise ValueError(msg)


def _stratify_labels(y: np.ndarray, bins: int) -> np.ndarray | None:
    if len(y) < max(4, bins):
        return None
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)[1:-1]
    edges = np.unique(np.quantile(y, quantiles))
    if edges.size < 1:
        return None
    labels = np.digitize(y, edges)
    counts = np.bincount(labels)
    return labels if counts.size > 0 and counts.min() >= 2 else None


def _safe_train_test_split(
    indices: np.ndarray,
    *,
    train_size: float,
    random_state: int,
    stratify: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        train, test = train_test_split(
            indices,
            train_size=train_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        train, test = train_test_split(
            indices,
            train_size=train_size,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def _condition_groups(data: GeneBags, gene_index: int) -> tuple[np.ndarray, ...]:
    n_cells = data.input_bags[gene_index].shape[0]
    cell_types = (
        data.cell_type_bags[gene_index]
        if data.cell_type_bags is not None
        else np.full(n_cells, "K562", dtype=object)
    )
    batches = (
        data.batch_bags[gene_index]
        if data.batch_bags is not None
        else np.full(n_cells, "", dtype=object)
    )
    keys: dict[tuple[str, str], list[int]] = {}
    for index, (cell_type, batch) in enumerate(zip(cell_types, batches, strict=True)):
        keys.setdefault((str(cell_type), str(batch)), []).append(index)
    return tuple(np.asarray(indices, dtype=np.int64) for indices in keys.values())


def _sample_control_indices(
    data: GeneBags,
    *,
    n_rows: int,
    rng: np.random.Generator,
    target_batch: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    if target_batch is None or data.control_batch is None:
        return sample_indices(data.control_input.shape[0], n_rows, rng), 0
    selected: list[int] = []
    fallback_count = 0
    global_indices = np.arange(data.control_input.shape[0], dtype=np.int64)
    for label in target_batch:
        matching = np.flatnonzero(data.control_batch.astype(str) == str(label))
        if matching.size == 0:
            matching = global_indices
            fallback_count += 1
        selected.append(int(rng.choice(matching)))
    return np.asarray(selected, dtype=np.int64), fallback_count


def _data_config(values: dict[str, Any]) -> DataConfig:
    return DataConfig(
        h5ad_path=Path(values["h5ad_path"]),
        overlap_csv=Path(values["overlap_csv"]),
        output_dir=Path(values["output_dir"]),
        obs_perturbation_col=str(values.get("obs_perturbation_col", "gene")),
        control_label=str(values.get("control_label", "non-targeting")),
        obs_cell_type_col=values.get("obs_cell_type_col"),
        obs_batch_col=values.get("obs_batch_col"),
        state_embed_key=values.get("state_embed_key"),
        state_hvg_n_top_genes=_int_or_none(values.get("state_hvg_n_top_genes")),
        scvi_obsm_key=values.get("scvi_obsm_key", "X_scVI"),
        depmap_label_col=str(values.get("depmap_label_col", "depmap_gene_effect")),
        matched_label_col=str(values.get("matched_label_col", "has_depmap_label")),
        min_cells_per_gene=int(values.get("min_cells_per_gene", 2)),
    )


def _external_test_config(values: Any) -> ExternalTestConfig | None:
    if values is None:
        return None
    return ExternalTestConfig(
        name=str(values.get("name", "external_test")),
        overlap_csv=Path(values["overlap_csv"]),
        sources=tuple(
            ExternalSourceConfig(
                name=str(source["name"]),
                h5ad_path=Path(source["h5ad_path"]),
                obs_perturbation_col=str(
                    source.get("obs_perturbation_col", "perturbation")
                ),
                control_label=(
                    str(source["control_label"])
                    if source.get("control_label") is not None
                    else None
                ),
                var_gene_symbol_col=str(source.get("var_gene_symbol_col", "gene_name")),
                obs_batch_col=(
                    str(source["obs_batch_col"])
                    if source.get("obs_batch_col") is not None
                    else None
                ),
            )
            for source in values.get("sources", ())
        ),
        control_label=_str_or_none(values.get("control_label")),
        obs_batch_col=_str_or_none(values.get("obs_batch_col")),
    )


def _split_config(values: dict[str, Any]) -> SplitConfig:
    return SplitConfig(
        train_fraction=float(values.get("train_fraction", 0.7)),
        val_fraction=float(values.get("val_fraction", 0.15)),
        test_fraction=float(values.get("test_fraction", 0.15)),
        random_state=int(values.get("random_state", 42)),
        stratify_bins=int(values.get("stratify_bins", 5)),
        train_genes=_tuple_or_none(values.get("train_genes")),
        val_genes=_tuple_or_none(values.get("val_genes")),
        test_genes=_tuple_or_none(values.get("test_genes")),
    )


def _state_config(values: dict[str, Any]) -> StateConfig:
    return StateConfig(
        backend=str(values.get("backend", "state_checkpoint")),
        checkpoint_path=_path_or_none(values.get("checkpoint_path")),
        model_dir=_path_or_none(values.get("model_dir")),
        embed_key=str(values.get("embed_key", "X_hvg")),
        input_dim=_int_or_none(values.get("input_dim")),
        output_dim=_int_or_none(values.get("output_dim")),
        pert_dim=_int_or_none(values.get("pert_dim")),
        hidden_dim=int(values.get("hidden_dim", 128)),
        cell_set_len=int(values.get("cell_set_len", 128)),
        allow_mock=bool(values.get("allow_mock", False)),
        known_perturbation_vectors=_path_or_none(
            values.get("known_perturbation_vectors")
        ),
    )


def _projector_config(values: dict[str, Any]) -> ProjectorConfig:
    return ProjectorConfig(
        teacher=str(values.get("teacher", "obsm")),
        latent_dim=int(values.get("latent_dim", 128)),
        ridge_alpha=float(values.get("ridge_alpha", 1.0)),
        trainable=_bool_value(values.get("trainable", True)),
        scvi_max_epochs=int(values.get("scvi_max_epochs", 100)),
        scvi_batch_size=int(values.get("scvi_batch_size", 256)),
        scvi_num_workers=_int_or_none(values.get("scvi_num_workers")),
        scvi_hidden_units=int(values.get("scvi_hidden_units", 128)),
        scvi_layers=int(values.get("scvi_layers", 2)),
        scvi_dropout=float(values.get("scvi_dropout", 0.1)),
        scvi_disable_lightning_logger=_bool_value(
            values.get("scvi_disable_lightning_logger", True)
        ),
        scvi_suppress_slurm_warning=_bool_value(
            values.get("scvi_suppress_slurm_warning", True)
        ),
    )


def _gmm_config(values: dict[str, Any]) -> GmmConfig:
    return GmmConfig(
        n_components=int(values.get("n_components", 32)),
        covariance_floor=float(values.get("covariance_floor", 1e-4)),
        max_fit_cells=_int_or_none(values.get("max_fit_cells", 20000)),
    )


def _model_config(values: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        c_hidden_units=tuple(int(v) for v in values.get("c_hidden_units", [64, 32])),
        dropout=float(values.get("dropout", 0.1)),
    )


def _loss_config(values: dict[str, Any]) -> LossConfig:
    return LossConfig(
        latent_mean_delta_weight=float(
            values.get("latent_mean_delta_weight", values.get("a_to_b_weight", 1.0))
        ),
        latent_energy_weight=float(
            values.get("latent_energy_weight", values.get("a_to_b_weight", 1.0))
        ),
        hvg_mean_delta_weight=float(values.get("hvg_mean_delta_weight", 0.1)),
        hvg_energy_weight=float(values.get("hvg_energy_weight", 0.1)),
        pred_c_weight=float(values.get("pred_c_weight", 1.0)),
        obs_c_weight=float(values.get("obs_c_weight", 0.25)),
        occupancy_weight=float(values.get("occupancy_weight", 0.1)),
    )


def _train_config(values: dict[str, Any]) -> TrainConfig:
    cell_set_len = values.get("cell_set_len", values.get("cells_per_gene", 128))
    return TrainConfig(
        run_id=values.get("run_id"),
        seed=int(values.get("seed", 42)),
        max_epochs=int(values.get("max_epochs", 50)),
        learning_rate=float(values.get("learning_rate", 1e-4)),
        weight_decay=float(values.get("weight_decay", 1e-4)),
        cell_set_len=int(cell_set_len),
        device=str(values.get("device", "auto")),
        float32_matmul_precision=_str_or_none(
            values.get("float32_matmul_precision", "high")
        ),
    )


def _tuple_or_none(values: Any) -> tuple[str, ...] | None:
    if values is None:
        return None
    return tuple(str(value) for value in values)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _path_or_none(value: Any) -> Path | None:
    return Path(value) if value else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None


def _torch_load_mapping(path: Path) -> dict[Any, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        msg = f"Expected mapping in {path}, got {type(payload).__name__}"
        raise TypeError(msg)
    return payload


def _as_vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _load_state_onehot_map(
    model_dir: Path | None,
    basename: str,
) -> dict[Any, Any] | None:
    if model_dir is None:
        return None
    candidates = (
        model_dir / f"{basename}.torch",
        model_dir / f"{basename}.pt",
        model_dir / f"{basename}.pkl",
    )
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".pkl":
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict):
                msg = f"Expected mapping in {path}, got {type(payload).__name__}"
                raise TypeError(msg)
            return payload
        return _torch_load_mapping(path)
    return None


def _argmax_index(value: Any) -> int | None:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        return int(array.item())
    if array.size == 0:
        return None
    return int(np.argmax(array.reshape(-1)))


def _state_hvg_indices(adata: ad.AnnData, config: AivcConfig) -> np.ndarray:
    checkpoint_indices = _checkpoint_gene_indices(adata, config.state.model_dir)
    if checkpoint_indices is not None:
        return checkpoint_indices
    if config.data.state_hvg_n_top_genes is None:
        msg = (
            "AnnData is missing obsm['X_hvg']; configure data.state_hvg_n_top_genes "
            "or provide a STATE model_dir with var_dims.pkl gene_names"
        )
        raise ValueError(msg)
    n_top = min(int(config.data.state_hvg_n_top_genes), int(adata.n_vars))
    variances = _column_variance(adata.X)
    indices = np.argsort(variances)[-n_top:]
    return np.sort(indices.astype(np.int64))


def _checkpoint_gene_indices(
    adata: ad.AnnData,
    model_dir: Path | None,
) -> np.ndarray | None:
    if model_dir is None:
        return None
    path = model_dir / "var_dims.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    names = payload.get("gene_names") if isinstance(payload, dict) else None
    if names is None:
        return None
    var_to_index = {str(name): index for index, name in enumerate(adata.var_names)}
    selected: list[int] = []
    for name in names:
        index = var_to_index.get(str(name))
        if index is None:
            return None
        selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def _column_variance(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).reshape(-1)
        mean_square = np.asarray(matrix.power(2).mean(axis=0)).reshape(-1)
        return np.maximum(mean_square - mean**2, 0.0)
    return np.var(np.asarray(matrix), axis=0)


def _dense_slice(matrix: Any, indices: np.ndarray) -> np.ndarray:
    subset = matrix[:, indices]
    if sparse.issparse(subset):
        subset = subset.toarray()
    return np.asarray(subset, dtype=np.float32)


def _feature_names(prefix: str, n_features: int) -> np.ndarray:
    return np.asarray(
        [f"{prefix}_{index}" for index in range(n_features)],
        dtype=object,
    )


def _import_scvi() -> object:
    try:
        import scipy.linalg as scipy_linalg

        if not hasattr(scipy_linalg, "tril"):
            scipy_linalg.tril = np.tril
        import scvi
    except ImportError as error:
        msg = "scvi-tools is required when projector.teacher is 'scvi'"
        raise ImportError(msg) from error
    return scvi


@contextmanager
def _suppress_normalized_x_warning() -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"adata\.X does not contain unnormalized count data.*",
            category=UserWarning,
        )
        yield
