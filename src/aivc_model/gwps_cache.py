"""Fold-neutral, fingerprinted GWPS expression cache."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from aivc_model.expression import (
    assert_finite_npy,
    compute_finite_feature_means,
    replace_nonfinite,
)
from aivc_model.gene_splits import (
    CANONICAL_GENE_COUNT,
    load_canonical_outer_manifest,
    sha256_file,
)
from aivc_model.prepare import (
    AivcConfig,
    GeneBags,
    _load_metadata,
    resolve_state_gene_order,
)

_SCHEMA_VERSION = 2
_STATE_FEATURE_COUNT = 2000
_ROW_CHUNK_SIZE = 1024
_ARRAY_FILENAMES = (
    "cells.npy",
    "offsets.npy",
    "genes.npy",
    "gene_outer_folds.npy",
    "batch_labels.npy",
    "control_cells.npy",
    "control_batch.npy",
    "feature_names.npy",
    "feature_fill_values.npy",
)


@dataclass(frozen=True)
class _CacheContract:
    gene_count: int
    state_dim: int


_PRODUCTION_CONTRACT = _CacheContract(
    gene_count=CANONICAL_GENE_COUNT,
    state_dim=_STATE_FEATURE_COUNT,
)


def file_signature(path: Path) -> dict[str, object]:
    """Return a cheap identity signature suitable for a large source file."""
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "resolved_path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def sha256_strings(values: np.ndarray) -> str:
    """Hash an ordered string array without ambiguous concatenation."""
    digest = hashlib.sha256()
    for value in np.asarray(values).astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def sidecar_signature(model_dir: Path, basename: str) -> dict[str, str] | None:
    """Hash the first supported checkpoint sidecar for a basename."""
    for suffix in (".torch", ".pt", ".pkl"):
        path = model_dir / f"{basename}{suffix}"
        if path.exists():
            return {
                "resolved_path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
    return None


def source_fingerprint(
    *,
    h5ad: Path,
    checkpoint: Path,
    var_dims: Path,
    pert_onehot_map: Path,
    batch_sidecar: Path | None,
    cell_type_sidecar: Path | None,
    feature_names: np.ndarray,
    canonical_split: Path,
    canonical_split_sha256_file: Path,
    overlap_csv: Path,
    canonical_gene_count: int,
    cache_seed: int,
    cache_cells_per_gene: int,
    depmap_label_col: str,
    matched_label_col: str,
    var_gene_symbol_col: str,
    obs_perturbation_col: str,
    control_label: str,
    obs_batch_col: str | None,
) -> str:
    """Hash every source that can affect the deterministic raw cache."""
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "h5ad": file_signature(h5ad),
        "checkpoint": file_signature(checkpoint),
        "var_dims": sha256_file(var_dims),
        "pert_onehot_map": sha256_file(pert_onehot_map),
        "batch_sidecar": _optional_sha256(batch_sidecar),
        "cell_type_sidecar": _optional_sha256(cell_type_sidecar),
        "feature_names_sha256": sha256_strings(feature_names),
        "canonical_split_sha256": sha256_file(canonical_split),
        "canonical_split_sha256_authority": sha256_file(canonical_split_sha256_file),
        "overlap_csv_sha256": sha256_file(overlap_csv),
        "canonical_gene_count": int(canonical_gene_count),
        "cache_seed": int(cache_seed),
        "cache_cells_per_gene": int(cache_cells_per_gene),
        "depmap_label_col": depmap_label_col,
        "matched_label_col": matched_label_col,
        "var_gene_symbol_col": var_gene_symbol_col,
        "obs_perturbation_col": obs_perturbation_col,
        "control_label": control_label,
        "obs_batch_col": obs_batch_col,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_gwps_cache(config: AivcConfig, cache_dir: Path) -> Path:
    """Build deterministic raw STATE-aligned GWPS arrays and their manifest."""
    return _build_gwps_cache(config, cache_dir, _PRODUCTION_CONTRACT)


def _build_gwps_cache(
    config: AivcConfig,
    cache_dir: Path,
    contract: _CacheContract,
) -> Path:
    model_dir, checkpoint, canonical_path, canonical_sha_path = _required_paths(config)
    adata = ad.read_h5ad(config.data.h5ad_path, backed="r")
    try:
        indices, feature_names = resolve_state_gene_order(
            adata,
            model_dir,
            config.data.var_gene_symbol_col,
        )
        _validate_state_contract(model_dir, feature_names, contract)
        canonical = _load_canonical_manifest(config, canonical_path, contract)
        fingerprint = _config_fingerprint(
            config,
            feature_names,
            model_dir=model_dir,
            checkpoint=checkpoint,
            canonical_path=canonical_path,
            canonical_sha_path=canonical_sha_path,
            contract=contract,
        )
        manifest_path = cache_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("source_fingerprint") != fingerprint:
                raise ValueError("GWPS cache fingerprint mismatch")
            _validate_manifest_arrays(cache_dir, manifest, contract)
            return manifest_path

        labels = (
            adata.obs[config.data.obs_perturbation_col]
            .astype(str)
            .str.upper()
            .to_numpy()
        )
        batches = _source_batches(adata, config)
        control_label = config.data.control_label.upper()
        canonical_genes = canonical["perturbation_gene"].to_numpy(dtype=str)
        missing = sorted(set(canonical_genes).difference(labels))
        if missing:
            raise ValueError(
                f"source is missing canonical response genes: {missing[:10]}"
            )
        selected_rows, offsets = _sample_response_rows(
            labels,
            canonical_genes,
            seed=config.data.cache_seed,
            cells_per_gene=config.data.cache_cells_per_gene,
        )
        control_rows = np.flatnonzero(labels == control_label)
        if not len(control_rows):
            raise ValueError(
                f"No control cells found for {config.data.control_label!r}"
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        fill_values = compute_finite_feature_means(
            adata.X,
            control_rows,
            indices,
            chunk_size=_ROW_CHUNK_SIZE,
        )
        _write_matrix_rows(
            cache_dir / "cells.npy",
            adata.X,
            selected_rows,
            indices,
            fill_values,
        )
        _write_matrix_rows(
            cache_dir / "control_cells.npy",
            adata.X,
            control_rows,
            indices,
            fill_values,
        )
        _write_array(cache_dir / "feature_fill_values.npy", fill_values)
    finally:
        adata.file.close()

    _write_array(cache_dir / "offsets.npy", offsets)
    _write_array(cache_dir / "genes.npy", canonical_genes)
    _write_array(
        cache_dir / "gene_outer_folds.npy",
        canonical["outer_fold"].to_numpy(dtype=np.int64),
    )
    _write_array(cache_dir / "batch_labels.npy", batches[selected_rows])
    _write_array(cache_dir / "control_batch.npy", batches[control_rows])
    _write_array(cache_dir / "feature_names.npy", feature_names.astype(str))
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": _SCHEMA_VERSION,
                "source_fingerprint": fingerprint,
                "arrays": _array_manifest(cache_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest_path


def load_gwps_cache(
    config: AivcConfig,
    cache_dir: Path,
    *,
    verify_hashes: bool = True,
) -> GeneBags:
    """Load and validate the memory-mappable raw GWPS cache."""
    return _load_gwps_cache(
        config,
        cache_dir,
        _PRODUCTION_CONTRACT,
        verify_hashes=verify_hashes,
    )


def _load_gwps_cache(
    config: AivcConfig,
    cache_dir: Path,
    contract: _CacheContract,
    *,
    verify_hashes: bool = True,
) -> GeneBags:
    model_dir, checkpoint, canonical_path, canonical_sha_path = _required_paths(config)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    arrays = _validate_manifest_arrays(
        cache_dir,
        manifest,
        contract,
        verify_hashes=verify_hashes,
    )
    feature_names = arrays["feature_names.npy"]
    _validate_state_contract(model_dir, feature_names, contract)
    canonical = _load_canonical_manifest(config, canonical_path, contract)
    fingerprint = _config_fingerprint(
        config,
        feature_names,
        model_dir=model_dir,
        checkpoint=checkpoint,
        canonical_path=canonical_path,
        canonical_sha_path=canonical_sha_path,
        contract=contract,
    )
    if manifest.get("source_fingerprint") != fingerprint:
        raise ValueError("GWPS cache fingerprint mismatch")

    genes = arrays["genes.npy"]
    outer_folds = arrays["gene_outer_folds.npy"]
    expected_genes = canonical["perturbation_gene"].to_numpy(dtype=str)
    expected_folds = canonical["outer_fold"].to_numpy(dtype=np.int64)
    if not np.array_equal(genes.astype(str), expected_genes) or not np.array_equal(
        outer_folds,
        expected_folds,
    ):
        raise ValueError("GWPS cache genes/folds do not match canonical manifest")

    cells = arrays["cells.npy"]
    offsets = arrays["offsets.npy"]
    batch_labels = arrays["batch_labels.npy"]
    control_cells = arrays["control_cells.npy"]
    control_batch = arrays["control_batch.npy"]
    feature_fill_values = arrays["feature_fill_values.npy"]
    input_bags = tuple(
        cells[int(offsets[index]) : int(offsets[index + 1])]
        for index in range(len(genes))
    )
    batch_bags = tuple(
        batch_labels[int(offsets[index]) : int(offsets[index + 1])]
        for index in range(len(genes))
    )
    metadata = _ordered_metadata(config, expected_genes)
    y = metadata[config.data.depmap_label_col].to_numpy(dtype=np.float32)
    return GeneBags(
        genes=genes.astype(object),
        y=y,
        input_bags=input_bags,
        latent_bags=input_bags,
        control_input=control_cells,
        control_latent=control_cells,
        cell_type_bags=None,
        control_cell_type=None,
        batch_bags=batch_bags,
        control_batch=control_batch,
        feature_names=feature_names.astype(object),
        feature_fill_values=feature_fill_values,
        metadata=metadata,
        input_dim=int(feature_names.shape[0]),
        latent_dim=int(feature_names.shape[0]),
        gene_outer_folds=outer_folds,
    )


def _required_paths(config: AivcConfig) -> tuple[Path, Path, Path, Path]:
    model_dir = config.state.model_dir
    checkpoint = config.state.checkpoint_path
    canonical_path = config.cv.outer_split_manifest
    canonical_sha_path = config.cv.outer_split_sha256_file
    if (
        model_dir is None
        or checkpoint is None
        or canonical_path is None
        or canonical_sha_path is None
    ):
        raise ValueError(
            "GWPS cache requires STATE, canonical manifest, and SHA-256 paths"
        )
    return model_dir, checkpoint, canonical_path, canonical_sha_path


def _config_fingerprint(
    config: AivcConfig,
    feature_names: np.ndarray,
    *,
    model_dir: Path,
    checkpoint: Path,
    canonical_path: Path,
    canonical_sha_path: Path,
    contract: _CacheContract,
) -> str:
    return source_fingerprint(
        h5ad=config.data.h5ad_path,
        checkpoint=checkpoint,
        var_dims=model_dir / "var_dims.pkl",
        pert_onehot_map=model_dir / "pert_onehot_map.pt",
        batch_sidecar=_find_sidecar(model_dir, "batch_onehot_map"),
        cell_type_sidecar=_find_sidecar(model_dir, "cell_type_onehot_map"),
        feature_names=feature_names,
        canonical_split=canonical_path,
        canonical_split_sha256_file=canonical_sha_path,
        overlap_csv=config.data.overlap_csv,
        canonical_gene_count=contract.gene_count,
        cache_seed=config.data.cache_seed,
        cache_cells_per_gene=config.data.cache_cells_per_gene,
        depmap_label_col=config.data.depmap_label_col,
        matched_label_col=config.data.matched_label_col,
        var_gene_symbol_col=config.data.var_gene_symbol_col,
        obs_perturbation_col=config.data.obs_perturbation_col,
        control_label=config.data.control_label,
        obs_batch_col=config.data.obs_batch_col,
    )


def _find_sidecar(model_dir: Path, basename: str) -> Path | None:
    for suffix in (".torch", ".pt", ".pkl"):
        path = model_dir / f"{basename}{suffix}"
        if path.exists():
            return path
    return None


def _optional_sha256(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    return {"resolved_path": str(path.resolve()), "sha256": sha256_file(path)}


def _load_canonical_manifest(
    config: AivcConfig,
    path: Path,
    contract: _CacheContract,
) -> pd.DataFrame:
    expected_sha256 = _read_expected_sha256(config.cv.outer_split_sha256_file)
    labels = _load_metadata(config.data)
    labels["perturbation_gene"] = labels["perturbation_gene"].astype(str).str.upper()
    if contract == _PRODUCTION_CONTRACT:
        if (
            len(labels) != contract.gene_count
            or labels["perturbation_gene"].nunique() != contract.gene_count
        ):
            raise ValueError(
                "canonical label authority must contain exactly 9338 unique genes"
            )
        return load_canonical_outer_manifest(path, labels, expected_sha256)
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"canonical manifest SHA-256 mismatch: {observed_sha256}")
    manifest = pd.read_csv(path)
    if manifest.columns.tolist() != ["perturbation_gene", "outer_fold"]:
        raise ValueError("canonical manifest has invalid columns")
    manifest["perturbation_gene"] = (
        manifest["perturbation_gene"].astype(str).str.upper()
    )
    if (
        len(manifest) != contract.gene_count
        or manifest["perturbation_gene"].nunique() != contract.gene_count
        or len(labels) != contract.gene_count
        or labels["perturbation_gene"].nunique() != contract.gene_count
        or set(manifest["perturbation_gene"]) != set(labels["perturbation_gene"])
    ):
        raise ValueError(
            f"canonical manifest must contain exactly {contract.gene_count} genes"
        )
    folds = pd.to_numeric(manifest["outer_fold"], errors="raise")
    if not np.equal(folds, np.floor(folds)).all():
        raise ValueError("canonical manifest outer_fold must contain integers")
    manifest["outer_fold"] = folds.astype(np.int64)
    return manifest


def _read_expected_sha256(path: Path | None) -> str:
    if path is None:
        raise ValueError("cv.outer_split_sha256_file is required")
    text = path.read_text()
    digest = text[:-1] if text.endswith("\n") else ""
    if (
        len(text) != 65
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "outer split SHA-256 file must contain one lowercase digest and newline"
        )
    return digest


def _validate_state_contract(
    model_dir: Path,
    feature_names: np.ndarray,
    contract: _CacheContract,
) -> None:
    path = model_dir / "var_dims.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("STATE var_dims.pkl must contain checkpoint metadata")
    input_dim = payload.get("input_dim")
    output_dim = payload.get("output_dim")
    feature_count = int(len(feature_names))
    expected_dim = contract.state_dim
    if input_dim != expected_dim:
        raise ValueError(
            f"STATE checkpoint input_dim={input_dim}; expected {expected_dim}"
        )
    if output_dim != expected_dim:
        raise ValueError(
            f"STATE checkpoint output_dim={output_dim}; expected {expected_dim}"
        )
    if feature_count != expected_dim:
        raise ValueError(
            f"STATE aligned feature count={feature_count}; expected {expected_dim}"
        )


def _source_batches(adata: ad.AnnData, config: AivcConfig) -> np.ndarray:
    column = config.data.obs_batch_col
    if column is None or column not in adata.obs:
        raise ValueError("GWPS cache requires a configured source batch column")
    return adata.obs[column].astype(str).to_numpy()


def _sample_response_rows(
    labels: np.ndarray,
    genes: np.ndarray,
    *,
    seed: int,
    cells_per_gene: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cells_per_gene < 1:
        raise ValueError("cache_cells_per_gene must be positive")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    offsets = [0]
    for gene in genes:
        indices = np.flatnonzero(labels == gene)
        if len(indices) > cells_per_gene:
            indices = np.sort(rng.choice(indices, size=cells_per_gene, replace=False))
        selected.append(indices)
        offsets.append(offsets[-1] + len(indices))
    flat = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    return flat, np.asarray(offsets, dtype=np.int64)


def _write_matrix_rows(
    path: Path,
    matrix: object,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    fill_values: np.ndarray,
) -> None:
    """Stream selected backed rows and columns into a float32 NPY memmap."""
    target = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(row_indices), len(column_indices)),
    )
    for start in range(0, len(row_indices), _ROW_CHUNK_SIZE):
        stop = min(start + _ROW_CHUNK_SIZE, len(row_indices))
        rows = row_indices[start:stop]
        order = np.argsort(rows)
        chunk = matrix[rows[order], :]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()
        values = np.asarray(chunk)[np.argsort(order)][:, column_indices]
        target[start:stop] = replace_nonfinite(values, fill_values)
    target.flush()


def _write_array(path: Path, values: np.ndarray) -> None:
    array = np.asarray(values)
    if array.dtype.kind in {"O", "U", "S"}:
        strings = array.astype(str)
        width = max((len(value) for value in strings.flat), default=1)
        array = strings.astype(f"<U{width}")
    target = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=array.dtype,
        shape=array.shape,
    )
    target[...] = array
    target.flush()


def _load_array(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _array_manifest(cache_dir: Path) -> dict[str, dict[str, object]]:
    """Bind every completed array without materializing memory-mapped payloads."""
    return {
        filename: _array_metadata(cache_dir / filename)
        for filename in _ARRAY_FILENAMES
    }


def _array_metadata(path: Path) -> dict[str, object]:
    array = _load_array(path)
    return {
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def validate_cache_arrays(
    cache_dir: Path,
    *,
    gene_count: int,
    state_dim: int,
) -> dict[str, np.ndarray]:
    """Validate schema-v2 array authority for strict preflight."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("prepared GWPS cache manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    return _validate_manifest_arrays(
        cache_dir,
        manifest,
        _CacheContract(gene_count=gene_count, state_dim=state_dim),
    )


def _validate_manifest_arrays(
    cache_dir: Path,
    manifest: object,
    contract: _CacheContract,
    *,
    verify_hashes: bool = True,
) -> dict[str, np.ndarray]:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError(f"GWPS cache schema must be version {_SCHEMA_VERSION}")
    fingerprint = manifest.get("source_fingerprint")
    if (
        set(manifest) != {"schema_version", "source_fingerprint", "arrays"}
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("GWPS cache manifest contract is invalid")
    metadata = manifest.get("arrays")
    if not isinstance(metadata, dict) or set(metadata) != set(_ARRAY_FILENAMES):
        raise ValueError("GWPS cache manifest array contract is invalid")

    arrays: dict[str, np.ndarray] = {}
    for filename in _ARRAY_FILENAMES:
        expected = metadata[filename]
        if not isinstance(expected, dict) or set(expected) != {
            "sha256",
            "shape",
            "dtype",
        }:
            raise ValueError("GWPS cache manifest array contract is invalid")
        path = cache_dir / filename
        if not path.is_file():
            raise ValueError(f"GWPS cache array is missing: {filename}")
        array = _load_array(path)
        if expected["shape"] != list(array.shape):
            raise ValueError(f"{filename} shape differs from manifest")
        if expected["dtype"] != array.dtype.str:
            raise ValueError(f"{filename} dtype differs from manifest")
        if verify_hashes and expected["sha256"] != sha256_file(path):
            raise ValueError(f"{filename} SHA-256 mismatch")
        arrays[filename] = array

    _validate_array_structure(cache_dir, arrays, contract)
    return arrays


def _validate_array_structure(
    cache_dir: Path,
    arrays: dict[str, np.ndarray],
    contract: _CacheContract,
) -> None:
    cells = arrays["cells.npy"]
    offsets = arrays["offsets.npy"]
    genes = arrays["genes.npy"]
    folds = arrays["gene_outer_folds.npy"]
    batches = arrays["batch_labels.npy"]
    controls = arrays["control_cells.npy"]
    control_batch = arrays["control_batch.npy"]
    features = arrays["feature_names.npy"]
    fills = arrays["feature_fill_values.npy"]

    if cells.ndim != 2 or cells.shape[1] != contract.state_dim:
        raise ValueError("GWPS cache cells must be 2-D with the STATE width")
    if cells.dtype.kind not in {"b", "i", "u", "f", "c"}:
        raise ValueError("GWPS cache cells must be numeric")
    if offsets.dtype.kind not in {"i", "u"} or offsets.shape != (
        contract.gene_count + 1,
    ):
        raise ValueError("GWPS cache offsets must be one integer boundary per gene")
    if int(offsets[0]) != 0:
        raise ValueError("GWPS cache offsets must start at zero")
    if np.any(np.diff(np.asarray(offsets, dtype=np.int64)) <= 0):
        raise ValueError("GWPS cache every gene bag must be non-empty")
    if int(offsets[-1]) != len(cells):
        raise ValueError("GWPS cache offsets must end at the response row count")
    if genes.shape != (contract.gene_count,):
        raise ValueError("GWPS cache genes must match the canonical gene count")
    if folds.shape != (contract.gene_count,):
        raise ValueError("GWPS cache folds must match the canonical gene count")
    if batches.shape != (len(cells),):
        raise ValueError("GWPS cache response batches must match response cells")
    if (
        controls.ndim != 2
        or not len(controls)
        or controls.shape[1] != contract.state_dim
    ):
        raise ValueError("GWPS cache controls must be non-empty with the STATE width")
    if controls.dtype.kind not in {"b", "i", "u", "f", "c"}:
        raise ValueError("GWPS cache controls must be numeric")
    if control_batch.shape != (len(controls),):
        raise ValueError("GWPS cache control batches must match control cells")
    if features.shape != (contract.state_dim,):
        raise ValueError("GWPS cache features must match the STATE dimension")
    if fills.shape != (contract.state_dim,) or fills.dtype.kind not in {
        "b",
        "i",
        "u",
        "f",
        "c",
    } or not np.isfinite(fills).all():
        raise ValueError(
            "GWPS cache feature fill values must match the finite STATE dimension"
        )
    assert_finite_npy(cache_dir / "cells.npy")
    assert_finite_npy(cache_dir / "control_cells.npy")


def _ordered_metadata(config: AivcConfig, genes: np.ndarray) -> pd.DataFrame:
    metadata = _load_metadata(config.data).copy()
    metadata["perturbation_gene"] = (
        metadata["perturbation_gene"].astype(str).str.upper()
    )
    indexed = metadata.set_index("perturbation_gene", drop=False)
    missing = [gene for gene in genes if gene not in indexed.index]
    if missing:
        raise ValueError(f"canonical genes missing DepMap labels: {missing[:10]}")
    return indexed.loc[genes].reset_index(drop=True)
