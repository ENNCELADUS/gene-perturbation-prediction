"""Build a label-blind, STATE-aligned HCT116 X-Atlas expression cache."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import pickle
from typing import Iterator

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aivc_model.expression import assert_finite_npy
from aivc_model.gene_splits import sha256_file

LOGGER = logging.getLogger(__name__)
_COLUMNS = (
    "gene_token_id",
    "gene_expression",
    "sample",
    "gene_target",
    "pass_guide_filter",
)
_SCHEMA_VERSION = 1


def _ensembl(value: object) -> str:
    return str(value).split(".", maxsplit=1)[0]


def build_feature_map(
    gene_metadata_path: Path,
    k562_h5ad_path: Path,
    state_model_dir: Path,
    *,
    symbol_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map ordered STATE features to X-Atlas tokens through Ensembl IDs."""
    with (state_model_dir / "var_dims.pkl").open("rb") as handle:
        state_names = np.asarray(pickle.load(handle)["gene_names"]).astype(str)
    k562 = ad.read_h5ad(k562_h5ad_path, backed="r")
    try:
        if symbol_col not in k562.var:
            raise ValueError(f"K562 var is missing {symbol_col!r}")
        symbols = k562.var[symbol_col].astype(str).to_numpy()
        symbol_counts = Counter(symbols)
        symbol_to_ensembl = {
            symbol: _ensembl(ensembl)
            for symbol, ensembl in zip(symbols, k562.var_names, strict=True)
            if symbol_counts[symbol] == 1
        }
    finally:
        k562.file.close()

    metadata = pd.read_parquet(
        gene_metadata_path,
        columns=["ensembl_id", "gene_name", "gene_token_id"],
    )
    if metadata["gene_token_id"].duplicated().any():
        raise ValueError("X-Atlas gene_token_id values must be unique")
    metadata["ensembl_id"] = metadata["ensembl_id"].map(_ensembl)
    token_by_ensembl: dict[str, list[int]] = {}
    for ensembl, token in metadata[["ensembl_id", "gene_token_id"]].itertuples(
        index=False, name=None
    ):
        token_by_ensembl.setdefault(str(ensembl), []).append(int(token))
    symbol_counts = metadata["gene_name"].astype(str).value_counts()
    token_by_unique_symbol = {
        str(symbol): [int(token)]
        for symbol, token in metadata[["gene_name", "gene_token_id"]].itertuples(
            index=False, name=None
        )
        if int(symbol_counts[str(symbol)]) == 1
    }

    feature_ensembl = np.asarray(
        [symbol_to_ensembl.get(name, "") for name in state_names]
    )
    flat_tokens: list[int] = []
    offsets = [0]
    mapping_methods: list[str] = []
    for name, ensembl in zip(state_names, feature_ensembl, strict=True):
        tokens = token_by_ensembl.get(str(ensembl), [])
        method = "ensembl"
        if not tokens:
            tokens = token_by_unique_symbol.get(str(name), [])
            method = "exact_symbol" if tokens else "unresolved"
        flat_tokens.extend(sorted(tokens))
        offsets.append(len(flat_tokens))
        mapping_methods.append(method)
    return (
        state_names,
        feature_ensembl,
        np.asarray(flat_tokens, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(mapping_methods),
    )


def normalize_cell(
    token_ids: object,
    expressions: object,
    token_to_feature: dict[int, int],
    feature_fill_values: np.ndarray,
    *,
    target_sum: float,
    log1p: bool,
) -> np.ndarray:
    """Align one sparse raw-count cell and apply the frozen input transform."""
    tokens = np.asarray(token_ids, dtype=np.int64)
    counts = np.asarray(expressions, dtype=np.float64)
    if tokens.ndim != 1 or counts.ndim != 1 or tokens.shape != counts.shape:
        raise ValueError("gene_token_id and gene_expression must be equal-length lists")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("X-Atlas expression must contain finite nonnegative counts")
    library_size = float(counts.sum())
    if library_size <= 0:
        raise ValueError("X-Atlas cell has zero library size")
    values = np.zeros(len(feature_fill_values), dtype=np.float64)
    for token, count in zip(tokens, counts, strict=True):
        feature = token_to_feature.get(int(token))
        if feature is not None:
            values[feature] += count
    values *= target_sum / library_size
    if log1p:
        np.log1p(values, out=values)
    resolved = np.zeros(len(values), dtype=bool)
    resolved[list(set(token_to_feature.values()))] = True
    unresolved = ~resolved
    values[unresolved] = feature_fill_values[unresolved]
    result = values.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("normalized STATE input contains nonfinite values")
    return result


def _batches(paths: list[Path], *, row_batch_size: int) -> Iterator[pa.RecordBatch]:
    for path in sorted(paths):
        parquet = pq.ParquetFile(path)
        missing = sorted(set(_COLUMNS).difference(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
        yield from parquet.iter_batches(
            batch_size=row_batch_size,
            columns=list(_COLUMNS),
            use_threads=False,
        )


def _eligible_rows(batch: pa.RecordBatch) -> Iterator[tuple[object, ...]]:
    columns = batch.to_pydict()
    for row in zip(*(columns[column] for column in _COLUMNS), strict=True):
        if int(row[4]) == 1:
            yield row


def _selected_group_counts(
    paths: list[Path],
    *,
    control_label: str,
    min_cells_per_group: int,
    max_cells_per_group: int | None,
    row_batch_size: int,
) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    responses: Counter[tuple[str, str]] = Counter()
    controls: Counter[str] = Counter()
    for batch in _batches(paths, row_batch_size=row_batch_size):
        for _tokens, _counts, sample, target, _passed in _eligible_rows(batch):
            sample = str(sample)
            target = str(target)
            if target.casefold() == control_label.casefold():
                controls[sample] += 1
            else:
                responses[(sample, target.upper())] += 1
    if not responses:
        raise ValueError("no pass_guide_filter=1 perturbation cells found")
    missing_controls = sorted({sample for sample, _ in responses}.difference(controls))
    if missing_controls:
        raise ValueError(
            f"samples lack Non-Targeting controls: {missing_controls[:10]}"
        )
    limit = max_cells_per_group
    eligible_responses = {
        key: count for key, count in responses.items() if count >= min_cells_per_group
    }
    if not eligible_responses:
        raise ValueError("no perturbation groups meet min_cells_per_group")
    selected_responses = {
        key: min(count, limit) if limit is not None else count
        for key, count in eligible_responses.items()
    }
    response_samples = {key[0] for key in selected_responses}
    selected_controls = {
        sample: count
        for sample, count in controls.items()
        if sample in response_samples
    }
    return selected_responses, selected_controls


def _control_library_size_target(
    paths: list[Path],
    *,
    samples: set[str],
    control_label: str,
    row_batch_size: int,
) -> float:
    library_sizes: list[float] = []
    for batch in _batches(paths, row_batch_size=row_batch_size):
        for _tokens, counts, sample, target, _passed in _eligible_rows(batch):
            if str(sample) not in samples:
                continue
            if str(target).casefold() != control_label.casefold():
                continue
            library_size = float(np.asarray(counts, dtype=np.float64).sum())
            if library_size > 0:
                library_sizes.append(library_size)
    if not library_sizes:
        raise ValueError("no positive-library Non-Targeting controls found")
    return float(np.median(np.asarray(library_sizes, dtype=np.float64)))


def _open_array(path: Path, rows: int, columns: int) -> np.memmap:
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(rows, columns)
    )


def _write_array(path: Path, values: np.ndarray) -> None:
    array = np.asarray(values)
    if array.dtype.kind in {"O", "U", "S"}:
        width = max((len(value) for value in array.astype(str)), default=1)
        array = array.astype(f"<U{width}")
    np.save(path, array, allow_pickle=False)


def _feature_moments(
    path: Path, *, chunk_size: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("K562 controls must be a non-empty 2-D NPY")
    sums = np.zeros(matrix.shape[1], dtype=np.float64)
    squares = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, len(matrix), chunk_size):
        values = np.asarray(matrix[start : start + chunk_size], dtype=np.float64)
        sums += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
    means = sums / len(matrix)
    variances = np.maximum(squares / len(matrix) - np.square(means), 0.0)
    return means, np.sqrt(variances)


def _control_context_qa(
    hct_controls: np.ndarray,
    k562_controls: np.ndarray,
) -> dict[str, float]:
    """Describe, without changing, the HCT116-to-K562 control difference."""
    hct_mean, _ = _feature_moments_from_array(hct_controls)
    k562_mean, _ = _feature_moments_from_array(k562_controls)
    difference = hct_mean - k562_mean
    denominator = float(np.linalg.norm(hct_mean) * np.linalg.norm(k562_mean))
    cosine = float(np.dot(hct_mean, k562_mean) / denominator) if denominator else 0.0
    return {
        "mean_difference_l2": float(np.linalg.norm(difference)),
        "mean_difference_mean_absolute": float(np.mean(np.abs(difference))),
        "control_mean_cosine_similarity": cosine,
    }


def _feature_moments_from_array(
    matrix: np.ndarray, *, chunk_size: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("control matrix must be a non-empty 2-D array")
    sums = np.zeros(matrix.shape[1], dtype=np.float64)
    squares = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, len(matrix), chunk_size):
        values = np.asarray(matrix[start : start + chunk_size], dtype=np.float64)
        sums += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
    means = sums / len(matrix)
    variances = np.maximum(squares / len(matrix) - np.square(means), 0.0)
    return means, np.sqrt(variances)


def _zscore_by_sample_controls(
    response_matrix: np.ndarray,
    control_matrix: np.ndarray,
    response_samples: list[str],
    control_samples: list[str],
) -> dict[str, int]:
    """Apply Replogle-style within-sample Non-Targeting relative Z-scores."""
    response_labels = np.asarray(response_samples)
    control_labels = np.asarray(control_samples)
    zero_variance_counts: dict[str, int] = {}
    for sample in sorted(set(control_samples)):
        control_mask = control_labels == sample
        controls = np.asarray(control_matrix[control_mask], dtype=np.float64)
        means = controls.mean(axis=0)
        stds = controls.std(axis=0, ddof=0)
        zero_variance = stds == 0
        zero_variance_counts[sample] = int(np.count_nonzero(zero_variance))
        safe_stds = stds.copy()
        safe_stds[zero_variance] = 1.0
        response_mask = response_labels == sample
        response_matrix[response_mask] = (
            np.asarray(response_matrix[response_mask], dtype=np.float64) - means
        ) / safe_stds
        control_matrix[control_mask] = (controls - means) / safe_stds
    return zero_variance_counts


def _distribution_qa(matrix: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    """Return deterministic sampled quantile and reference-range QA."""
    size = min(len(matrix), 4096)
    indices = np.linspace(0, len(matrix) - 1, num=size, dtype=np.int64)
    values = np.asarray(matrix[indices], dtype=np.float32)
    reference_size = min(len(reference), 4096)
    reference_indices = np.linspace(
        0, len(reference) - 1, num=reference_size, dtype=np.int64
    )
    reference_values = np.asarray(reference[reference_indices], dtype=np.float32)
    probabilities = [0.001, 0.01, 0.5, 0.99, 0.999]
    lower = np.quantile(reference_values, 0.001, axis=0)
    upper = np.quantile(reference_values, 0.999, axis=0)
    return {
        "sample_rows": int(size),
        "value_quantiles": np.quantile(values, probabilities).tolist(),
        "reference_value_quantiles": np.quantile(
            reference_values, probabilities
        ).tolist(),
        "fraction_outside_reference_0.1_99.9pct": float(
            np.mean((values < lower) | (values > upper))
        ),
        "minimum": float(values.min()),
        "negative_fraction": float(np.mean(values < 0)),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(values))),
    }


def _array_metadata(path: Path) -> dict[str, object]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def _load_transform_contract(path: Path) -> dict[str, object]:
    contract = json.loads(path.read_text())
    expected = {
        "schema_version",
        "contract_id",
        "target_sum_policy",
        "log1p",
        "relative_control_zscore",
        "zscore_groupby",
        "zscore_ddof",
        "zero_variance_policy",
        "provenance_verified",
        "source",
        "k562_h5ad_sha256",
        "k562_reference_controls_sha256",
        "frozen_checkpoint_sha256",
        "feature_fill_values_sha256",
        "state_var_dims_sha256",
    }
    if not isinstance(contract, dict) or set(contract) != expected:
        raise ValueError(f"transform contract must contain exactly {sorted(expected)}")
    if contract["schema_version"] != 1 or not isinstance(contract["contract_id"], str):
        raise ValueError("transform contract identity is invalid")
    if not isinstance(contract["log1p"], bool):
        raise ValueError("transform contract log1p must be boolean")
    if contract["target_sum_policy"] != "median_non_targeting_library_size":
        raise ValueError("unsupported transform contract target_sum_policy")
    if contract["log1p"] is not False:
        raise ValueError("Replogle normalized single-cell input does not use log1p")
    if contract["relative_control_zscore"] is not True:
        raise ValueError("Replogle relative control Z-normalization is required")
    if contract["zscore_groupby"] != "sample" or contract["zscore_ddof"] != 0:
        raise ValueError("Replogle Z-normalization must use sample groups and ddof=0")
    if contract["zero_variance_policy"] != "unit_denominator":
        raise ValueError("unsupported zero_variance_policy")
    if contract["provenance_verified"] is not True:
        raise ValueError("transform contract provenance must be verified before use")
    if not isinstance(contract["source"], str) or not contract["source"]:
        raise ValueError("transform contract source must be non-empty")
    for key in (
        "k562_h5ad_sha256",
        "k562_reference_controls_sha256",
        "frozen_checkpoint_sha256",
        "feature_fill_values_sha256",
        "state_var_dims_sha256",
    ):
        value = contract[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"transform contract {key} must be a SHA-256 digest")
    return contract


def _verify_transform_artifacts(
    contract: dict[str, object],
    *,
    k562_h5ad_path: Path,
    k562_reference_controls_path: Path,
    frozen_checkpoint_path: Path,
    feature_fill_values_path: Path,
    state_var_dims_path: Path,
) -> None:
    expected_paths = {
        "k562_h5ad_sha256": k562_h5ad_path,
        "k562_reference_controls_sha256": k562_reference_controls_path,
        "frozen_checkpoint_sha256": frozen_checkpoint_path,
        "feature_fill_values_sha256": feature_fill_values_path,
        "state_var_dims_sha256": state_var_dims_path,
    }
    mismatches = [
        key
        for key, path in expected_paths.items()
        if sha256_file(path) != contract[key]
    ]
    if mismatches:
        raise ValueError(f"transform contract artifact hash mismatch: {mismatches}")


def build_hct116_cache(
    parquet_paths: list[Path],
    gene_metadata_path: Path,
    k562_h5ad_path: Path,
    state_model_dir: Path,
    frozen_checkpoint_path: Path,
    k562_reference_controls_path: Path,
    feature_fill_values_path: Path,
    transform_contract_path: Path,
    output_dir: Path,
    *,
    symbol_col: str = "gene_name",
    control_label: str = "Non-Targeting",
    min_cells_per_group: int,
    max_cells_per_group: int | None = None,
    row_batch_size: int = 1024,
) -> Path:
    """Stream a deterministic label-blind HCT116 cache and QA manifest."""
    if not parquet_paths or any(not path.is_file() for path in parquet_paths):
        raise ValueError("at least one existing X-Atlas parquet is required")
    if len({path.resolve() for path in parquet_paths}) != len(parquet_paths):
        raise ValueError("X-Atlas parquet paths must be unique")
    if not frozen_checkpoint_path.is_file():
        raise ValueError("an existing frozen inference checkpoint is required")
    if row_batch_size < 1 or min_cells_per_group < 1:
        raise ValueError("row_batch_size and min_cells_per_group must be positive")
    if max_cells_per_group is not None and max_cells_per_group < 1:
        raise ValueError("max_cells_per_group must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    contract = _load_transform_contract(transform_contract_path)
    _verify_transform_artifacts(
        contract,
        k562_h5ad_path=k562_h5ad_path,
        k562_reference_controls_path=k562_reference_controls_path,
        frozen_checkpoint_path=frozen_checkpoint_path,
        feature_fill_values_path=feature_fill_values_path,
        state_var_dims_path=state_model_dir / "var_dims.pkl",
    )
    source_hashes = {
        str(path.resolve()): sha256_file(path) for path in sorted(parquet_paths)
    }

    names, ensembl, flat_tokens, token_offsets, mapping_methods = build_feature_map(
        gene_metadata_path,
        k562_h5ad_path,
        state_model_dir,
        symbol_col=symbol_col,
    )
    fills = np.load(feature_fill_values_path, allow_pickle=False).astype(np.float32)
    if fills.shape != names.shape or not np.isfinite(fills).all():
        raise ValueError(
            "feature fill values must match the finite STATE feature width"
        )
    k562_control_means, _ = _feature_moments(k562_reference_controls_path)
    if k562_control_means.shape != names.shape or not np.allclose(
        k562_control_means, fills, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("K562 control cells and feature fill means disagree")
    token_to_feature = {
        int(token): feature
        for feature in range(len(names))
        for token in flat_tokens[token_offsets[feature] : token_offsets[feature + 1]]
    }
    if len(token_to_feature) != len(flat_tokens):
        raise ValueError("an X-Atlas token maps to multiple STATE features")
    response_counts, control_counts = _selected_group_counts(
        parquet_paths,
        control_label=control_label,
        min_cells_per_group=min_cells_per_group,
        max_cells_per_group=max_cells_per_group,
        row_batch_size=row_batch_size,
    )
    target_sum = _control_library_size_target(
        parquet_paths,
        samples=set(control_counts),
        control_label=control_label,
        row_batch_size=row_batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    response_matrix = _open_array(
        output_dir / "response_cells.npy", sum(response_counts.values()), len(names)
    )
    control_matrix = _open_array(
        output_dir / "control_cells.npy", sum(control_counts.values()), len(names)
    )
    response_keys = sorted(response_counts)
    response_group_counts = np.asarray(
        [response_counts[key] for key in response_keys], dtype=np.int64
    )
    response_group_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(response_group_counts)]
    )
    response_bases = {
        key: int(response_group_offsets[index])
        for index, key in enumerate(response_keys)
    }
    response_samples = [
        sample
        for sample, gene in response_keys
        for _ in range(response_counts[(sample, gene)])
    ]
    response_genes = [
        gene
        for sample, gene in response_keys
        for _ in range(response_counts[(sample, gene)])
    ]
    control_keys = sorted(control_counts)
    control_group_counts = np.asarray(
        [control_counts[sample] for sample in control_keys], dtype=np.int64
    )
    control_group_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(control_group_counts)]
    )
    control_bases = {
        sample: int(control_group_offsets[index])
        for index, sample in enumerate(control_keys)
    }
    control_samples = [
        sample for sample in control_keys for _ in range(control_counts[sample])
    ]
    used_responses: Counter[tuple[str, str]] = Counter()
    used_controls: Counter[str] = Counter()
    for batch in _batches(parquet_paths, row_batch_size=row_batch_size):
        for row in _eligible_rows(batch):
            tokens, counts, sample_value, target_value, _passed = row
            sample, target = str(sample_value), str(target_value)
            is_control = target.casefold() == control_label.casefold()
            key = sample if is_control else (sample, target.upper())
            selected = control_counts if is_control else response_counts
            used = used_controls if is_control else used_responses
            if key not in selected or used[key] >= selected[key]:
                continue
            values = normalize_cell(
                tokens,
                counts,
                token_to_feature,
                fills,
                target_sum=target_sum,
                log1p=False,
            )
            if is_control:
                control_index = control_bases[sample] + used_controls[sample]
                control_matrix[control_index] = values
            else:
                response_index = response_bases[key] + used_responses[key]
                response_matrix[response_index] = values
            used[key] += 1
    response_matrix.flush()
    control_matrix.flush()
    if sum(used_responses.values()) != len(response_matrix) or sum(
        used_controls.values()
    ) != len(control_matrix):
        raise RuntimeError("written cache rows differ from the streaming count pass")
    post_hashes = {
        str(path.resolve()): sha256_file(path) for path in sorted(parquet_paths)
    }
    if post_hashes != source_hashes:
        raise RuntimeError("X-Atlas parquet changed during cache construction")
    zero_variance_counts = _zscore_by_sample_controls(
        response_matrix,
        control_matrix,
        response_samples,
        control_samples,
    )
    response_matrix.flush()
    control_matrix.flush()
    arrays = {
        "feature_names.npy": names,
        "feature_ensembl_ids.npy": ensembl,
        "feature_token_ids.npy": flat_tokens,
        "feature_token_offsets.npy": token_offsets,
        "feature_mapping_methods.npy": mapping_methods,
        "feature_fill_values.npy": fills,
        "response_genes.npy": np.asarray(response_genes),
        "response_samples.npy": np.asarray(response_samples),
        "control_samples.npy": np.asarray(control_samples),
        "response_group_samples.npy": np.asarray([key[0] for key in response_keys]),
        "response_group_genes.npy": np.asarray([key[1] for key in response_keys]),
        "response_group_offsets.npy": response_group_offsets,
        "response_group_counts.npy": response_group_counts,
        "control_group_samples.npy": np.asarray(control_keys),
        "control_group_offsets.npy": control_group_offsets,
        "control_group_counts.npy": control_group_counts,
    }
    for filename, values in arrays.items():
        _write_array(output_dir / filename, values)
    assert_finite_npy(output_dir / "response_cells.npy")
    assert_finite_npy(output_dir / "control_cells.npy")
    unresolved = np.flatnonzero(np.diff(token_offsets) == 0)
    qa = {
        "label_blind": True,
        "read_columns": list(_COLUMNS),
        "state_feature_count": int(len(names)),
        "mapped_feature_count": int(len(names) - len(unresolved)),
        "unresolved_features": names[unresolved].tolist(),
        "mapping_method_counts": {
            method: int(np.count_nonzero(mapping_methods == method))
            for method in ("ensembl", "exact_symbol", "unresolved")
        },
        "response_cells": int(len(response_matrix)),
        "control_cells": int(len(control_matrix)),
        "response_groups": int(len(response_counts)),
        "samples": sorted(control_counts),
        "normalization": {
            **contract,
            "resolved_target_sum": target_sum,
            "k562_training_transform_verified": True,
            "contract_sha256": sha256_file(transform_contract_path),
        },
        "state_batch_embedding": "not_used_observed_response_transport",
        "control_alignment": {
            "method": "replogle_sample_relative_non_targeting_zscore",
            "hct116_control_state": "sample_matched_non_targeting_cells",
            "hct116_to_k562_mean_std_matching": False,
            "raw_cell_line_baseline_preserved": False,
            "relative_perturbation_context_preserved": True,
            "zero_variance_feature_counts": zero_variance_counts,
            **_control_context_qa(
                control_matrix,
                np.load(
                    k562_reference_controls_path,
                    mmap_mode="r",
                    allow_pickle=False,
                ),
            ),
        },
        "response_distribution": _distribution_qa(
            response_matrix,
            np.load(k562_reference_controls_path, mmap_mode="r", allow_pickle=False),
        ),
        "control_distribution": _distribution_qa(
            control_matrix,
            np.load(k562_reference_controls_path, mmap_mode="r", allow_pickle=False),
        ),
        "response_nonzero_fraction": float(
            np.count_nonzero(response_matrix) / response_matrix.size
        ),
        "control_nonzero_fraction": float(
            np.count_nonzero(control_matrix) / control_matrix.size
        ),
    }
    qa_path = output_dir / "qa.json"
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n")
    cache_files = ["response_cells.npy", "control_cells.npy", *arrays, "qa.json"]
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "label_blind": True,
        "sources": {
            "parquet": source_hashes,
            "gene_metadata": sha256_file(gene_metadata_path),
            "k562_h5ad": sha256_file(k562_h5ad_path),
            "state_var_dims": sha256_file(state_model_dir / "var_dims.pkl"),
            "frozen_checkpoint": sha256_file(frozen_checkpoint_path),
            "k562_reference_controls": sha256_file(k562_reference_controls_path),
            "feature_fill_values": sha256_file(feature_fill_values_path),
            "transform_contract": sha256_file(transform_contract_path),
        },
        "parameters": {
            "symbol_col": symbol_col,
            "control_label": control_label,
            "min_cells_per_group": min_cells_per_group,
            "max_cells_per_group": max_cells_per_group,
            "row_batch_size": row_batch_size,
        },
        "files": {
            filename: (
                _array_metadata(output_dir / filename)
                if filename.endswith(".npy")
                else {"sha256": sha256_file(output_dir / filename)}
            )
            for filename in cache_files
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, action="append", default=[])
    parser.add_argument("--parquet-dir", type=Path)
    parser.add_argument("--expected-parquet-count", type=int)
    parser.add_argument("--gene-metadata", type=Path, required=True)
    parser.add_argument("--k562-h5ad", type=Path, required=True)
    parser.add_argument("--state-model-dir", type=Path, required=True)
    parser.add_argument("--frozen-checkpoint", type=Path, required=True)
    parser.add_argument("--k562-reference-controls", type=Path, required=True)
    parser.add_argument("--feature-fill-values", type=Path, required=True)
    parser.add_argument("--transform-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--symbol-col", default="gene_name")
    parser.add_argument("--control-label", default="Non-Targeting")
    parser.add_argument("--min-cells-per-group", type=int, required=True)
    parser.add_argument("--max-cells-per-group", type=int)
    parser.add_argument("--row-batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    """Build the configured HCT116 cache."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parquet_paths = list(args.parquet)
    if args.parquet_dir is not None:
        parquet_paths.extend(sorted(args.parquet_dir.glob("*.parquet")))
    if args.expected_parquet_count is not None and (
        len(parquet_paths) != args.expected_parquet_count
    ):
        raise ValueError(
            f"expected {args.expected_parquet_count} parquet files, "
            f"found {len(parquet_paths)}"
        )
    manifest = build_hct116_cache(
        parquet_paths,
        args.gene_metadata,
        args.k562_h5ad,
        args.state_model_dir,
        args.frozen_checkpoint,
        args.k562_reference_controls,
        args.feature_fill_values,
        args.transform_contract,
        args.output_dir,
        symbol_col=args.symbol_col,
        control_label=args.control_label,
        min_cells_per_group=args.min_cells_per_group,
        max_cells_per_group=args.max_cells_per_group,
        row_batch_size=args.row_batch_size,
    )
    LOGGER.info("HCT116 cache ready: %s", manifest)


if __name__ == "__main__":
    main()
