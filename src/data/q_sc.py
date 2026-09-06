"""data / q sc."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import sparse
from src.data.geneeffect import RAW_UMI_SEMANTICS
from src.data.geneeffect import _REGISTRY_COLUMNS
from src.data.geneeffect import _unique_strings


@dataclass(frozen=True)
class QScFeatures:
    """Per-gene raw-count population summaries for one cell line."""

    symbols: tuple[str, ...]
    values: np.ndarray
    available: np.ndarray


def load_q_sc_line(
    cache_dir: Path, model_id: str, expected_genes: Sequence[str]
) -> QScFeatures:
    """Open one prepared q_sc shard without consulting raw source data.

    The historical ``source_sha256`` scalar is part of the file layout but is
    deliberately not read or recomputed by the training input path.
    """
    symbols = tuple(str(gene).strip().upper() for gene in expected_genes)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError(
            "expected q_sc genes must be non-empty and unique"
            "; run `hpc/run.sh prepare <config>`"
        )
    path = Path(cache_dir) / f"{model_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing prepared q_sc shard {path}; run `hpc/run.sh prepare <config>`"
        )
    try:
        with np.load(path, allow_pickle=False) as shard:
            expected_keys = {
                "model_id",
                "gene_symbols",
                "values",
                "available",
            }
            if not expected_keys.issubset(shard.files):
                raise ValueError(
                    f"q_sc shard {path} fields {sorted(shard.files)} do not match "
                    f"the prepared layout {sorted(expected_keys)}"
                    "; run `hpc/run.sh prepare <config>`"
                )
            recorded_model_id = str(shard["model_id"].item())
            recorded_symbols = tuple(shard["gene_symbols"].astype(str).tolist())
            values = np.asarray(shard["values"])
            available = np.asarray(shard["available"])
    except (OSError, EOFError) as exc:
        raise ValueError(
            f"unable to read prepared q_sc shard {path}: {exc}"
            "; run `hpc/run.sh prepare <config>`"
        ) from exc
    if recorded_model_id != model_id:
        raise ValueError(
            f"q_sc shard {path} records model_id {recorded_model_id!r}, "
            f"expected {model_id!r}"
            "; run `hpc/run.sh prepare <config>`"
        )
    if recorded_symbols != symbols:
        raise ValueError(
            f"q_sc shard {path} gene order does not match prepared panel"
            "; run `hpc/run.sh prepare <config>`"
        )
    if values.shape != (len(symbols), 3) or values.dtype != np.dtype(np.float32):
        raise ValueError(
            f"q_sc shard {path} values must be float32 [{len(symbols)}, 3], "
            f"got shape={values.shape} dtype={values.dtype}"
            "; run `hpc/run.sh prepare <config>`"
        )
    if available.shape != (len(symbols),) or available.dtype != np.dtype(bool):
        raise ValueError(
            f"q_sc shard {path} available must be bool [{len(symbols)}], "
            f"got shape={available.shape} dtype={available.dtype}"
            "; run `hpc/run.sh prepare <config>`"
        )
    if bool((~available).any()) and not bool(np.isnan(values[~available]).all()):
        raise ValueError(
            f"q_sc shard {path} has non-NaN unavailable rows"
            "; run `hpc/run.sh prepare <config>`"
        )
    if bool(available.any()):
        present = values[available]
        if (
            not bool(np.isfinite(present).all())
            or bool((present[:, 0] < 0).any())
            or bool(((present[:, 1] < 0) | (present[:, 1] > 1)).any())
            or bool((present[:, 2] < 0).any())
        ):
            raise ValueError(
                f"q_sc shard {path} has invalid available values"
                "; run `hpc/run.sh prepare <config>`"
            )
    return QScFeatures(
        symbols=symbols,
        values=values.astype(np.float32, copy=False),
        available=available.astype(bool, copy=False),
    )


def compute_q_sc(
    adata: Any,
    requested_symbols: Sequence[str],
    *,
    gene_symbol_column: str = "auto",
) -> QScFeatures:
    """Compute mean, detected fraction and population variance from raw counts."""
    symbols = _unique_strings(requested_symbols, "requested symbols")
    if any(symbol != symbol.strip().upper() for symbol in symbols):
        raise ValueError("requested symbols must use canonical uppercase spelling")
    if gene_symbol_column == "auto":
        candidates = [
            name
            for name in ("gene_symbol", "gene_symbols", "gene_name")
            if name in adata.var.columns
        ]
        if len(candidates) != 1:
            raise ValueError(
                "AnnData var must contain exactly one recognized gene-symbol "
                f"column in auto mode, found {candidates}"
            )
        gene_symbol_column = candidates[0]
    if gene_symbol_column not in adata.var.columns:
        raise ValueError(f"AnnData var is missing {gene_symbol_column!r}")
    raw_source_symbols = adata.var[gene_symbol_column]
    if raw_source_symbols.isna().any():
        raise ValueError("AnnData contains missing or empty gene symbols")
    source_symbols = raw_source_symbols.astype(str).str.strip().str.upper()
    if source_symbols.eq("").any():
        raise ValueError("AnnData contains missing or empty gene symbols")
    if not set(symbols).intersection(source_symbols):
        raise ValueError(
            "AnnData gene symbols have zero overlap with requested symbols"
        )
    if int(adata.X.shape[0]) == 0:
        raise ValueError("AnnData contains no cells")
    if hasattr(adata, "obs") and "model_id" in adata.obs.columns:
        model_ids = set(adata.obs["model_id"].astype(str))
        if len(model_ids) != 1:
            raise ValueError("AnnData obs contains multiple model_id values")
    matrix = adata.X
    data = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if data.size and (
        not np.isfinite(data).all()
        or np.any(data < 0)
        or not np.equal(data, np.floor(data)).all()
    ):
        raise ValueError("q_sc requires finite, nonnegative, integer raw UMI counts")
    positions: dict[str, list[int]] = {}
    for index, symbol in enumerate(source_symbols):
        positions.setdefault(symbol, []).append(index)
    values = np.full((len(symbols), 3), np.nan, dtype=np.float32)
    available = np.zeros(len(symbols), dtype=bool)
    for output_index, symbol in enumerate(symbols):
        source_indices = positions.get(symbol)
        if source_indices is None:
            continue
        if len(source_indices) == 1:
            column = matrix[:, source_indices[0]]
        else:
            column = matrix[:, source_indices]
            column = column.sum(axis=1)
        if sparse.issparse(column):
            column = column.toarray()
        array = np.asarray(column, dtype=np.float64).reshape(-1)
        values[output_index] = (array.mean(), np.mean(array > 0), array.var(ddof=0))
        available[output_index] = True
    return QScFeatures(symbols=symbols, values=values, available=available)


def build_q_sc_shards(
    registry: pd.DataFrame,
    output_dir: Path,
    requested_symbols: Sequence[str],
    *,
    reader: Callable[[Path], Any] | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Build atomic per-line q_sc NPZ shards from a validated registry."""
    if registry.index.name != "model_id" or not registry.index.is_unique:
        raise ValueError("registry must be uniquely indexed by model_id")
    missing_columns = sorted(set(_REGISTRY_COLUMNS[1:]) - set(registry.columns))
    if missing_columns:
        raise ValueError(f"registry is missing columns: {missing_columns}")
    if (registry["matrix_semantics"] != RAW_UMI_SEMANTICS).any():
        raise ValueError("registry contains non-raw-UMI semantics")
    if (registry["source_kind"] != "h5ad").any():
        raise ValueError("registry contains unsupported source_kind")
    symbols = _unique_strings(requested_symbols, "requested symbols")
    if reader is None:
        import anndata as ad

        reader = ad.read_h5ad
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"refusing to overwrite nonempty q_sc output directory {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, object]] = {}
    for model_id, row in registry.iterrows():
        source_path = Path(row["source_path"])
        final_path = output_dir / f"{model_id}.npz"
        ready = False
        if resume and final_path.is_file():
            try:
                load_q_sc_line(output_dir, str(model_id), symbols)
                ready = True
            except (ValueError, OSError):
                pass
        if not ready:
            adata = reader(source_path)
            if not hasattr(adata, "obs") or "model_id" not in adata.obs.columns:
                raise ValueError(f"{model_id}: source AnnData obs is missing model_id")
            observed_model_ids = set(adata.obs["model_id"].astype(str))
            if observed_model_ids != {str(model_id)}:
                raise ValueError(
                    f"{model_id}: source AnnData model_id values are "
                    f"{sorted(observed_model_ids)}"
                )
            features = compute_q_sc(adata, symbols)
            tmp_path = output_dir / f".{model_id}-{uuid.uuid4().hex}.npz"
            np.savez(
                tmp_path,
                model_id=np.asarray(str(model_id)),
                gene_symbols=np.asarray(symbols),
                values=features.values,
                available=features.available,
            )
            os.replace(tmp_path, final_path)
        entries[str(model_id)] = {
            "path": final_path.name,
            "source_path": str(source_path),
        }
    manifest = {
        "schema_version": "exp13-q-sc-v1",
        "gene_symbols": list(symbols),
        "line_count": len(entries),
        "lines": entries,
    }
    tmp_manifest = output_dir / f".manifest-{uuid.uuid4().hex}.json"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_manifest, output_dir / "manifest.json")
    return manifest
