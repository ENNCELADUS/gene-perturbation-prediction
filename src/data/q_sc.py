"""data / q sc."""

from __future__ import annotations

import hashlib
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
                "source_sha256",
            }
            if set(shard.files) != expected_keys:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_one_shard(
    path: Path, model_id: str, symbols: tuple[str, ...], source_sha256: str
) -> list[str]:
    problems: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as shard:
            keys = set(shard.files)
            expected_keys = {
                "model_id",
                "gene_symbols",
                "values",
                "available",
                "source_sha256",
            }
            if keys != expected_keys:
                return [
                    f"{model_id}: shard keys {sorted(keys)} != {sorted(expected_keys)}"
                ]
            if str(shard["model_id"].item()) != model_id:
                problems.append(f"{model_id}: embedded model_id mismatch")
            observed_symbols = tuple(shard["gene_symbols"].astype(str))
            if observed_symbols != symbols:
                problems.append(f"{model_id}: gene order mismatch")
            values = shard["values"]
            available_raw = shard["available"]
            valid_values_shape = values.shape == (len(symbols), 3)
            valid_available_shape = available_raw.shape == (len(symbols),)
            if not valid_values_shape:
                problems.append(f"{model_id}: values shape mismatch")
            if not valid_available_shape:
                problems.append(f"{model_id}: availability shape mismatch")
            if available_raw.dtype != np.dtype(bool):
                problems.append(f"{model_id}: availability dtype is not bool")
            if str(shard["source_sha256"].item()) != source_sha256:
                problems.append(f"{model_id}: source SHA-256 mismatch")
            if (
                valid_values_shape
                and valid_available_shape
                and np.issubdtype(values.dtype, np.number)
            ):
                available = available_raw.astype(bool)
                unavailable = ~available
                if unavailable.any() and not np.isnan(values[unavailable]).all():
                    problems.append(f"{model_id}: unavailable genes are not NaN")
                if available.any():
                    present = values[available]
                    invalid = (
                        not np.isfinite(present).all()
                        or np.any(present[:, 0] < 0)
                        or np.any((present[:, 1] < 0) | (present[:, 1] > 1))
                        or np.any(present[:, 2] < 0)
                    )
                    if invalid:
                        problems.append(
                            f"{model_id}: available q_sc values are invalid"
                        )
            elif valid_values_shape:
                problems.append(f"{model_id}: values dtype is not numeric")
    except (OSError, ValueError, EOFError, IndexError, TypeError) as exc:
        problems.append(f"{model_id}: unreadable shard: {exc}")
    return problems


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
    prior_lines: dict[str, object] = {}
    prior_manifest_path = output_dir / "manifest.json"
    if resume and prior_manifest_path.is_file():
        try:
            prior_manifest = json.loads(prior_manifest_path.read_text())
            if (
                isinstance(prior_manifest, dict)
                and prior_manifest.get("gene_symbols") == list(symbols)
                and isinstance(prior_manifest.get("lines"), dict)
            ):
                prior_lines = prior_manifest["lines"]
        except json.JSONDecodeError:
            pass
    entries: dict[str, dict[str, object]] = {}
    for model_id, row in registry.iterrows():
        source_path = Path(row["source_path"])
        source_sha256 = _sha256(source_path)
        final_path = output_dir / f"{model_id}.npz"
        prior_entry = prior_lines.get(str(model_id), {})
        can_resume = (
            resume
            and final_path.is_file()
            and not _verify_one_shard(final_path, str(model_id), symbols, source_sha256)
            and isinstance(prior_entry, dict)
            and prior_entry.get("sha256") == _sha256(final_path)
        )
        if can_resume:
            pass
        else:
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
                source_sha256=np.asarray(source_sha256),
            )
            os.replace(tmp_path, final_path)
        entries[str(model_id)] = {
            "path": final_path.name,
            "sha256": _sha256(final_path),
            "source_sha256": source_sha256,
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


def verify_q_sc_shards(
    registry: pd.DataFrame, output_dir: Path, requested_symbols: Sequence[str]
) -> dict[str, object]:
    """Unrestricted full-directory verification of every expected shard."""
    symbols = _unique_strings(requested_symbols, "requested symbols")
    output_dir = Path(output_dir)
    expected = {f"{model_id}.npz" for model_id in registry.index}
    observed = {path.name for path in output_dir.glob("*.npz")}
    problems = [f"missing shard: {name}" for name in sorted(expected - observed)]
    problems.extend(f"extra shard: {name}" for name in sorted(observed - expected))
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"manifest is missing or unreadable: {exc}")
    if not isinstance(manifest, dict):
        problems.append("manifest root is not an object")
        manifest = {}
    if manifest.get("schema_version") != "exp13-q-sc-v1":
        problems.append("manifest schema_version mismatch")
    if manifest.get("line_count") != len(expected):
        problems.append("manifest line_count mismatch")
    manifest_lines = manifest.get("lines")
    if not isinstance(manifest_lines, dict):
        problems.append("manifest lines metadata is missing")
        manifest_lines = {}
    if set(manifest_lines) != set(registry.index.astype(str)):
        problems.append("manifest line membership mismatch")
    if manifest.get("gene_symbols") != list(symbols):
        problems.append("manifest gene order mismatch")
    for model_id, row in registry.iterrows():
        path = output_dir / f"{model_id}.npz"
        if path.is_file():
            problems.extend(
                _verify_one_shard(
                    path, str(model_id), symbols, _sha256(Path(row["source_path"]))
                )
            )
            entry = manifest_lines.get(str(model_id), {})
            if not isinstance(entry, dict) or entry.get("sha256") != _sha256(path):
                problems.append(f"{model_id}: shard SHA-256 mismatch")
            if not isinstance(entry, dict) or entry.get("path") != path.name:
                problems.append(f"{model_id}: manifest shard path mismatch")
            source_sha256 = _sha256(Path(row["source_path"]))
            if (
                not isinstance(entry, dict)
                or entry.get("source_sha256") != source_sha256
            ):
                problems.append(f"{model_id}: manifest source SHA-256 mismatch")
    return {
        "status": "passed" if not problems else "failed",
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "lines_expected": len(expected),
        "lines_present": len(observed & expected),
        "shard_sha256": {
            str(model_id): str(entry["sha256"])
            for model_id, entry in sorted(manifest_lines.items())
            if isinstance(entry, dict) and "sha256" in entry
        },
        "discrepancies": problems,
    }
