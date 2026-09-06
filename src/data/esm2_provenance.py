"""Authenticated provenance for generated ESM-2 embedding tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "esm2-embedding-provenance-v1"
UNIPROT_MAPPING_SCHEMA = "esm2-uniprot-mapping-v1"
ISOFORM_POLICY = "canonical_reviewed_top_hit"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ordered_strings(values: Sequence[str]) -> str:
    """Hash a string sequence with unambiguous length framing."""
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def sequence_sha256_by_symbol(
    symbols: Sequence[str], sequences: Mapping[str, str]
) -> dict[str, str | None]:
    """Bind each requested symbol to the exact sequence used, or to absence."""
    return {
        symbol: (
            hashlib.sha256(sequences[symbol].encode("utf-8")).hexdigest()
            if symbol in sequences
            else None
        )
        for symbol in symbols
    }


def build_embedding_artifact_record(path: str | Path) -> dict[str, object]:
    """Describe and hash the exact arrays stored in an ESM-2 NPZ."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=True) as payload:
            if set(payload.files) != {"symbols", "vectors", "resolved"}:
                raise ValueError("ESM2 NPZ must contain symbols, vectors, and resolved")
            raw_symbols = payload["symbols"]
            vectors = payload["vectors"]
            resolved = payload["resolved"]
    except Exception as exc:
        raise ValueError(f"invalid ESM2 NPZ {path}: {exc}") from exc
    if raw_symbols.ndim != 1:
        raise ValueError("ESM2 symbols must be one-dimensional")
    symbols = tuple(str(value) for value in raw_symbols.tolist())
    if any(not symbol or symbol != symbol.upper() for symbol in symbols):
        raise ValueError("ESM2 symbols must be nonempty uppercase strings")
    if len(set(symbols)) != len(symbols):
        raise ValueError("ESM2 symbols must be unique")
    if vectors.ndim != 2 or vectors.shape[0] != len(symbols):
        raise ValueError("ESM2 vectors must be rows aligned to symbols")
    if resolved.dtype != np.dtype(bool) or resolved.shape != (len(symbols),):
        raise ValueError("ESM2 resolved must be a bool vector aligned to symbols")
    if not np.isfinite(vectors).all():
        raise ValueError("ESM2 vectors must be finite")
    resolved_symbols = [
        symbol
        for symbol, is_resolved in zip(symbols, resolved, strict=True)
        if bool(is_resolved)
    ]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "symbols": list(symbols),
        "symbol_order_sha256": sha256_ordered_strings(symbols),
        "row_count": len(symbols),
        "vector_width": int(vectors.shape[1]),
        "vectors_dtype": vectors.dtype.str,
        "resolved_dtype": resolved.dtype.str,
        "resolved_count": int(resolved.sum()),
        "resolved_symbols": resolved_symbols,
        "resolved_mask_sha256": hashlib.sha256(
            np.ascontiguousarray(resolved).tobytes()
        ).hexdigest(),
    }


def authenticate_uniprot_mapping(
    sequence_source: Mapping[str, object],
    embedding_artifact: Mapping[str, object],
    mapping_json_path: str | Path,
    mapping_csv_path: str | Path,
) -> dict[str, Any]:
    """Authenticate exact UniProt JSON/CSV identities for an embedding artifact."""
    mapping_json_path = Path(mapping_json_path)
    mapping_csv_path = Path(mapping_csv_path)
    if sequence_source.get("uniprot_mapping_json_sha256") != sha256_file(
        mapping_json_path
    ) or sequence_source.get("uniprot_mapping_csv_sha256") != sha256_file(
        mapping_csv_path
    ):
        raise ValueError("ESM2 UniProt mapping artifact SHA256 mismatch")
    try:
        mapping_payload = json.loads(mapping_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid ESM2 UniProt mapping JSON: {exc}") from exc
    if (
        not isinstance(mapping_payload, dict)
        or mapping_payload.get("schema_version") != UNIPROT_MAPPING_SCHEMA
        or not isinstance(mapping_payload.get("records"), list)
    ):
        raise ValueError("unsupported ESM2 UniProt mapping schema")
    symbols = embedding_artifact.get("symbols")
    resolved = embedding_artifact.get("resolved_symbols")
    if (
        not isinstance(symbols, list)
        or any(not isinstance(symbol, str) for symbol in symbols)
        or not isinstance(resolved, list)
        or any(not isinstance(symbol, str) for symbol in resolved)
        or not set(resolved).issubset(symbols)
    ):
        raise ValueError("invalid ESM2 embedding symbol provenance")
    mapping_records = mapping_payload["records"]
    if (
        any(not isinstance(record, dict) for record in mapping_records)
        or [record.get("gene_symbol") for record in mapping_records] != symbols
    ):
        raise ValueError("ESM2 UniProt mapping membership/order mismatch")
    try:
        with mapping_csv_path.open(newline="", encoding="utf-8") as handle:
            csv_records = list(csv.DictReader(handle))
    except Exception as exc:
        raise ValueError(f"invalid ESM2 UniProt mapping CSV: {exc}") from exc
    if [record.get("gene_symbol") for record in csv_records] != symbols:
        raise ValueError("ESM2 UniProt CSV membership/order mismatch")
    sequence_hashes = sequence_source.get("sequence_sha256_by_symbol")
    if not isinstance(sequence_hashes, dict) or set(sequence_hashes) != set(symbols):
        raise ValueError("ESM2 provenance sequence mapping membership mismatch")
    for symbol, digest in sequence_hashes.items():
        if digest is not None and not _is_sha256(digest):
            raise ValueError(f"invalid sequence SHA-256 for {symbol}")
    resolved_symbols = set(resolved)
    inconsistent = [
        symbol
        for symbol, digest in sequence_hashes.items()
        if (digest is not None) != (symbol in resolved_symbols)
    ]
    if inconsistent:
        raise ValueError(
            "ESM2 provenance sequence mapping disagrees with resolved mask: "
            f"{inconsistent[:10]}"
        )
    for record, csv_record in zip(mapping_records, csv_records, strict=True):
        symbol = record.get("gene_symbol")
        is_resolved = symbol in resolved_symbols
        expected_identity = (
            record.get("primary_accession"),
            record.get("entry_id"),
            record.get("isoform_identifier"),
            record.get("isoform_policy"),
            record.get("sequence_sha256"),
        )
        if record.get("resolved") is not is_resolved:
            raise ValueError(f"ESM2 UniProt resolved flag mismatch for {symbol}")
        if is_resolved:
            if not all(isinstance(value, str) and value for value in expected_identity):
                raise ValueError(f"resolved symbol lacks UniProt identity: {symbol}")
            if (
                record.get("isoform_policy") != ISOFORM_POLICY
                or record.get("isoform_identifier") != record.get("primary_accession")
                or record.get("sequence_sha256") != sequence_hashes[symbol]
            ):
                raise ValueError(f"UniProt identity/sequence mismatch for {symbol}")
        elif any(value is not None for value in expected_identity):
            raise ValueError(
                f"unresolved symbol has asserted UniProt identity: {symbol}"
            )
        expected_csv = {
            key: "" if value is None else str(value) for key, value in record.items()
        }
        if csv_record != expected_csv:
            raise ValueError(f"UniProt JSON/CSV mapping mismatch for {symbol}")
    return mapping_payload


def load_and_authenticate_esm2_provenance(
    manifest_path: str | Path,
    npz_path: str | Path,
    *,
    expected_width: int | None = None,
    mapping_json_path: str | Path | None = None,
    mapping_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a provenance sidecar and authenticate it against the exact NPZ."""
    manifest_path = Path(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"invalid ESM2 provenance manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported ESM2 provenance schema")
    artifact = payload.get("embedding_artifact")
    if not isinstance(artifact, dict):
        raise ValueError("ESM2 provenance is missing embedding_artifact")
    observed = build_embedding_artifact_record(npz_path)
    for key in (
        "sha256",
        "symbols",
        "symbol_order_sha256",
        "row_count",
        "vector_width",
        "vectors_dtype",
        "resolved_dtype",
        "resolved_count",
        "resolved_symbols",
        "resolved_mask_sha256",
    ):
        if artifact.get(key) != observed[key]:
            raise ValueError(f"ESM2 provenance {key} mismatch")
    if expected_width is not None and observed["vector_width"] != expected_width:
        raise ValueError(
            f"ESM2 vector width {observed['vector_width']} != {expected_width}"
        )
    source = payload.get("sequence_source")
    if not isinstance(source, dict):
        raise ValueError("ESM2 provenance is missing sequence_source")
    for field in ("benchmark_csv_sha256", "sequence_cache_sha256"):
        if not _is_sha256(source.get(field)):
            raise ValueError(f"ESM2 provenance has invalid {field}")
    recorded_json_path = source.get("uniprot_mapping_json_path")
    recorded_csv_path = source.get("uniprot_mapping_csv_path")
    if mapping_json_path is None and not isinstance(recorded_json_path, str):
        raise ValueError("ESM2 provenance lacks UniProt mapping JSON path")
    if mapping_csv_path is None and not isinstance(recorded_csv_path, str):
        raise ValueError("ESM2 provenance lacks UniProt mapping CSV path")
    authenticate_uniprot_mapping(
        source,
        observed,
        mapping_json_path or recorded_json_path,
        mapping_csv_path or recorded_csv_path,
    )
    model = payload.get("loaded_model")
    tokenizer = payload.get("tokenizer")
    for record, label, field in (
        (model, "loaded_model", "state_sha256"),
        (model, "loaded_model", "config_sha256"),
        (tokenizer, "tokenizer", "vocabulary_config_sha256"),
    ):
        if not isinstance(record, dict):
            raise ValueError(f"ESM2 provenance is missing {label}")
        digest = record.get(field)
        if not _is_sha256(digest):
            raise ValueError(f"ESM2 provenance has invalid {label} hash")
    requested_model = payload.get("requested_model_id")
    if not isinstance(requested_model, str) or not requested_model:
        raise ValueError("ESM2 provenance requested_model_id is invalid")
    return payload
