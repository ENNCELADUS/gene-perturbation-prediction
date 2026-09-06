"""Fetch UniProt sequences for a supplied gene universe and embed with ESM2.

Run with ``python -m src.data.prepare.precompute_esm2_embeddings``.
Use explicit benchmark, sequence-cache and output paths; see ``--help``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple
import uuid

import numpy as np
import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer

from src.data.embeddings import Esm2EmbeddingTable, require_complete_esm_coverage
from src.data.esm2_provenance import (
    SCHEMA_VERSION as PROVENANCE_SCHEMA_VERSION,
    build_embedding_artifact_record,
    sequence_sha256_by_symbol,
    sha256_file,
)

logger = logging.getLogger("precompute_esm2")
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
SEQUENCE_CACHE_SCHEMA = "uniprot-sequence-cache-v2"
UNIPROT_MAPPING_SCHEMA = "esm2-uniprot-mapping-v1"
ISOFORM_POLICY = "canonical_reviewed_top_hit"
REVIEWED_ENTRY_TYPE = "UniProtKB reviewed (Swiss-Prot)"


class UniProtSequenceRecord(NamedTuple):
    primary_accession: str
    entry_id: str
    isoform_identifier: str
    isoform_policy: str
    sequence: str

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    return str(value)


def hash_model_state(model: object) -> str:
    """Hash sorted parameter/buffer names, dtypes, shapes, and exact bytes."""
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def hash_tokenizer_vocabulary_config(tokenizer: object) -> str:
    """Hash the loaded tokenizer vocabulary and effective configuration."""
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocabulary": tokenizer.get_vocab(),
        "init_kwargs": getattr(tokenizer, "init_kwargs", {}),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
    }
    encoded = json.dumps(
        _canonical_json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_model_config(model: object) -> str:
    """Hash the effective configuration of the loaded model instance."""
    config = model.config.to_dict()
    encoded = json.dumps(
        _canonical_json_value(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def symbols_from_csv(csv_path: Path, symbol_columns: tuple[str, ...]) -> list[str]:
    """Return sorted unique upper-case symbols from selected CSV columns."""
    frame = pd.read_csv(csv_path, usecols=list(symbol_columns))
    symbols: set[str] = set()
    for column in symbol_columns:
        symbols.update(frame[column].dropna().astype(str).str.upper())
    return sorted(symbols)


def universe_symbols(benchmark_csv: Path) -> list[str]:
    """Return sorted upper-case gene symbols from both columns of the benchmark CSV.

    Args:
        benchmark_csv: Path to the SL benchmark CSV with ``gene_a_symbol``
            and ``gene_b_symbol`` columns.

    Returns:
        Sorted list of unique upper-case gene symbols.
    """
    return symbols_from_csv(benchmark_csv, ("gene_a_symbol", "gene_b_symbol"))


def fetch_sequence(
    symbol: str, identifier: str | None = None
) -> UniProtSequenceRecord | None:
    """Return the reviewed canonical UniProt record for a gene symbol.

    Queries UniProt REST for the top reviewed human hit. On any network
    error, logs a warning and returns ``None``.

    Args:
        symbol: Upper-case HGNC gene symbol.

    Returns:
        Amino-acid sequence string, or ``None`` if not found or on error.
    """
    gene_query = f"xref:GeneID-{identifier}" if identifier else f"gene_exact:{symbol}"
    query = f"({gene_query}) AND (organism_id:9606) AND (reviewed:true)"
    params = urllib.parse.urlencode({"query": query, "format": "json", "size": 1})
    url = f"{UNIPROT_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - network best-effort, logged
        logger.warning("fetch failed for %s: %s", symbol, exc)
        return None
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        return None
    hit = results[0]
    if not isinstance(hit, dict):
        return None
    genes = hit.get("genes")
    primary_gene = None
    if isinstance(genes, list) and genes and isinstance(genes[0], dict):
        gene_name = genes[0].get("geneName")
        if isinstance(gene_name, dict):
            primary_gene = gene_name.get("value")
    if primary_gene != symbol or hit.get("entryType") != REVIEWED_ENTRY_TYPE:
        logger.warning("UniProt identity mismatch for %s", symbol)
        return None
    if identifier is not None:
        cross_references = hit.get("uniProtKBCrossReferences")
        has_gene_id = isinstance(cross_references, list) and any(
            isinstance(reference, dict)
            and reference.get("database") == "GeneID"
            and str(reference.get("id")) == identifier
            for reference in cross_references
        )
        if not has_gene_id:
            logger.warning("UniProt GeneID mismatch for %s", symbol)
            return None
    accession = hit.get("primaryAccession")
    entry_id = hit.get("uniProtkbId")
    sequence_record = hit.get("sequence")
    sequence = (
        sequence_record.get("value") if isinstance(sequence_record, dict) else None
    )
    if not all(
        isinstance(value, str) and value for value in (accession, entry_id, sequence)
    ):
        logger.warning("incomplete UniProt identity for %s", symbol)
        return None
    return UniProtSequenceRecord(
        primary_accession=accession,
        entry_id=entry_id,
        isoform_identifier=accession,
        isoform_policy=ISOFORM_POLICY,
        sequence=sequence,
    )


def load_or_fetch_sequences(
    symbols: list[str],
    cache: Path,
    identifiers: dict[str, str] | None = None,
    *,
    refetch_legacy_cache: bool = False,
    read_only: bool = False,
) -> dict[str, UniProtSequenceRecord]:
    """Load cached symbol→sequence map; fetch missing symbols from UniProt.

    Writes incrementally to ``cache`` every 100 new symbols.

    Args:
        symbols: Upper-case gene symbols to resolve.
        cache: Path to the JSON cache file (created if absent).

    Returns:
        Mapping from upper-case symbol to amino-acid sequence.
    """
    records: dict[str, UniProtSequenceRecord] = {}
    if cache.exists():
        try:
            payload = json.loads(cache.read_text())
        except json.JSONDecodeError:
            logger.warning("corrupt JSON cache at %s; starting fresh", cache)
            payload = None
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == SEQUENCE_CACHE_SCHEMA
        ):
            raw_records = payload.get("records")
            if not isinstance(raw_records, dict):
                raise ValueError("UniProt sequence cache records must be a mapping")
            for symbol, raw in raw_records.items():
                if not isinstance(raw, dict):
                    raise ValueError(f"invalid UniProt cache record for {symbol}")
                identity_values = (
                    raw.get("primary_accession"),
                    raw.get("entry_id"),
                    raw.get("isoform_identifier"),
                    raw.get("isoform_policy"),
                    raw.get("sequence"),
                )
                if not all(
                    isinstance(value, str) and value for value in identity_values
                ):
                    raise ValueError(f"incomplete UniProt cache record for {symbol}")
                record = UniProtSequenceRecord(
                    primary_accession=raw.get("primary_accession"),
                    entry_id=raw.get("entry_id"),
                    isoform_identifier=raw.get("isoform_identifier"),
                    isoform_policy=raw.get("isoform_policy"),
                    sequence=raw.get("sequence"),
                )
                if (
                    record.isoform_policy != ISOFORM_POLICY
                    or record.isoform_identifier != record.primary_accession
                ):
                    raise ValueError(f"unsupported UniProt isoform record for {symbol}")
                if record.sequence_sha256 != raw.get("sequence_sha256"):
                    raise ValueError(f"cached sequence SHA-256 mismatch for {symbol}")
                records[str(symbol)] = record
        elif isinstance(payload, dict) and all(
            isinstance(value, str) for value in payload.values()
        ):
            if not refetch_legacy_cache:
                raise ValueError(
                    "legacy sequence-only cache lacks UniProt accessions; rerun with "
                    "--refetch-legacy-cache to refetch authoritative records"
                )
            logger.warning("discarding legacy sequence-only cache and refetching")
        elif payload is not None:
            raise ValueError("unsupported UniProt sequence cache schema")

    missing_symbols = [symbol for symbol in symbols if symbol not in records]
    if read_only:
        if missing_symbols:
            raise ValueError(
                "read-only UniProt sequence cache is incomplete: "
                f"{len(missing_symbols)} missing symbols"
            )
        return records

    def write_cache() -> None:
        _atomic_write_text(
            cache,
            json.dumps(
                {
                    "schema_version": SEQUENCE_CACHE_SCHEMA,
                    "records": {
                        symbol: {
                            **record._asdict(),
                            "sequence_sha256": record.sequence_sha256,
                        }
                        for symbol, record in sorted(records.items())
                    },
                },
                sort_keys=True,
            ),
        )

    for i, symbol in enumerate(symbols):
        if symbol in records:
            continue
        identifier = identifiers.get(symbol) if identifiers is not None else None
        record = (
            fetch_sequence(symbol, identifier)
            if identifier is not None
            else fetch_sequence(symbol)
        )
        if record:
            records[symbol] = record
            if len(records) % 100 == 0:
                write_cache()
                logger.info("resolved %d/%d sequences", len(records), len(symbols))
        time.sleep(0.1)  # be polite to UniProt
    write_cache()
    return records


def identifiers_from_csv(
    csv_path: Path, symbol_column: str, identifier_column: str
) -> dict[str, str]:
    """Map upper-case symbols to integer identifiers from one CSV."""
    frame = pd.read_csv(csv_path, usecols=[symbol_column, identifier_column])
    identifiers: dict[str, str] = {}
    for symbol, identifier in frame.itertuples(index=False, name=None):
        if pd.isna(symbol) or pd.isna(identifier):
            continue
        key = str(symbol).upper()
        value = str(int(identifier))
        existing = identifiers.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"conflicting identifiers for {key}: {existing}, {value}")
        identifiers[key] = value
    return identifiers


def truncate_sequence(seq: str, symbol: str, max_len: int = 1022) -> str:
    """Truncate a protein sequence to ``max_len`` tokens; warns when truncation occurs.

    Args:
        seq: Amino-acid sequence string.
        symbol: Gene symbol for log messages.
        max_len: Maximum allowed sequence length (default: 1022, ESM2 limit).

    Returns:
        The original sequence if within ``max_len``, otherwise the first
        ``max_len`` characters.
    """
    if len(seq) > max_len:
        logger.warning(
            "truncating %s from %d to %d residues for ESM2 input",
            symbol,
            len(seq),
            max_len,
        )
        return seq[:max_len]
    return seq


def check_resolution(resolved: np.ndarray, n_symbols: int) -> None:
    """Validate that a sufficient fraction of symbols were resolved to sequences.

    Args:
        resolved: Boolean array indicating which symbols have embeddings.
        n_symbols: Total number of symbols in the universe.

    Raises:
        RuntimeError: If no symbols resolved (all-zero ``resolved`` array).
    """
    n_resolved = int(resolved.sum())
    if n_resolved == 0:
        raise RuntimeError("no sequences resolved; aborting embedding write")
    frac = n_resolved / n_symbols if n_symbols > 0 else 0.0
    if frac < 0.5:
        logger.warning(
            "low resolution: only %d/%d symbols resolved (%.1f%%)",
            n_resolved,
            n_symbols,
            frac * 100,
        )


def require_complete_asset_coverage(
    symbols: list[str], vectors: np.ndarray, resolved: np.ndarray
) -> None:
    """Require complete canonical coverage before an ESM-2 asset is written."""
    table = Esm2EmbeddingTable(
        dim=int(vectors.shape[1]),
        vectors_by_symbol={
            symbol: vector
            for symbol, vector, is_resolved in zip(
                symbols, vectors, resolved, strict=True
            )
            if bool(is_resolved)
        },
    )
    require_complete_esm_coverage(symbols, table)


def mean_pool_residues(
    hidden: np.ndarray, special_tokens_mask: np.ndarray
) -> np.ndarray:
    """Mean-pool token embeddings over residues, excluding special tokens.

    ESM2 prepends a BOS (``<cls>``) and appends an EOS (``<eos>``) token; padding
    tokens may also be present. The spec calls for a residue mean, so special
    tokens (``special_tokens_mask == 1``) are excluded from the average.

    Args:
        hidden: Token embeddings, shape ``(n_tokens, dim)``.
        special_tokens_mask: Per-token mask, shape ``(n_tokens,)``; ``1`` marks a
            special (non-residue) token to exclude.

    Returns:
        Residue-mean embedding, shape ``(dim,)``. If every token is special
        (degenerate input), falls back to a full mean over all tokens to avoid
        a NaN.
    """
    hidden = np.asarray(hidden, dtype=np.float32)
    keep = np.asarray(special_tokens_mask) == 0
    if not keep.any():
        return hidden.mean(axis=0)
    return hidden[keep].mean(axis=0)


def embed_sequences(
    symbols: list[str],
    seqs: dict[str, str],
    model_name: str,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Embed resolved sequences with ESM2; unresolved rows stay zero.

    Args:
        symbols: Upper-case gene symbols in universe order.
        seqs: Mapping from symbol to amino-acid sequence.
        model_name: HuggingFace model ID, e.g. ``"facebook/esm2_t33_650M_UR50D"``.
        cache_dir: Optional Hugging Face cache directory for model/tokenizer files.
        local_files_only: If True, require the model/tokenizer to already be in
            the local Hugging Face cache.

    Returns:
        ``(vectors, resolved, runtime_identity)`` where ``vectors`` has shape
        ``(n_gene, hidden_size)``, ``resolved`` is a boolean array, and the
        identity record hashes the actual loaded model and tokenizer.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from_pretrained_kwargs = {
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
    }
    from_pretrained_kwargs = {
        key: value for key, value in from_pretrained_kwargs.items() if value is not None
    }
    tokenizer = EsmTokenizer.from_pretrained(model_name, **from_pretrained_kwargs)
    model = EsmModel.from_pretrained(model_name, **from_pretrained_kwargs)
    runtime_identity = {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_state_sha256": hash_model_state(model),
        "model_config_sha256": hash_model_config(model),
        "tokenizer_class": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        ),
        "tokenizer_vocabulary_config_sha256": (
            hash_tokenizer_vocabulary_config(tokenizer)
        ),
    }
    model = model.to(device).eval()
    dim = model.config.hidden_size
    vectors = np.zeros((len(symbols), dim), dtype=np.float32)
    resolved = np.zeros(len(symbols), dtype=bool)
    with torch.no_grad():
        for row, symbol in enumerate(symbols):
            seq = seqs.get(symbol)
            if not seq:
                continue
            toks = tokenizer(
                truncate_sequence(seq, symbol),
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special = toks.pop("special_tokens_mask")[0].cpu().numpy()
            toks = toks.to(device)
            out = model(**toks).last_hidden_state[0]  # (L, dim)
            vectors[row] = mean_pool_residues(out.cpu().numpy(), special)
            resolved[row] = True
            if row % 200 == 0:
                logger.info("embedded %d/%d", row, len(symbols))
    return vectors, resolved, runtime_identity


def write_embedding_with_provenance(
    output: Path,
    provenance_output: Path,
    *,
    symbols: list[str],
    vectors: np.ndarray,
    resolved: np.ndarray,
    sequences: dict[str, str],
    uniprot_records: dict[str, UniProtSequenceRecord],
    sequence_cache: Path,
    mapping_json_output: Path,
    mapping_csv_output: Path,
    benchmark_csv: Path,
    symbol_columns: tuple[str, ...],
    requested_model_id: str,
    runtime_identity: dict[str, str],
) -> dict[str, object]:
    """Atomically publish an embedding table and its authenticated sidecar."""
    outputs = (output, provenance_output, mapping_json_output, mapping_csv_output)
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("embedding, provenance, and mapping outputs must be distinct")
    if resolved.shape != (len(symbols),):
        raise ValueError("resolved mask must align with symbols")
    resolved_symbols = {
        symbol
        for symbol, is_resolved in zip(symbols, resolved, strict=True)
        if bool(is_resolved)
    }
    if set(sequences) != resolved_symbols or set(uniprot_records) != resolved_symbols:
        raise ValueError(
            "resolved mask, sequences, and UniProt records must have exact membership"
        )
    for symbol in resolved_symbols:
        if sequences[symbol] != uniprot_records[symbol].sequence:
            raise ValueError(f"sequence differs from UniProt record for {symbol}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.npz")
    try:
        np.savez(
            temporary,
            symbols=np.asarray(symbols, dtype=object),
            vectors=vectors,
            resolved=resolved,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    artifact = build_embedding_artifact_record(output)
    mapping_records = []
    for symbol in symbols:
        record = uniprot_records.get(symbol)
        if symbol in resolved_symbols and record is None:
            raise ValueError(f"resolved symbol lacks UniProt identity: {symbol}")
        mapping_records.append(
            {
                "gene_symbol": symbol,
                "resolved": symbol in resolved_symbols,
                "primary_accession": (
                    record.primary_accession if record is not None else None
                ),
                "entry_id": record.entry_id if record is not None else None,
                "isoform_identifier": (
                    record.isoform_identifier if record is not None else None
                ),
                "isoform_policy": record.isoform_policy if record is not None else None,
                "sequence_sha256": (
                    record.sequence_sha256 if record is not None else None
                ),
            }
        )
    mapping_payload = {
        "schema_version": UNIPROT_MAPPING_SCHEMA,
        "records": mapping_records,
    }
    _atomic_write_text(
        mapping_json_output,
        json.dumps(mapping_payload, indent=2, sort_keys=True) + "\n",
    )
    csv_buffer = io.StringIO(newline="")
    fieldnames = list(mapping_records[0]) if mapping_records else ["gene_symbol"]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(mapping_records)
    _atomic_write_text(mapping_csv_output, csv_buffer.getvalue())
    manifest: dict[str, object] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "requested_model_id": requested_model_id,
        "loaded_model": {
            "class": runtime_identity["model_class"],
            "state_sha256": runtime_identity["model_state_sha256"],
            "config_sha256": runtime_identity["model_config_sha256"],
        },
        "tokenizer": {
            "class": runtime_identity["tokenizer_class"],
            "vocabulary_config_sha256": runtime_identity[
                "tokenizer_vocabulary_config_sha256"
            ],
        },
        "sequence_source": {
            "benchmark_csv_path": str(benchmark_csv),
            "benchmark_csv_sha256": sha256_file(benchmark_csv),
            "symbol_columns": list(symbol_columns),
            "sequence_cache_path": str(sequence_cache),
            "sequence_cache_sha256": sha256_file(sequence_cache),
            "sequence_sha256_by_symbol": sequence_sha256_by_symbol(symbols, sequences),
            "uniprot_mapping_json_path": str(mapping_json_output),
            "uniprot_mapping_json_sha256": sha256_file(mapping_json_output),
            "uniprot_mapping_csv_path": str(mapping_csv_output),
            "uniprot_mapping_csv_sha256": sha256_file(mapping_csv_output),
        },
        "embedding_artifact": artifact,
    }
    _atomic_write_text(
        provenance_output, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    """CLI entry point: fetch sequences and embed with ESM2."""
    parser = argparse.ArgumentParser(
        description="Fetch UniProt sequences and embed with ESM2."
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        required=True,
        help="SL benchmark CSV with gene_a_symbol and gene_b_symbol columns.",
    )
    parser.add_argument(
        "--symbol-column",
        action="append",
        default=None,
        help=(
            "CSV symbol column; repeat for multiple columns. Defaults to the "
            "gene_a_symbol and gene_b_symbol columns."
        ),
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Optional integer identifier column used for sequence lookup.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .npz path.",
    )
    parser.add_argument(
        "--provenance-out",
        type=Path,
        default=None,
        help=("Atomic provenance sidecar. Defaults to <out>.provenance.json."),
    )
    parser.add_argument(
        "--seq-cache",
        type=Path,
        required=True,
        help="JSON cache for symbol→sequence (incremental, reuse across runs).",
    )
    parser.add_argument(
        "--refetch-legacy-cache",
        action="store_true",
        help="Discard a legacy sequence-only cache and refetch UniProt identities.",
    )
    parser.add_argument(
        "--sequence-cache-read-only",
        action="store_true",
        help="Require a complete sequence cache and never fetch or rewrite it.",
    )
    parser.add_argument("--mapping-json-out", type=Path, default=None)
    parser.add_argument("--mapping-csv-out", type=Path, default=None)
    parser.add_argument(
        "--model",
        default="facebook/esm2_t33_650M_UR50D",
        help="HuggingFace ESM2 model ID.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory for model/tokenizer downloads.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only already-cached Hugging Face files; do not download.",
    )
    parser.add_argument(
        "--require-complete-coverage",
        action="store_true",
        help=(
            "Fail before writing unless every requested symbol has an embedding. "
            "Required for canonical exp05 assets; optional for exploratory assets."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    symbol_columns = (
        tuple(args.symbol_column)
        if args.symbol_column
        else ("gene_a_symbol", "gene_b_symbol")
    )
    symbols = symbols_from_csv(args.benchmark_csv, symbol_columns)
    logger.info("universe size: %d genes", len(symbols))
    identifiers = None
    if args.id_column is not None:
        if len(symbol_columns) != 1:
            raise ValueError("--id-column requires exactly one --symbol-column")
        identifiers = identifiers_from_csv(
            args.benchmark_csv, symbol_columns[0], args.id_column
        )
    records = (
        load_or_fetch_sequences(
            symbols,
            args.seq_cache,
            identifiers,
            refetch_legacy_cache=args.refetch_legacy_cache,
            read_only=args.sequence_cache_read_only,
        )
        if identifiers is not None
        else load_or_fetch_sequences(
            symbols,
            args.seq_cache,
            refetch_legacy_cache=args.refetch_legacy_cache,
            read_only=args.sequence_cache_read_only,
        )
    )
    seqs = {symbol: record.sequence for symbol, record in records.items()}
    vectors, resolved, runtime_identity = embed_sequences(
        symbols,
        seqs,
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if args.require_complete_coverage:
        require_complete_asset_coverage(symbols, vectors, resolved)
    else:
        check_resolution(resolved, n_symbols=len(symbols))
    provenance_out = args.provenance_out or args.out.with_suffix(
        args.out.suffix + ".provenance.json"
    )
    mapping_json_out = args.mapping_json_out or args.out.with_suffix(
        args.out.suffix + ".uniprot_mapping.json"
    )
    mapping_csv_out = args.mapping_csv_out or args.out.with_suffix(
        args.out.suffix + ".uniprot_mapping.csv"
    )
    write_embedding_with_provenance(
        args.out,
        provenance_out,
        symbols=symbols,
        vectors=vectors,
        resolved=resolved,
        sequences=seqs,
        uniprot_records=records,
        sequence_cache=args.seq_cache,
        mapping_json_output=mapping_json_out,
        mapping_csv_output=mapping_csv_out,
        benchmark_csv=args.benchmark_csv,
        symbol_columns=symbol_columns,
        requested_model_id=args.model,
        runtime_identity=runtime_identity,
    )
    logger.info(
        "wrote %s and %s (%d resolved / %d)",
        args.out,
        provenance_out,
        int(resolved.sum()),
        len(symbols),
    )


if __name__ == "__main__":
    main()
