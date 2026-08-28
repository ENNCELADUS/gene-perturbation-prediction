#!/usr/bin/env python3
"""Build an authenticated Exp13 UniProt cache from local reviewed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence
import uuid

import pandas as pd

SCHEMA_VERSION = "uniprot-sequence-cache-v2"
ISOFORM_POLICY = "canonical_reviewed_top_hit"
SOURCE_SCHEMA_VERSION = "exp13-uniprot-offline-sources-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticate_sources(
    source_manifest: Path, artifacts: dict[str, Path]
) -> dict[str, object]:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("offline source manifest must be an object")
    if payload.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("offline source manifest schema mismatch")
    if payload.get("reviewed") is not True or payload.get("taxonomy_id") != 9606:
        raise ValueError("offline source manifest must bind reviewed human UniProt")
    if payload.get("uniprot_query") != "reviewed:true AND organism_id:9606":
        raise ValueError("offline source manifest UniProt query mismatch")
    expected = payload.get("artifacts")
    if not isinstance(expected, dict) or set(expected) != set(artifacts):
        raise ValueError("offline source manifest artifact membership mismatch")
    observed_hashes: dict[str, str] = {}
    for name, path in artifacts.items():
        record = expected.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise ValueError(f"offline source manifest lacks {name} SHA-256")
        observed = _sha256_file(path)
        if observed != record["sha256"]:
            raise ValueError(f"offline source artifact SHA-256 mismatch: {name}")
        observed_hashes[name] = observed
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "manifest_sha256": _sha256_file(source_manifest),
        "reviewed": True,
        "taxonomy_id": 9606,
        "uniprot_query": "reviewed:true AND organism_id:9606",
        "artifact_sha256": observed_hashes,
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_cache(
    *,
    union_csv: Path,
    sequence_tsv: Path,
    identity_tsv: Path,
    legacy_cache: Path,
    aliases_json: Path,
    source_manifest: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {output}")
    symbols = tuple(
        pd.read_csv(union_csv, usecols=["gene_symbol"])["gene_symbol"]
        .astype(str)
        .str.upper()
    )
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("union gene symbols must be non-empty and unique")
    source_provenance = _authenticate_sources(
        source_manifest,
        {
            "union_csv": union_csv,
            "sequence_tsv": sequence_tsv,
            "identity_tsv": identity_tsv,
            "legacy_cache": legacy_cache,
            "aliases_json": aliases_json,
        },
    )

    sequences = pd.read_csv(sequence_tsv, sep="\t")
    required_sequence = {"Entry", "Gene Names (primary)", "Sequence"}
    if not required_sequence.issubset(sequences.columns):
        raise ValueError("reviewed sequence TSV fields mismatch")
    sequences = sequences.dropna(subset=list(required_sequence)).copy()
    sequences["Entry"] = sequences["Entry"].astype(str)
    sequences["Gene Names (primary)"] = (
        sequences["Gene Names (primary)"].astype(str).str.upper()
    )
    if sequences["Entry"].duplicated().any():
        raise ValueError("reviewed sequence TSV has duplicate accessions")

    identities = pd.read_csv(identity_tsv, sep="\t")
    required_identity = {"Entry", "Entry Name", "Gene Names (primary)"}
    if not required_identity.issubset(identities.columns):
        raise ValueError("reviewed identity TSV fields mismatch")
    identities = identities.dropna(subset=["Entry", "Entry Name"]).copy()
    identities["Entry"] = identities["Entry"].astype(str)
    identities["Entry Name"] = identities["Entry Name"].astype(str)
    identities["Gene Names (primary)"] = (
        identities["Gene Names (primary)"].astype(str).str.upper()
    )
    if identities["Entry"].duplicated().any():
        raise ValueError("reviewed identity TSV has duplicate accessions")

    sequence_by_accession = (
        sequences.set_index("Entry")["Sequence"].astype(str).to_dict()
    )
    entry_id_by_accession = (
        identities.set_index("Entry")["Entry Name"].astype(str).to_dict()
    )
    identity_symbol_by_accession = identities.set_index("Entry")[
        "Gene Names (primary)"
    ].to_dict()
    sequence_symbol_by_accession = sequences.set_index("Entry")[
        "Gene Names (primary)"
    ].to_dict()
    primary_accessions: dict[str, list[str]] = {}
    for accession, symbol in sequences[["Entry", "Gene Names (primary)"]].itertuples(
        index=False, name=None
    ):
        primary_accessions.setdefault(symbol, []).append(accession)

    legacy = json.loads(legacy_cache.read_text(encoding="utf-8"))
    if not isinstance(legacy, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in legacy.items()
    ):
        raise ValueError("legacy sequence cache must be a string mapping")
    legacy = {key.upper(): value for key, value in legacy.items()}
    aliases = json.loads(aliases_json.read_text(encoding="utf-8"))
    if not isinstance(aliases, dict) or not all(
        isinstance(key, str)
        and isinstance(value, dict)
        and set(value) == {"accession", "primary_symbol", "sequence_sha256"}
        and all(isinstance(field, str) for field in value.values())
        for key, value in aliases.items()
    ):
        raise ValueError("accession aliases have invalid identity records")
    aliases = {key.upper(): value for key, value in aliases.items()}

    accessions_by_sequence: dict[str, list[str]] = {}
    for accession, sequence in sequence_by_accession.items():
        accessions_by_sequence.setdefault(sequence, []).append(accession)

    resolved: dict[str, str] = {}
    for symbol in symbols:
        legacy_sequence = legacy.get(symbol)
        if symbol in aliases:
            alias = aliases[symbol]
            accession = alias["accession"]
        else:
            candidates = primary_accessions.get(symbol)
            if candidates is None:
                candidates = accessions_by_sequence.get(legacy_sequence, [])
            matches = [
                candidate
                for candidate in candidates
                if sequence_by_accession[candidate] == legacy_sequence
            ]
            if len(candidates) == 1:
                matches = candidates
            if len(matches) != 1:
                raise ValueError(
                    f"symbol {symbol} requires an explicit accession alias; "
                    f"candidates={candidates}, sequence matches={matches}"
                )
            accession = matches[0]
        if accession not in sequence_by_accession:
            raise ValueError(f"symbol {symbol}: accession lacks reviewed sequence")
        if accession not in entry_id_by_accession:
            raise ValueError(f"symbol {symbol}: accession lacks reviewed entry ID")
        if (
            identity_symbol_by_accession[accession]
            != sequence_symbol_by_accession[accession]
        ):
            raise ValueError(f"symbol {symbol}: reviewed source identity mismatch")
        if symbol in aliases and (
            sequence_symbol_by_accession[accession] != alias["primary_symbol"].upper()
        ):
            raise ValueError(f"symbol {symbol}: alias primary symbol mismatch")
        sequence_sha256 = hashlib.sha256(
            sequence_by_accession[accession].encode("utf-8")
        ).hexdigest()
        if symbol in aliases and sequence_sha256 != alias["sequence_sha256"]:
            raise ValueError(f"symbol {symbol}: alias sequence SHA-256 mismatch")
        if (
            symbol in aliases
            and legacy_sequence
            and sequence_by_accession[accession] != legacy_sequence
        ):
            raise ValueError(
                f"symbol {symbol}: accession sequence differs from legacy cache"
            )
        resolved[symbol] = accession

    records = {
        symbol: {
            "primary_accession": accession,
            "entry_id": entry_id_by_accession[accession],
            "isoform_identifier": accession,
            "isoform_policy": ISOFORM_POLICY,
            "sequence": sequence_by_accession[accession],
            "sequence_sha256": hashlib.sha256(
                sequence_by_accession[accession].encode("utf-8")
            ).hexdigest(),
        }
        for symbol, accession in resolved.items()
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_provenance": source_provenance,
        "records": records,
    }
    _atomic_write_json(output, payload)
    return {
        "status": "built",
        "symbols": len(symbols),
        "records": len(records),
        "aliases": sorted(aliases),
        "output": str(output),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-csv", type=Path, required=True)
    parser.add_argument("--sequence-tsv", type=Path, required=True)
    parser.add_argument("--identity-tsv", type=Path, required=True)
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument("--aliases-json", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = build_cache(**vars(parse_args(argv)))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
