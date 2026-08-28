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
    if identities["Entry"].duplicated().any():
        raise ValueError("reviewed identity TSV has duplicate accessions")

    sequence_by_accession = (
        sequences.set_index("Entry")["Sequence"].astype(str).to_dict()
    )
    entry_id_by_accession = (
        identities.set_index("Entry")["Entry Name"].astype(str).to_dict()
    )
    primary_accession: dict[str, str] = {}
    for accession, symbol in sequences[["Entry", "Gene Names (primary)"]].itertuples(
        index=False, name=None
    ):
        if symbol in primary_accession:
            raise ValueError(
                f"reviewed sequence TSV has duplicate primary symbol: {symbol}"
            )
        primary_accession[symbol] = accession

    legacy = json.loads(legacy_cache.read_text(encoding="utf-8"))
    if not isinstance(legacy, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in legacy.items()
    ):
        raise ValueError("legacy sequence cache must be a string mapping")
    legacy = {key.upper(): value for key, value in legacy.items()}
    aliases = json.loads(aliases_json.read_text(encoding="utf-8"))
    if not isinstance(aliases, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in aliases.items()
    ):
        raise ValueError("accession aliases must be a string mapping")
    aliases = {key.upper(): value for key, value in aliases.items()}

    accessions_by_sequence: dict[str, list[str]] = {}
    for accession, sequence in sequence_by_accession.items():
        accessions_by_sequence.setdefault(sequence, []).append(accession)

    resolved: dict[str, str] = {}
    for symbol in symbols:
        if symbol in aliases:
            accession = aliases[symbol]
        elif symbol in primary_accession:
            accession = primary_accession[symbol]
        else:
            matches = accessions_by_sequence.get(legacy.get(symbol, ""), [])
            if len(matches) != 1:
                raise ValueError(
                    f"symbol {symbol} requires an explicit accession alias; "
                    f"sequence matches={matches}"
                )
            accession = matches[0]
        if accession not in sequence_by_accession:
            raise ValueError(f"symbol {symbol}: accession lacks reviewed sequence")
        if accession not in entry_id_by_accession:
            raise ValueError(f"symbol {symbol}: accession lacks reviewed entry ID")
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
    payload = {"schema_version": SCHEMA_VERSION, "records": records}
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = build_cache(**vars(parse_args(argv)))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
