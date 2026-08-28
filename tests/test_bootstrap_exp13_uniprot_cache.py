from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.bootstrap_exp13_uniprot_cache import build_cache


def _alias(accession: str, primary_symbol: str, sequence: str) -> dict[str, str]:
    return {
        "accession": accession,
        "primary_symbol": primary_symbol,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
    }


def _write_source_manifest(tmp_path: Path, artifacts: dict[str, Path]) -> Path:
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "exp13-uniprot-offline-sources-v1",
                "reviewed": True,
                "taxonomy_id": 9606,
                "uniprot_query": "reviewed:true AND organism_id:9606",
                "artifacts": {
                    name: {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                    }
                    for name, path in artifacts.items()
                },
            }
        )
    )
    return manifest


def test_build_cache_resolves_primary_sequence_and_explicit_alias(
    tmp_path: Path,
) -> None:
    pd.DataFrame({"gene_symbol": ["DIRECT", "OLD", "AMBIG"]}).to_csv(
        tmp_path / "union.csv", index=False
    )
    pd.DataFrame(
        {
            "Entry": ["P1", "P2", "P3", "P4"],
            "Gene Names (primary)": ["DIRECT", "NEW", "OTHER", "OTHER2"],
            "Sequence": ["MA", "MB", "MB", "MX"],
        }
    ).to_csv(tmp_path / "sequences.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "Entry": ["P1", "P2", "P3", "P4"],
            "Entry Name": ["D_HUMAN", "N_HUMAN", "O_HUMAN", "DX_HUMAN"],
            "Gene Names (primary)": ["DIRECT", "NEW", "OTHER", "OTHER2"],
        }
    ).to_csv(tmp_path / "identities.tsv", sep="\t", index=False)
    (tmp_path / "legacy.json").write_text(
        json.dumps({"OLD": "MB", "AMBIG": "MB"})
    )
    (tmp_path / "aliases.json").write_text(
        json.dumps(
            {
                "OLD": _alias("P3", "OTHER", "MB"),
                "AMBIG": _alias("P2", "NEW", "MB"),
            }
        )
    )
    source_manifest = _write_source_manifest(
        tmp_path,
        {
            "union_csv": tmp_path / "union.csv",
            "sequence_tsv": tmp_path / "sequences.tsv",
            "identity_tsv": tmp_path / "identities.tsv",
            "legacy_cache": tmp_path / "legacy.json",
            "aliases_json": tmp_path / "aliases.json",
        },
    )

    report = build_cache(
        union_csv=tmp_path / "union.csv",
        sequence_tsv=tmp_path / "sequences.tsv",
        identity_tsv=tmp_path / "identities.tsv",
        legacy_cache=tmp_path / "legacy.json",
        aliases_json=tmp_path / "aliases.json",
        source_manifest=source_manifest,
        output=tmp_path / "cache.json",
    )

    payload = json.loads((tmp_path / "cache.json").read_text())
    assert report["records"] == 3
    assert payload["records"]["DIRECT"]["primary_accession"] == "P1"
    assert payload["records"]["OLD"]["primary_accession"] == "P3"
    assert payload["records"]["AMBIG"]["entry_id"] == "N_HUMAN"
    assert payload["source_provenance"]["reviewed"] is True
    assert payload["source_provenance"]["taxonomy_id"] == 9606


def test_build_cache_rejects_valid_alias_with_wrong_sequence(tmp_path: Path) -> None:
    pd.DataFrame({"gene_symbol": ["OLD"]}).to_csv(
        tmp_path / "union.csv", index=False
    )
    pd.DataFrame(
        {
            "Entry": ["P2", "P3"],
            "Gene Names (primary)": ["NEW", "OTHER"],
            "Sequence": ["MB", "MX"],
        }
    ).to_csv(tmp_path / "sequences.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "Entry": ["P2", "P3"],
            "Entry Name": ["N_HUMAN", "O_HUMAN"],
            "Gene Names (primary)": ["NEW", "OTHER"],
        }
    ).to_csv(tmp_path / "identities.tsv", sep="\t", index=False)
    (tmp_path / "legacy.json").write_text(json.dumps({"OLD": "MB"}))
    (tmp_path / "aliases.json").write_text(
        json.dumps({"OLD": _alias("P3", "OTHER", "MB")})
    )
    source_manifest = _write_source_manifest(
        tmp_path,
        {
            "union_csv": tmp_path / "union.csv",
            "sequence_tsv": tmp_path / "sequences.tsv",
            "identity_tsv": tmp_path / "identities.tsv",
            "legacy_cache": tmp_path / "legacy.json",
            "aliases_json": tmp_path / "aliases.json",
        },
    )

    with pytest.raises(ValueError, match="sequence SHA-256 mismatch"):
        build_cache(
            union_csv=tmp_path / "union.csv",
            sequence_tsv=tmp_path / "sequences.tsv",
            identity_tsv=tmp_path / "identities.tsv",
            legacy_cache=tmp_path / "legacy.json",
            aliases_json=tmp_path / "aliases.json",
            source_manifest=source_manifest,
            output=tmp_path / "cache.json",
        )
