from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.prepare.bootstrap_exp13_uniprot_cache import build_cache


def _alias(accession: str, primary_symbol: str) -> dict[str, str]:
    return {"accession": accession, "primary_symbol": primary_symbol}


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
    (tmp_path / "legacy.json").write_text(json.dumps({"OLD": "MB", "AMBIG": "MB"}))
    (tmp_path / "aliases.json").write_text(
        json.dumps(
            {
                "OLD": _alias("P3", "OTHER"),
                "AMBIG": _alias("P2", "NEW"),
            }
        )
    )

    report = build_cache(
        union_csv=tmp_path / "union.csv",
        sequence_tsv=tmp_path / "sequences.tsv",
        identity_tsv=tmp_path / "identities.tsv",
        legacy_cache=tmp_path / "legacy.json",
        aliases_json=tmp_path / "aliases.json",
        output=tmp_path / "cache.json",
    )

    payload = json.loads((tmp_path / "cache.json").read_text())
    assert report["records"] == 3
    assert payload["records"]["DIRECT"]["primary_accession"] == "P1"
    assert payload["records"]["OLD"]["primary_accession"] == "P3"
    assert payload["records"]["AMBIG"]["entry_id"] == "N_HUMAN"


def test_build_cache_rejects_alias_whose_sequence_differs_from_legacy(
    tmp_path: Path,
) -> None:
    pd.DataFrame({"gene_symbol": ["OLD"]}).to_csv(tmp_path / "union.csv", index=False)
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
    (tmp_path / "aliases.json").write_text(json.dumps({"OLD": _alias("P3", "OTHER")}))

    with pytest.raises(ValueError, match="differs from legacy cache"):
        build_cache(
            union_csv=tmp_path / "union.csv",
            sequence_tsv=tmp_path / "sequences.tsv",
            identity_tsv=tmp_path / "identities.tsv",
            legacy_cache=tmp_path / "legacy.json",
            aliases_json=tmp_path / "aliases.json",
            output=tmp_path / "cache.json",
        )


def test_build_cache_rejects_unique_direct_with_conflicting_legacy(
    tmp_path: Path,
) -> None:
    pd.DataFrame({"gene_symbol": ["DIRECT"]}).to_csv(
        tmp_path / "union.csv", index=False
    )
    pd.DataFrame(
        {
            "Entry": ["P1"],
            "Gene Names (primary)": ["DIRECT"],
            "Sequence": ["MA"],
        }
    ).to_csv(tmp_path / "sequences.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "Entry": ["P1"],
            "Entry Name": ["D_HUMAN"],
            "Gene Names (primary)": ["DIRECT"],
        }
    ).to_csv(tmp_path / "identities.tsv", sep="\t", index=False)
    (tmp_path / "legacy.json").write_text(json.dumps({"DIRECT": "MX"}))
    (tmp_path / "aliases.json").write_text("{}")

    with pytest.raises(ValueError, match="requires an explicit accession alias"):
        build_cache(
            union_csv=tmp_path / "union.csv",
            sequence_tsv=tmp_path / "sequences.tsv",
            identity_tsv=tmp_path / "identities.tsv",
            legacy_cache=tmp_path / "legacy.json",
            aliases_json=tmp_path / "aliases.json",
            output=tmp_path / "cache.json",
        )
