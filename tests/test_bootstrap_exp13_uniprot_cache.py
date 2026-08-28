from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.bootstrap_exp13_uniprot_cache import build_cache


def test_build_cache_resolves_primary_sequence_and_explicit_alias(
    tmp_path: Path,
) -> None:
    pd.DataFrame({"gene_symbol": ["DIRECT", "OLD", "AMBIG"]}).to_csv(
        tmp_path / "union.csv", index=False
    )
    pd.DataFrame(
        {
            "Entry": ["P1", "P2", "P3"],
            "Gene Names (primary)": ["DIRECT", "NEW", "OTHER"],
            "Sequence": ["MA", "MB", "MB"],
        }
    ).to_csv(tmp_path / "sequences.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "Entry": ["P1", "P2", "P3"],
            "Entry Name": ["D_HUMAN", "N_HUMAN", "O_HUMAN"],
            "Gene Names (primary)": ["DIRECT", "NEW", "OTHER"],
        }
    ).to_csv(tmp_path / "identities.tsv", sep="\t", index=False)
    (tmp_path / "legacy.json").write_text(
        json.dumps({"OLD": "MC", "AMBIG": "MB"})
    )
    (tmp_path / "aliases.json").write_text(
        json.dumps({"OLD": "P3", "AMBIG": "P2"})
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
