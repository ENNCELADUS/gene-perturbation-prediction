"""Plan one second deposit per labelled single-cell training line for the noise
diagnosis's deposit swap (protocol §14), from tracked files only; the plan is what
``build_extra_single_cell_sources --second-source`` builds."""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import pandas as pd

from src.data.builders.build_extra_single_cell_lines import choose_sources, read_map
from src.data.splits import load_geneeffect_split

SCHEMA_VERSION = 1
#: The 226 table's names of deposits the identity map knows under another key.
PRIMARY_KEYS = {"kinker_sccle": "kinker_scp542", "Tahoe-100M DMSO": "tahoe100m"}
POLICY = (
    "One second deposit per labelled single-cell training line: an accepted "
    "raw-count source of the identity map that the extras spec builds, other than "
    "the line's own and from another accession, chosen by the membership policy."
)


def plan(
    candidates: pd.DataFrame,
    primary: Mapping[str, str],
    buildable: Mapping[str, int],
) -> dict[str, dict]:
    """Per line of ``primary`` (ModelID to its deposit's source key) with a second
    deposit: the primary, the second source, its wave and accession."""
    accepted = candidates.loc[
        (candidates["review"] == "ok") & (candidates["raw_counts"] == "true")
    ]
    accession = {
        (row.model_id, row.source): row.accession
        for row in accepted.itertuples(index=False)
    }
    eligible = []
    for row in accepted.itertuples(index=False):
        own = primary.get(row.model_id)
        if own is None or row.source == own or row.source not in buildable:
            continue
        if accession.get((row.model_id, own)) == row.accession:
            continue  # the same deposit counted another way
        eligible.append(row)
    if not eligible:
        return {}
    chosen = choose_sources(pd.DataFrame(eligible))
    return {
        model_id: {
            "primary": primary[model_id],
            "second": source,
            "wave": int(buildable[source]),
            "accession": accession[(model_id, source)],
        }
        for model_id, source in sorted(chosen.items())
    }


def primary_sources(table: pd.DataFrame, lines: Collection[str]) -> dict[str, str]:
    """ModelID to its deposit's identity-map source key, for ``lines``."""
    sources = table.set_index("model_id")["expression_source"].astype(str)
    return {m: PRIMARY_KEYS.get(sources[m], sources[m]) for m in lines}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    split = load_geneeffect_split(args.split)
    table = pd.read_csv(args.split.with_suffix(".csv"))
    spec = json.loads(args.spec.read_text())
    buildable = {name: int(source["wave"]) for name, source in spec["sources"].items()}
    lines = plan(
        read_map(args.candidates),
        primary_sources(table, split.supervised_train),
        buildable,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "lines": lines,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(lines)} lines with a second deposit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
