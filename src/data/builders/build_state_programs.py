"""Fetch and pin the reference cell-state programs of the noise diagnosis.

Runs on the Mac, like ``build_context_reference``; the tables are small and reach
the H20 through git. Writes under configs/context_prior/reference/

- ``cell_cycle.tsv`` (``phase`` in S, G2M; ``gene``): Tirosh et al. 2016's 43 S and
  54 G2/M genes, as Seurat ships them (``cc.genes``);
- ``meta_programs.tsv`` (``program``, ``gene``): Gavish et al. 2023's 41 malignant
  meta-programs, 50 genes each, from the authors' repository;
- ``meta_program_sources.tsv`` (``program``, ``studies``, ``cell_line_studies``):
  how many studies each meta-program was derived from, and how many of them are
  cell-line studies (Kinker et al. 2020's CCLE pools and the other cell-line sets),

and records each file's source and row count in ``provenance.json``. A
config chooses which programs a run reads; nothing here drops one.

uv run --extra data python -m src.data.builders.build_state_programs \
    --out configs/context_prior/reference
"""  # noqa: E501

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

SEURAT_COMMIT = "00e55c9b54c8b12b5ae85ffafc1c18d683c73a9f"
THREECA_COMMIT = "21a784103528cdf15008761f9669e2025159d6d2"
CELL_CYCLE_URL = (
    f"https://raw.githubusercontent.com/satijalab/seurat/{SEURAT_COMMIT}/"
    "data/cc.genes.rda"
)
META_PROGRAMS_URL = (
    f"https://raw.githubusercontent.com/tiroshlab/3ca/{THREECA_COMMIT}/"
    "ITH_hallmarks/MPs_distribution/MP_list.RDS.gz"
)
SOURCES_URL = (
    f"https://raw.githubusercontent.com/tiroshlab/3ca/{THREECA_COMMIT}/"
    "ITH_hallmarks/TCGA_analysis/study_contribution_per_MP.RDS"
)
#: Seurat's list names and the phase each holds.
PHASES = {"s.genes": "S", "g2m.genes": "G2M"}
#: Study names of the 3CA collection that are cell-line studies: Kinker et al.
#: 2020's CCLE pools (``<LINEAGE>_Kinker_2020``) and the cell-line sets of others.
_CELL_LINE_STUDY = re.compile(r"kinker|cell_?lines?", re.IGNORECASE)
#: What was checked about where the programs come from (provenance ``checks``).
CHECKS = {
    "meta_program_cell_lines": (
        "meta_program_sources.tsv counts each program's cell-line studies (Kinker et "
        "al. 2020's CCLE pools and other cell-line sets). Kinker's pools hold the "
        "cells of validation and test lines, the exposure ruled out for Kinker's own "
        "programs on 2026-10-04, so a config reads only programs with no cell-line "
        "study (21 of 41; decided 2026-10-12)."
    ),
    "tirosh_cell_cycle": (
        "Tirosh et al. 2016 (Science 352:189, PMC4944528): 'we used gene signatures "
        "previously shown to denote G1/S or G2/M phases in both synchronization and "
        "single-cell experiments in cell lines', citing Whitfield et al. 2002 "
        "(synchronised HeLa; HeLa is a test line here) and Macosko et al. 2015. "
        "Canonical cell-cycle genes from an earlier experiment, no deposit this "
        "project reads; kept and noted."
    ),
}


def fetch(url: str) -> bytes:
    """The body of ``url`` through curl (the system trust store)."""
    result = subprocess.run(
        ["curl", "-sSfL", "--max-time", "600", url], capture_output=True, check=True
    )
    return result.stdout


def program_name(name: str) -> str:
    """One spelling per program: the authors' names differ only in spaces."""
    return " ".join(str(name).split())


def is_cell_line_study(study: str) -> bool:
    return bool(_CELL_LINE_STUDY.search(str(study)))


def cell_cycle_table(lists: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """``phase``, ``gene`` from Seurat's ``cc.genes`` lists."""
    if set(lists) != set(PHASES):
        raise ValueError(f"cc.genes holds {sorted(lists)}, not {sorted(PHASES)}")
    return pd.DataFrame(
        [(PHASES[key], str(gene)) for key in PHASES for gene in lists[key]],
        columns=["phase", "gene"],
    )


def meta_program_table(programs: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """``program``, ``gene`` in the authors' program order."""
    return pd.DataFrame(
        [
            (program_name(name), str(gene))
            for name, genes in programs.items()
            for gene in genes
        ],
        columns=["program", "gene"],
    )


def program_number(name: str) -> str:
    """``MP<n>``: the two tables agree on numbers, not always on names (MP41 is
    "Unassigned" in one and "Unassigned-II" in the other)."""
    match = re.match(r"MP\d+\b", program_name(name))
    if match is None:
        raise ValueError(f"no MP number in program name {name!r}")
    return match.group(0)


def source_table(
    contributions: Mapping[str, Sequence[str]], programs: Sequence[str]
) -> pd.DataFrame:
    """Per program (named as in ``programs``), its contributing studies and how
    many are cell-line studies, joined on the MP number.

    Raises unless the two tables hold exactly the same MP numbers.
    """
    by_number = {
        program_number(k): [str(s) for s in v] for k, v in contributions.items()
    }
    names = {program_number(name): name for name in programs}
    if set(by_number) != set(names):
        raise ValueError(
            "the study table and the program list hold different programs: "
            f"{sorted(set(by_number) ^ set(names))}"
        )
    return pd.DataFrame(
        [
            (
                name,
                len(by_number[number]),
                sum(is_cell_line_study(s) for s in by_number[number]),
            )
            for number, name in names.items()
        ],
        columns=["program", "studies", "cell_line_studies"],
    )


def _read_r(payload: bytes) -> object:
    import rdata

    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return rdata.conversion.convert(rdata.parser.parse_data(payload))


def _write(out: Path, name: str, frame: pd.DataFrame, source: str) -> dict:
    path = out / name
    frame.to_csv(path, sep="\t", index=False)
    return {"source": source, "rows": len(frame)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out

    cell_cycle = cell_cycle_table(_read_r(fetch(CELL_CYCLE_URL))["cc.genes"])
    programs = _read_r(fetch(META_PROGRAMS_URL))["Cancer"]
    meta = meta_program_table(programs)
    names = list(dict.fromkeys(meta["program"]))
    sources = source_table(_read_r(fetch(SOURCES_URL)), names)

    provenance_path = out / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["files"]["cell_cycle.tsv"] = _write(
        out, "cell_cycle.tsv", cell_cycle, CELL_CYCLE_URL
    )
    provenance["files"]["meta_programs.tsv"] = _write(
        out, "meta_programs.tsv", meta, META_PROGRAMS_URL
    )
    provenance["files"]["meta_program_sources.tsv"] = _write(
        out, "meta_program_sources.tsv", sources, SOURCES_URL
    )
    provenance.setdefault("checks", {}).update(CHECKS)
    provenance["state_programs_retrieved"] = date.today().isoformat()
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"{len(cell_cycle)} cell-cycle rows, {len(names)} meta-programs "
        f"({int((sources['cell_line_studies'] == 0).sum())} without a cell-line study)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
