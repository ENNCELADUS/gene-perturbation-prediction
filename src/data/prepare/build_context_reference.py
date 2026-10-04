"""Fetch and pin the reference tables of the linear context prior.

Runs on the Mac: the H20 host has no internet, so the small tables are committed
under configs/context_prior/reference/ and reach it through git. Line tables hold
training-side lines only: validation and test lines and every line sharing their
patients never enter one.

uv run python -m src.data.prepare.build_context_reference \
    --split configs/benchmarks/cell_line_geneeffect_226_split.json \
    --extra-lines configs/benchmarks/extra_bulk_lines_26Q1.json \
    --hotspot data/sl_dependency_v0/raw/depmap/OmicsSomaticMutationsMatrixHotspot.csv \
    --damaging data/sl_dependency_v0/raw/depmap/OmicsSomaticMutationsMatrixDamaging.csv \
    --out configs/context_prior/reference
"""  # noqa: E501

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import urllib.parse
from collections.abc import Collection, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.depmap import read_depmap_matrix
from src.data.extra_lines import load_extra_lines
from src.data.splits import load_geneeffect_226_split

BIOMART_URL = "https://useast.ensembl.org/biomart/martservice"
CORUM_URL = (
    "https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/"
    "download_current_file?file_id=human&file_format=txt"
)
HALLMARK_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/"
    "h.all.v2024.1.Hs.symbols.gmt"
)
PROGENY_URL = "https://omnipathdb.org/annotations?resources=PROGENy&format=tsv"
#: DepMap 24Q4 OmicsSignatures.csv (MSIScore), figshare 10.6084/m9.figshare.27993248.
SIGNATURES_URL = "https://ndownloader.figshare.com/files/51065726"
CHROMOSOMES = (*(str(n) for n in range(1, 23)), "X", "Y")
PARALOG_MIN_IDENTITY = 20.0
PARALOGS_PER_GENE = 10
PROGENY_TOP = 100
DRIVER_MIN_FRACTION = 0.05


def fetch(url: str) -> str:
    """The body of ``url`` through curl, which verifies TLS against the system
    trust store (CORUM's server sends an incomplete chain Python cannot verify)."""
    result = subprocess.run(
        ["curl", "-sSfL", "--max-time", "600", url], capture_output=True, check=True
    )
    return result.stdout.decode("utf-8")


def paralog_url(chromosome: str) -> str:
    query = (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="1" '
        'uniqueRows="1"><Dataset name="hsapiens_gene_ensembl" interface="default">'
        f'<Filter name="chromosome_name" value="{chromosome}"/>'
        '<Attribute name="external_gene_name"/>'
        '<Attribute name="hsapiens_paralog_associated_gene_name"/>'
        '<Attribute name="hsapiens_paralog_perc_id"/>'
        '<Attribute name="hsapiens_paralog_perc_id_r1"/>'
        "</Dataset></Query>"
    )
    return f"{BIOMART_URL}?{urllib.parse.urlencode({'query': query})}"


def parse_paralogs(text: str) -> pd.DataFrame:
    """BioMart rows -> gene, paralog, identity (the lower %id of both directions)."""
    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    if frame.shape[1] != 4:
        raise ValueError(f"unexpected BioMart response: {text[:200]!r}")
    frame.columns = ["gene", "paralog", "identity", "identity_reverse"]
    frame = frame.dropna(subset=["gene", "paralog"])
    gene, paralog = frame["gene"].str.upper(), frame["paralog"].str.upper()
    keep = gene != paralog
    pairs = pd.DataFrame(
        {
            "gene": gene[keep],
            "paralog": paralog[keep],
            "identity": np.minimum(
                frame.loc[keep, "identity"].astype(float),
                frame.loc[keep, "identity_reverse"].astype(float),
            ),
        }
    )
    return pairs.groupby(["gene", "paralog"], as_index=False)["identity"].max()


def closest_paralogs(
    pairs: pd.DataFrame, *, min_identity: float, per_gene: int
) -> pd.DataFrame:
    """Each gene's ``per_gene`` closest paralogs at or above ``min_identity``."""
    kept = pairs.loc[pairs["identity"] >= min_identity].sort_values(
        ["gene", "identity", "paralog"], ascending=[True, False, True]
    )
    return kept.groupby("gene", sort=False).head(per_gene).reset_index(drop=True)


def parse_corum(text: str) -> pd.DataFrame:
    """CORUM's current human file -> one (complex_id, gene) row per subunit."""
    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    human = frame.loc[
        frame["organism"] == "Human", ["complex_id", "subunits_gene_name"]
    ]
    rows = [
        (int(complex_id), gene.strip().upper())
        for complex_id, members in human.dropna().itertuples(index=False)
        for gene in members.split(";")
        if gene.strip()
    ]
    return (
        pd.DataFrame(rows, columns=["complex_id", "gene"])
        .drop_duplicates()
        .sort_values(["complex_id", "gene"])
        .reset_index(drop=True)
    )


def parse_gmt(text: str) -> pd.DataFrame:
    """GMT -> one (gene_set, gene) row per member, symbols upper-cased."""
    rows = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            rows += [(fields[0], gene.upper()) for gene in fields[2:] if gene]
    return pd.DataFrame(rows, columns=["gene_set", "gene"]).drop_duplicates()


def parse_progeny(text: str, *, top: int) -> pd.DataFrame:
    """OmniPath PROGENy annotations -> each pathway's ``top`` genes by p-value."""
    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    wide = (
        frame.set_index(["genesymbol", "record_id", "label"])["value"]
        .unstack("label")
        .reset_index()
        .dropna(subset=["pathway", "weight", "p_value"])
    )
    wide["weight"] = wide["weight"].astype(float)
    wide["p_value"] = wide["p_value"].astype(float)
    best = (
        wide.sort_values(["pathway", "p_value", "genesymbol"])
        .groupby("pathway", sort=True)
        .head(top)
    )
    return pd.DataFrame(
        {
            "pathway": best["pathway"].to_numpy(),
            "gene": best["genesymbol"].str.upper().to_numpy(),
            "weight": best["weight"].to_numpy(),
        }
    )


def driver_status(
    hotspot: pd.DataFrame,
    damaging: pd.DataFrame,
    lines: Sequence[str],
    *,
    min_fraction: float,
) -> pd.DataFrame:
    """Lines x driver genes, 1 where a hotspot or damaging mutation is called.

    Drivers are the hotspot matrix's genes called in at least ``min_fraction`` of
    ``lines``; GeneEffect is never read.
    """
    rows = [m for m in lines if m in hotspot.index]
    genes = list(hotspot.columns)
    called = (hotspot.loc[rows, genes].fillna(0) > 0) | (
        damaging.reindex(index=rows, columns=genes).fillna(0) > 0
    )
    drivers = [g for g in genes if called[g].mean() >= min_fraction]
    status = called.loc[:, drivers].astype(int)
    status.index.name = "model_id"
    return status


def training_msi(text: str, blocked: Collection[str]) -> pd.Series:
    frame = pd.read_csv(io.StringIO(text), index_col=0)
    msi = frame["MSIScore"].dropna()
    msi.index = msi.index.astype(str)
    msi = msi.loc[~msi.index.isin(set(blocked))].sort_index()
    msi.index.name, msi.name = "model_id", "msi_score"
    return msi


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split", "extra-lines", "hotspot", "damaging", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    split = load_geneeffect_226_split(args.split)
    extra = load_extra_lines(args.extra_lines, split)
    blocked = {*split.val, *split.test, *extra.excluded}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.concat(parse_paralogs(fetch(paralog_url(c))) for c in CHROMOSOMES)
    pairs = pairs.groupby(["gene", "paralog"], as_index=False)["identity"].max()
    hotspot = read_depmap_matrix(args.hotspot)
    damaging = read_depmap_matrix(args.damaging)
    tables = {
        "paralogs.tsv.gz": (
            closest_paralogs(
                pairs, min_identity=PARALOG_MIN_IDENTITY, per_gene=PARALOGS_PER_GENE
            ),
            BIOMART_URL,
        ),
        "complexes.tsv": (parse_corum(fetch(CORUM_URL)), CORUM_URL),
        "hallmark.tsv": (parse_gmt(fetch(HALLMARK_URL)), HALLMARK_URL),
        "progeny.tsv": (
            parse_progeny(fetch(PROGENY_URL), top=PROGENY_TOP),
            PROGENY_URL,
        ),
        "drivers.tsv": (
            driver_status(
                hotspot,
                damaging,
                sorted(set(hotspot.index) - blocked),
                min_fraction=DRIVER_MIN_FRACTION,
            ).reset_index(),
            str(args.hotspot),
        ),
        "msi.tsv": (
            training_msi(fetch(SIGNATURES_URL), blocked).reset_index(),
            SIGNATURES_URL,
        ),
    }
    provenance = {
        "retrieved": date.today().isoformat(),
        "settings": {
            "paralog_min_identity": PARALOG_MIN_IDENTITY,
            "paralogs_per_gene": PARALOGS_PER_GENE,
            "progeny_top": PROGENY_TOP,
            "driver_min_fraction": DRIVER_MIN_FRACTION,
        },
        "files": {},
    }
    for name, (frame, source) in tables.items():
        frame.to_csv(out / name, sep="\t", index=False)
        provenance["files"][name] = {
            "source": source,
            "rows": len(frame),
            "sha256": _sha256(out / name),
        }
        print(name, len(frame))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
