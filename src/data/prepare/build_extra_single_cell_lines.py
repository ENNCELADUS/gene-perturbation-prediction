"""The frozen single-cell identity map and the extra single-cell membership.

``configs/benchmarks/single_cell_atlas_candidates.csv`` maps each source's own
spelling of a line (``identifier``) to a DepMap ModelID once, with the review of
that mapping; nothing joins lines by name at run time. Two steps:

``propose`` (Mac) gives every map row without a review a ModelID: the source's own
ModelID or RRID, else a Model.csv name (``StrippedCellLineName`` or
``CellLineName``, case and punctuation ignored), else an exact Cellosaurus name or
synonym whose accession is a Model.csv RRID. Rows it cannot settle are flagged
(``ambiguous``, ``no_match``, ``problematic``, ``derived``) for review by hand.

uv run python -m src.data.prepare.build_extra_single_cell_lines propose \
    --candidates configs/benchmarks/single_cell_atlas_candidates.csv \
    --model data/sl_dependency_v0/raw/depmap/Model.csv

``build`` (H20) writes the membership file from the reviewed map, the 26Q1 files
and the built sources' QC reports (``<basal_raw_umi>/qc/<source>.json``).

uv run python -m src.data.prepare.build_extra_single_cell_lines build \
    --candidates configs/benchmarks/single_cell_atlas_candidates.csv \
    --model data/sl_dependency_v0/raw/depmap/Model.csv \
    --gene-effect data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv \
    --split configs/benchmarks/cell_line_geneeffect_226_split.json \
    --qc-dir data/sl_dependency_v0/processed/basal_raw_umi/qc \
    --out configs/benchmarks/extra_single_cell_lines_26Q1.json
"""  # noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.depmap import read_model_ids, read_models
from src.data.extra_lines import SINGLE_CELL_SCHEMA_VERSION
from src.data.splits import FixedSplit, load_geneeffect_split

MAP_COLUMNS = (
    "source",
    "accession",
    "model_id",
    "identifier",
    "identifier_kind",
    "cells",
    "raw_counts",
    "condition",
    "size_gb",
    "condition_class",
    "review",
    "wave",
    "note",
)
#: Best first: the order a line's sources are preferred in.
CONDITION_CLASSES = ("basal_2d", "control_guide", "cutting_control", "other")
FLAGS = ("ambiguous", "no_match", "problematic", "derived")
POLICY = (
    "Labelled lines outside the 226 whose basal single cells are built, one source "
    "per line (untreated or vehicle 2D first, then raw counts, then most cells). A "
    "line sharing a PatientID with a validation or test line is excluded; a chosen "
    "line without 26Q1 GeneEffect is unlabelled and not built; a labelled line whose "
    "source is not built yet is pending."
)

#: Cellosaurus entries: ``search(name)`` and ``entry(accession)`` return dicts
#: with ``accession``, ``names`` (identifier and synonyms), ``derived_from``
#: (accessions) and ``problematic`` (the comment, or "").
Search = Callable[[str], list[dict[str, Any]]]
Entry = Callable[[str], dict[str, Any] | None]


def read_map(path: Path) -> pd.DataFrame:
    """The identity map, checked: exact columns, known classes and reviews, one
    row per (source, identifier), a ModelID on every accepted row."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != MAP_COLUMNS:
        raise ValueError(f"{path}: columns must be {MAP_COLUMNS}")
    bad = sorted(set(frame["condition_class"]) - set(CONDITION_CLASSES))
    if bad:
        raise ValueError(f"{path}: unknown condition_class {bad}")
    reviews = frame["review"]
    known = reviews.isin(("", "ok", *FLAGS)) | reviews.str.startswith("dropped: ")
    if not known.all():
        raise ValueError(f"{path}: unknown review {sorted(set(reviews[~known]))}")
    if frame.duplicated(["source", "identifier"]).any():
        raise ValueError(f"{path}: duplicate (source, identifier) rows")
    if (frame.loc[reviews == "ok", "model_id"] == "").any():
        raise ValueError(f"{path}: an accepted row has no model_id")
    return frame


def choose_sources(candidates: pd.DataFrame) -> dict[str, str]:
    """ModelID to the one source its cells come from, over accepted rows of an
    ingested wave: best condition class, then raw counts, then most cells, then
    source name."""
    rows = candidates.loc[(candidates["review"] == "ok") & (candidates["wave"] != "")]
    per = (
        rows.assign(
            rank=rows["condition_class"].map(CONDITION_CLASSES.index),
            cooked=rows["raw_counts"] != "true",
            cells=pd.to_numeric(rows["cells"], errors="coerce").fillna(0),
        )
        .groupby(["model_id", "source"], as_index=False)
        .agg(rank=("rank", "min"), cooked=("cooked", "max"), cells=("cells", "sum"))
        .assign(fewer=lambda frame: -frame["cells"])
        .sort_values(["model_id", "rank", "cooked", "fewer", "source"])
    )
    return dict(per.groupby("model_id", sort=True)["source"].first())


@dataclass(frozen=True)
class Admission:
    """The chosen lines outside the 226, by what membership does with them.

    Attributes:
        labelled: Lines with 26Q1 GeneEffect, each with its chosen source.
        unlabelled: Lines without 26Q1 GeneEffect.
        excluded: Lines sharing a patient with a validation or test line.
    """

    labelled: Mapping[str, str]
    unlabelled: tuple[str, ...]
    excluded: Mapping[str, str]


def admit(
    candidates: pd.DataFrame,
    models: pd.DataFrame,
    *,
    labelled_ids: Collection[str],
    split: FixedSplit,
) -> Admission:
    """Sort the chosen lines outside the 226 into labelled, unlabelled, excluded."""
    chosen = choose_sources(candidates)
    unknown = sorted(set(chosen) - set(models.index))
    if unknown:
        raise ValueError(f"ModelIDs absent from Model.csv: {unknown[:10]}")
    ours = set(split.all_model_ids)
    held = {models.at[m, "patient_id"]: m for m in (*split.val, *split.test)}
    labelled, unlabelled, excluded = {}, [], {}
    for model_id, source in chosen.items():
        patient = models.at[model_id, "patient_id"]
        if model_id in ours:
            continue
        if patient in held:
            excluded[model_id] = (
                f"shares PatientID {patient} with held-out line {held[patient]}"
            )
        elif model_id in labelled_ids:
            labelled[model_id] = source
        else:
            unlabelled.append(model_id)
    return Admission(labelled, tuple(unlabelled), excluded)


def build_membership(
    candidates: pd.DataFrame,
    models: pd.DataFrame,
    *,
    labelled_ids: Collection[str],
    split: FixedSplit,
    qc: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Membership payload; a labelled line is built when its chosen source's QC
    report holds it with at least one cell, and pending otherwise."""
    admission = admit(candidates, models, labelled_ids=labelled_ids, split=split)
    accepted = candidates.loc[candidates["review"] == "ok"]
    labelled, pending = {}, {}
    for model_id, source in admission.labelled.items():
        report = qc.get(source)
        line = None if report is None else report["lines"].get(model_id)
        if report is None:
            pending[model_id] = f"source {source} not built"
            continue
        if not line or not line["cells"]:
            pending[model_id] = f"no cells after QC in {source}"
            continue
        rows = accepted.loc[accepted["model_id"] == model_id]
        own = rows.loc[rows["source"] == source]
        labelled[model_id] = {
            "source": source,
            "accession": own["accession"].iloc[0],
            "identifiers": sorted(own["identifier"]),
            "condition": "; ".join(sorted(set(own["condition"]))),
            "condition_class": min(own["condition_class"], key=CONDITION_CLASSES.index),
            "patient_id": models.at[model_id, "patient_id"],
            "lineage": models.at[model_id, "lineage"],
            "cells": int(line["cells"]),
            "arms": dict(line["arms"]),
            "other_sources": sorted(set(rows["source"]) - {source}),
        }
    return {
        "schema_version": SINGLE_CELL_SCHEMA_VERSION,
        "policy": POLICY,
        "labelled": labelled,
        "unlabelled": list(admission.unlabelled),
        "pending": pending,
        "excluded": dict(admission.excluded),
    }


def _norm(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _accept(row: dict, model_id: str, kind: str, note: str = "") -> None:
    row.update(model_id=model_id, identifier_kind=kind, review="ok", note=note)


def propose(
    candidates: pd.DataFrame,
    model_table: pd.DataFrame,
    *,
    search: Search,
    entry: Entry,
) -> pd.DataFrame:
    """Fill ``model_id``, ``identifier_kind``, ``review`` and ``note`` of every row
    with an empty review; reviewed rows are left as they are.

    ``model_table`` is Model.csv (``ModelID``, ``CellLineName``,
    ``StrippedCellLineName``, ``RRID``).
    """
    by_name: dict[str, set[str]] = {}
    for column in ("StrippedCellLineName", "CellLineName"):
        for model_id, name in zip(model_table["ModelID"], model_table[column]):
            if isinstance(name, str) and name:
                by_name.setdefault(_norm(name), set()).add(model_id)
    rrids = model_table.dropna(subset=["RRID"]).groupby("RRID")["ModelID"].agg(list)
    by_rrid = {rrid: ids[0] for rrid, ids in rrids.items() if len(ids) == 1}
    known = set(model_table["ModelID"])
    rrid_of = dict(zip(model_table["ModelID"], model_table["RRID"]))
    rows = candidates.to_dict("records")
    for row in rows:
        if row["review"]:
            continue
        name = row["identifier"]
        hits = sorted(by_name.get(_norm(name), ()))
        if re.fullmatch(r"ACH-\d{6}", name) and name in known:
            _accept(row, name, "model_id")
        elif name in by_rrid:
            _accept(row, by_rrid[name], "rrid")
        elif len(hits) == 1:
            _accept(row, hits[0], "name")
        elif hits:
            row.update(review="ambiguous", note="Model.csv: " + ", ".join(hits))
        else:
            exact = [
                found
                for found in search(name)
                if _norm(name) in {_norm(n) for n in found["names"]}
            ]
            ours = [found for found in exact if found["accession"] in by_rrid]
            parents = [
                (found["accession"], by_rrid[parent])
                for found in exact
                for parent in found["derived_from"]
                if parent in by_rrid
            ]
            if len(ours) == 1:
                accession = ours[0]["accession"]
                _accept(row, by_rrid[accession], "synonym", f"Cellosaurus {accession}")
            elif parents:
                row.update(
                    review="derived",
                    note="; ".join(f"{child} derives from {m}" for child, m in parents),
                )
            else:
                found = ", ".join(e["accession"] for e in exact) or "nothing exact"
                row.update(review="no_match", note=f"Cellosaurus: {found}")
        if row["review"] == "ok":
            rrid = rrid_of[row["model_id"]]
            details = entry(rrid) if isinstance(rrid, str) and rrid else None
            if details and details["problematic"]:
                row.update(review="problematic", note=details["problematic"])
    return pd.DataFrame(rows, columns=list(MAP_COLUMNS))


_CELLOSAURUS = "https://api.cellosaurus.org"


def _cellosaurus_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for line in payload["Cellosaurus"]["cell-line-list"]:
        entries.append(
            {
                "accession": next(
                    a["value"] for a in line["accession-list"] if a["type"] == "primary"
                ),
                "names": [n["value"] for n in line.get("name-list", [])],
                "derived_from": [d["accession"] for d in line.get("derived-from", [])],
                "problematic": "; ".join(
                    c["value"]
                    for c in line.get("comment-list", [])
                    if c["category"] == "Problematic cell line"
                ),
            }
        )
    return entries


def _get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())


@cache
def cellosaurus_search(name: str) -> list[dict[str, Any]]:
    """Cellosaurus cell lines a free-text search for ``name`` returns."""
    query = urllib.parse.urlencode(
        {"q": name, "fields": "ac,id,sy,hi,cc", "format": "json", "rows": 20}
    )
    return _cellosaurus_entries(_get(f"{_CELLOSAURUS}/search/cell-line?{query}"))


@cache
def cellosaurus_entry(accession: str) -> dict[str, Any] | None:
    """One Cellosaurus cell line by accession."""
    entries = _cellosaurus_entries(
        _get(f"{_CELLOSAURUS}/cell-line/{accession}?fields=ac,id,sy,hi,cc&format=json")
    )
    return entries[0] if entries else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    steps = parser.add_subparsers(dest="step", required=True)
    proposing = steps.add_parser("propose")
    building = steps.add_parser("build")
    for step in (proposing, building):
        step.add_argument("--candidates", type=Path, required=True)
        step.add_argument("--model", type=Path, required=True)
    for name in ("gene-effect", "split", "qc-dir", "out"):
        building.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    candidates = read_map(args.candidates)
    if args.step == "propose":
        proposed = propose(
            candidates,
            pd.read_csv(args.model, dtype=str),
            search=cellosaurus_search,
            entry=cellosaurus_entry,
        )
        proposed.to_csv(args.candidates, index=False)
        flagged = proposed.loc[~proposed["review"].isin(("ok", "")), MAP_COLUMNS[:5]]
        print(proposed["review"].str.split(":").str[0].value_counts().to_string())
        print(flagged.to_string(index=False))
        return 0
    if not args.qc_dir.is_dir():
        raise FileNotFoundError(f"{args.qc_dir}: no QC directory")
    qc = {}
    for path in sorted(args.qc_dir.glob("*.json")):
        report = json.loads(path.read_text())
        qc[report["source"]] = report
    payload = build_membership(
        candidates,
        read_models(args.model),
        labelled_ids=read_model_ids(args.gene_effect),
        split=load_geneeffect_split(args.split),
        qc=qc,
    )
    inputs = {
        "candidates": args.candidates,
        "model": args.model,
        "gene_effect": args.gene_effect,
        "split": args.split,
        **{f"qc/{source}": args.qc_dir / f"{source}.json" for source in sorted(qc)},
    }
    payload = {
        **payload,
        "inputs": {
            k: {"path": str(p), "sha256": _sha256(p)} for k, p in inputs.items()
        },
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"{len(payload['labelled'])} labelled, {len(payload['pending'])} pending, "
        f"{len(payload['unlabelled'])} unlabelled, {len(payload['excluded'])} excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
