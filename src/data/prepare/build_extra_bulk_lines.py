"""Build configs/benchmarks/extra_bulk_lines_26Q1.json from the pinned 26Q1 files.

uv run python -m src.data.prepare.build_extra_bulk_lines \
    --model data/sl_dependency_v0/raw/depmap/Model.csv \
    --gene-effect data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv \
    --bulk data/sl_dependency_v0/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv \
    --split configs/benchmarks/cell_line_geneeffect_226_split.json \
    --out configs/benchmarks/extra_bulk_lines_26Q1.json
"""  # noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Collection
from pathlib import Path

import pandas as pd

from src.data.depmap import read_model_ids, read_models
from src.data.extra_lines import SCHEMA_VERSION
from src.data.splits import FixedSplit, load_geneeffect_226_split

POLICY = (
    "Lines outside the 226 with 26Q1 bulk RNA; labelled ones also have 26Q1 "
    "GeneEffect. Every line outside the 226 sharing a PatientID with a validation "
    "or test line is excluded from every fit."
)


def build_extra_lines(
    models: pd.DataFrame,
    *,
    labelled_ids: Collection[str],
    bulk_ids: Collection[str],
    split: FixedSplit,
) -> dict:
    """Membership payload: labelled and unlabelled extra lines, and exclusions."""
    ours = set(split.all_model_ids)
    missing = sorted((set(labelled_ids) | set(bulk_ids) | ours) - set(models.index))
    if missing:
        raise ValueError(f"ModelIDs absent from Model.csv: {missing[:10]}")
    held = {models.at[m, "patient_id"]: m for m in (*split.val, *split.test)}
    excluded = {
        m: (
            f"shares PatientID {models.at[m, 'patient_id']} with held-out line "
            f"{held[models.at[m, 'patient_id']]}"
        )
        for m in sorted(set(models.index) - ours)
        if models.at[m, "patient_id"] in held
    }
    usable = set(bulk_ids) - ours - set(excluded)
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "labelled": sorted(usable & set(labelled_ids)),
        "unlabelled": sorted(usable - set(labelled_ids)),
        "excluded": excluded,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "gene-effect", "bulk", "split", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_extra_lines(
        read_models(args.model),
        labelled_ids=read_model_ids(args.gene_effect),
        bulk_ids=read_model_ids(args.bulk),
        split=load_geneeffect_226_split(args.split),
    )
    payload["sources"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in (
            ("model", args.model),
            ("gene_effect", args.gene_effect),
            ("bulk", args.bulk),
        )
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"labelled {len(payload['labelled'])}, unlabelled "
        f"{len(payload['unlabelled'])}, excluded {len(payload['excluded'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
