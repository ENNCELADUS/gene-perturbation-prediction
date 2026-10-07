"""Build configs/benchmarks/extra_single_cell_lines_26Q1.json from the pinned
single-cell atlas candidates and the 26Q1 files.

uv run python -m src.data.prepare.build_extra_single_cell_lines \
    --candidates configs/benchmarks/single_cell_atlas_candidates.csv \
    --model data/sl_dependency_v0/raw/depmap/Model.csv \
    --gene-effect data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv \
    --split configs/benchmarks/cell_line_geneeffect_226_split.json \
    --out configs/benchmarks/extra_single_cell_lines_26Q1.json
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Sequence
from pathlib import Path

import pandas as pd

from src.data.depmap import read_model_ids, read_models
from src.data.splits import FixedSplit, load_geneeffect_226_split

SCHEMA_VERSION = 1
POLICY = (
    "Lines outside the 226 with public basal single-cell RNA; labelled ones have 26Q1 "
    "GeneEffect. A line in the 226 or sharing a PatientID with a validation or test "
    "line is excluded. Membership is pinned; ingesting the lines is a separate step."
)


def build_extra_single_cell_lines(
    candidates: pd.DataFrame,
    models: pd.DataFrame,
    *,
    labelled_ids: Collection[str],
    split: FixedSplit,
) -> dict:
    """Membership payload: labelled and unlabelled candidate lines, the excluded
    ones with their reason, and each usable line's sources."""
    missing = sorted(set(candidates["model_id"]) - set(models.index))
    if missing:
        raise ValueError(f"ModelIDs absent from Model.csv: {missing[:10]}")
    ours = set(split.all_model_ids)
    held = {models.at[m, "patient_id"]: m for m in (*split.val, *split.test)}
    excluded = {}
    for model_id in sorted(set(candidates["model_id"])):
        patient = models.at[model_id, "patient_id"]
        if model_id in ours:
            excluded[model_id] = "already one of the 226"
        elif patient in held:
            excluded[model_id] = (
                f"shares PatientID {patient} with held-out line {held[patient]}"
            )
    usable = set(candidates["model_id"]) - set(excluded)
    sources = candidates.groupby("model_id")["source"].apply(lambda s: sorted(set(s)))
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "labelled": sorted(usable & set(labelled_ids)),
        "unlabelled": sorted(usable - set(labelled_ids)),
        "excluded": excluded,
        "sources": {m: sources[m] for m in sorted(usable)},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("candidates", "model", "gene-effect", "split", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_extra_single_cell_lines(
        pd.read_csv(args.candidates, dtype=str),
        read_models(args.model),
        labelled_ids=read_model_ids(args.gene_effect),
        split=load_geneeffect_226_split(args.split),
    )
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"{len(payload['labelled'])} labelled, {len(payload['unlabelled'])} "
        f"unlabelled, {len(payload['excluded'])} excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
