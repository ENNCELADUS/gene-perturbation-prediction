"""DepMap lines outside the 226 that join the training side through bulk RNA."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from src.data.splits import FixedSplit

SCHEMA_VERSION = "extra-bulk-lines-26q1-v1"


@dataclass(frozen=True)
class ExtraLines:
    """Training-side lines outside the split, and the lines no fit may read.

    Attributes:
        labelled: Lines with 26Q1 GeneEffect and bulk RNA.
        unlabelled: Lines with bulk RNA and no GeneEffect.
        excluded: Lines outside the split sharing a patient with a validation or
            test line, with the reason; never read by any fit.
    """

    labelled: tuple[str, ...]
    unlabelled: tuple[str, ...]
    excluded: Mapping[str, str]


def load_extra_lines(path: Path, split: FixedSplit) -> ExtraLines:
    """Load the membership file; raise on duplicates or any overlap with the split."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    lines = ExtraLines(
        tuple(payload["labelled"]),
        tuple(payload["unlabelled"]),
        dict(payload["excluded"]),
    )
    groups = {
        "labelled": lines.labelled,
        "unlabelled": lines.unlabelled,
        "excluded": tuple(lines.excluded),
        "split": split.all_model_ids,
    }
    for name, values in groups.items():
        if len(set(values)) != len(values):
            raise ValueError(f"{path}: {name} has duplicate ModelIDs")
    for (left, first), (right, second) in combinations(groups.items(), 2):
        overlap = sorted(set(first) & set(second))
        if overlap:
            raise ValueError(f"{path}: {left} and {right} overlap: {overlap[:10]}")
    return lines


__all__ = ["SCHEMA_VERSION", "ExtraLines", "load_extra_lines"]
