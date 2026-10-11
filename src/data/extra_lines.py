"""DepMap lines outside the 226 that join the training side: through bulk RNA, or
through basal single cells."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from src.data.splits import FixedSplit

SCHEMA_VERSION = "extra-bulk-lines-26q1-v1"
SINGLE_CELL_SCHEMA_VERSION = "extra-single-cell-lines-26q1-v2"


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


@dataclass(frozen=True)
class ExtraSingleCellLines:
    """Lines outside the split whose basal single cells are, or will be, built.

    Attributes:
        labelled: Lines with 26Q1 GeneEffect whose raw-UMI file is built, each
            with its record (source, accession, condition, cells, arms, ...).
        unlabelled: Chosen lines without 26Q1 GeneEffect; never built.
        pending: Chosen labelled lines whose source is not built yet, with the
            reason.
        excluded: Chosen lines sharing a patient with a validation or test line,
            with the reason; never built.
    """

    labelled: Mapping[str, Mapping[str, Any]]
    unlabelled: tuple[str, ...]
    pending: Mapping[str, str]
    excluded: Mapping[str, str]


def _check_disjoint(path: Path, groups: Mapping[str, Sequence[str]]) -> None:
    for name, values in groups.items():
        if len(set(values)) != len(values):
            raise ValueError(f"{path}: {name} has duplicate ModelIDs")
    for (left, first), (right, second) in combinations(groups.items(), 2):
        overlap = sorted(set(first) & set(second))
        if overlap:
            raise ValueError(f"{path}: {left} and {right} overlap: {overlap[:10]}")


def _payload(path: Path, schema_version: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"{path}: schema_version must be {schema_version}")
    return payload


def load_extra_lines(path: Path, split: FixedSplit) -> ExtraLines:
    """Load the membership file; raise on duplicates or any overlap with the split."""
    payload = _payload(path, SCHEMA_VERSION)
    lines = ExtraLines(
        tuple(payload["labelled"]),
        tuple(payload["unlabelled"]),
        dict(payload["excluded"]),
    )
    _check_disjoint(
        path,
        {
            "labelled": lines.labelled,
            "unlabelled": lines.unlabelled,
            "excluded": tuple(lines.excluded),
            "split": split.all_model_ids,
        },
    )
    return lines


def load_extra_single_cell_lines(path: Path, split: FixedSplit) -> ExtraSingleCellLines:
    """Load the single-cell membership file; raise on duplicates or any overlap
    between its sets or with the split."""
    payload = _payload(path, SINGLE_CELL_SCHEMA_VERSION)
    lines = ExtraSingleCellLines(
        dict(payload["labelled"]),
        tuple(payload["unlabelled"]),
        dict(payload["pending"]),
        dict(payload["excluded"]),
    )
    _check_disjoint(
        path,
        {
            "labelled": tuple(lines.labelled),
            "unlabelled": lines.unlabelled,
            "pending": tuple(lines.pending),
            "excluded": tuple(lines.excluded),
            "split": split.all_model_ids,
        },
    )
    return lines


__all__ = [
    "SCHEMA_VERSION",
    "SINGLE_CELL_SCHEMA_VERSION",
    "ExtraLines",
    "ExtraSingleCellLines",
    "load_extra_lines",
    "load_extra_single_cell_lines",
]
