"""Pinned DepMap matrices as ModelID x gene-symbol frames, Model.csv, and the
reviewed identity table of the lines without a ModelID."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

#: The reviewed identity table of the SL lines without a DepMap ModelID.
LINE_IDENTITIES = Path("configs/benchmarks/sl_line_identities.csv")
IDENTITY_COLUMNS = (
    "line_id", "name", "spellings", "cellosaurus", "parent", "patient_id", "lineage",
    "non_cancer", "note",
)  # fmt: skip
#: DepMap header ``SYMBOL (Entrez)``; some Omics columns carry ``(Unknown)``.
_COLUMN = re.compile(r"^(?P<symbol>\S+) \((?:\d+|Unknown)\)$")
#: Profile metadata of DepMap Omics matrices (one row per sequencing profile).
OMICS_METADATA = frozenset(
    {
        "SequencingID",
        "ModelConditionID",
        "ModelID",
        "IsDefaultEntryForMC",
        "IsDefaultEntryForModel",
    }
)


def depmap_symbol(column: str) -> str:
    """Upper-case symbol of a DepMap ``SYMBOL (Entrez)`` column."""
    match = _COLUMN.fullmatch(str(column).strip())
    if match is None:
        raise ValueError(f"not a DepMap gene column: {column!r}")
    return match.group("symbol").upper()


def read_depmap_matrix(path: Path) -> pd.DataFrame:
    """A DepMap matrix as float64, indexed by ModelID, columns upper-case symbols.

    Omics matrices (a ``ModelID`` column, one row per profile) keep each model's
    default entry; CRISPR matrices are indexed by ModelID in their first column.
    """
    frame = pd.read_csv(path)
    if "ModelID" in frame.columns:
        default = frame["IsDefaultEntryForModel"].astype(str).str.lower() == "yes"
        frame = frame.loc[default].set_index("ModelID")
        frame = frame.drop(
            columns=[
                column
                for column in frame.columns
                if column in OMICS_METADATA or str(column).startswith("Unnamed")
            ]
        )
    else:
        frame = frame.set_index(frame.columns[0])
    frame.index = frame.index.astype(str)
    frame.index.name = "model_id"
    if not frame.index.is_unique:
        repeated = sorted(frame.index[frame.index.duplicated()].unique())
        raise ValueError(f"{path}: duplicate ModelIDs {repeated[:10]}")
    symbols = [depmap_symbol(column) for column in frame.columns]
    repeated = sorted(s for s, n in Counter(symbols).items() if n > 1)
    if repeated:
        raise ValueError(f"{path}: columns map to duplicate symbols {repeated[:10]}")
    frame.columns = pd.Index(symbols, name="gene_symbol")
    return frame.astype(np.float64)


def read_model_ids(path: Path) -> frozenset[str]:
    """ModelIDs of a matrix (an Omics matrix's default entries), values unread."""
    header = pd.read_csv(path, nrows=0).columns
    if "ModelID" in header:
        frame = pd.read_csv(
            path, usecols=["ModelID", "IsDefaultEntryForModel"], dtype=str
        )
        default = frame["IsDefaultEntryForModel"].str.lower() == "yes"
        return frozenset(frame.loc[default, "ModelID"])
    return frozenset(pd.read_csv(path, usecols=[0], dtype=str).iloc[:, 0])


def read_line_identities(path: Path = LINE_IDENTITIES) -> pd.DataFrame:
    """The reviewed identity table of the SL lines without a DepMap ModelID, by
    ``line_id`` (Cellosaurus accession, else ``SL-<study>-<line>``)."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != IDENTITY_COLUMNS:
        raise ValueError(f"{path}: columns must be {IDENTITY_COLUMNS}")
    if (frame[["line_id", "patient_id", "lineage"]] == "").any().any():
        raise ValueError(f"{path}: a row has no line_id, patient_id or lineage")
    if set(frame["non_cancer"]) - {"true", "false"}:
        raise ValueError(f"{path}: non_cancer must be true or false")
    return frame.set_index("line_id", verify_integrity=True)


def identity_spellings(identities: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(study_id, line as the study's reader spells it) -> line_id."""
    out = {}
    for line_id, field in identities["spellings"].items():
        for entry in filter(None, (part.strip() for part in field.split(";"))):
            study, spelling = entry.split(":", 1)
            if (study, spelling) in out:
                raise ValueError(f"spelling {study}:{spelling} has two line_ids")
            out[(study, spelling)] = line_id
    return out


def read_models(path: Path, identities: Path | None = LINE_IDENTITIES) -> pd.DataFrame:
    """Model.csv: ``patient_id`` (a missing one becomes the ModelID) and ``lineage``,
    followed by the rows of the line identity table (lines without a ModelID)."""
    frame = pd.read_csv(
        path, usecols=["ModelID", "PatientID", "OncotreeLineage"], dtype=str
    ).set_index("ModelID", verify_integrity=True)
    frame.index.name = "model_id"
    patient = frame["PatientID"].fillna(pd.Series(frame.index, index=frame.index))
    models = pd.DataFrame({"patient_id": patient, "lineage": frame["OncotreeLineage"]})
    if identities is None:
        return models
    extra = read_line_identities(identities)[["patient_id", "lineage"]]
    extra.index.name = "model_id"
    return pd.concat([models, extra], verify_integrity=True)
