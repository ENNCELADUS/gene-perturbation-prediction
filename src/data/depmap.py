"""Pinned DepMap matrices as ModelID x gene-symbol frames, and Model.csv."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

#: DepMap header ``SYMBOL (Entrez)``; some Omics columns carry ``(Unknown)``.
_COLUMN = re.compile(r"^(?P<symbol>\S+) \((?:\d+|Unknown)\)$")
#: Profile metadata of DepMap Omics matrices (one row per sequencing profile).
_OMICS_METADATA = frozenset(
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
                if column in _OMICS_METADATA or str(column).startswith("Unnamed")
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


def read_models(path: Path) -> pd.DataFrame:
    """Model.csv: ``patient_id`` (a missing one becomes the ModelID) and ``lineage``."""
    frame = pd.read_csv(
        path, usecols=["ModelID", "PatientID", "OncotreeLineage"], dtype=str
    ).set_index("ModelID", verify_integrity=True)
    frame.index.name = "model_id"
    patient = frame["PatientID"].fillna(pd.Series(frame.index, index=frame.index))
    return pd.DataFrame({"patient_id": patient, "lineage": frame["OncotreeLineage"]})
