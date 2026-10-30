"""Readers of the three core combinatorial screens, wrapped around the three-screen
benchmark's readers (build_sl_screen_benchmark)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.builders.build_sl_screen_benchmark import (
    RAW_FILES,
    read_harle,
    read_ito,
    read_klingbeil,
)

COLUMNS = (
    "study_id", "list_id", "line", "anchor_gene", "anchor_genotype", "gene_a",
    "gene_b", "score", "fdr", "label", "unlabelled_reason",
)  # fmt: skip


def standard(
    study_id: str, frame: pd.DataFrame, score: str, reason: pd.Series | None = None
) -> pd.DataFrame:
    """A core screen reader's rows in the standard columns (combinatorial)."""
    out = pd.DataFrame(
        {
            "study_id": study_id,
            "list_id": study_id + "|" + frame["screen_line"].astype(str),
            "line": frame["screen_line"].astype(str),
            "anchor_gene": "",
            "anchor_genotype": "",
            "gene_a": frame["screen_gene_a"].astype(str),
            "gene_b": frame["screen_gene_b"].astype(str),
            "score": frame[score].astype(float),
            "fdr": frame["fdr"].astype(float),
            "label": frame["label"].astype("Int8"),
            "unlabelled_reason": "" if reason is None else reason,
        }
    )
    return out[list(COLUMNS)]


def read_Ito2021(raw: Path) -> pd.DataFrame:
    return standard(
        "Ito2021", read_ito(raw / "sl_screens/raw" / RAW_FILES["ito"]), "gemini"
    )


def read_Klingbeil2024(raw: Path) -> pd.DataFrame:
    base = raw / "sl_screens/raw"
    frame = read_klingbeil(
        base / RAW_FILES["klingbeil_gemini"],
        base / RAW_FILES["klingbeil_lfc"],
        base / RAW_FILES["klingbeil_fdr"],
    )
    return standard("Klingbeil2024", frame, "gemini")


def read_Harle2025(raw: Path) -> pd.DataFrame:
    frame = read_harle(raw / "sl_screens/raw" / RAW_FILES["harle"])
    reason = frame["label"].isna().map({True: "single-depleted gene", False: ""})
    return standard("Harle2025", frame, "gi", reason)
