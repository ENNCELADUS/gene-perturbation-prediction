"""Load and slice the K562 SL-pair benchmark CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "fold_id",
    "split_role",
    "sl_label",
    "gene_a_symbol",
    "gene_b_symbol",
    "gene_a_k562_gene_effect",
    "gene_b_k562_gene_effect",
)
VALID_SPLIT_TYPES: tuple[str, ...] = ("CV1", "CV2", "CV3")


def load_benchmark(path: Path) -> pd.DataFrame:
    """Load the SL-pair benchmark CSV and validate its schema.

    Args:
        path: Path to an all-CV (CV1/CV2/CV3) balanced benchmark CSV, a
            per-split balanced CSV, or any compatible CSV. CSVs without a
            ``split_type`` column are treated as CV1.

    Returns:
        The validated benchmark DataFrame.

    Raises:
        ValueError: If required columns are missing, labels are not binary,
            split roles are unexpected, or gene-effect values contain NaN.
    """
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"benchmark CSV missing columns: {missing}")
    if "split_type" not in frame.columns:
        frame = frame.copy()
        frame["split_type"] = "CV1"
    split_types = set(frame["split_type"].unique())
    invalid_split_types = split_types.difference(VALID_SPLIT_TYPES)
    if invalid_split_types:
        raise ValueError(
            "split_type must be one of "
            f"{VALID_SPLIT_TYPES}, got {sorted(invalid_split_types)}"
        )
    labels = set(frame["sl_label"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"sl_label must be in {{0, 1}}, got {sorted(labels)}")
    roles = set(frame["split_role"].unique())
    if not roles.issubset({"train", "test"}):
        raise ValueError(f"split_role must be train/test, got {sorted(roles)}")
    effect_cols = ["gene_a_k562_gene_effect", "gene_b_k562_gene_effect"]
    if frame[effect_cols].isna().any().any():
        raise ValueError("gene-effect columns must not contain NaN")
    return frame


def fold_split(
    frame: pd.DataFrame, split_type: str, fold_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice train and test rows for one split/fold evaluation unit.

    Args:
        frame: Validated benchmark DataFrame.
        split_type: CV split type to extract.
        fold_id: Fold id to extract.

    Returns:
        A ``(train_df, test_df)`` tuple, each reset-indexed.
    """
    fold = frame[(frame["split_type"] == split_type) & (frame["fold_id"] == fold_id)]
    train_df = fold[fold["split_role"] == "train"].reset_index(drop=True)
    test_df = fold[fold["split_role"] == "test"].reset_index(drop=True)
    return train_df, test_df


def assert_rand_only(frame: pd.DataFrame) -> None:
    """Reject non-``Rand`` negative sampling (leakage guard for selectivity).

    Args:
        frame: Benchmark DataFrame.

    Raises:
        ValueError: If a ``negative_sampling_method`` column is present and any
            value is not ``"Rand"``.
    """
    if "negative_sampling_method" not in frame.columns:
        return
    methods = set(frame["negative_sampling_method"].unique())
    if methods != {"Rand"}:
        raise ValueError(
            "selectivity mode requires Rand negatives only; found "
            f"negative_sampling_method values: {sorted(methods)}"
        )
