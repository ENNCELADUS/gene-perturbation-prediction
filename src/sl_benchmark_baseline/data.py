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


def load_benchmark(path: Path) -> pd.DataFrame:
    """Load the minimal benchmark CSV and validate its schema.

    Args:
        path: Path to ``k562_SL_benchmark_minimal.csv`` or a compatible CSV.

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
    frame: pd.DataFrame, fold_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice train and test rows for one CV1 fold.

    Args:
        frame: Validated benchmark DataFrame.
        fold_id: CV1 fold id to extract.

    Returns:
        A ``(train_df, test_df)`` tuple, each reset-indexed.
    """
    fold = frame[frame["fold_id"] == fold_id]
    train_df = fold[fold["split_role"] == "train"].reset_index(drop=True)
    test_df = fold[fold["split_role"] == "test"].reset_index(drop=True)
    return train_df, test_df
