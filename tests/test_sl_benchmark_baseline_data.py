from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_load_benchmark_validates_and_fold_split(synthetic_benchmark_csv: Path) -> None:
    from sl_benchmark_baseline.data import (
        REQUIRED_COLUMNS,
        fold_split,
        load_benchmark,
    )

    df = load_benchmark(synthetic_benchmark_csv)
    for column in REQUIRED_COLUMNS:
        assert column in df.columns
    assert set(df["sl_label"].unique()) == {0, 1}

    train_df, test_df = fold_split(df, fold_id=0)
    assert set(train_df["split_role"].unique()) == {"train"}
    assert set(test_df["split_role"].unique()) == {"test"}
    assert (train_df["fold_id"] == 0).all()
    assert (test_df["fold_id"] == 0).all()


def test_load_benchmark_rejects_missing_column(tmp_path: Path) -> None:
    from sl_benchmark_baseline.data import load_benchmark

    bad = pd.DataFrame({"pair_id": ["P0"], "fold_id": [0]})
    bad_path = tmp_path / "bad.csv"
    bad.to_csv(bad_path, index=False)
    try:
        load_benchmark(bad_path)
    except ValueError as error:
        assert "missing" in str(error).lower()
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing columns")
