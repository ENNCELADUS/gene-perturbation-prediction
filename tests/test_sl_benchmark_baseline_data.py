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
    assert set(df["split_type"].unique()) == {"CV1"}

    train_df, test_df = fold_split(df, split_type="CV1", fold_id=0)
    assert set(train_df["split_role"].unique()) == {"train"}
    assert set(test_df["split_role"].unique()) == {"test"}
    assert (train_df["fold_id"] == 0).all()
    assert (test_df["fold_id"] == 0).all()
    assert set(train_df["split_type"].unique()) == {"CV1"}
    assert set(test_df["split_type"].unique()) == {"CV1"}


def test_fold_split_preserves_split_type_boundaries(
    synthetic_all_cv_benchmark_csv: Path,
) -> None:
    from sl_benchmark_baseline.data import fold_split, load_benchmark

    df = load_benchmark(synthetic_all_cv_benchmark_csv)
    cv1_train, cv1_test = fold_split(df, split_type="CV1", fold_id=0)
    cv2_train, cv2_test = fold_split(df, split_type="CV2", fold_id=0)

    assert set(cv1_train["split_type"].unique()) == {"CV1"}
    assert set(cv1_test["split_type"].unique()) == {"CV1"}
    assert set(cv2_train["split_type"].unique()) == {"CV2"}
    assert set(cv2_test["split_type"].unique()) == {"CV2"}


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


def test_load_benchmark_rejects_invalid_split_type(tmp_path: Path) -> None:
    from sl_benchmark_baseline.data import load_benchmark

    bad = pd.DataFrame(
        {
            "pair_id": ["P0"],
            "split_type": ["CV4"],
            "fold_id": [0],
            "split_role": ["train"],
            "sl_label": [1],
            "gene_a_symbol": ["A"],
            "gene_b_symbol": ["B"],
            "gene_a_k562_gene_effect": [-0.1],
            "gene_b_k562_gene_effect": [-0.2],
        }
    )
    bad_path = tmp_path / "bad_split_type.csv"
    bad.to_csv(bad_path, index=False)

    try:
        load_benchmark(bad_path)
    except ValueError as error:
        assert "split_type" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for invalid split_type")
