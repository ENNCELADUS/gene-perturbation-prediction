from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_cli_main_runs_and_writes_summary(
    synthetic_benchmark_csv: Path, tmp_path: Path
) -> None:
    from sl_benchmark_baseline.__main__ import main

    output_dir = tmp_path / "cli_run"
    exit_code = main(
        [
            "--input-csv",
            str(synthetic_benchmark_csv),
            "--output-dir",
            str(output_dir),
            "--folds",
            "0",
            "1",
            "--ranking-k",
            "2",
            "5",
        ]
    )
    assert exit_code == 0
    assert (output_dir / "summary.csv").exists()


def test_cli_main_runs_requested_split_type_only(
    synthetic_all_cv_benchmark_csv: Path, tmp_path: Path
) -> None:
    from sl_benchmark_baseline.__main__ import main

    output_dir = tmp_path / "cli_cv2_run"
    exit_code = main(
        [
            "--input-csv",
            str(synthetic_all_cv_benchmark_csv),
            "--output-dir",
            str(output_dir),
            "--split-types",
            "CV2",
            "--folds",
            "0",
            "--ranking-k",
            "2",
            "4",
        ]
    )

    assert exit_code == 0
    summary = pd.read_csv(output_dir / "summary.csv")
    assert set(summary["split_type"].unique()) == {"CV2"}
