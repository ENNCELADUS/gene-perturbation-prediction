from __future__ import annotations

from pathlib import Path


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
