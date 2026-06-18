from __future__ import annotations

import subprocess
import sys


def test_help_works_without_torch() -> None:
    # --help must not require torch/cuda; exits 0.
    result = subprocess.run(
        [sys.executable, "-m", "ddgcn", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "run-cv" in result.stdout


def test_parser_run_cv_requires_config() -> None:
    from ddgcn.__main__ import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["run-cv", "--config", "x.yaml"])
    assert args.command == "run-cv"
    assert str(args.config) == "x.yaml"
    assert args.split_type is None


def test_parser_accepts_split_type_override() -> None:
    from ddgcn.__main__ import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["run-cv", "--config", "x.yaml", "--split-type", "CV2"]
    )
    assert args.split_type == "CV2"
