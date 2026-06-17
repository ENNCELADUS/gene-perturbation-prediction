# tests/sl_dl_model/test_cli.py
"""Tests for the sl_dl_model CLI (__main__.py)."""

import subprocess
import sys


def test_cli_help_lists_run_cv():
    """--help must list the run-cv subcommand without loading torch/accelerate."""
    out = subprocess.run(
        [sys.executable, "-m", "sl_dl_model", "--help"],
        capture_output=True,
        text=True,
    )
    assert "run-cv" in out.stdout, (
        f"Expected 'run-cv' in stdout.\nstdout: {out.stdout}\nstderr: {out.stderr}"
    )
