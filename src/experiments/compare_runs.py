"""Paired line bootstrap of selective Spearman between two runs of one split.

``python -m src.experiments.compare_runs RUN_A RUN_B --split val`` reads
``evaluation/<split>/predictions.parquet`` of two revision run directories and prints
A minus B with its 95% interval (1,000 resamples, seed 0) over the selective genes
of A's ``best.pt``. It reports; the reader decides.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.eval import metrics
from src.experiments.revision import BOOTSTRAP_REPEATS, BOOTSTRAP_SEED
from src.training.checkpoint import load_checkpoint


def compare(run_a: Path, run_b: Path, split: str) -> dict:
    selective = load_checkpoint(run_a / "train" / "best.pt")["preprocessing"][
        "selective_genes"
    ]
    frames = [
        pd.read_parquet(run / "evaluation" / split / "predictions.parquet")
        for run in (run_a, run_b)
    ]
    return metrics.paired_line_bootstrap(
        *frames, selective, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    args = parser.parse_args(argv)
    result = compare(args.run_a, args.run_b, args.split)
    low, high = result["interval"]
    print(
        f"{args.run_a.name} minus {args.run_b.name}, {args.split}: "
        f"{result['difference']:.4f} [{low:.4f}, {high:.4f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
