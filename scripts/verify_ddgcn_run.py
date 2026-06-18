"""Verify exp10 DDGCN run artifacts and print the per-split comparison table.

Usage:
    uv run python scripts/verify_ddgcn_run.py \
        results/experiments/10_k562_sl_pair_ddgcn/run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main(run_dir: str) -> None:
    """Assert acceptance criteria and print the DDGCN per-split metric table.

    Args:
        run_dir: Path to the DDGCN run output directory.
    """
    out = Path(run_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["candidate_gene_count"] == 9471, manifest["candidate_gene_count"]
    assert manifest["dropout"] == 0.5, manifest["dropout"]
    assert manifest["lr"] == 0.01, manifest["lr"]
    print("manifest OK: 9471 genes, dropout=0.5, lr=0.01")

    summary = pd.read_csv(out / "summary.csv")
    wanted = ["auroc", "aupr", "ndcg@10", "map@10"]
    table = summary[summary["metric"].isin(wanted)].pivot_table(
        index="split_type", columns="metric", values="mean"
    )
    print(table.to_string())


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "results/experiments/10_k562_sl_pair_ddgcn/run"
    )
