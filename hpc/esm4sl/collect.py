"""Score table of the SL benchmark from the official ESM4SL test pass: reads the newest
``csv_log/version_<n>/test_logits.csv`` (gene1, gene2, probs; gene ids), maps the ids
to symbols (``genes.csv`` of prepare.py), writes ``scores.parquet``, ``metrics.json``.

    .venv-esm4sl/bin/python hpc/esm4sl/collect.py --run outputs/sl/esm4sl/seed0 \
        --out outputs/sl/esm4sl/seed0
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hpc" / "common"))
import sl_common as common  # noqa: E402


def scores_from_logits(logits: pd.DataFrame, genes: pd.DataFrame) -> pd.Series:
    """Probability indexed by the sorted symbol pair (gene_1, gene_2)."""
    symbol = genes.set_index("id")["symbol"]
    first, second = logits["gene1"].map(symbol), logits["gene2"].map(symbol)
    index = pd.MultiIndex.from_arrays(
        [np.minimum(first, second), np.maximum(first, second)]
    )
    scores = pd.Series(logits["probs"].to_numpy(), index=index)
    return scores[~scores.index.duplicated()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/esm4sl")
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data/sl_studies/benchmark_v2/labels.parquet",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "configs/benchmarks/cell_line_sl_split.json",
    )
    args = parser.parse_args()

    logits_path = max(
        args.run.glob("csv_log/version_*/test_logits.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    scores = scores_from_logits(
        pd.read_csv(logits_path), pd.read_csv(args.data / "genes.csv")
    )
    rows = common.benchmark_rows(args.labels, args.split)
    report = common.write_pair_scores(args.out, rows, scores)
    prepared = json.loads((args.data / "metrics.json").read_text())
    metrics = {
        **report,
        "test_logits": str(logits_path),
        "test_logits_sha256": hashlib.sha256(logits_path.read_bytes()).hexdigest(),
        "best_checkpoints": sorted(p.name for p in args.run.glob("best_ckpts/*.ckpt")),
        "official_config": "esm4sl/configuration/attn/new.yaml",
        "inputs": prepared,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
