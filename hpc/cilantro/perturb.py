"""Released notebook 2_perturber, first half: Geneformer in-silico deletion of every
gene in every line.

One process runs one shard (genes i::n) on one GPU, with its own scratch directories,
because the released perturber clears its scratch directory before every gene. Per-gene
outputs land in one shared directory and finished genes are skipped, so shards can be
relaunched to resume. --limit runs the first N genes of the shard (throughput).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    GENEFORMER_MODEL,
    OFFICIAL_DATA,
    SYMBOL_MAP,
    configure,
    paths,
    run_config,
    seed_everything,
)


def most_paired_genes(genes: list[str], n: int) -> list[str]:
    """Smoke runs only: the n genes with the most SynLethDB SL or non-SL pairs."""
    import pandas as pd

    with open(SYMBOL_MAP, "rb") as f:
        to_ensembl = pickle.load(f)
    pairs = pd.concat(
        [
            pd.read_csv(OFFICIAL_DATA / name)
            for name in ("Human_SL.csv", "Human_nonSL.csv")
        ]
    )
    degree = (
        pd.concat([pairs["n1.name"], pairs["n2.name"]]).map(to_ensembl).value_counts()
    )
    universe = set(genes)
    return [g for g in degree.index if g in universe][:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    p = paths(args.run_dir)
    shard_dir = p["perturb"] / "shards" / f"{args.shard:02d}"
    configure(args.run_dir, res_dir=shard_dir / "res")
    seed_everything(0)
    for sub in ("res", "isp_out", "tmp"):
        (shard_dir / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("subtracted_embs", "perturbed_embs"):
        (p["perturb"] / sub).mkdir(parents=True, exist_ok=True)

    with open(p["genes_to_perturb"], "rb") as f:
        genes = pickle.load(f)
    subset = run_config(args.run_dir)["gene_subset"]
    if subset is not None:
        genes = most_paired_genes(genes, subset)
    genes = genes[args.shard :: args.num_shards]
    if args.limit is not None:
        genes = genes[: args.limit]
    gene_list = shard_dir / "genes.pkl"
    with open(gene_list, "wb") as f:
        pickle.dump(genes, f)

    from viability_perturber import via_perturber

    vp = via_perturber(
        str(p["tokenized"]),
        str(GENEFORMER_MODEL),
        0,
        "Pretrained",
        str(p["perturb"]) + "/",
        "cell",
        temp_files=str(shard_dir / "tmp"),
        out_dir=str(shard_dir / "isp_out") + "/",
        hvg_viable_file=str(gene_list),
    )
    start = time.time()
    vp.perturb_all_genes()
    elapsed = time.time() - start
    done = sum((p["perturb"] / "subtracted_embs" / f"{g}.csv").exists() for g in genes)
    record = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "genes": len(genes),
        "genes_written": done,
        "seconds": round(elapsed, 1),
        "limit": args.limit,
    }
    (shard_dir / ("throughput.json" if args.limit else "done.json")).write_text(
        json.dumps(record) + "\n"
    )
    print(json.dumps(record))


if __name__ == "__main__":
    main()
