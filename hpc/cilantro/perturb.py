"""Released notebook 2_perturber, first half: Geneformer in-silico deletion of every
gene in every line.

One process runs one shard (genes i::n) on one GPU. Per-gene outputs land in one
shared directory and finished genes are skipped, so shards can be relaunched to
resume. The released skip test is only that the file exists, so before the shard runs,
a gene's two output files are deleted (and recomputed) unless each has one complete
row per line whose tokens include the gene. --limit runs the first N genes of the
shard (throughput).

By default the deletion is computed by ``fast_perturb.Deleter``, the harness's direct
reproduction of the released per-gene computation (same model, batches, pooling and
CSV assembly; byte-identical files, see its docstring and the design's §4.4).
--released runs the released ``via_perturber`` instead, with its own scratch
directories, because it clears its scratch directory before every gene.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DICTIONARIES,
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


def expected_rows(run_dir: Path) -> dict[str, int]:
    """Per gene, the lines whose tokens include it: the rows of a complete output."""
    from datasets import load_from_disk

    with open(DICTIONARIES[run_config(run_dir)["dictionary"]]["token"], "rb") as f:
        token_of = pickle.load(f)
    lines_with = Counter()
    for tokens in load_from_disk(str(paths(run_dir)["tokenized"]))["input_ids"]:
        lines_with.update(set(tokens))
    return {gene: lines_with[token] for gene, token in token_of.items()}


def complete(path: Path, rows: int) -> bool:
    """A header plus ``rows`` rows, every one with the header's field count."""
    if not path.exists():
        return False
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        return False
    lines = data[:-1].split(b"\n")
    width = lines[0].count(b",")
    return len(lines) == rows + 1 and all(line.count(b",") == width for line in lines)


def remove_incomplete(perturb_dir: Path, genes: list[str], rows: dict[str, int]):
    removed = []
    for gene in genes:
        files = [
            perturb_dir / s / f"{gene}.csv"
            for s in ("subtracted_embs", "perturbed_embs")
        ]
        if any(f.exists() for f in files) and not all(
            complete(f, rows[gene]) for f in files
        ):
            for f in files:
                f.unlink(missing_ok=True)
            removed.append(gene)
    return removed


def released_perturber(run_dir: Path, shard_dir: Path, gene_list: Path):
    """The released ``via_perturber`` over the shard's gene list."""
    p = paths(run_dir)
    for sub in ("res", "isp_out", "tmp"):
        (shard_dir / sub).mkdir(parents=True, exist_ok=True)
    from viability_perturber import via_perturber

    return via_perturber(
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


def fast_deleter(run_dir: Path):
    """The harness's direct reproduction of the released per-gene deletion."""
    from fast_perturb import Deleter

    p = paths(run_dir)
    return Deleter(
        dataset_path=p["tokenized"],
        model_dir=GENEFORMER_MODEL,
        token_dictionary=DICTIONARIES[run_config(run_dir)["dictionary"]]["token"],
        ensembl_mapping=SYMBOL_MAP,
        gene_effect_csv=p["gene_effect"],
        out_dir=p["perturb"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--released", action="store_true")
    args = parser.parse_args()
    p = paths(args.run_dir)
    shard_dir = p["perturb"] / "shards" / f"{args.shard:02d}"
    configure(args.run_dir, res_dir=shard_dir / "res")
    seed_everything(0)
    shard_dir.mkdir(parents=True, exist_ok=True)
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
    removed = remove_incomplete(p["perturb"], genes, expected_rows(args.run_dir))
    print(f"incomplete outputs removed for recomputation: {removed}")
    gene_list = shard_dir / "genes.pkl"
    with open(gene_list, "wb") as f:
        pickle.dump(genes, f)

    start = time.time()
    if args.released:
        released_perturber(args.run_dir, shard_dir, gene_list).perturb_all_genes()
    else:
        fast_deleter(args.run_dir).perturb_all_genes(genes)
    elapsed = time.time() - start
    done = sum((p["perturb"] / "subtracted_embs" / f"{g}.csv").exists() for g in genes)
    record = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "genes": len(genes),
        "genes_written": done,
        "incomplete_removed_at_start": removed,
        "seconds": round(elapsed, 1),
        "limit": args.limit,
        "path": "released" if args.released else "fast",
    }
    (shard_dir / ("throughput.json" if args.limit else "done.json")).write_text(
        json.dumps(record) + "\n"
    )
    print(json.dumps(record))


if __name__ == "__main__":
    main()
