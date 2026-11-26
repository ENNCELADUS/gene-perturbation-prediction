"""Build the PrimeKG neighbor tables used by src/preprocess_datasets.py.

For each cell line, every gene pair in the fold tables gets one row with the
PrimeKG neighbor ids of each gene (all 1-hop neighbors, padded with 2-hop
neighbors up to at least ``MIN_COUNT``) and their shared neighbors. Writes
``{cell_line}_neighbors.tsv`` under ``PROMPTCASL_KG_NEIGHBORS``
(default ``data/KG_neighbors``).

Run from the repository root:

    python -m scripts.build_kg_neighbors
"""

import argparse
import os
import random

import pandas as pd
from pykeen.datasets import PrimeKG
from tqdm import tqdm

from src.config import CELL_LINES, FOLDS_DIR, KG_NEIGHBOR_DIR

SEED = 42
MIN_COUNT = 100


def build_adjacency(triples):
    adj = {}
    for h, _, t in triples.tolist():
        adj.setdefault(h, set()).add(t)
        adj.setdefault(t, set()).add(h)
    return adj


def n_hop(adj, start, n, exclude):
    seen = {start} | set(exclude)
    frontier = {start}
    out = set()
    for _ in range(n):
        nxt = set()
        for e in frontier:
            nxt |= adj.get(e, set())
        nxt -= seen
        out |= nxt
        seen |= nxt
        frontier = nxt
    return out


def pad_neighbors(adj, entity_id, exclude, rng, min_count=MIN_COUNT):
    """Return all 1-hop neighbors, or pad with 2-hop neighbors to ``min_count``."""
    one_hop = adj.get(entity_id, set()) - {entity_id} - set(exclude)
    neighbors = list(one_hop)
    if len(neighbors) >= min_count:
        return neighbors
    two_hop = list(n_hop(adj, entity_id, 2, exclude) - one_hop)
    rng.shuffle(two_hop)
    neighbors += two_hop[: min_count - len(neighbors)]
    return neighbors


def load_pairs(cell_line):
    frames = []
    for name in (f"{cell_line}_filtered_NP1_F.csv", f"{cell_line}_full.csv"):
        path = os.path.join(FOLDS_DIR, name)
        if os.path.exists(path):
            frames.append(pd.read_csv(path)[["gene_1", "gene_2"]])
    return pd.concat(frames).drop_duplicates() if frames else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_count", type=int, default=MIN_COUNT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    dataset = PrimeKG(create_inverse_triples=True)
    entity_to_id = dataset.training.entity_to_id
    adj = build_adjacency(dataset.training.mapped_triples)
    os.makedirs(KG_NEIGHBOR_DIR, exist_ok=True)

    for cl in CELL_LINES:
        pairs = load_pairs(cl)
        if pairs is None:
            print(f"[skip] {cl}: no fold file")
            continue
        out_path = os.path.join(KG_NEIGHBOR_DIR, f"{cl}_neighbors.tsv")
        n_written = 0
        with open(out_path, "w") as fout:
            fout.write("geneA\tgeneA_neighbors\tgeneB\tgeneB_neighbors\tshared_neighbors\n")
            for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc=cl):
                a = entity_to_id.get(row["gene_1"])
                b = entity_to_id.get(row["gene_2"])
                if a is None or b is None:
                    continue
                na = pad_neighbors(adj, a, {b}, rng, args.min_count)
                nb = pad_neighbors(adj, b, {a}, rng, args.min_count)
                shared = set(na) & set(nb)
                fout.write(
                    f"{a}\t{','.join(map(str, na))}\t{b}\t{','.join(map(str, nb))}\t"
                    f"{','.join(map(str, shared))}\n"
                )
                n_written += 1
        print(f"{cl}: {n_written} pairs -> {out_path}")


if __name__ == "__main__":
    main()
