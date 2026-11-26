"""SLMGAE inputs for the benchmark genes (the released 6,375 genes cover 21.7% of the
held-out rows). Nodes: the benchmark genes with a GO-BP term, a GO-CC term, a BioGRID
physical edge or a fit SL pair. Support views, in the official order: GO-BP and GO-CC
similarity (Wang's measure from the GAF and OBO files, k = 45 neighbours) and the
BioGRID physical network without every edge between the two genes of a benchmark pair.
The main view (the fit SL pairs) is built by train.py.

    .venv-slmgae/bin/python hpc/slmgae/prepare.py --out data/slmgae/benchmark
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hpc"))
sys.path.insert(0, str(Path(__file__).parent))

import go_similarity as go  # noqa: E402
from sl_common import (  # noqa: E402
    FIT_SIDES,
    HELD_SIDES,
    benchmark_rows,
    drop_pair_edges,
    pair_labels,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/sl_studies/benchmark_v2/labels.parquet"),
    )
    parser.add_argument(
        "--split", type=Path, default=Path("configs/benchmarks/cell_line_sl_split.json")
    )
    parser.add_argument(
        "--gaf", type=Path, default=Path("data/sl_baselines/go/goa_human.gaf.218.gz")
    )
    parser.add_argument(
        "--obo", type=Path, default=Path("data/sl_baselines/go/go-basic.2023-07-27.obo")
    )
    parser.add_argument(
        "--biogrid",
        type=Path,
        default=Path("data/sl_baselines/biogrid/BIOGRID-9606.csv"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=45)
    cli = parser.parse_args()
    start = time.time()

    rows = benchmark_rows(cli.labels, cli.split)
    bench = set(rows["gene_1"]) | set(rows["gene_2"])
    fit = pair_labels(rows, FIT_SIDES)
    held = pair_labels(rows, HELD_SIDES)
    parents = go.read_obo(cli.obo)
    annot = {a: go.read_gaf(cli.gaf, a) for a in go.ASPECTS}
    annot = {
        a: t[t["symbol"].isin(bench) & t["term"].isin(parents)]
        for a, t in annot.items()
    }

    ppi = pd.read_csv(cli.biogrid, usecols=[3, 4, 6])
    ppi.columns = ["gene1", "gene2", "type"]
    ppi = ppi[(ppi["type"] == "physical") & (ppi["gene1"] != ppi["gene2"])]
    ppi = ppi[ppi["gene1"].isin(bench) & ppi["gene2"].isin(bench)]
    n_ppi_before = len(ppi.drop_duplicates())
    ppi = drop_pair_edges(ppi[["gene1", "gene2"]], rows)
    ppi = pd.DataFrame(
        {
            "gene1": np.minimum(ppi["gene1"], ppi["gene2"]),
            "gene2": np.maximum(ppi["gene1"], ppi["gene2"]),
        }
    ).drop_duplicates()

    nodes = sorted(
        set().union(
            *(set(t["symbol"]) for t in annot.values()),
            set(ppi["gene1"]),
            set(ppi["gene2"]),
        )
        | set(fit.loc[fit["label"] == 1, "gene_1"])
        | set(fit.loc[fit["label"] == 1, "gene_2"])
    )
    node = {g: i for i, g in enumerate(nodes)}
    print(f"nodes {len(nodes)} of {len(bench)} benchmark genes", flush=True)

    views, counts = {}, {"nodes": len(nodes), "benchmark_genes": len(bench)}
    for aspect, table in annot.items():
        sim = go.go_similarity(table, parents, nodes)
        graph = go.knn_graph(sim, cli.k)
        views[aspect] = graph
        counts[f"go_{aspect.lower()}_genes_with_term"] = int(table["symbol"].nunique())
        print(
            f"GO-{aspect}: {graph.nnz // 2} edges  {time.time() - start:.0f}s",
            flush=True,
        )
        del sim
    i, j = ppi["gene1"].map(node).to_numpy(), ppi["gene2"].map(node).to_numpy()
    views["PPI"] = pd.DataFrame({"row": np.r_[i, j], "col": np.r_[j, i]})
    counts["edges"] = {
        "GO-BP": views["BP"].nnz // 2,
        "GO-CC": views["CC"].nnz // 2,
        "PPI": len(ppi),
        "PPI_before_pair_edges_removed": n_ppi_before,
    }

    args = {}
    for name, v in views.items():
        row, col = (
            (v.row, v.col)
            if hasattr(v, "row")
            else (v["row"].to_numpy(), v["col"].to_numpy())
        )
        args[f"{name}_row"], args[f"{name}_col"] = (
            row.astype(np.int32),
            col.astype(np.int32),
        )

    pairs = fit.rename(columns={"label": "fit_label"}).merge(
        held.rename(columns={"label": "held_label"}),
        on=["gene_1", "gene_2"],
        how="outer",
    )
    pairs = pairs[pairs["gene_1"].isin(node) & pairs["gene_2"].isin(node)].copy()
    pairs["i"], pairs["j"] = pairs["gene_1"].map(node), pairs["gene_2"].map(node)
    assert (pairs["i"] < pairs["j"]).all()
    pairs["in_fit"] = pairs["fit_label"].notna()
    pairs["in_held"] = pairs["held_label"].notna()
    pairs["fit_label"] = pairs["fit_label"].fillna(-1).astype(int)
    pairs = pairs.drop(columns="held_label")

    cli.out.mkdir(parents=True, exist_ok=True)
    (cli.out / "genes.json").write_text(json.dumps(nodes))
    np.savez_compressed(cli.out / "views.npz", **args)
    pairs.to_parquet(cli.out / "pairs.parquet", index=False)
    counts["pairs"] = {
        "fit": int(pairs["in_fit"].sum()),
        "fit_sl": int((pairs["fit_label"] == 1).sum()),
        "held_only": int((pairs["in_held"] & ~pairs["in_fit"]).sum()),
    }
    counts["seconds"] = time.time() - start
    counts["inputs_sha256"] = {
        p.name: sha256(p)
        for p in (cli.gaf, cli.obo, cli.biogrid, cli.labels, cli.split)
    }
    (cli.out / "prepare.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
