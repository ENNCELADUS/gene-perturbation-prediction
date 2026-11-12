"""MiT4SL inputs for the multi-study SL benchmark, transcribing the released tutorials
(third_party/mit4sl/tutorials at commit 68cfb5c):

- contextualized_PPI_construction.ipynb: per line, the genes with TPM >= 400 (DepMap
  26Q1 bulk RNA, TPM = 2^x - 1 of its log2(TPM + 1)), the reference PPI edges that touch
  them, the 32 nearest proteins of each such gene by its adjacency row, and the PPI
  subgraph on those proteins that PrimeKG and the sequence embeddings know.
- cross_cell_line_scenario_constrcution.ipynb: the rows as (PrimeKG index A, PrimeKG
  index B, cell id, label) in ``sl_train_0.csv``; here the training-side lines'
  genetic rows, and the validation and test lines' rows in ``sl_val_0.csv`` and
  ``sl_test_0.csv``.

A benchmark row is kept when both genes are PrimeKG genes with a sequence embedding
and its line has DepMap expression. One row per (line, pair): SL when half or more of
its labels are SL. Every PrimeKG and reference-PPI edge between the two genes of a
benchmark pair is removed, so no label leaks through the graphs (as hpc/kg4sl).

    PYTHONPATH=third_party/mit4sl/src .venv-mit4sl/bin/python hpc/mit4sl/prepare.py \
        --out data/mit4sl/benchmark
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.depmap import read_depmap_matrix  # noqa: E402

OFFICIAL = ROOT / "data/mit4sl/official/MultiOmics_feature"
FIT_SIDES, HELD_SIDES = ("train", "bulk_labelled"), ("val", "test")
TPM_THRESHOLD, K_NEIGHBORS = 400, 32
GENE = "gene/protein"


def pair_keys(a: np.ndarray, b: np.ndarray) -> set[tuple[int, int]]:
    return set(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))


def drop_pair_edges(kgdata, pairs: set[tuple[int, int]]) -> dict[str, int]:
    """Remove every gene-gene edge of the PrimeKG HeteroData whose two ends (in the
    gene type's own index) form a benchmark pair; the edge counts removed by type."""
    removed = {}
    for edge_type in kgdata.edge_types:
        source, _, target = edge_type
        if source != GENE or target != GENE:
            continue
        index = kgdata[edge_type].edge_index
        a, b = index[0].numpy(), index[1].numpy()
        key = np.minimum(a, b).astype(np.int64) * 10**6 + np.maximum(a, b)
        blocked = np.fromiter((x * 10**6 + y for x, y in pairs), np.int64)
        drop = np.isin(key, blocked)
        kgdata[edge_type].edge_index = index[:, torch.from_numpy(~drop)]
        removed["|".join(edge_type)] = int(drop.sum())
    return removed


def cell_graph(
    expressed: list[str],
    ppi: pd.DataFrame,
    name_to_index: dict[str, int],
    sequence_index: set[int],
):
    """The tutorial's build_contextualized_ppi_for_cell for one line's expressed genes:
    (proteins table, torch_geometric Data) or None."""
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors
    from torch_geometric.data import Data

    candidate = ppi[ppi["a_name"].isin(expressed) | ppi["b_name"].isin(expressed)]
    if candidate.empty:
        return None
    proteins = sorted(set(candidate["a_name"]) | set(candidate["b_name"]))
    pid = {p: i for i, p in enumerate(proteins)}
    initial = sorted({pid[g] for g in expressed if g in pid})
    adjacency = csr_matrix(
        (np.ones(len(candidate), int),
         (candidate["a_name"].map(pid), candidate["b_name"].map(pid))),
        shape=(len(proteins), len(proteins)), dtype=int,
    ).toarray()  # fmt: skip
    knn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, len(proteins))).fit(adjacency)
    # one query of all expressed genes: the same neighbours as one query each
    expanded = set(initial) | set(knn.kneighbors(adjacency[initial])[1].ravel())
    table = pd.DataFrame({"protein_name": [proteins[i] for i in sorted(expanded)]})
    table["candidate"] = table["protein_name"].isin(expressed).astype(int)
    table["primekg_index"] = table["protein_name"].map(name_to_index)
    table = table.dropna(subset=["primekg_index"])
    table = table[table["primekg_index"].isin(sequence_index)].reset_index(drop=True)
    if table.empty:
        return None
    table["primekg_index"] = table["primekg_index"].astype(int)
    kept = set(table["protein_name"])
    sub = ppi[ppi["a_name"].isin(kept) & ppi["b_name"].isin(kept)]
    if sub.empty:
        return None
    nodes = sorted(set(sub["protein_a"]) | set(sub["protein_b"]))
    sid = {n: i for i, n in enumerate(nodes)}
    names = dict(zip(sub["protein_a"], sub["a_name"])) | dict(
        zip(sub["protein_b"], sub["b_name"])
    )
    flags = dict(zip(table["protein_name"], table["candidate"]))
    candidate_flag = [int(flags.get(names[n], 0)) for n in nodes]
    graph = Data(
        x=torch.arange(len(nodes)).unsqueeze(1),
        edge_index=torch.tensor(
            np.stack([sub["protein_a"].map(sid), sub["protein_b"].map(sid)])
        ),
        edge_attr=torch.ones((len(sub), 1)),
        candidate=torch.tensor(candidate_flag).unsqueeze(1),
    )
    in_graph = {names[n] for n in nodes}
    return table[table["protein_name"].isin(in_graph)].reset_index(drop=True), graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
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
    parser.add_argument(
        "--expression",
        type=Path,
        default=ROOT
        / "data/sl_dependency_v0/raw/depmap"
        / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
    )
    cli = parser.parse_args()
    nodes = pd.read_csv(OFFICIAL / "kg_data/Primenode.csv")
    nodes = nodes[nodes["node_type"] == GENE]  # other types reuse gene names
    name_to_index = dict(zip(nodes["node_name"].str.upper(), nodes["node_index"]))
    with open(OFFICIAL / "kg_data/node_index_dic.json") as f:
        type_index = {int(k): v for k, v in json.load(f)[GENE].items()}
    with open(OFFICIAL / "seq_data/protein_sequence_embedding.pkl", "rb") as f:
        sequence_index = set(pickle.load(f))
    known = {
        s for s, i in name_to_index.items() if i in sequence_index and i in type_index
    }

    sides = json.loads(cli.split.read_text())
    side = {m: s for s in FIT_SIDES + HELD_SIDES for m in sides[s]}
    rows = pd.read_parquet(
        cli.labels, columns=["model_id", "gene_a", "gene_b", "label", "condition"]
    )
    rows = rows[rows["label"].notna() & rows["model_id"].isin(side)]
    rows = rows.assign(side=rows["model_id"].map(side), label=rows["label"].astype(int))
    expression = read_depmap_matrix(cli.expression)
    # every benchmark pair's edges go, the held-out ones included
    all_pairs = rows[rows["gene_a"].isin(known) & rows["gene_b"].isin(known)]
    blocked_index = pair_keys(
        all_pairs["gene_a"].map(name_to_index).to_numpy(),
        all_pairs["gene_b"].map(name_to_index).to_numpy(),
    )
    keep = (
        rows["gene_a"].isin(known)
        & rows["gene_b"].isin(known)
        & rows["model_id"].isin(expression.index)
        & ((rows["condition"] == "genetic") | rows["side"].isin(HELD_SIDES))
    )
    rows = rows[keep]
    lines = sorted(rows["model_id"].unique())

    ppi = pd.read_excel(OFFICIAL / "cell_line_data/protein-protein_network.xlsx")
    mapping = pd.read_csv(OFFICIAL / "cell_line_data/protein2node.tsv", sep="\t")
    to_name = dict(zip(mapping["node_id"], mapping["symbol"].str.upper()))
    ppi = ppi.assign(
        a_name=ppi["protein_a"].map(to_name), b_name=ppi["protein_b"].map(to_name)
    )
    ppi = ppi.dropna(subset=["a_name", "b_name"])
    a_idx, b_idx = ppi["a_name"].map(name_to_index), ppi["b_name"].map(name_to_index)
    ppi_key = pd.Series(
        list(zip(np.fmin(a_idx, b_idx), np.fmax(a_idx, b_idx))), index=ppi.index
    )
    ppi_blocked = ppi_key.isin(blocked_index)
    ppi = ppi[~ppi_blocked].reset_index(drop=True)

    tpm = 2.0 ** expression.loc[lines] - 1
    tables, graphs, cell_ids, skipped = [], [], {}, []
    for model_id in lines:
        expressed = sorted(tpm.columns[tpm.loc[model_id].to_numpy() >= TPM_THRESHOLD])
        built = cell_graph(expressed, ppi, name_to_index, sequence_index)
        if built is None:
            skipped.append(model_id)
            continue
        cell_ids[model_id] = len(graphs)
        tables.append(built[0].assign(cell_name=model_id, cell_id=cell_ids[model_id]))
        graphs.append(built[1])
        print(
            f"{model_id}: {len(built[0])} proteins, {built[1].num_edges} edges",
            flush=True,
        )
    rows = rows[rows["model_id"].isin(cell_ids)]

    out = cli.out
    (out / "task").mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(
        out / "cell_line_proteins.csv", index=False
    )
    with open(out / "cell_line_subgraphs.pkl", "wb") as f:
        pickle.dump(graphs, f)
    with open(OFFICIAL / "kg_data/kgdata.pkl", "rb") as f:
        kgdata = pickle.load(f)
    blocked_type = {(type_index[a], type_index[b]) for a, b in blocked_index}
    removed = drop_pair_edges(kgdata, pair_keys(*map(np.array, zip(*blocked_type))))
    with open(out / "kgdata.pkl", "wb") as f:
        pickle.dump(kgdata, f)

    files = {"train": FIT_SIDES, "val": ("val",), "test": ("test",)}
    counts = {}
    for name, members in files.items():
        part = rows[rows["side"].isin(members)]
        part = part.groupby(["model_id", "gene_a", "gene_b"], as_index=False)[
            "label"
        ].mean()
        table = pd.DataFrame(
            {"0": part["gene_a"].map(name_to_index),
             "1": part["gene_b"].map(name_to_index),
             "2": part["model_id"].map(cell_ids),
             "3": part["label"].ge(0.5).astype(int)}
        )  # fmt: skip
        table.to_csv(out / "task" / f"sl_{name}_0.csv", index=False)
        part.drop(columns="label").to_csv(
            out / "task" / f"keys_{name}_0.csv", index=False
        )
        counts[name] = {"rows": len(table), "sl": int(table["3"].sum()),
                        "lines": int(part["model_id"].nunique())}  # fmt: skip
    summary = {
        "lines": len(cell_ids),
        "lines_without_a_graph": skipped,
        "rows": counts,
        "benchmark_pairs_blocked": len(blocked_index),
        "ppi_edges_removed": int(ppi_blocked.sum()),
        "kg_edges_removed": removed,
    }
    (out / "prepare.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
