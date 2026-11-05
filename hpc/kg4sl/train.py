"""KG4SL (Wang et al. 2021; official code at third_party/kg4sl, pinned and patched by
setup.sh) on the multi-study SL benchmark. KG4SL scores a gene pair from a knowledge
graph; it does not read the line.

The graph is the official SynLethKG without its SL, non-SL and SR edges (as the official
preprocessing.py), and also without every edge between the two genes of a benchmark pair
(the official step removes the edges between its SL pairs). A benchmark gene is a graph
gene of the same symbol; a symbol with more than one gene entity takes the one with the
most edges. The model, its neighbour sampling and its settings are the official ones. It
trains on the training lines' genetic pairs: one sample per ordered pair, SL when at
least half of its rows are SL, and as many tested non-SL pairs as SL pairs (the official
data are SynLethDB SL pairs and as many random pairs). The epoch is the one with the
best AUROC on the validation lines' pairs (the official rule, read on our validation
lines), with the official early stop on the training loss. Output: one score per held-
out (gene_a, gene_b); a pair with a gene outside the graph gets the training SL rate, so
it ties.

    PYTHONPATH=third_party/kg4sl/src .venv-kg4sl/bin/python hpc/kg4sl/train.py \
        --out outputs/sl/kg4sl/seed0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from sklearn.metrics import roc_auc_score

from data_loader import construct_adj, construct_kg2dict
from model import KG4SL

tf.disable_v2_behavior()
#: Relations the official preprocessing removes: SL, SR and non-SL.
LABEL_RELATIONS = {"SL_GsG", "SR_GsrG", "NONSL_GnsG"}


def official_args(epochs: int) -> argparse.Namespace:
    """The defaults of the official main.py."""
    return argparse.Namespace(
        n_epochs=epochs,
        neighbor_sample_size=64,
        dim=256,
        n_hop=1,
        batch_size=512,
        l2_weight=0.0039,
        lr=2e-3,
    )


def benchmark_rows(labels: Path, split: Path) -> pd.DataFrame:
    side = {
        m: s
        for s, members in json.loads(split.read_text()).items()
        if s in ("train", "val", "test")
        for m in members
    }
    rows = pd.read_parquet(
        labels, columns=["model_id", "gene_a", "gene_b", "label", "condition"]
    )
    rows = rows[rows["label"].notna() & rows["model_id"].isin(side)]
    return rows.assign(
        split=rows["model_id"].map(side), label=rows["label"].astype(int)
    )


def graph_genes(kg: pd.DataFrame, names: pd.Series) -> pd.Series:
    """Symbol -> gene entity: the entities on a gene end of a relation (``X_GpBP``:
    the head; ``X_AeG``: the tail)."""
    kind = kg["r"].str.split("_").str[1]
    gene = pd.concat(
        [kg.loc[kind.str[0] == "G", "a"], kg.loc[kind.str[-1] == "G", "b"]]
    ).unique()
    degree = pd.concat([kg["a"], kg["b"]]).value_counts()
    table = pd.DataFrame(
        {
            "entity": gene,
            "symbol": names.reindex(gene).to_numpy(),
            "degree": degree.reindex(gene).to_numpy(),
        }
    )
    table = table.dropna(subset=["symbol"]).sort_values(
        ["symbol", "degree", "entity"], ascending=[True, False, True]
    )
    return table.drop_duplicates("symbol").set_index("symbol")["entity"]


def pair_key(a: pd.Series, b: pd.Series) -> pd.Series:
    low, high = np.minimum(a, b), np.maximum(a, b)
    return pd.Series(
        low.astype(np.int64) * 10**7 + high.astype(np.int64), index=a.index
    )


def score(sess, model, args, pairs: np.ndarray) -> np.ndarray:
    """The official model takes whole batches only: pad the last one."""
    n, size = len(pairs), args.batch_size
    padded = np.concatenate([pairs, np.repeat(pairs[:1], (-n) % size, axis=0)])
    out = [
        sess.run(
            model.scores_normalized,
            {
                model.nodea_indices: padded[i : i + size, 0],
                model.nodeb_indices: padded[i : i + size, 1],
            },
        )
        for i in range(0, len(padded), size)
    ]
    return np.concatenate(out)[:n]


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
    parser.add_argument("--kg", type=Path, default=Path("third_party/kg4sl/data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    cli = parser.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)
    args = official_args(cli.epochs)
    np.random.seed(cli.seed)
    tf.set_random_seed(cli.seed)

    kg = pd.read_csv(cli.kg / "kg_triplet.csv", encoding="utf-8-sig")
    kg.columns = ["a", "r", "b"]
    names = pd.read_csv(cli.kg / "dbid2name.csv", index_col=0).set_index("_id")["name"]
    kg = kg[~kg["r"].isin(LABEL_RELATIONS)]
    symbol_entity = graph_genes(kg, names)

    rows = benchmark_rows(cli.labels, cli.split)
    rows["a"] = rows["gene_a"].map(symbol_entity)
    rows["b"] = rows["gene_b"].map(symbol_entity)
    mapped = rows.dropna(subset=["a", "b"])
    benchmark_pairs = set(pair_key(mapped["a"], mapped["b"]))
    within = pair_key(kg["a"], kg["b"]).isin(benchmark_pairs)
    kg = kg[~within]

    entities = pd.Index(np.unique(np.concatenate([kg["a"], kg["b"]])))
    relations = pd.Index(sorted(kg["r"].unique()))
    kg2id = np.stack(
        [
            entities.get_indexer(kg["a"]),
            relations.get_indexer(kg["r"]),
            entities.get_indexer(kg["b"]),
        ],
        axis=1,
    ).astype(np.int64)
    adj_entity, adj_relation = construct_adj(
        args, construct_kg2dict(kg2id), len(entities)
    )

    # a gene that kept no edge has no entity: its pairs are not covered
    rows["ia"] = entities.get_indexer(rows["a"].fillna(-1).astype(np.int64))
    rows["ib"] = entities.get_indexer(rows["b"].fillna(-1).astype(np.int64))
    rows["covered"] = (rows["ia"] >= 0) & (rows["ib"] >= 0)

    def labelled_pairs(side: str) -> np.ndarray:
        chosen = rows[
            (rows["split"] == side) & (rows["condition"] == "genetic") & rows["covered"]
        ]
        rate = chosen.groupby(["ia", "ib"])["label"].mean()
        return np.stack(
            [
                rate.index.get_level_values(0),
                rate.index.get_level_values(1),
                (rate.to_numpy() >= 0.5).astype(int),
            ],
            axis=1,
        ).astype(np.int64)

    train, val = labelled_pairs("train"), labelled_pairs("val")
    # official ratio, one non-SL pair per SL pair; with all tested pairs (1% SL) the
    # official L2 shrinks every embedding and the score is one constant
    positive, negative = train[train[:, 2] == 1], train[train[:, 2] == 0]
    drawn = np.random.choice(len(negative), len(positive), replace=False)
    train = np.concatenate([positive, negative[drawn]])
    np.random.shuffle(train)
    held = rows[rows["split"].isin(["val", "test"])].drop_duplicates(
        ["gene_a", "gene_b"]
    )
    held_covered = held[held["covered"]]
    held_pairs = held_covered[["ia", "ib"]].to_numpy(np.int64)

    model = KG4SL(args, len(entities), len(relations), adj_entity, adj_relation)
    curve, best = [], {"val_auroc": -1.0}
    threads = tf.ConfigProto(
        intra_op_parallelism_threads=32, inter_op_parallelism_threads=4
    )
    with tf.Session(config=threads) as sess:
        sess.run(tf.global_variables_initializer())
        best_loss, stalled = np.inf, 0
        for epoch in range(args.n_epochs):
            start, losses = time.time(), []
            # the official loop skips the last incomplete batch
            for i in range(0, len(train) - args.batch_size + 1, args.batch_size):
                batch = train[i : i + args.batch_size]
                _, loss = model.train(
                    sess,
                    {
                        model.nodea_indices: batch[:, 0],
                        model.nodeb_indices: batch[:, 1],
                        model.labels: batch[:, 2],
                    },
                )
                losses.append(loss)
            loss = float(np.mean(losses))
            val_auroc = float(
                roc_auc_score(val[:, 2], score(sess, model, args, val[:, :2]))
            )
            curve.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "val_auroc": val_auroc,
                    "seconds": time.time() - start,
                }
            )
            print(json.dumps(curve[-1]), flush=True)
            if val_auroc > best["val_auroc"]:
                best = {
                    "epoch": epoch,
                    "val_auroc": val_auroc,
                    "scores": score(sess, model, args, held_pairs),
                }
            if loss < best_loss:
                best_loss, stalled = loss, 0
            else:
                stalled += 1
                if stalled >= 2:  # the official early stop
                    break

    out = held[["gene_a", "gene_b"]].assign(
        score=0.5, covered=held["covered"].to_numpy()
    )
    out.loc[held_covered.index, "score"] = best["scores"]
    out.to_parquet(cli.out / "scores.parquet", index=False)
    held_rows = rows[rows["split"].isin(["val", "test"])]
    report = {
        "best_epoch": best["epoch"],
        "best_val_auroc": best["val_auroc"],
        "curve": curve,
        "entities": len(entities),
        "relations": len(relations),
        "triples": len(kg2id),
        "edges_removed_within_benchmark_pairs": int(within.sum()),
        "symbols_mapped": int(symbol_entity.size),
        "training_pairs": len(train),
        "tested_training_pairs": len(positive) + len(negative),
        "validation_pairs": len(val),
        "held_pairs": len(held),
        "held_pairs_covered": int(held["covered"].sum()),
        "held_rows_covered_share": float(held_rows["covered"].mean()),
        "args": vars(args),
        "seed": cli.seed,
    }
    (cli.out / "metrics.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
