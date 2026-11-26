"""SLMGAE (Ding et al., official code at third_party/slmgae) on the multi-study SL
benchmark. SLMGAE scores a gene pair from a graph auto-encoder over three support views
(GO-BP, GO-CC, BioGRID; hpc/slmgae/prepare.py) and a main view, the fit SL pairs; it
does not read the line. The model, its loss, optimiser and settings are the official
ones.

Differences from the official train.py: one fit on all fit pairs in place of five folds;
the loss index is the official one (every pair i < j, except the pairs the model is
tested on) held as an array: here the pairs that only held-out lines tested; the
official 300 epochs, the last epoch (the official script has no validation: the AUROC on
the validation lines' pairs every 25 epochs is telemetry). An untested pair counts as
non-SL, as in the official code. A pair with a gene outside the nodes is not covered.

GO views: our own Wang similarities computed for the benchmark genes, not the official
loader's output. The released Human_GOsim*.txt are ragged upper triangles (line i holds
the similarities of gene i to genes i + 1 onward), but the official load_dense_feature
reads them as if each line started at column i, so it misaligns every line after the
first; --check-official-go reports our agreement with both readings.

    PYTHONPATH=third_party/slmgae/code .venv-slmgae/bin/python hpc/slmgae/train.py \
        --data data/slmgae/benchmark --out outputs/sl/slmgae/seed0 --check-official-go
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hpc" / "common"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "third_party" / "slmgae" / "code"))

import go_similarity as go  # noqa: E402
from sl_common import SCORED_SIDES, benchmark_rows, pair_labels, write_pair_scores  # noqa: E402

#: The defaults of the official train.py.
FLAGS = {
    "log_file": ("string", "log/LossAblation.txt"),
    "learning_rate": ("float", 0.001),
    "epochs": ("integer", 300),
    "eva_epochs": ("integer", 25),
    "hidden1": ("integer", 512),
    "hidden2": ("integer", 256),
    "nn_size": ("integer", 45),
    "dropout": ("float", 0.20),
    "Alpha": ("float", 2.0),
    "Coe": ("float", 2.0),
    "Beta": ("float", 4.0),
    "early_stopping": ("integer", 20),
}


def loss_index(n: int, excluded_i: np.ndarray, excluded_j: np.ndarray) -> np.ndarray:
    """The official ``train_index`` as an int32 array: every pair i < j of n nodes but
    the excluded pairs (i < j), in the order of np.triu_indices. The official code
    builds it from Python sets."""
    x, y = np.triu_indices(n, k=1)
    keep = ~np.isin(x * n + y, excluded_i.astype(np.int64) * n + excluded_j)
    return np.stack([x[keep], y[keep]], axis=1).astype(np.int32)


def neighbour_jaccard(ours: sp.spmatrix, official: sp.spmatrix) -> tuple[float, int]:
    """Mean Jaccard index of the two graphs' neighbour sets over the genes with a
    neighbour in either, and the number of those genes."""
    a, b = sp.csr_matrix(ours, dtype=bool), sp.csr_matrix(official, dtype=bool)
    inter = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    union = (
        np.asarray(a.sum(axis=1)).ravel() + np.asarray(b.sum(axis=1)).ravel() - inter
    )
    has = union > 0
    return float((inter[has] / union[has]).mean()), int(has.sum())


def check_official_go(official: Path, gaf: Path, obo: Path, k: int) -> dict:
    """Agreement of our GO graphs with the released matrices on the released genes
    (the genes of List_Proteins_in_SL.txt, in that order): our Wang similarity and the
    same k nearest neighbours rule against (a) the graph of the official
    load_dense_feature, which is what the released code trains on, and (b) the graph of
    the released file read with the correct alignment (go.read_released). Neighbour
    Jaccard index, and for (b) also the entry correlations and the share of zeros."""
    from scipy.stats import spearmanr
    from utils import load_dense_feature

    genes = (official / "List_Proteins_in_SL.txt").read_text().split("\n")[:-1]
    n = len(genes)
    parents = go.read_obo(obo)
    rng = np.random.default_rng(0)
    sample = (rng.integers(0, n, 2_000_000), rng.integers(0, n, 2_000_000))
    keep = sample[0] < sample[1]
    sample = (sample[0][keep], sample[1][keep])
    out = {"released_genes": n}
    for aspect, name in (("BP", "Human_GOsim.txt"), ("CC", "Human_GOsim_CC.txt")):
        table = go.read_gaf(gaf, aspect)
        table = table[table["symbol"].isin(set(genes)) & table["term"].isin(parents)]
        sim = go.go_similarity(table, parents, genes)
        ours = go.knn_graph(sim, k)
        loader = load_dense_feature(str(official / name), knn=True, nnSize=k)
        released = go.read_released(official / name, n)
        aligned = go.knn_graph(released, k)
        x, y = sim[sample], released[sample]
        out[aspect] = {
            "released_genes_with_a_term": int(table["symbol"].nunique()),
            "edges_ours": ours.nnz // 2,
            "edges_official_loader": loader.nnz // 2,
            "edges_released_aligned": aligned.nnz // 2,
            "jaccard_vs_official_loader": neighbour_jaccard(ours, loader)[0],
            "jaccard_vs_released_aligned": neighbour_jaccard(ours, aligned)[0],
            "entry_pearson_vs_released_aligned": float(np.corrcoef(x, y)[0, 1]),
            "entry_spearman_vs_released_aligned": float(spearmanr(x, y)[0]),
            "released_zero_share": float((released[np.triu_indices(n, 1)] == 0).mean()),
            "entries_sampled": len(x),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/sl_studies/benchmark_v2/labels.parquet"),
    )
    parser.add_argument(
        "--split", type=Path, default=Path("configs/benchmarks/cell_line_sl_split.json")
    )
    parser.add_argument("--official", type=Path, default=Path("data/slmgae/official"))
    parser.add_argument(
        "--gaf", type=Path, default=Path("data/sl_baselines/go/goa_human.gaf.218.gz")
    )
    parser.add_argument(
        "--obo", type=Path, default=Path("data/sl_baselines/go/go-basic.2023-07-27.obo")
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--check-official-go", action="store_true")
    cli = parser.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)

    import tensorflow.compat.v1 as tf
    from sklearn.metrics import roc_auc_score

    tf.disable_v2_behavior()
    for name, (kind, default) in FLAGS.items():
        getattr(tf.app.flags, f"DEFINE_{kind}")(name, default, name)
    tf.app.flags.FLAGS.mark_as_parsed()
    from models import SLMGAE
    from objective import Optimizer
    from utils import construct_feed_dict, preprocess_graph, sparse_to_tuple

    np.random.seed(cli.seed)
    tf.set_random_seed(cli.seed)

    nodes = json.loads((cli.data / "genes.json").read_text())
    node = {g: i for i, g in enumerate(nodes)}
    n = len(nodes)
    views = np.load(cli.data / "views.npz")
    adjs = [
        sp.coo_matrix(
            (np.ones(len(views[f"{v}_row"])), (views[f"{v}_row"], views[f"{v}_col"])),
            shape=(n, n),
        )
        for v in ("BP", "CC", "PPI")
    ]
    pairs = pd.read_parquet(cli.data / "pairs.parquet")
    sl = pairs[pairs["fit_label"] == 1]
    adj = sp.coo_matrix((np.ones(len(sl)), (sl["i"], sl["j"])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    adjs.append(adj)
    held_only = pairs[pairs["in_held"] & ~pairs["in_fit"]]
    index = loss_index(n, held_only["i"].to_numpy(), held_only["j"].to_numpy())
    print(f"nodes {n}  main-view edges {len(sl)}  loss pairs {len(index)}", flush=True)

    features = sparse_to_tuple(adj)
    supports = [preprocess_graph(a) for a in adjs]
    placeholders = {
        "support": [
            tf.sparse_placeholder(tf.float32, name=f"adj_{i}")
            for i in range(len(supports))
        ],
        "features": tf.sparse_placeholder(tf.float32, name="features"),
        "adj_orig": tf.sparse_placeholder(tf.float32, name="adj_orig"),
        "dropout": tf.placeholder_with_default(0.0, shape=(), name="dropout"),
    }
    model = SLMGAE(
        placeholders,
        features[2][1],
        features[1].shape[0],
        n,
        len(supports) - 1,
        name="SLMGAE_1",
    )
    # above 1 GB: a graph constant (2 GB limit) would fail, so a variable holds it
    index_in = tf.placeholder(tf.int32, shape=index.shape)
    index_var = tf.Variable(index_in, trainable=False, collections=[])
    with tf.name_scope("optimizer"):
        opt = Optimizer(
            supp=model.support_recs,
            main=model.main_rec,
            preds=model.reconstructions,
            labels=tf.sparse_tensor_to_dense(
                placeholders["adj_orig"], validate_indices=False
            ),
            num_nodes=n,
            num_edges=adj.sum(),
            index=index_var,
        )
    config = tf.ConfigProto(
        intra_op_parallelism_threads=cli.threads, inter_op_parallelism_threads=4
    )
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(index_var.initializer, {index_in: index})
    del index

    feed = construct_feed_dict(supports, features, sparse_to_tuple(adj), placeholders)
    eval_feed = dict(feed)
    train_feed = {**feed, placeholders["dropout"]: tf.app.flags.FLAGS.dropout}

    rows = benchmark_rows(cli.labels, cli.split)
    scored = rows[rows["side"].isin(SCORED_SIDES)][
        ["gene_1", "gene_2"]
    ].drop_duplicates()
    scored = scored[scored["gene_1"].isin(node) & scored["gene_2"].isin(node)]
    si, sj = (
        scored["gene_1"].map(node).to_numpy(),
        scored["gene_2"].map(node).to_numpy(),
    )
    val = pair_labels(rows, ("val",))
    val = val[val["gene_1"].isin(node) & val["gene_2"].isin(node)]
    vi, vj = val["gene_1"].map(node).to_numpy(), val["gene_2"].map(node).to_numpy()

    def predict() -> np.ndarray:
        rec = sess.run(model.predict(), feed_dict=eval_feed)
        return rec

    def val_auroc(rec: np.ndarray) -> float:
        return float(roc_auc_score(val["label"], (rec[vi, vj] + rec[vj, vi]) / 2))

    cost, curve, seconds = [], [], []
    for epoch in range(cli.epochs):
        t = time.time()
        _, loss = sess.run([opt.opt_op, opt.cost], feed_dict=train_feed)
        seconds.append(time.time() - t)
        cost.append(float(loss))
        print(
            f"epoch {epoch + 1:04d} train_loss={loss:.5f} time={seconds[-1]:.2f}s",
            flush=True,
        )
        if (epoch + 1) % 25 == 0 and epoch + 1 < cli.epochs:
            curve.append({"epoch": epoch + 1, "val_auroc": val_auroc(predict())})
            print(curve[-1], flush=True)
    rec = predict()
    curve.append({"epoch": cli.epochs, "val_auroc": val_auroc(rec)})
    scores = pd.Series(
        (rec[si, sj] + rec[sj, si]) / 2,
        index=pd.MultiIndex.from_frame(scored[["gene_1", "gene_2"]]),
    )
    report = write_pair_scores(cli.out, rows, scores)

    metrics = {
        "settings": {k: tf.app.flags.FLAGS[k].value for k in FLAGS}
        | {"epochs": cli.epochs, "seed": cli.seed},
        "nodes": n,
        "main_view_edges": len(sl),
        "loss_pairs": int(n * (n - 1) // 2 - len(held_only)),
        "scores": report,
        "val_auroc_curve": curve,
        "train_loss": cost,
        "seconds_per_epoch_mean": float(np.mean(seconds)),
        "seconds_per_epoch_first": seconds[0],
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "prepare": json.loads((cli.data / "prepare.json").read_text()),
    }
    if cli.check_official_go:
        metrics["official_go_agreement"] = check_official_go(
            cli.official, cli.gaf, cli.obo, 45
        )
    (cli.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "train_loss"}, indent=2))


if __name__ == "__main__":
    main()
