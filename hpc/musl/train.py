"""MuSL (official code third_party/musl, commit f8021cf) on the multi-study SL
benchmark, from the inputs of hpc/musl/prepare.py. One model over all training lines:
the official MuSL (CNN on the pair's 32 x 32 expression histograms, GraphSAGE on the
PPI graph from released ESM-2 node embeddings, 35 statistical features), its official
train step and optimizer, and the cell-line setting of the README (learning rate 1e-4,
weight decay 1e-3, batch 512, patience 10; default config otherwise). Each row reads
its own line's imputed cells. The validation AUPR of the final head over the validation
lines' samples picks the epoch (the official monitor, there the test split). Scores:
the SL probability of every row of the validation and test lines;
scores_pooled_input.parquet reads one pooled matrix of training-line cells for every
row instead (the same weights). Training lines are not scored (their rows get the
median score, covered false).

    OMP_NUM_THREADS=16 .venv-musl/bin/python hpc/musl/train.py \
        --data data/musl/benchmark --out outputs/sl/musl/seed0 --device 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hpc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare  # noqa: E402
import sl_common as sl  # noqa: E402

utils = prepare.official("utils")
module = prepare.official("module")

CONFIG = {  # README cell-line command; the rest is get_default_config
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "batch_size": 512,
    "patience": 10,
    "lr_factor": 0.5,
    "lr_patience": 3,
    "min_lr": 1e-6,
    "bins": 32,
    "seed": 0,
}


class Lines:
    """Matrices and (normalized) features of the lines of one data directory."""

    def __init__(self, data: Path, params: dict | None):
        self.data, self.params, self.non_finite = data, params, 0
        self.exp = {}

    def raw_features(self, name: str) -> np.ndarray:
        path = "pooled_features.npy" if name == "pooled" else f"features/{name}.npy"
        return np.load(self.data / path)

    def features(self, name: str, mask: np.ndarray) -> torch.Tensor:
        """The raw features of the masked rows, normalized with the fit rows' parameters
        (the official normalize_with_params edits its input: it gets a copy)."""
        raw = torch.from_numpy(self.raw_features(name)[mask]).clone()
        out = utils.normalize_with_params(raw, self.params)
        self.non_finite += int((~torch.isfinite(out)).sum())
        return torch.nan_to_num(out, nan=0.0).float()

    def matrix(self, name: str) -> torch.Tensor:
        file = "pooled.npy" if name == "pooled" else f"lines/{name}.npy"
        return torch.from_numpy(np.load(self.data / file))


def edge_dataset(pairs: pd.DataFrame, labels, matrix, state, bins: int):
    edges = torch.from_numpy(pairs[["i", "j"]].to_numpy().T.copy()).long()
    return module.EdgeDataset(
        edges, torch.as_tensor(np.asarray(labels)).long(), matrix, state=state, bins=bins
    )


@torch.no_grad()
def predict(model, dataset, edge_index, device, batch_size, workers) -> np.ndarray:
    """softmax(final_pred)[:, 1] of every sample, in order."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    out = []
    for batch in loader:
        result = model(
            batch["image"].to(device),
            batch["state"].to(device),
            [edge_index],
            batch["edge"].to(device),
        )
        out.append(F.softmax(result["final_pred"], dim=1)[:, 1].cpu())
    return torch.cat(out).numpy()


def aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8, help="data loader workers")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "data/musl/official/protein_embeddings.pt",
    )
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
    cli = parser.parse_args()
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device(f"cuda:{cli.device}" if cli.device.isdigit() else cli.device)
    data = cli.data
    manifest = json.loads((data / "manifest.json").read_text())
    side = {m: v["side"] for m, v in manifest["lines"].items()}
    nodes = pd.read_csv(data / "nodes.csv")["gene"].tolist()
    edge_index = torch.from_numpy(np.load(data / "edges.npy")).long().to(device)

    # released ESM-2 node embeddings; a node without one gets a zero vector
    released = torch.load(cli.embeddings, map_location="cpu", weights_only=True)
    dim = next(iter(released.values())).shape[0]
    embeddings = torch.zeros(len(nodes), dim)
    missing = 0
    for k, gene in enumerate(nodes):
        if gene in released:
            embeddings[k] = released[gene].to(torch.float32)
        else:
            missing += 1

    rows = {m: pd.read_parquet(data / "rows" / f"{m}.parquet") for m in side}
    train = [m for m in side if side[m] == "train" and rows[m]["fit"].any()]
    validation = [m for m in side if side[m] == "val" and rows[m]["val"].any()]
    held = [m for m in side if side[m] in sl.HELD_SIDES]

    # normalization parameters from the fit rows of the training lines
    fit_raw = np.concatenate(
        [Lines(data, None).raw_features(m)[rows[m]["fit"].to_numpy()] for m in train]
    )
    params = utils.get_normalize_params(torch.from_numpy(fit_raw))
    lines = Lines(data, params)

    def dataset(m: str, flag: str):
        mask = rows[m][flag].to_numpy()
        return edge_dataset(
            rows[m][mask], rows[m]["label"][mask], lines.matrix(m),
            lines.features(m, mask), cfg["bins"],
        )  # fmt: skip

    fit_set = ConcatDataset([dataset(m, "fit") for m in train])
    val_set = ConcatDataset([dataset(m, "val") for m in validation])
    val_labels = np.concatenate(
        [rows[m]["label"][rows[m]["val"]].to_numpy() for m in validation]
    )

    model = module.MuSL(
        num_nodes=len(nodes), node_embedding_dim=dim, hidden_dim=128,
        target_node_dim=1024, node_embeddings=embeddings.to(device), unified_dim=256,
        enable_cross_attention=False, enable_adaptive_fusion=False,
        enable_contrastive_learning=False, use_cnn=True, use_gnn=True, use_stat=True,
    ).to(device)  # fmt: skip
    optimizer = utils.customAdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["lr_factor"],
        patience=cfg["lr_patience"], min_lr=cfg["min_lr"],
    )  # fmt: skip
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        fit_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=cli.workers
    )

    history, best, stale = [], None, 0
    for epoch in range(1, cli.epochs + 1):
        start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        for batch in loader:
            optimizer.zero_grad()
            step = module.train_step(model, batch, edge_index, criterion, device)
            step["total_loss"].backward()  # all three branches on: the official sum
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += step["total_loss"].item()
        train_seconds = time.perf_counter() - start
        val_aupr = aupr(
            val_labels,
            predict(model, val_set, edge_index, device, 1024, cli.workers),
        )
        scheduler.step(val_aupr)
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / len(loader),
                "val_aupr": val_aupr,
                "lr": optimizer.param_groups[0]["lr"],
                "steps_per_second": len(loader) / train_seconds,
                "seconds": time.perf_counter() - start,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if best is None or val_aupr > best["val_aupr"]:
            best = {
                "epoch": epoch,
                "val_aupr": val_aupr,
                "state": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
            }
            stale = 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break

    model.load_state_dict(best["state"])
    cli.out.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], cli.out / "best.pt")

    # scores of the held-out lines on their own cells and on the pooled matrix
    pooled_rows = pd.read_parquet(data / "pooled_rows.parquet")
    pooled_state = lines.features("pooled", np.ones(len(pooled_rows), bool))
    pooled_matrix = lines.matrix("pooled")
    pooled_at = pd.MultiIndex.from_frame(pooled_rows[["i", "j"]])
    own, pooled = {}, {}
    start = time.perf_counter()
    for m in held:
        mask = rows[m]["score"].to_numpy()
        pairs = rows[m][mask]
        zeros = np.zeros(len(pairs), int)
        keys = pd.MultiIndex.from_frame(
            pairs[["model_id", "gene_1", "gene_2"]].reset_index(drop=True)
        )
        own_set = edge_dataset(
            pairs, zeros, lines.matrix(m), lines.features(m, mask), cfg["bins"]
        )
        at = pooled_at.get_indexer(pd.MultiIndex.from_frame(pairs[["i", "j"]]))
        pooled_set = edge_dataset(
            pairs, zeros, pooled_matrix, pooled_state[at], cfg["bins"]
        )
        for table, d in ((own, own_set), (pooled, pooled_set)):
            table[m] = pd.Series(
                predict(model, d, edge_index, device, 1024, cli.workers), index=keys
            )
    score_seconds = time.perf_counter() - start

    benchmark = sl.benchmark_rows(cli.labels, cli.split)
    own_scores = pd.concat(own.values())
    pooled_scores = pd.concat(pooled.values())
    report = sl.write_line_scores(cli.out, benchmark, own_scores)
    report_pooled = sl.write_line_scores(
        cli.out, benchmark, pooled_scores, name="scores_pooled_input"
    )
    metrics = {
        "settings": {
            **cfg,
            "epochs": cli.epochs,
            "device": str(device),
            "workers": cli.workers,
        },  # fmt: skip
        "best_epoch": best["epoch"],
        "best_val_aupr": best["val_aupr"],
        "epochs_run": len(history),
        "history": history,
        "steps_per_second_mean": float(
            np.mean([h["steps_per_second"] for h in history])
        ),
        "fit_samples": len(fit_set),
        "val_samples": len(val_set),
        "nodes": len(nodes),
        "edges": int(edge_index.shape[1]),
        "embedding_dim": dim,
        "nodes_without_embedding": missing,
        "feature_values_not_finite_after_normalization": lines.non_finite,
        "scores": report,
        "scores_pooled_input": report_pooled,
        "own_vs_pooled_spearman": float(
            own_scores.corr(pooled_scores, method="spearman")
        ),
        "score_seconds": score_seconds,
        "prepare": {k: v for k, v in manifest.items() if k != "lines"},
        "sha256": {
            "protein_embeddings": prepare.sha256(cli.embeddings),
            "manifest": prepare.sha256(data / "manifest.json"),
        },
    }
    (cli.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
