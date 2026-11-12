"""MiT4SL (Tao et al. 2025; official code at third_party/mit4sl, commit 68cfb5c) on the
multi-study SL benchmark, from the inputs of hpc/mit4sl/prepare.py. It runs the official
model, graph loaders and training step (train_MiT4SL.train) with the cross-cell-line
protocol (configs/cross_cell_line/protocol.yaml: the large-data preset, 20 epochs at
learning rate 1e-3, each epoch all SL pairs and as many non-SL pairs drawn again) for
one run, seed 1024 (the official MODEL_SEED). The official script takes 10% of its
training rows as validation; here the validation lines are the validation set, and the
epoch is the one with the best validation AUROC (the official monitor). Output: one
score per held-out (model_id, gene_a, gene_b), the softmax SL probability; a row
prepare.py could not keep (a gene outside PrimeKG, a line without DepMap expression)
gets the median of the scored held-out rows, so these rows tie.

    PYTHONPATH=third_party/mit4sl/src .venv-mit4sl/bin/python hpc/mit4sl/train.py \
        --data data/mit4sl/benchmark --out outputs/sl/mit4sl/seed0 --device 0
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score

import model as official_model
import torch_geometric.typing
from config_loader import load_cfg_from_paths
from model import MiT4SL
from train_MiT4SL import resolve_effective_hyperparameters, train
from util import Construct_loader, init_graph_data, set_seed

# torch-scatter is built for the CPU only (no nvcc on the H20); PyG then uses torch's
# own scatter on the GPU. torch-sparse (HGTLoader's sampler) still runs on the CPU.
torch_geometric.typing.WITH_TORCH_SCATTER = False
_HGTConv = official_model.HGTConv


def _hgt_conv(*args, group: str = "sum", **kwargs):
    """The official HGTConv(..., group='sum'): PyG >= 2.3 dropped the argument and
    always sums over edge types."""
    if group != "sum":
        raise ValueError(f"HGTConv group {group!r}")
    return _HGTConv(*args, **kwargs)


official_model.HGTConv = _hgt_conv

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CODE = ROOT / "third_party/mit4sl"
OFFICIAL_DATA = ROOT / "data/mit4sl/official/MultiOmics_feature"
NODE_COUNT = 27671  # util.Downstream_data_preprocess_cell: genes of the PrimeKG graph
HELD = ("val", "test")


def split_data(task: Path, name: str, node_type: dict):
    """util.Downstream_data_preprocess_cell for one file, without its random 10%
    validation split: (rows mapped to the gene type's index, mask, node count, the
    rows as read)."""
    data = pd.read_csv(task / f"sl_{name}_0.csv")
    ori = data.copy()
    for frame in (data, ori):
        frame.columns = [0, 1, 2, 3]
    data[0] = data[0].astype(str).map(node_type).astype(int)
    data[1] = data[1].astype(str).map(node_type).astype(int)
    nodes = list(set(data[0]) | set(data[1]))
    mask = torch.zeros(NODE_COUNT, dtype=torch.bool)
    mask[nodes] = True
    return data, mask, len(nodes), ori


def sequences(data: Path, task: Path) -> tuple[dict, pd.DataFrame]:
    """util.overlapping_with_sequence over the training, validation and test rows."""
    with open(OFFICIAL_DATA / "seq_data/protein_sequence_embedding.pkl", "rb") as f:
        embedding = pickle.load(f)
    proteins = pd.read_csv(data / "cell_line_proteins.csv")
    nodes = set(proteins["primekg_index"])
    for name in ("train", "val", "test"):
        rows = pd.read_csv(task / f"sl_{name}_0.csv")
        nodes |= set(rows["0"]) | set(rows["1"])
    return {i: embedding[i] for i in nodes}, proteins


@torch.no_grad()
def predict(model, loader, pair_data, num_nodes, ori_data, device) -> np.ndarray:
    """The SL probability of each row (the official evaluate_split's scores)."""
    model.eval()
    out = []
    for batch in loader:
        _, _, _, logits, _ = model(pair_data, batch.to(device), num_nodes, ori_data)
        out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    if len(out) != 1:
        raise ValueError(f"expected one loader batch, got {len(out)}")
    return out[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, help="smoke run: fewer epochs")
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
    cfg, _ = load_cfg_from_paths(
        [str(OFFICIAL_CODE / "configs/cross_cell_line/protocol.yaml")]
    )
    torch.manual_seed(0)  # the official global initialisation
    np.random.seed(0)
    device = torch.device(f"cuda:{cli.device}" if cli.device.isdigit() else cli.device)
    task = cli.data / "task"
    with open(OFFICIAL_DATA / "kg_data/node_index_dic.json") as f:
        node_type = json.load(f)[cfg.SOLVER.NODE_TYPE]
    kgdata, cell_ppidata = init_graph_data(
        str(cli.data / "kgdata.pkl"), str(cli.data / "cell_line_subgraphs.pkl")
    )
    sequence_data, cell_proteins = sequences(cli.data, task)
    parts = {n: split_data(task, n, node_type) for n in ("train", "val", "test")}
    train_data, train_mask, num_train, ori_train = parts["train"]
    hparams = resolve_effective_hyperparameters(
        cfg, len(parts["train"][0]) + len(parts["val"][0])
    )
    epochs = cli.epochs or hparams["effective_max_epochs"]
    seed = cfg.SOLVER.MODEL_SEED  # run 0 of the official seed repeats
    set_seed(seed)
    model = MiT4SL(
        kgdata, cell_ppidata, sequence_data, cell_proteins,
        cfg.KG.HIDEEN_DIM, cfg.KG.EMB_DIM, cfg.KG.NUM_HEADS, cfg.KG.NUM_LAYERS,
        cfg.Cell_Line.HIDDEN_DIM, cfg.Cell_Line.NUM_LAYERS,
        cfg.KG.USE_KG, cfg.Cell_Line.USE_Cell, cfg.ProteinSeq.USE_Seq, device,
    )  # fmt: skip
    optimizer = optim.Adam(
        model.parameters(),
        lr=hparams["effective_lr"],
        betas=(cfg.OPTIM.BETA1, cfg.OPTIM.BETA2),
        eps=cfg.OPTIM.EPS,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
    )
    loaders = Construct_loader(
        cfg.KG_SAMPLER.SAMPLE_NODES, cfg.KG_SAMPLER.SAMPLE_LAYERS,
        cfg.SOLVER.NUM_WORKERS, kgdata, train_mask, parts["val"][1],
        parts["test"][1], cfg.SOLVER.NODE_TYPE,
        num_train, parts["val"][2], parts["test"][2],
    )  # fmt: skip
    loader = dict(zip(("train", "val", "test"), loaders))
    history, best = [], None
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        log = train(
            cfg, epoch, seed, model, loader["train"], optimizer,
            train_data, num_train, ori_train, device,
        )  # fmt: skip
        scores = {
            n: predict(model, loader[n], parts[n][0], parts[n][2], parts[n][3], device)
            for n in HELD
        }
        auroc = {n: float(roc_auc_score(parts[n][0][3], scores[n])) for n in HELD}
        history.append(
            {"epoch": epoch, "loss": log["loss"], "val_auroc": auroc["val"],
             "test_auroc": auroc["test"], "seconds": time.perf_counter() - start}
        )  # fmt: skip
        print(json.dumps(history[-1]), flush=True)
        if best is None or auroc["val"] > best["val_auroc"]:
            best = {"epoch": epoch, "val_auroc": auroc["val"], "scores": scores}

    keys = pd.concat(
        [pd.read_csv(task / f"keys_{n}_0.csv").assign(score=best["scores"][n])
         for n in HELD],
        ignore_index=True,
    )  # fmt: skip
    sides = json.loads(cli.split.read_text())
    held_lines = {m for s in HELD for m in sides[s]}
    labels = pd.read_parquet(
        cli.labels, columns=["model_id", "gene_a", "gene_b", "label"]
    )
    held = labels[labels["label"].notna() & labels["model_id"].isin(held_lines)]
    held = held[["model_id", "gene_a", "gene_b"]].drop_duplicates()
    out = held.merge(keys, on=["model_id", "gene_a", "gene_b"], how="left")
    fill = float(out["score"].median())
    out = out.assign(covered=out["score"].notna(), score=out["score"].fillna(fill))
    cli.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cli.out / "scores.parquet", index=False)
    metrics = {
        "epochs": epochs,
        "learning_rate": hparams["effective_lr"],
        "seed": seed,
        "best_epoch": best["epoch"],
        "best_val_auroc": best["val_auroc"],
        "history": history,
        "held_rows": len(out),
        "held_rows_covered_share": float(out["covered"].mean()),
        "uncovered_score": fill,
    }
    (cli.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
