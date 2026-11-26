"""Train one MVGCN-iSL model over the fit lines and score the train, val and test rows.

The official model (GCN_pool), loss, optimiser step and early stopping are imported
from the vendored code; one epoch is one official training step for each fit line, on
that line's node features and expression-filtered graphs. Checkpoint selection reads
the validation lines' loss only. Run after prepare.py:

    PYTHONPATH=third_party/mvgcn_isl/code OMP_NUM_THREADS=16 \
        .venv-mvgcn/bin/python hpc/mvgcn_isl/train.py \
        --data data/mvgcn_isl/benchmark --out outputs/sl/mvgcn_isl/seed0 --device 0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import scale

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hpc"))
sys.path.insert(0, str(ROOT / "hpc" / "mvgcn_isl"))
sys.path.insert(0, str(ROOT / "third_party" / "mvgcn_isl" / "code"))

import main as official  # noqa: E402  (code/main.py: train_model, test_model)
import prepare  # noqa: E402
import sl_common  # noqa: E402
from model import GCN_pool, EarlyStopping  # noqa: E402
from utils import generate_torch_edges  # noqa: E402

# the official functions read the module global ``args`` of main.py (its defaults)
official.args = SimpleNamespace(
    pooling="max", batch_size=512, balanced=1, pos_weight=50, model="GCN_pool"
)
EXP = prepare.FEATURES.index("exp")


def line_pairs(rows: pd.DataFrame, index: dict[str, int]) -> dict[str, pd.DataFrame]:
    """The genetic rows of each line, one row for each pair (SL when half of its rows
    are): gene1, gene2 as node indices, label."""
    genetic = rows[rows["condition"] == "genetic"]
    rate = genetic.groupby(["model_id", "gene_1", "gene_2"])["label"].mean().ge(0.5)
    pairs = rate.astype(int).rename("label").reset_index()
    pairs["gene1"] = pairs["gene_1"].map(index)
    pairs["gene2"] = pairs["gene_2"].map(index)
    return {line: frame for line, frame in pairs.groupby("model_id")}


def pooled_z(model, data) -> torch.Tensor:
    """The official pooling of the graph-specific encodings (main.test_model)."""
    z = torch.cat([model.encode(data.x, e) for e in data.edge_index_list], 1)
    z = (
        z.unsqueeze(1)
        .reshape(z.shape[0], len(data.edge_index_list), -1)
        .transpose(1, 2)
    )
    return F.max_pool2d(z, (1, len(data.edge_index_list))).squeeze(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    cli = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(f"cuda:{cli.device}" if torch.cuda.is_available() else "cpu")
    cli.out.mkdir(parents=True, exist_ok=True)

    genes = json.loads((cli.data / "genes.json").read_text())
    index = {g: i for i, g in enumerate(genes)}
    full_edges = {
        k: torch.tensor(v, device=device)
        for k, v in np.load(cli.data / "graphs.npz").items()
    }
    raw = dict(np.load(cli.data / "features.npz"))
    rows = pd.read_parquet(cli.data / "rows.parquet")
    side = rows.drop_duplicates("model_id").set_index("model_id")["side"]
    in_graph = np.zeros(len(genes), bool)
    for e in full_edges.values():
        in_graph[e.cpu().numpy().ravel()] = True

    def line_data(x_raw: np.ndarray) -> SimpleNamespace:
        """Scaled node features; the graphs cut to the genes expressed in the line."""
        expressed = torch.tensor(x_raw[:, EXP] > 0, device=device)
        return SimpleNamespace(
            # float64 as in the official code (float32 gives numerical warnings)
            x=torch.tensor(
                scale(x_raw.astype(np.float64)), dtype=torch.float, device=device
            ),
            edge_index_list=[
                prepare.line_edges(e, expressed) for e in full_edges.values()
            ],
        )

    pairs = line_pairs(rows, index)
    fit = [m for m in raw if side.get(m) in sl_common.FIT_SIDES and m in pairs]
    val = [m for m in raw if side.get(m) == "val" and m in pairs]
    rng = np.random.default_rng(0)

    def usable(frame):  # a line needs SL and non-SL rows for a balanced draw
        return frame["label"].nunique() == 2

    fit = [m for m in fit if usable(pairs[m])]
    val_edges = {}
    for m in val:
        if usable(pairs[m]):
            val_edges[m] = generate_torch_edges(
                sl_common.balanced(pairs[m], rng), False, False, device
            )
    print(
        f"{len(fit)} fit lines, {len(val_edges)} validation lines, {len(genes)} nodes",
        flush=True,
    )

    model = GCN_pool(len(prepare.FEATURES), 64, len(full_edges)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    stopper = EarlyStopping(
        patience=150, verbose=True, reverse=True, path=str(cli.out / "best.pt")
    )
    curve, seconds = [], []
    for epoch in range(1, cli.epochs + 1):
        start = time.time()
        train_loss = 0.0
        for m in np.random.permutation(fit):
            frame = pairs[m]
            positive = frame[frame["label"] == 1]
            negative = frame[frame["label"] == 0]
            if len(negative) < len(
                positive
            ):  # the official balanced draw needs enough non-SL rows
                frame = pd.concat([positive.sample(len(negative)), negative])
            pos, neg = generate_torch_edges(frame, True, True, device)
            train_loss += official.train_model(
                model, optimizer, line_data(raw[m]), device, pos, neg
            )
        val_loss = float(
            np.mean(
                [
                    official.test_model(
                        model, optimizer, line_data(raw[m]), device, *edges
                    )[0]
                    for m, edges in val_edges.items()
                ]
            )
        )
        seconds.append(time.time() - start)
        curve.append(
            {"epoch": epoch, "train_loss": train_loss / len(fit), "val_loss": val_loss}
        )
        print(
            f"epoch {epoch} train {curve[-1]['train_loss']:.4f} val {val_loss:.4f} "
            f"{seconds[-1]:.0f}s",
            flush=True,
        )
        stopper(val_loss, model)
        if stopper.early_stop:
            print("early stop", flush=True)
            break
    model.load_state_dict(torch.load(cli.out / "best.pt", map_location=device))
    model.eval()

    scored = rows[rows["side"].isin(sl_common.SCORED_SIDES)]
    held = set(sl_common.HELD_SIDES)

    @torch.no_grad()
    def score(mean_input: bool) -> pd.Series:
        """Sigmoid of the official decoder on every scored row; a gene with no graph
        edge and no feature in the line is not scored. ``mean_input``: held-out
        lines read the fit lines' mean features."""
        parts = []
        for m, frame in scored.drop_duplicates(
            ["model_id", "gene_1", "gene_2"]
        ).groupby("model_id"):
            key = "mean_input" if mean_input and side[m] in held else m
            if key not in raw:
                continue
            data = line_data(raw[key])
            z = pooled_z(model, data)
            a, b = (
                frame["gene_1"].map(index).to_numpy(),
                frame["gene_2"].map(index).to_numpy(),
            )
            idx = torch.tensor(np.stack([a, b]), device=device)
            prob = (
                torch.cat(
                    [
                        model.decode(z, idx[:, i : i + 100_000]).reshape(-1).sigmoid()
                        for i in range(0, idx.shape[1], 100_000)
                    ]
                )
                .cpu()
                .numpy()
            )
            known = in_graph | (raw[key] != 0).any(1)
            prob[~(known[a] & known[b])] = np.nan
            parts.append(
                pd.Series(
                    prob,
                    index=pd.MultiIndex.from_frame(
                        frame[["model_id", "gene_1", "gene_2"]]
                    ),
                )
            )
        return pd.concat(parts)

    report = sl_common.write_line_scores(cli.out, rows, score(False))
    control = sl_common.write_line_scores(
        cli.out, rows, score(True), name="scores_mean_input"
    )
    best = min(curve, key=lambda c: c["val_loss"])
    metrics = {
        "settings": {
            "epochs": cli.epochs,
            "patience": 150,
            "lr": 1e-4,
            "out_channels": 64,
            "batch_size": 512,
            "seed": 0,
            "device": str(device),
        },
        "fit_lines": len(fit),
        "validation_lines": len(val_edges),
        "epochs_run": len(curve),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "seconds_per_epoch": float(np.mean(seconds)),
        "scores": report,
        "scores_mean_input": control,
        "curve": curve,
        "prepare": json.loads((cli.data / "prepare.json").read_text()),
    }
    (cli.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(
        json.dumps(
            {k: v for k, v in metrics.items() if k not in ("curve", "prepare")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
