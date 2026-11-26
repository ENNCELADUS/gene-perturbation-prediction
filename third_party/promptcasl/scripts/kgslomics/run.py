"""Train / evaluate the KG-SLomics baseline on PromptCASL's splits.

  MCL  --split std|gplo   pooled 6 cell lines, random or gene-pair-disjoint 5-fold split
  MCTL --target CL        train on the other 5 lines, evaluate on CL
                          --loco_mode np1|full   balanced vs imbalanced tables
                          --aligned              restrict to SLBench's 9845 genes

  python run.py --scenario mcl --split gplo --fold 0
  python run.py --scenario mctl --target PK1
"""
import argparse
import json
import os
import time
from collections import defaultdict

# Reduce allocator fragmentation caused by the variable-size subgraphs; must be
# set before CUDA initialises.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from common import (CACHE, CELL_LINES, N_FOLDS, apply_alignment,
                    build_subset_map, full_metrics, load_entity_to_id,
                    load_full, load_split, norm_pair)
from model import KGSLomics


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def build_instances(split, ent, loco_mode="np1", aligned=False):
    """Pooled (gene1_id, gene2_id, cell_idx, label, fold) table for 6 lines."""
    loader = load_split if loco_mode == "np1" else load_full
    df = pd.concat([loader(cl, split) for cl in CELL_LINES], ignore_index=True)
    if aligned:
        n0 = len(df)
        df = apply_alignment(df)
        print(f"[aligned] SLBench 9845-gene filter: {n0} -> {len(df)} instances")
    df["g1"] = df.gene_1.map(ent)
    df["g2"] = df.gene_2.map(ent)
    missing = df.g1.isna() | df.g2.isna()
    if missing.any():
        print(f"dropping {int(missing.sum())} instances with genes outside PrimeKG")
        df = df[~missing]
    df["cell_idx"] = df.cell_line.map({c: i for i, c in enumerate(CELL_LINES)})
    df["pair"] = [norm_pair(a, b) for a, b in zip(df.gene_1, df.gene_2)]
    return df.reset_index(drop=True).astype({"g1": np.int64, "g2": np.int64})


def split_indices(df, scenario, fold, target, val_mode, seed=1234):
    """-> (train_idx, val_idx, test_idx) as positional indices into df."""
    if scenario == "mcl":
        test = np.where(df.fold.values == fold + 1)[0]
        rest = np.where(df.fold.values != fold + 1)[0]
    else:
        t = CELL_LINES.index(target)
        test = np.where(df.cell_idx.values == t)[0]
        rest = np.where(df.cell_idx.values != t)[0]

    if val_mode == "heldout":
        # Use the held-out fold as the early-stopping set (PromptCASL's protocol).
        return rest, test, test
    # stricter: carve an inner validation fold out of the training portion
    folds = df.fold.values[rest]
    if scenario == "mcl":
        inner = (fold + 1) % N_FOLDS + 1
        val = rest[folds == inner]
        train = rest[folds != inner]
    else:
        val = rest[folds == 1]
        train = rest[folds != 1]
    if len(val) == 0 or len(train) == 0:
        # {cl}_full.csv carries fold==1 on every row, so there is nothing to
        # carve by fold id -- fall back to a seeded random 1/N_FOLDS slice.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(rest))
        cut = max(1, len(rest) // N_FOLDS)
        val, train = rest[perm[:cut]], rest[perm[cut:]]
    return train, val, test


def make_loader(sub, graph, batch, fanout, shuffle):
    """Returns (loader, pos) where pos maps a batch's `input_id` -- an index
    into this loader's edge_label_index -- back to a row of the pooled table."""
    eli = torch.tensor(np.stack([sub.g1.values, sub.g2.values]), dtype=torch.long)
    loader = LinkNeighborLoader(
        graph, num_neighbors=fanout, edge_label_index=eli,
        edge_label=torch.tensor(sub.label.values, dtype=torch.float),
        batch_size=batch, shuffle=shuffle, neg_sampling_ratio=0.0)
    return loader, sub.index.values


# --------------------------------------------------------------------------
# weights (their compute_class_weights / compute_sl_rarity_weights)
# --------------------------------------------------------------------------
def class_weights(sub):
    cnt = sub.cell_idx.value_counts().to_dict()
    raw = {c: 1.0 / (n ** 0.5) for c, n in cnt.items()}
    tot = sum(raw.values())
    return {c: w * len(raw) / tot for c, w in raw.items()}


def rarity_weights(sub):
    """total_cell_lines / (#lines in which the pair is a positive), on positives."""
    pos = sub[sub.label == 1]
    freq = pos.groupby("pair").cell_idx.nunique()
    n_cl = sub.cell_idx.nunique()
    return {p: n_cl / f for p, f in freq.items()}


# --------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------
def run_epoch(model, loader, pos, df, kg_emb, omics, dev, opt=None, cw=None,
              rw=None, amp=True):
    train = opt is not None
    model.train() if train else model.eval()
    crit = nn.BCEWithLogitsLoss(reduction="none")
    tot_loss, n_batch = 0.0, 0
    rows = []

    for batch in loader:
        ids = pos[batch.input_id.cpu().numpy()]
        cls = df.cell_idx.values[ids]
        batch = batch.to(dev)
        n_id = batch.n_id
        losses = []
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                for c in np.unique(cls):
                    m = torch.from_numpy(cls == c).to(dev)
                    sel = ids[cls == c]
                    z = model.encode(kg_emb[n_id], omics[c][n_id],
                                     batch.edge_index, batch.edge_type)
                    out = model.decode(z, batch.edge_label_index[:, m])
                    y = batch.edge_label[m]
                    loss = crit(out, y)
                    if rw is not None:
                        w = torch.ones_like(loss)
                        for k, gi in enumerate(sel):
                            if df.label.values[gi] == 1:
                                w[k] = rw.get(df.pair.values[gi], 1.0)
                        loss = loss * w
                    loss = loss.mean() * (cw.get(int(c), 1.0) if cw else 1.0)
                    losses.append(loss)
                    if not train:
                        rows.append(pd.DataFrame({
                            "idx": sel,
                            "prob": out.float().sigmoid().detach().cpu().numpy()}))
        if losses:
            loss = sum(losses) / len(losses)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot_loss += float(loss)
            n_batch += 1
    pred = pd.concat(rows) if rows else None
    return tot_loss / max(n_batch, 1), pred


def evaluate(df, pred, subset_map, tag):
    """Overall / per-cell-line / conflict-consistent metrics."""
    d = df.iloc[pred.idx.values].copy()
    d["prob"] = pred.prob.values
    d["subset"] = [subset_map.get(p, "single_cl") for p in d.pair]
    res = {"tag": tag, "overall": full_metrics(d.label.values, d.prob.values),
           "per_cell": {}, "per_subset": {}}
    for cl in CELL_LINES:
        s = d[d.cell_line == cl]
        if len(s):
            res["per_cell"][cl] = full_metrics(s.label.values, s.prob.values)
    for name in ["conflict", "consistent", "single_cl"]:
        s = d[d.subset == name]
        if len(s):
            res["per_subset"][name] = full_metrics(s.label.values, s.prob.values)
    return res, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["mcl", "mctl"], default="mcl")
    ap.add_argument("--split", choices=["std", "gplo"], default="gplo")
    ap.add_argument("--fold", type=int, default=0, help="MCL test fold, 0-based")
    ap.add_argument("--target", default=None, help="MCTL held-out cell line")
    ap.add_argument("--loco_mode", choices=["np1", "full"], default="np1",
                    help="MCTL tables: np1 = balanced {cl}_filtered_NP1_F.csv, "
                         "full = imbalanced {cl}_full.csv (train and test both)")
    ap.add_argument("--aligned", action="store_true",
                    help="restrict to SLBench's 9845-gene universe so n matches "
                         "the DDGCN/KG4SL/SLMGAE/NSF4SL/GCATSL LOCO baselines")
    ap.add_argument("--protein", default="procan", choices=["procan", "zero"])
    ap.add_argument("--val_mode", default="heldout", choices=["heldout", "inner"])
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--fanout", type=int, nargs=2, default=[25, 10])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--es", type=int, default=10)
    ap.add_argument("--es_metric", default="auc", choices=["auc", "loss"],
                    help="early-stopping metric")
    ap.add_argument("--num_bases", type=int, default=None)
    ap.add_argument("--no_class_weight", action="store_true")
    ap.add_argument("--no_rarity_weight", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(CACHE, "..", "results"))
    ap.add_argument("--dry_run", type=int, default=0,
                    help="stop after N training batches; prints peak GPU memory")
    args = ap.parse_args()
    if args.scenario == "mctl" and not args.target:
        ap.error("--scenario mctl requires --target")
    if args.scenario == "mcl" and args.loco_mode == "full":
        ap.error("--loco_mode full is MCTL-only: the full tables carry a single "
                 "fold id, so there is no 5-fold MCL split to make from them")
    if args.scenario == "mctl" and args.split == "std":
        print("[note] MCTL uses the gplo fold ids")
        args.split = "gplo"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    kg = torch.load(os.path.join(CACHE, "primekg.pt"), weights_only=False)
    graph = Data(edge_index=kg["edge_index"], edge_type=kg["edge_type"],
                 num_nodes=kg["num_nodes"])
    kg_emb = torch.from_numpy(np.load(os.path.join(CACHE, "kg_emb128.npy"))).to(dev)
    omics = [torch.from_numpy(
        np.load(os.path.join(CACHE, f"omics_{cl}_{args.protein}.npy"))).to(dev)
        for cl in CELL_LINES]

    ent = load_entity_to_id()
    df = build_instances(args.split, ent, args.loco_mode, args.aligned)
    subset_map = build_subset_map(args.split, args.loco_mode)
    tr_i, va_i, te_i = split_indices(df, args.scenario, args.fold, args.target,
                                     args.val_mode, args.seed)
    tr, va, te = df.iloc[tr_i], df.iloc[va_i], df.iloc[te_i]
    dataset = "aligned" if args.aligned else "fullpairs"
    if args.scenario == "mcl":
        tag = f"mcl_{args.split}_fold{args.fold}" + ("_aligned" if args.aligned else "")
    else:
        tag = f"mctl_{args.target}_{args.loco_mode}_{dataset}"
    print(f"[{tag}] train {len(tr)} | val {len(va)} | test {len(te)} "
          f"(test prev {te.label.mean():.4f}) | val_mode={args.val_mode} "
          f"| protein={args.protein}")

    cw = None if args.no_class_weight else class_weights(tr)
    rw = None if args.no_rarity_weight else rarity_weights(tr)

    tr_loader, tr_pos = make_loader(tr, graph, args.batch, args.fanout, True)
    va_loader, va_pos = make_loader(va, graph, args.batch, args.fanout, False)
    te_loader, te_pos = make_loader(te, graph, args.batch, args.fanout, False)

    model = KGSLomics(num_relations=len(kg["rel_to_id"]), heads=args.heads,
                      dropout=args.dropout, num_bases=args.num_bases).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.dry_run:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i, batch in enumerate(tr_loader):
            if i >= args.dry_run:
                break
            ids = tr_pos[batch.input_id.cpu().numpy()]
            cls = df.cell_idx.values[ids]
            batch = batch.to(dev)
            # mirrors run_epoch: every class' graph is alive until one backward
            with torch.autocast("cuda", dtype=torch.float16, enabled=not args.no_amp):
                losses = []
                for c in np.unique(cls):
                    m = torch.from_numpy(cls == c).to(dev)
                    z = model.encode(kg_emb[batch.n_id], omics[c][batch.n_id],
                                     batch.edge_index, batch.edge_type)
                    losses.append(model.decode(z, batch.edge_label_index[:, m]).mean())
            (sum(losses) / len(losses)).backward()
            print(f"  batch {i}: nodes={batch.num_nodes} "
                  f"edges={batch.edge_index.shape[1]} classes={len(np.unique(cls))} "
                  f"peak_alloc={torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
                  f"peak_reserved={torch.cuda.max_memory_reserved()/2**30:.2f} GiB")
        print(f"{args.dry_run} batches in {time.time()-t0:.1f}s "
              f"({len(tr_loader)} batches/epoch)")
        return

    best, best_state, bad = -np.inf, None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, tr_loader, tr_pos, df, kg_emb, omics, dev,
                               opt=opt, cw=cw, rw=rw, amp=not args.no_amp)
        va_loss, va_pred = run_epoch(model, va_loader, va_pos, df, kg_emb, omics,
                                     dev, cw=cw, rw=rw, amp=not args.no_amp)
        va_auc = full_metrics(df.iloc[va_pred.idx.values].label.values,
                              va_pred.prob.values)["AUC"]
        # release cached-but-unused blocks so the card can be shared
        alloc = torch.cuda.max_memory_allocated() / 2 ** 30
        resv = torch.cuda.max_memory_reserved() / 2 ** 30
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"  ep{ep:03d} train {tr_loss:.4f} val {va_loss:.4f} "
              f"val_AUC {va_auc:.4f} ({time.time()-t0:.0f}s) "
              f"[alloc {alloc:.1f} / reserved {resv:.1f} GiB]")
        score = -va_loss if args.es_metric == "loss" else va_auc
        if np.isfinite(score) and score > best + 1e-5:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.es:
                print(f"  early stop at epoch {ep} "
                      f"(best {args.es_metric} {abs(best):.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    _, te_pred = run_epoch(model, te_loader, te_pos, df, kg_emb, omics, dev,
                           cw=cw, rw=rw, amp=not args.no_amp)
    res, table = evaluate(df, te_pred, subset_map, tag)

    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    res["args"] = vars(args)
    res["dataset"] = dataset
    res["subset_universe"] = args.loco_mode
    res["n_train"], res["n_val"] = len(tr), len(va)
    with open(os.path.join(outdir, f"{tag}_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    table[["gene_1", "gene_2", "cell_line", "label", "prob", "subset"]].to_csv(
        os.path.join(outdir, f"{tag}_pred.csv"), index=False)

    o = res["overall"]
    print(f"\n[{tag}] AUC {o['AUC']:.4f}  AUPR {o['AUPR']:.4f} "
          f"(prev {o['prevalence']:.3f})  macroF1 {o['macro_F1']:.4f}")
    for name, m in res["per_subset"].items():
        print(f"    {name:11s} n={m['n']:5d} AUC {m['AUC']:.4f} AUPR {m['AUPR']:.4f}")
    for cl, m in res["per_cell"].items():
        print(f"    {cl:11s} n={m['n']:5d} AUC {m['AUC']:.4f} AUPR {m['AUPR']:.4f}")
    print(f"-> {outdir}/{tag}_*.{{json,csv}}")


if __name__ == "__main__":
    main()
