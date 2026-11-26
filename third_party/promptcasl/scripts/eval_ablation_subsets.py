#!/usr/bin/env python3
"""Evaluate trained ablation checkpoints per cell line and per context subset.

Run: python -m scripts.eval_ablation_subsets --device 0 [--model_ids ...]
"""

import argparse
import glob
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModel

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import BIOMEDBERT, FOLDS_DIR, SAVE_DIR  # noqa: E402
from src.eval import _load_pathway_encoder, load_models_from_log  # noqa: E402
from src.train_model import (  # noqa: E402
    load_omics_dict,
    load_omics_emb_dict,
    validate_model,
)

CELL_LINES = ["A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"]
SCENARIO = "MCL"
MODEL_CELL_LINE = "mixed-6cl"
BASE_MODEL_DIR = os.path.join(SAVE_DIR, SCENARIO)


def norm_pair(gp):
    """'G1|G2' or (g1, g2) -> sorted 'A|B' to match the pair list."""
    if isinstance(gp, str):
        parts = gp.split("|")
    else:
        parts = list(gp)
    return "|".join(sorted(str(p) for p in parts))


# --------------------------------------------------------------------------
# Subset map: variant B (within NP1) + variant A (global full)
# --------------------------------------------------------------------------
def build_subset_map():
    """Return pair(norm) -> {'B': type, 'A': is-global-full-conflict}.

    Variant B (relevant to MCL): computed from the filtered NP1 data,
      single_cl  : pair appears in only one cell line (no cross-cell-line contrast)
      conflict   : pair shows >=2 distinct labels across cell lines (needs context)
      consistent : pair has the same label across cell lines (pair memorization suffices)
    Variant A (control): global full-conflict pairs from
      splits_v2/label_conflict/label_conflict_pairs.csv.
    """
    pair_cl_label = defaultdict(dict)
    for cl in CELL_LINES:
        fn = os.path.join(FOLDS_DIR, f"{cl}_filtered_NP1_F.csv")
        df = pd.read_csv(fn)
        for g1, g2, y in zip(df.gene_1, df.gene_2, df.SL_or_not):
            pair_cl_label[norm_pair((g1, g2))][cl] = int(y == "SL")

    subset = {}
    for p, labs in pair_cl_label.items():
        if len(labs) == 1:
            t = "single_cl"
        elif len(set(labs.values())) > 1:
            t = "conflict"
        else:
            t = "consistent"
        subset[p] = {"B": t}

    conf_csv = os.path.join(FOLDS_DIR, "splits_v2/label_conflict/label_conflict_pairs.csv")
    global_conf = set()
    if os.path.exists(conf_csv):
        cdf = pd.read_csv(conf_csv)
        global_conf = {norm_pair((r.gene_1, r.gene_2)) for r in cdf.itertuples()}
    for p in subset:
        subset[p]["A"] = p in global_conf
    return subset


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def auc_aupr(labels, probs):
    """Return (n, n_pos, prevalence, AUC, AUPR); AUC/AUPR are NaN for single-class."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    n, n_pos = len(labels), int(labels.sum())
    prev = n_pos / n if n else float("nan")
    if n_pos == 0 or n_pos == n:
        return n, n_pos, prev, float("nan"), float("nan")
    roc = roc_auc_score(labels, probs)
    pr, rc, _ = precision_recall_curve(labels, probs)
    return n, n_pos, prev, roc, sk_auc(rc, pr)


# --------------------------------------------------------------------------
# Evaluate one model: return a per-sample DataFrame
# --------------------------------------------------------------------------
def eval_one_model(model_id, bioLLM, device, dataset_path_override=None):
    model_dir = os.path.join(BASE_MODEL_DIR, MODEL_CELL_LINE, model_id)
    fold_files = sorted(
        f for f in os.listdir(model_dir)
        if f.startswith(f"model_{SCENARIO}_F") and f.endswith(".pt")
    )
    folds = sorted(int(f.split("_F")[1].split(".")[0]) for f in fold_files)
    if not folds:
        print(f"  [skip] {model_id}: no fold checkpoint")
        return None

    records = []
    for fold in folds:
        (llm_encoder, gene_agg, gene_ctx_attn, predictor,
         ge_encoder, cnv_encoder, mu_encoder, film_layer, args) = load_models_from_log(
            model_dir, bioLLM, SCENARIO, fold, device, single_model=False)

        assert not getattr(args, "use_lora", False), "LoRA is not handled by this script."
        dataset_path = dataset_path_override or args.dataset_path

        omics_emb_dict = load_omics_emb_dict() if getattr(args, "omics_data", 0) else None
        omics_dict = load_omics_dict() if getattr(args, "omics_encoder", 0) else None
        pathway_encoder, pathway_dict, geneset_dict, text_pathway_fusion, pair_builder = _load_pathway_encoder(
            model_dir, args, SCENARIO, fold, device, single_model=False)
        omics_mode = getattr(args, "omics_mode", None) or None
        if str(omics_mode) == "None":
            omics_mode = None

        abl = dict(
            no_context=bool(getattr(args, "no_context", False)),
            no_kg=bool(getattr(args, "no_kg", False)),
            random_genept=bool(getattr(args, "random_genept", False)),
            no_genept=bool(getattr(args, "no_genept", False)),
            no_text=bool(getattr(args, "no_text", False)),
        )

        for cl in CELL_LINES:
            pt = os.path.join(dataset_path, f"{cl}_F{fold}.pt")
            if not os.path.exists(pt):
                print(f"    [warn] missing {pt}")
                continue
            val_ds = torch.load(pt, weights_only=False)
            val_loader = DataLoader(
                val_ds, batch_size=getattr(args, "batch_size", 24), shuffle=False)
            res = validate_model(
                val_loader, llm_encoder, gene_agg, gene_ctx_attn, predictor, device,
                omics_emb_dict=omics_emb_dict, omics_dict=omics_dict,
                ge_encoder=ge_encoder, cnv_encoder=cnv_encoder, mu_encoder=mu_encoder,
                film_layer=film_layer, pathway_encoder=pathway_encoder, pathway_dict=pathway_dict,
                omics_mode=omics_mode, geneset_dict=geneset_dict,
                text_pathway_fusion=text_pathway_fusion, pair_builder=pair_builder,
                return_preds=True, **abl)
            for gp, c, y, pr in zip(res["gene_pairs"], res["cell_lines"], res["labels"], res["probs"]):
                records.append({"model": model_id, "fold": fold, "cell_line": c,
                                "gene_pair": gp, "label": int(y), "prob": float(pr)})
        print(f"    fold {fold} done ({len(records)} cumulative rows)  flags={abl}  omics_mode={omics_mode}")
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Per-cell-line / subset metrics from the per-sample DataFrame
# --------------------------------------------------------------------------
def metrics_per_cell(df):
    rows = []
    aucs, auprs = [], []
    for cl in CELL_LINES:
        sub = df[df.cell_line == cl]
        if sub.empty:
            continue
        n, npos, prev, roc, pr = auc_aupr(sub.label, sub.prob)
        rows.append({"cell_line": cl, "n": n, "AUC": roc, "AUPR": pr})
        if not np.isnan(roc):
            aucs.append(roc)
            auprs.append(pr)
    rows.append({"cell_line": "CL_mean", "n": "", "AUC": np.mean(aucs), "AUPR": np.mean(auprs)})
    n, npos, prev, roc, pr = auc_aupr(df.label, df.prob)
    rows.append({"cell_line": "Overall", "n": n, "AUC": roc, "AUPR": pr})
    return pd.DataFrame(rows)


def metrics_subset(df, subset_map):
    nb = df.gene_pair.map(lambda g: subset_map.get(norm_pair(g), {}).get("B", "NA"))
    na = df.gene_pair.map(lambda g: subset_map.get(norm_pair(g), {}).get("A", False))
    rows = []

    def add(name, mask):
        s = df[mask]
        n, npos, prev, roc, pr = auc_aupr(s.label, s.prob)
        rows.append({"subset": name, "n": n, "n_pos": npos, "prevalence": round(prev, 3),
                     "AUC": roc, "AUPR": pr})

    add("overall", df.index == df.index)
    add("conflict_B", nb == "conflict")
    add("consistent_B", nb == "consistent")
    add("single_cl_B", nb == "single_cl")
    add("conflict_A_global", na.values)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--biobert_path", type=str, default=BIOMEDBERT)
    ap.add_argument("--model_ids", nargs="+", default=None,
                    help="Run directory names under save/MCL/<model_cell_line>; "
                         "if omitted, every 'abl' run is evaluated.")
    ap.add_argument("--dataset_path", type=str, default=None,
                    help="Override dataset_path (default: from each run's training.log).")
    ap.add_argument("--model_cell_line", type=str, default="mixed-6cl",
                    help="Sub-dir under save/MCL/: mixed-6cl (standard) or mixed-6cl-gplo (GPLO).")
    ap.add_argument("--out_root", type=str, default=os.path.join(PROJECT_ROOT, "results/ablation_eval"))
    args = ap.parse_args()

    global MODEL_CELL_LINE
    MODEL_CELL_LINE = args.model_cell_line

    device = torch.device(f"cuda:{args.device}")
    bioLLM = AutoModel.from_pretrained(args.biobert_path).to(device)

    if args.model_ids:
        model_ids = args.model_ids
    else:
        model_ids = sorted(os.path.basename(p) for p in
                           glob.glob(os.path.join(BASE_MODEL_DIR, MODEL_CELL_LINE, "*abl*")))
    print(f"Evaluating {len(model_ids)} runs:")
    for m in model_ids:
        print("  -", m)

    subset_map = build_subset_map()
    os.makedirs(args.out_root, exist_ok=True)
    summary_rows = []

    for model_id in model_ids:
        print(f"\n{'='*70}\n[eval] {model_id}\n{'='*70}")
        df = eval_one_model(model_id, bioLLM, device, args.dataset_path)
        if df is None or df.empty:
            continue
        out_dir = os.path.join(args.out_root, model_id)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

        mpc = metrics_per_cell(df)
        msub = metrics_subset(df, subset_map)
        mpc.to_csv(os.path.join(out_dir, "metrics_per_cell.csv"), index=False)
        msub.to_csv(os.path.join(out_dir, "metrics_subset.csv"), index=False)
        print("\n-- per cell line --")
        print(mpc.to_string(index=False))
        print("\n-- subsets --")
        print(msub.to_string(index=False))

        row = {"model": model_id, "n_folds": df.fold.nunique()}
        for _, r in mpc.iterrows():
            row[f"{r.cell_line}_AUC"] = round(r.AUC, 4) if pd.notna(r.AUC) else np.nan
            row[f"{r.cell_line}_AUPR"] = round(r.AUPR, 4) if pd.notna(r.AUPR) else np.nan
        for _, r in msub.iterrows():
            row[f"{r.subset}_AUC"] = round(r.AUC, 4) if pd.notna(r.AUC) else np.nan
            row[f"{r.subset}_AUPR"] = round(r.AUPR, 4) if pd.notna(r.AUPR) else np.nan
        summary_rows.append(row)

    if summary_rows:
        summ = pd.DataFrame(summary_rows)
        summ.to_csv(os.path.join(args.out_root, "SUMMARY.csv"), index=False)
        print(f"\n{'='*70}\nSUMMARY saved to {os.path.join(args.out_root, 'SUMMARY.csv')}\n{'='*70}")
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
