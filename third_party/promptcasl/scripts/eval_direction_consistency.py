#!/usr/bin/env python3
"""Within-pair direction consistency on the conflict subset (paired bootstrap).

Run: python -m scripts.eval_direction_consistency
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.config import FOLDS_DIR  # noqa: E402

CELL_LINES = ["A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"]
EVAL_ROOT = os.path.join(PROJECT_ROOT, "results/ablation_eval")


def norm_pair(gp):
    parts = str(gp).split("|") if isinstance(gp, str) else list(gp)
    return "|".join(sorted(str(p) for p in parts))


def conflict_pairs():
    """Variant B conflict: a pair in >=2 cell lines with non-unique labels."""
    pcl = defaultdict(dict)
    for cl in CELL_LINES:
        df = pd.read_csv(os.path.join(FOLDS_DIR, f"{cl}_filtered_NP1_F.csv"))
        for g1, g2, y in zip(df.gene_1, df.gene_2, df.SL_or_not):
            pcl[norm_pair((g1, g2))][cl] = int(y == "SL")
    return {p for p, l in pcl.items() if len(l) >= 2 and len(set(l.values())) > 1}


def clean_pairs(pred_df, conflict):
    """Clean pair = conflict pair whose instances all lie in one fold and include both labels.

    The fold assignment is fixed by the NP1 split and identical across runs, so the
    clean set derived from any run is the same; the order is fixed for paired tests.
    """
    df = pred_df.copy()
    df["np"] = df.gene_pair.map(norm_pair)
    df = df[df.np.isin(conflict)]
    out = [p for p, g in df.groupby("np") if g.fold.nunique() == 1 and set(g.label) == {0, 1}]
    return sorted(out)


def per_pair_scores(pred_df, clean):
    """Return (per-pair win-rate vector, micro, tie_rate) in the fixed clean order."""
    df = pred_df.copy()
    df["np"] = df.gene_pair.map(norm_pair)
    g = df[df.np.isin(clean)]
    rate = {}
    W = T = L = 0
    for p, sub in g.groupby("np"):
        sl = sub.loc[sub.label == 1, "prob"].values
        ns = sub.loc[sub.label == 0, "prob"].values
        w = sum(a > b for a in sl for b in ns)
        t = sum(a == b for a in sl for b in ns)
        l = len(sl) * len(ns) - w - t
        rate[p] = (w + 0.5 * t) / (w + t + l)
        W += w
        T += t
        L += l
    vec = np.array([rate[p] for p in clean], dtype=float)
    tot = W + T + L
    return vec, ((W + 0.5 * T) / tot if tot else np.nan), (T / tot if tot else np.nan)


def _stars(ci_lo, ci_hi, p):
    if ci_lo <= 0 <= ci_hi:
        return "ns"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_ids", nargs="+", default=None,
                    help="Run directory names under ablation_eval; all if omitted.")
    ap.add_argument("--ref_substr", default="nocontext",
                    help="Reference run for the significance test (substring match).")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_root", default=EVAL_ROOT)
    ap.add_argument("--out_csv", default=os.path.join(EVAL_ROOT, "direction_consistency.csv"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    conflict = conflict_pairs()
    if args.model_ids:
        model_ids = args.model_ids
    else:
        model_ids = sorted(os.path.basename(os.path.dirname(p))
                           for p in glob.glob(os.path.join(args.eval_root, "*", "test_predictions.csv")))

    avail = [m for m in model_ids
             if os.path.exists(os.path.join(args.eval_root, m, "test_predictions.csv"))]
    for m in set(model_ids) - set(avail):
        print(f"  [skip] {m}: no test_predictions.csv (run eval_ablation_subsets.py first)")
    if not avail:
        return
    clean = clean_pairs(
        pd.read_csv(os.path.join(args.eval_root, avail[0], "test_predictions.csv")), conflict)
    n = len(clean)
    # Shared resample indices so the CI and the paired test are consistent.
    boot_idx = rng.integers(0, n, size=(args.n_boot, n))

    vecs, recs = {}, []
    for mid in avail:
        vec, micro, tie = per_pair_scores(
            pd.read_csv(os.path.join(args.eval_root, mid, "test_predictions.csv")), clean)
        vecs[mid] = vec
        bs = vec[boot_idx].mean(axis=1)
        recs.append({"model": mid, "n_clean_pairs": n, "macro": vec.mean(), "micro": micro,
                     "tie_rate": tie, "macro_ci_lo": np.percentile(bs, 2.5),
                     "macro_ci_hi": np.percentile(bs, 97.5)})

    ref = next((m for m in avail if args.ref_substr in m), None)
    ref_tag = os.path.basename(ref).split("_abl_")[-1] if ref else None
    for rec in recs:
        mid = rec["model"]
        if ref is None or mid == ref:
            rec["delta_vs_ref"] = 0.0 if mid == ref else np.nan
            rec["sig_vs_ref"] = "ref" if mid == ref else "NA"
            continue
        d = vecs[mid] - vecs[ref]
        bs = d[boot_idx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
        rec["delta_vs_ref"] = d.mean()
        rec["sig_vs_ref"] = _stars(lo, hi, p)

    out = pd.DataFrame(recs)[["model", "n_clean_pairs", "macro", "micro", "tie_rate",
                              "macro_ci_lo", "macro_ci_hi", "delta_vs_ref", "sig_vs_ref"]]
    for c in ["macro", "micro", "tie_rate", "macro_ci_lo", "macro_ci_hi", "delta_vs_ref"]:
        out[c] = out[c].round(4)
    out.to_csv(args.out_csv, index=False)
    print(f"reference run = {ref_tag}  (sig column = paired bootstrap test vs. it)\n")
    print(out.to_string(index=False))
    print(f"\nSaved to {args.out_csv}")


if __name__ == "__main__":
    main()
