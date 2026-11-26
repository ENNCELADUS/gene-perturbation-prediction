"""Build the 4-dimensional per-cell-line omics node features.

Channels per gene: [Expression, CNA, Mut, Protein] = CCLE GE and CN, CCLE mutation
count, Sanger ProCan intensity (missing -> 0). Nodes without a measurement keep 0;
the remaining KG nodes inherit the iterative neighbour average.

    python build_features.py --protein procan|zero
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from common import (ACH, CACHE, CCLE_CN, CCLE_GE, CCLE_MU, CELL_LINES,
                    PROCAN, SIDM, load_entity_to_id)

MAX_ITERS = 10


def _strip_entrez(cols):
    """'TSPAN6 (7105)' -> 'TSPAN6'"""
    return [c.split(" (")[0].strip() for c in cols]


def load_expression_cn():
    ge = pd.read_csv(CCLE_GE, index_col=0)
    cn = pd.read_csv(CCLE_CN, index_col=0)
    ge.columns = _strip_entrez(ge.columns)
    cn.columns = _strip_entrez(cn.columns)
    ge = ge.loc[:, ~ge.columns.duplicated()]
    cn = cn.loc[:, ~cn.columns.duplicated()]
    return ge, cn


def load_mutation_counts():
    """ACH -> Series(gene -> #variant annotations)."""
    mu = pd.read_csv(CCLE_MU, usecols=["ModelID", "HugoSymbol", "VariantInfo"],
                     low_memory=False)
    mu = mu[mu.ModelID.isin(ACH.values())].copy()
    mu["VariantInfo"] = mu["VariantInfo"].fillna("").astype(str).str.split("&")
    mu = mu.explode("VariantInfo")
    mu = mu[mu.VariantInfo.str.len() > 0]
    counts = mu.groupby(["ModelID", "HugoSymbol"]).size()
    return {ach: counts.loc[ach] if ach in counts.index.get_level_values(0) else
            pd.Series(dtype=float) for ach in ACH.values()}


def load_protein(kind):
    """cell line -> Series(gene -> intensity); empty Series when unavailable."""
    if kind == "zero":
        return {cl: pd.Series(dtype=float) for cl in CELL_LINES}
    if kind == "procan":
        cols = pd.read_csv(PROCAN, nrows=0).columns
        want = [SIDM[cl] for cl in CELL_LINES if SIDM[cl] in set(cols)]
        df = pd.read_csv(PROCAN, usecols=["GeneSymbol"] + want).set_index("GeneSymbol")
        out = {}
        for cl in CELL_LINES:
            sid = SIDM[cl]
            out[cl] = df[sid].dropna() if sid in df.columns else pd.Series(dtype=float)
        return out
    raise ValueError(kind)


def propagate(x, known, edge_index, num_nodes):
    """Port of KG-SLomics' scatter_multiomics_augmentation.

    Nodes without a measurement inherit the running sum of their already-filled
    neighbours divided by (degree + 1), expanding one hop per iteration.
    """
    row, col = edge_index
    deg = torch.bincount(col, minlength=num_nodes).float().unsqueeze(1)
    denom = deg + 1.0
    known = known.clone()
    frontier = known.clone()
    total = torch.zeros_like(x)

    def accumulate(front):
        mask = front[row]
        if not bool(mask.any()):
            return False
        total.index_add_(0, col[mask], x[row[mask]])
        return True

    accumulate(frontier)
    for it in range(MAX_ITERS):
        cand = (~known) & (total.abs().sum(dim=1) > 0)
        if not bool(cand.any()):
            break
        x[cand] = total[cand] / denom[cand]
        frontier = cand
        known |= cand
        if not accumulate(frontier):
            break
    return x, known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default="procan", choices=["procan", "zero"])
    args = ap.parse_args()

    ent = load_entity_to_id()
    kg = torch.load(os.path.join(CACHE, "primekg.pt"), weights_only=False)
    edge_index, N = kg["edge_index"], kg["num_nodes"]

    ge, cn = load_expression_cn()
    mut = load_mutation_counts()
    prot = load_protein(args.protein)

    print(f"{'cell':8s} {'expr':>7s} {'cna':>7s} {'mut':>7s} {'prot':>7s} "
          f"{'KG nodes measured':>18s} {'after propagation':>18s}")
    for cl in CELL_LINES:
        ach = ACH[cl]
        feats = {}
        feats["expr"] = ge.loc[ach] if ach in ge.index else pd.Series(dtype=float)
        feats["cna"] = cn.loc[ach] if ach in cn.index else pd.Series(dtype=float)
        feats["mut"] = mut.get(ach, pd.Series(dtype=float))
        feats["prot"] = prot[cl]

        x = torch.zeros((N, 4), dtype=torch.float32)
        measured = torch.zeros(N, dtype=torch.bool)
        n_hit = {}
        for j, key in enumerate(["expr", "cna", "mut", "prot"]):
            s = feats[key]
            s = s[~s.index.duplicated()] if len(s) else s
            ids, vals = [], []
            for g, v in s.items():
                i = ent.get(str(g))
                if i is not None and np.isfinite(v):
                    ids.append(i)
                    vals.append(float(v))
            n_hit[key] = len(ids)
            if ids:
                idx = torch.tensor(ids, dtype=torch.long)
                x[idx, j] = torch.tensor(vals, dtype=torch.float32)
                measured[idx] = True

        n_measured = int(measured.sum())
        x, filled = propagate(x, measured, edge_index, N)
        out = os.path.join(CACHE, f"omics_{cl}_{args.protein}.npy")
        np.save(out, x.numpy())
        print(f"{cl:8s} {n_hit['expr']:7d} {n_hit['cna']:7d} {n_hit['mut']:7d} "
              f"{n_hit['prot']:7d} {n_measured:18d} {int(filled.sum()):18d}")

    print(f"\n-> cache/omics_<cell>_{args.protein}.npy  ({N} x 4 each)")


if __name__ == "__main__":
    main()
