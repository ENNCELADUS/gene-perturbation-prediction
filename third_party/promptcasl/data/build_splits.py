"""Build the data splits for PromptCASL.

Scope (per project decision):
  * MCL is NP1-only. No train-NP1/test-full mixing on the MCL side.
  * The train-NP1 / test-full setup belongs to LOCO (zero-shot on a held-out
    cell line's full, imbalanced data).

Produces CSV split definitions under
    data/folds_assignment/splits_v2/
Existing files in folds_assignment/ are not modified.

Three deliverables:
  1. MCL-GPLO  (Global Gene-Pair Leave-Out, NP1 only)
     Merge the cell lines' filtered NP1 data and re-assign the `fold`
     column GLOBALLY BY GENE-PAIR: every (cell_line) instance of a given
     normalized pair lands in the SAME fold. So when fold k is the test
     fold, that pair appears in NO cell line's training folds -> tests
     true new-pair generalization, not pair memorization. Both train and
     test come from NP1 (balanced).
       -> MCL_GPLO/{cl}_GPLO_NP1_F.csv     (same schema as filtered, GPLO fold)
       -> MCL_GPLO/MCL_GPLO_all_NP1.csv     (merged, for inspection)

  2. LOCO test-subset tagging (5 -> 1), THREE versions per target:
       NP1      : test = target NP1,  seen-basis = 5 lines NP1   (all-NP1, balanced)
       full     : test = target full, seen-basis = 5 lines full  (all-full)
       testfull : test = target full, seen-basis = 5 lines NP1   (train NP1 / test full,
                  = the real LOCO experiment)
     Tag columns:
       subset_pair  : seen / unseen
       subset_gene  : both_seen / one_gene_unseen / both_gene_unseen
       conflict_type: consistent / conflict / ambiguous / NA_unseen
       label_conflict = conflict or ambiguous
       -> LOCO_subsets/LOCO_{target}_test_tagged_{NP1,full,testfull}.csv

  3. label-conflict global subset
     Pairs that appear in >=2 cell lines with >=2 distinct labels.
       -> label_conflict/label_conflict_pairs.csv

Run with any python that has pandas:
    python -m data.build_splits
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import FOLDS_DIR as _FOLDS_DIR  # noqa: E402

FOLDS_DIR = Path(_FOLDS_DIR)
OUT_DIR = FOLDS_DIR / "splits_v2"
CELL_LINES = ["A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"]
N_FOLDS = 5


def norm_pair(g1, g2):
    return tuple(sorted((str(g1), str(g2))))


def load(cl, kind):
    """kind: 'filtered' (NP1 balanced) or 'full' (complete, imbalanced)."""
    fn = f"{cl}_filtered_NP1_F.csv" if kind == "filtered" else f"{cl}_full.csv"
    df = pd.read_csv(FOLDS_DIR / fn)
    df["__pair"] = [norm_pair(a, b) for a, b in zip(df["gene_1"], df["gene_2"])]
    df["__is_SL"] = (df["SL_or_not"] == "SL").astype(int)
    df["cell_line_origin"] = cl
    return df


# --------------------------------------------------------------------------
# 1. MCL-GPLO  (NP1 only)
# --------------------------------------------------------------------------
def assign_pair_folds(pair_stats):
    """Greedy balanced assignment of unique pairs to N_FOLDS folds.

    pair_stats: dict pair -> (n_pos, n_neg). Returns dict pair -> fold(1..N).
    Deterministic: sort by (-total, -n_pos, pair); assign each pair to the fold
    with the smallest current (total_load, pos_load, count). Primary key is
    TOTAL load (keeps folds equal-sized); secondary is POS load (keeps each fold
    ~50% positive). Using pos as the primary key would permanently starve a fold
    of negatives once its pos load drifts by 1.
    """
    order = sorted(
        pair_stats.items(),
        key=lambda kv: (-(kv[1][0] + kv[1][1]), -kv[1][0], kv[0]),
    )
    load_pos = [0] * N_FOLDS
    load_neg = [0] * N_FOLDS
    load_cnt = [0] * N_FOLDS
    pair2fold = {}
    for pair, (npos, nneg) in order:
        f = min(range(N_FOLDS),
                key=lambda i: (load_pos[i] + load_neg[i], load_pos[i], load_cnt[i]))
        pair2fold[pair] = f + 1  # 1-indexed to match existing fold column
        load_pos[f] += npos
        load_neg[f] += nneg
        load_cnt[f] += 1
    return pair2fold


def build_mcl_gplo():
    out = OUT_DIR / "MCL_GPLO"
    out.mkdir(parents=True, exist_ok=True)

    dfs = {cl: load(cl, "filtered") for cl in CELL_LINES}
    merged = pd.concat(dfs.values(), ignore_index=True)

    # per-pair pos/neg counts across all cell lines
    pair_stats = defaultdict(lambda: [0, 0])
    for pair, y in zip(merged["__pair"], merged["__is_SL"]):
        pair_stats[pair][int(y)] += 1  # index 1 -> pos, 0 -> neg
    pair_stats = {p: (v[1], v[0]) for p, v in pair_stats.items()}  # (npos, nneg)

    pair2fold = assign_pair_folds(pair_stats)

    # write per-cell-line CSVs with GPLO fold (same schema as filtered files)
    drop_cols = ["__pair", "__is_SL"]
    for cl, df in dfs.items():
        df = df.copy()
        df["fold"] = [pair2fold[p] for p in df["__pair"]]
        df.drop(columns=drop_cols).to_csv(out / f"{cl}_GPLO_NP1_F.csv", index=False)

    merged_out = merged.copy()
    merged_out["fold"] = [pair2fold[p] for p in merged_out["__pair"]]
    merged_out.drop(columns=drop_cols).to_csv(out / "MCL_GPLO_all_NP1.csv", index=False)

    # ---- verification: no pair shared across folds; fold balance ----
    fold_of_pair = {}
    leak = 0
    for p, y in zip(merged_out["__pair"], merged_out["fold"]):
        if p in fold_of_pair and fold_of_pair[p] != y:
            leak += 1
        fold_of_pair[p] = y
    assert leak == 0, f"GPLO LEAK: {leak} pairs span multiple folds!"

    stats = merged_out.assign(is_SL=merged["__is_SL"].values)
    fold_tab = stats.groupby("fold").agg(n=("is_SL", "size"), SL=("is_SL", "sum"))
    fold_tab["NotSL"] = fold_tab["n"] - fold_tab["SL"]
    fold_tab["pos_ratio"] = (fold_tab["SL"] / fold_tab["n"]).round(3)

    crosscheck = {}
    pairs_by_fold = {f: set(stats.loc[stats.fold == f, "__pair"]) for f in range(1, N_FOLDS + 1)}
    for f in range(1, N_FOLDS + 1):
        test_p = pairs_by_fold[f]
        train_p = set().union(*[pairs_by_fold[g] for g in range(1, N_FOLDS + 1) if g != f])
        crosscheck[f] = len(test_p & train_p)

    return {
        "n_total": len(merged_out),
        "n_unique_pairs": len(pair2fold),
        "fold_table": fold_tab,
        "leak_pairs": leak,
        "crosscheck_train_test_pair_overlap": crosscheck,
        "per_cl_counts": {cl: len(df) for cl, df in dfs.items()},
    }


# --------------------------------------------------------------------------
# 2. LOCO test-subset tagging (train NP1 / test full)
# --------------------------------------------------------------------------
def _loco_basis(train_cls, data):
    """Build (pair->{labels}, genes) for the training pool of given data type."""
    tpl, tg = defaultdict(set), set()
    for cl in train_cls:
        d = data[cl]
        tg |= set(d["gene_1"]) | set(d["gene_2"])
        for p, y in zip(d["__pair"], d["__is_SL"]):
            tpl[p].add(int(y))
    return tpl, tg


def _loco_tag(test_df, train_pair_labels, train_genes):
    """Tag one target test frame against a training pool."""
    sp, sg, ct_, lc = [], [], [], []
    for p, y, g1, g2 in zip(test_df["__pair"], test_df["__is_SL"],
                            test_df["gene_1"], test_df["gene_2"]):
        seen = p in train_pair_labels
        sp.append("seen" if seen else "unseen")
        if seen:
            labs = train_pair_labels[p]
            ct = "ambiguous" if len(labs) > 1 else ("consistent" if y in labs else "conflict")
            ct_.append(ct)
            lc.append(ct in ("conflict", "ambiguous"))
        else:
            ct_.append("NA_unseen")
            lc.append(False)
        a, b = g1 in train_genes, g2 in train_genes
        sg.append("both_seen" if (a and b) else "one_gene_unseen" if (a or b) else "both_gene_unseen")
    t = test_df.copy()
    t["subset_pair"], t["subset_gene"], t["conflict_type"], t["label_conflict"] = sp, sg, ct_, lc
    return t


def build_loco_subsets():
    """LOCO 5->1 test-subset tagging, THREE versions per target:
      NP1      : test = target NP1,  seen-basis = 5 lines NP1   (all-NP1, balanced)
      full     : test = target full, seen-basis = 5 lines full  (all-full)
      testfull : test = target full, seen-basis = 5 lines NP1   (train NP1 / test full,
                 = the real experiment)
    """
    out = OUT_DIR / "LOCO_subsets"
    out.mkdir(parents=True, exist_ok=True)
    np1 = {cl: load(cl, "filtered") for cl in CELL_LINES}
    full = {cl: load(cl, "full") for cl in CELL_LINES}

    # (suffix, test_data, basis_data)
    configs = [("NP1", np1, np1), ("full", full, full), ("testfull", full, np1)]
    summary = {sfx: {} for sfx, _, _ in configs}
    for sfx, test_data, basis_data in configs:
        for tgt in CELL_LINES:
            train_cls = [cl for cl in CELL_LINES if cl != tgt]
            tpl, tg = _loco_basis(train_cls, basis_data)
            t = _loco_tag(test_data[tgt], tpl, tg)
            t.drop(columns=["__pair", "__is_SL"]).to_csv(
                out / f"LOCO_{tgt}_test_tagged_{sfx}.csv", index=False)
            pu = int((t.subset_pair == "unseen").sum())
            ogu = int((t.subset_gene == "one_gene_unseen").sum())
            bgu = int((t.subset_gene == "both_gene_unseen").sum())
            summary[sfx][tgt] = {
                "n_test": len(t),
                "pair_seen": int((t.subset_pair == "seen").sum()),
                "consistent": int((t.conflict_type == "consistent").sum()),
                "conflict": int((t.conflict_type == "conflict").sum()),
                "ambiguous": int((t.conflict_type == "ambiguous").sum()),
                "pair_unseen": pu,
                "novel_combo": pu - ogu - bgu,   # unseen pair, both genes seen
                "one_gene_unseen": ogu,
                "both_gene_unseen": bgu,
            }
    return summary


# --------------------------------------------------------------------------
# 3. label-conflict global subset
# --------------------------------------------------------------------------
def build_label_conflict():
    out = OUT_DIR / "label_conflict"
    out.mkdir(parents=True, exist_ok=True)

    full = {cl: load(cl, "full") for cl in CELL_LINES}
    pair_cl_label = defaultdict(dict)   # pair -> {cl: label}
    pair_genes = {}
    for cl in CELL_LINES:
        for p, y, g1, g2 in zip(full[cl]["__pair"], full[cl]["__is_SL"],
                                full[cl]["gene_1"], full[cl]["gene_2"]):
            pair_cl_label[p][cl] = int(y)
            pair_genes[p] = (g1, g2)

    rows = []
    for p, labs in pair_cl_label.items():
        if len(labs) >= 2 and len(set(labs.values())) > 1:
            g1, g2 = pair_genes[p]
            row = {"gene_1": g1, "gene_2": g2, "n_cell_lines": len(labs),
                   "n_SL": sum(labs.values()), "n_NotSL": len(labs) - sum(labs.values())}
            for cl in CELL_LINES:
                row[cl] = ("SL" if labs[cl] else "NotSL") if cl in labs else ""
            rows.append(row)

    cdf = pd.DataFrame(rows).sort_values(["n_cell_lines", "gene_1", "gene_2"],
                                         ascending=[False, True, True])
    cdf.to_csv(out / "label_conflict_pairs.csv", index=False)
    return {"n_conflict_pairs": len(cdf)}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 1. MCL-GPLO (NP1 train + NP1 test) ===")
    gplo = build_mcl_gplo()
    print(f"  total={gplo['n_total']} unique_pairs={gplo['n_unique_pairs']} leak={gplo['leak_pairs']}")
    print(f"  train/test pair overlap per fold: {gplo['crosscheck_train_test_pair_overlap']}")
    print(gplo["fold_table"].to_string())

    print("\n=== 2. LOCO subsets (3 versions: NP1 / full / testfull) ===")
    loco = build_loco_subsets()
    for sfx in ["NP1", "full", "testfull"]:
        print(f"  [{sfx}]")
        for tgt, s in loco[sfx].items():
            print(f"    {tgt}: {s}")

    print("\n=== 3. label-conflict ===")
    conflict = build_label_conflict()
    print(f"  conflict pairs: {conflict['n_conflict_pairs']}")


if __name__ == "__main__":
    main()
