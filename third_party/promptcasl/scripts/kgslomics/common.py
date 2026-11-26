"""Shared paths, constants and metrics for the KG-SLomics re-implementation.

Runs the KG-SLomics RGAT architecture on PrimeKG + ConvE, the graph PromptCASL uses.
Run the scripts in this directory directly, e.g. ``python scripts/kgslomics/run.py``.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (auc as sk_auc, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_recall_curve, recall_score,
                             roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (  # noqa: E402
    CCLE_DIR,
    FOLDS_DIR as _FOLDS_DIR,
    PRIMEKG_EMB128,
    PRIMEKG_ENTITY_JSON,
    PRIMEKG_TRIPLES_TSV,
    SANGER_PROCAN,
    SLBENCH_META,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")

FOLDS_DIR = _FOLDS_DIR
GPLO_DIR = os.path.join(FOLDS_DIR, "splits_v2/MCL_GPLO")
LOCO_DIR = os.path.join(FOLDS_DIR, "splits_v2/LOCO_subsets")

ENTITY_JSON = PRIMEKG_ENTITY_JSON
TRIPLES_TSV = PRIMEKG_TRIPLES_TSV
CONVE_128 = PRIMEKG_EMB128
META_9845 = SLBENCH_META

CCLE_GE = os.path.join(CCLE_DIR, "GE.csv")
CCLE_CN = os.path.join(CCLE_DIR, "CNGene.csv")
CCLE_MU = os.path.join(CCLE_DIR, "MU.csv")

PROCAN = SANGER_PROCAN

CELL_LINES = ["A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"]

ACH = {"A549": "ACH-000681", "PK1": "ACH-000307", "IPC298": "ACH-000915",
       "MEWO": "ACH-000987", "GI1": "ACH-000756", "HS936T": "ACH-000801"}
SIDM = {"A549": "SIDM00903", "PK1": "SIDM01321", "IPC298": "SIDM01059",
        "MEWO": "SIDM00545", "GI1": "SIDM00273", "HS936T": "SIDM01750"}

N_FOLDS = 5


def norm_pair(g1, g2):
    """Undirected, order-normalised pair key -- matches data/build_splits.py."""
    return tuple(sorted((str(g1), str(g2))))


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def load_split(cell_line, split):
    """NP1 (balanced) table for one cell line.

    split='std'  -> {cl}_filtered_NP1_F.csv            (random folds)
    split='gplo' -> splits_v2/MCL_GPLO/{cl}_GPLO_NP1_F.csv (pair-disjoint folds)
    """
    if split == "std":
        path = os.path.join(FOLDS_DIR, f"{cell_line}_filtered_NP1_F.csv")
    elif split == "gplo":
        path = os.path.join(GPLO_DIR, f"{cell_line}_GPLO_NP1_F.csv")
    else:
        raise ValueError(f"unknown split {split!r}")
    df = pd.read_csv(path)
    df = df[["gene_1", "gene_2", "SL_or_not", "fold"]].copy()
    df["label"] = (df["SL_or_not"] == "SL").astype(np.int64)
    df["cell_line"] = cell_line
    return df.drop(columns=["SL_or_not"])


def load_full(cell_line, split=None):
    """Imbalanced full table for one cell line (the 'full' LOCO/MCTL scenario).

    ``split`` is accepted and ignored so this is drop-in for :func:`load_split`:
    there is no GPLO variant of the full tables, and every row carries fold==1.
    """
    path = os.path.join(FOLDS_DIR, f"{cell_line}_full.csv")
    df = pd.read_csv(path)
    df = df[["gene_1", "gene_2", "SL_or_not", "fold"]].copy()
    df["label"] = (df["SL_or_not"] == "SL").astype(np.int64)
    df["cell_line"] = cell_line
    return df.drop(columns=["SL_or_not"])


def load_loco_test(cell_line, mode="NP1"):
    """Held-out target-line table for MCTL; mode in {NP1, full, testfull}.

    The pair sets equal :func:`load_split` / :func:`load_full` for NP1 / full;
    ``testfull`` differs from ``full`` only in the subset tag columns.
    """
    path = os.path.join(LOCO_DIR, f"LOCO_{cell_line}_test_tagged_{mode}.csv")
    df = pd.read_csv(path)
    df = df[["gene_1", "gene_2", "SL_or_not"]].copy()
    df["label"] = (df["SL_or_not"] == "SL").astype(np.int64)
    df["cell_line"] = cell_line
    df["fold"] = 0
    return df.drop(columns=["SL_or_not"])


_ALIGNED_SYMBOLS = None


def aligned_symbols():
    """The 9845 gene symbols SLBench's baselines are restricted to."""
    global _ALIGNED_SYMBOLS
    if _ALIGNED_SYMBOLS is None:
        _ALIGNED_SYMBOLS = set(pd.read_csv(META_9845)["symbol"].astype(str))
    return _ALIGNED_SYMBOLS


def apply_alignment(df):
    """Keep only pairs whose both genes are in the 9845-gene universe."""
    sym = aligned_symbols()
    keep = df.gene_1.astype(str).isin(sym) & df.gene_2.astype(str).isin(sym)
    return df[keep].copy()


# --------------------------------------------------------------------------
# conflict / consistent subsets (definition B = NP1-internal, matching
# scripts/eval_ablation_subsets.py so the numbers line up with PromptCASL's)
# --------------------------------------------------------------------------
def build_subset_map(split="std", loco_mode="np1"):
    """pair -> 'single_cl' | 'conflict' | 'consistent'.

    loco_mode='np1' uses the NP1 tables; loco_mode='full' uses the full tables.
    Recorded as ``subset_universe`` in the run's metrics.json.
    """
    loader = load_split if loco_mode == "np1" else load_full
    pair_cl_label = defaultdict(dict)
    for cl in CELL_LINES:
        df = loader(cl, split)
        for g1, g2, y in zip(df.gene_1, df.gene_2, df.label):
            pair_cl_label[norm_pair(g1, g2)][cl] = int(y)
    out = {}
    for p, labs in pair_cl_label.items():
        if len(labs) == 1:
            out[p] = "single_cl"
        elif len(set(labs.values())) > 1:
            out[p] = "conflict"
        else:
            out[p] = "consistent"
    return out


# --------------------------------------------------------------------------
# metrics (mirrors scripts/eval_ablation_subsets.py so results are comparable)
# --------------------------------------------------------------------------
def auc_aupr(labels, probs):
    """(n, n_pos, prevalence, AUROC, AUPR); AUPR = trapezoid on the PR curve,
    the same estimator eval_ablation_subsets.py uses. NaN when single-class."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    n, n_pos = len(labels), int(labels.sum())
    prev = n_pos / n if n else float("nan")
    if n == 0 or n_pos == 0 or n_pos == n:
        return n, n_pos, prev, float("nan"), float("nan")
    pr, rc, _ = precision_recall_curve(labels, probs)
    return n, n_pos, prev, roc_auc_score(labels, probs), sk_auc(rc, pr)


def full_metrics(labels, probs):
    """AUROC/AUPR + threshold-based scores at the macro-F1-optimal cut in
    [0.05, 0.95], matching PromptCASL's reporting protocol."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    n, n_pos, prev, roc, aupr = auc_aupr(labels, probs)
    out = {"n": n, "n_pos": n_pos, "prevalence": prev, "AUC": roc, "AUPR": aupr,
           "AP": float("nan"), "macro_F1": float("nan"), "threshold": float("nan"),
           "accuracy": float("nan"), "balanced_accuracy": float("nan"),
           "recall": float("nan")}
    if n == 0 or n_pos == 0 or n_pos == n:
        return out
    out["AP"] = average_precision_score(labels, probs)
    best_f1, best_t = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(labels, (probs >= t).astype(int), average="macro",
                      zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    pred = (probs >= best_t).astype(int)
    out.update(macro_F1=best_f1, threshold=best_t,
               accuracy=float((pred == labels).mean()),
               balanced_accuracy=balanced_accuracy_score(labels, pred),
               recall=recall_score(labels, pred, zero_division=0))
    return out


def load_entity_to_id():
    with open(ENTITY_JSON) as fh:
        return json.load(fh)
