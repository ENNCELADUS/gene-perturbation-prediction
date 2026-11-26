"""Pathway-level omics utilities: Reactome gene sets and per-pathway activity scores."""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.config import CCLE_DIR, GMT_C2CP

# MSigDB C2:CP gene sets (Reactome).
_GMT_C2CP = GMT_C2CP

# CCLE raw data paths.
_CCLE_GE = os.path.join(CCLE_DIR, "GE.csv")
_CCLE_CNV = os.path.join(CCLE_DIR, "CNGene.csv")
_CCLE_MU = os.path.join(CCLE_DIR, "MU.csv")


# ─────────────────────────────────────────────────────────────────────────────
# GMT loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gmt(
    gmt_path: str,
    prefix_filter: Optional[str] = None,
    min_ccle_overlap: int = 15,
    ccle_genes: Optional[set] = None,
) -> dict[str, list[str]]:
    """Load pathway gene sets from a GMT file.

    Args:
        gmt_path: path to .gmt file
        prefix_filter: only keep pathways whose name starts with this prefix
                       (e.g. 'KEGG_', 'HALLMARK_')
        min_ccle_overlap: drop pathways with fewer than this many genes in CCLE
        ccle_genes: set of CCLE gene symbols used for overlap filtering;
                    if None, no overlap filtering is applied

    Returns:
        Ordered dict {pathway_name: [gene_symbol, ...]}
    """
    pathways: dict[str, list[str]] = {}
    with open(gmt_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            if prefix_filter and not name.startswith(prefix_filter):
                continue
            genes = parts[2:]  # skip name and description URL
            if ccle_genes is not None:
                genes = [g for g in genes if g in ccle_genes]
            if len(genes) >= min_ccle_overlap:
                pathways[name] = genes
    return pathways


def load_reactome(min_ccle_overlap: int = 15, ccle_genes: Optional[set] = None) -> dict[str, list[str]]:
    """Reactome gene sets (MSigDB C2:CP:REACTOME, ~1615 pathways before overlap filtering)."""
    return load_gmt(_GMT_C2CP, prefix_filter="REACTOME_",
                    min_ccle_overlap=min_ccle_overlap, ccle_genes=ccle_genes)


# ─────────────────────────────────────────────────────────────────────────────
# CCLE data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_gene_symbol(col: str) -> str:
    """Extract gene symbol from 'SYMBOL (ENTREZ_ID)' column name."""
    return col.split(" (")[0].strip()


def load_ccle_ge(verbose: bool = True) -> tuple[pd.DataFrame, set[str]]:
    """Load full CCLE GE matrix.

    Returns:
        ge_df  : DataFrame (cell_lines × genes), columns are gene symbols
        genes  : set of gene symbols present in GE
    """
    if verbose:
        print("Loading CCLE GE …")
    ge_raw = pd.read_csv(_CCLE_GE, index_col=0)
    ge_raw.columns = [_strip_gene_symbol(c) for c in ge_raw.columns]
    ge_raw.index.name = "cell_line"
    return ge_raw, set(ge_raw.columns)


def load_ccle_cnv(ge_index: pd.Index, verbose: bool = True) -> pd.DataFrame:
    """Load CCLE CNV matrix aligned to ge_index rows.

    Returns:
        cn_df : DataFrame aligned to same cell lines as GE, columns are gene symbols
    """
    if verbose:
        print("Loading CCLE CNV …")
    cn_raw = pd.read_csv(_CCLE_CNV, index_col=0)
    cn_raw.columns = [_strip_gene_symbol(c) for c in cn_raw.columns]
    cn_raw.index.name = "cell_line"
    # Align to the cell lines present in GE
    common_cls = ge_index.intersection(cn_raw.index)
    return cn_raw.loc[common_cls]


def load_ccle_mu(verbose: bool = True) -> pd.DataFrame:
    """Load CCLE mutation file and pivot to binary (cell_line × gene) DataFrame.

    A gene is marked 1 for a given cell line if any mutation record exists.

    Returns:
        mu_binary : DataFrame of 0/1 values (cell_lines × genes)
    """
    if verbose:
        print("Loading CCLE MU (may take a moment) …")
    mu_raw = pd.read_csv(_CCLE_MU, usecols=["ModelID", "HugoSymbol"])
    mu_raw = mu_raw.dropna(subset=["ModelID", "HugoSymbol"])
    # Build binary matrix
    mu_raw["mutated"] = 1
    mu_binary = (
        mu_raw.groupby(["ModelID", "HugoSymbol"])["mutated"]
        .max()
        .unstack(fill_value=0)
    )
    mu_binary.index.name = "cell_line"
    return mu_binary


# ─────────────────────────────────────────────────────────────────────────────
# Z-score normalisation (per gene, across cell lines)
# ─────────────────────────────────────────────────────────────────────────────

def zscore_ge(ge_df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene column across all cell lines.

    Genes with zero variance are set to 0 after z-scoring.
    """
    mu = ge_df.mean(axis=0)
    sigma = ge_df.std(axis=0)
    sigma = sigma.replace(0, np.nan)
    z = (ge_df - mu) / sigma
    return z.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Pathway score computation for a single cell line
# ─────────────────────────────────────────────────────────────────────────────

def compute_pathway_scores(
    ach_id: str,
    ge_z: pd.DataFrame,
    cn_df: pd.DataFrame,
    mu_df: pd.DataFrame,
    pathways: dict[str, list[str]],
) -> np.ndarray:
    """Compute [P, 3] pathway score matrix for one cell line.

    Args:
        ach_id   : CCLE ACH cell line ID (e.g. 'ACH-000681')
        ge_z     : z-scored GE DataFrame (all cell lines × genes)
        cn_df    : CNV DataFrame (cell lines × genes)
        mu_df    : binary mutation DataFrame (cell lines × genes)
        pathways : ordered dict {pathway_name: [gene_symbols]}

    Returns:
        scores : np.ndarray of shape [P, 3]
                 scores[:, 0] = expr_score
                 scores[:, 1] = cnv_score
                 scores[:, 2] = mut_score
    """
    P = len(pathways)
    scores = np.zeros((P, 3), dtype=np.float32)

    # Row availability
    has_ge  = ach_id in ge_z.index
    has_cnv = ach_id in cn_df.index
    has_mu  = ach_id in mu_df.index

    ge_row  = ge_z.loc[ach_id]  if has_ge  else None
    cnv_row = cn_df.loc[ach_id] if has_cnv else None
    mu_row  = mu_df.loc[ach_id] if has_mu  else None

    for i, (pw_name, pw_genes) in enumerate(pathways.items()):
        # ── expr_score: mean z-score over pathway genes ───────────────────
        if has_ge:
            overlap_ge = [g for g in pw_genes if g in ge_row.index]
            if overlap_ge:
                scores[i, 0] = float(ge_row[overlap_ge].mean())

        # ── cnv_score: mean |CNV - 1| over pathway genes ─────────────────
        if has_cnv:
            overlap_cn = [g for g in pw_genes if g in cnv_row.index]
            if overlap_cn:
                cnv_vals = cnv_row[overlap_cn].fillna(1.0).values
                scores[i, 1] = float(np.abs(cnv_vals - 1.0).mean())

        # ── mut_score: fraction of mutated pathway genes ──────────────────
        if has_mu:
            overlap_mu = [g for g in pw_genes if g in mu_row.index]
            total = len(pw_genes)
            if overlap_mu and total > 0:
                scores[i, 2] = float(mu_row[overlap_mu].sum()) / total

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# High-level: compute pathway omics for a set of cell lines
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_pathway_omics(
    cell_line_name_to_ach: dict[str, str],
    pathways: dict[str, list[str]],
    verbose: bool = True,
) -> dict[str, dict]:
    """Compute pathway scores for each cell line.

    Args:
        cell_line_name_to_ach : mapping from project cell line name → ACH ID
        pathways              : output of load_hallmark() or load_kegg()
        verbose               : print progress

    Returns:
        result dict:
          {cell_line_name: {"ach_id": str, "pathway_names": [str], "scores": [[P, 3] as list]}}
    """
    # Load CCLE data
    ge_raw, ccle_genes = load_ccle_ge(verbose)
    ge_z   = zscore_ge(ge_raw)
    cn_df  = load_ccle_cnv(ge_raw.index, verbose)
    mu_df  = load_ccle_mu(verbose)

    pathway_names = list(pathways.keys())
    if verbose:
        print(f"Computing scores for {len(cell_line_name_to_ach)} cell lines × "
              f"{len(pathway_names)} pathways …")

    result = {}
    for cl_name, ach_id in cell_line_name_to_ach.items():
        scores = compute_pathway_scores(ach_id, ge_z, cn_df, mu_df, pathways)
        result[cl_name] = {
            "ach_id": ach_id,
            "pathway_names": pathway_names,
            "scores": scores.tolist(),  # [P, 3]
        }
        if verbose:
            print(f"  {cl_name} ({ach_id}): shape {scores.shape}, "
                  f"expr_mean={scores[:,0].mean():.3f}, "
                  f"cnv_mean={scores[:,1].mean():.3f}, "
                  f"mut_mean={scores[:,2].mean():.3f}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Runtime loading helper (used in train_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_pathway_omics_dict(json_path: str) -> dict[str, np.ndarray]:
    """Load precomputed pathway scores from JSON.

    Returns:
        {cell_line_name: np.ndarray of shape [P, 3]}
    """
    with open(json_path) as f:
        data = json.load(f)
    return {
        cl: np.array(v["scores"], dtype=np.float32)
        for cl, v in data.items()
    }


def get_pathway_names(json_path: str) -> list[str]:
    """Return the ordered list of pathway names from a precomputed JSON."""
    with open(json_path) as f:
        data = json.load(f)
    first = next(iter(data.values()))
    return first["pathway_names"]


# ─────────────────────────────────────────────────────────────────────────────
# Gene-level pathway omics (for GeneSetEncoder)
# ─────────────────────────────────────────────────────────────────────────────

def select_pathway_genes(
    pathways: dict[str, list[str]],
    ge_raw: pd.DataFrame,
    g_max: int = 200,
) -> dict[str, list[str]]:
    """For each pathway, select up to g_max genes.

    When G_p > g_max, keep the g_max genes with highest expression variance
    across CCLE cell lines (most discriminative for cell-line differences).
    Gene order is fixed regardless of cell line.
    """
    gene_var = ge_raw.var(axis=0)
    selected = {}
    for pw_name, pw_genes in pathways.items():
        if len(pw_genes) <= g_max:
            selected[pw_name] = list(pw_genes)
        else:
            var_sub = gene_var.reindex(pw_genes, fill_value=0.0)
            selected[pw_name] = var_sub.nlargest(g_max).index.tolist()
    return selected


def compute_gene_level_pathway_omics(
    cell_line_name_to_ach: dict[str, str],
    pathways: dict[str, list[str]],
    g_max: int = 200,
    verbose: bool = True,
) -> dict:
    """Compute gene-level pathway omics tensors for GeneSetEncoder.

    For each cell line and each pathway, builds a [G_max, 3] matrix of
    per-gene omics values:
        channel 0 : GE z-score (standardised across all CCLE cell lines)
        channel 1 : |CNV - 1|  (deviation from diploid)
        channel 2 : mutation   (binary 0/1)

    Padding (positions beyond G_p) is zero-filled with mask=True.

    Returns dict with keys:
        gene_omics     : np.ndarray [N_cl, P, G_max, 3]  float32
        gene_mask      : np.ndarray [N_cl, P, G_max]      bool (True=padding)
        cell_line_names: list[str]
        pathway_names  : list[str]
        gene_names     : list[list[str]]  — [P][G_p] selected gene symbols
    """
    ge_raw, _ = load_ccle_ge(verbose=verbose)
    ge_z      = zscore_ge(ge_raw)
    cn_df     = load_ccle_cnv(ge_raw.index, verbose=verbose)
    mu_df     = load_ccle_mu(verbose=verbose)

    gene_selections = select_pathway_genes(pathways, ge_raw, g_max)
    pathway_names   = list(gene_selections.keys())
    P               = len(pathway_names)
    cell_line_names = list(cell_line_name_to_ach.keys())
    N_cl            = len(cell_line_names)

    gene_omics_all = np.zeros((N_cl, P, g_max, 3), dtype=np.float32)
    gene_mask_all  = np.ones((N_cl, P, g_max),      dtype=bool)

    if verbose:
        n_truncated = sum(1 for pw, genes in pathways.items() if len(genes) > g_max)
        print(f"Gene-level omics: P={P}, G_max={g_max}, "
              f"pathways truncated (G_p>{g_max}): {n_truncated}")

    for k, (cl_name, ach_id) in enumerate(cell_line_name_to_ach.items()):
        has_ge  = ach_id in ge_z.index
        has_cnv = ach_id in cn_df.index
        has_mu  = ach_id in mu_df.index

        ge_row  = ge_z.loc[ach_id]  if has_ge  else None
        cnv_row = cn_df.loc[ach_id] if has_cnv else None
        mu_row  = mu_df.loc[ach_id] if has_mu  else None

        for i, pw_name in enumerate(pathway_names):
            sel = gene_selections[pw_name]
            G_p = len(sel)
            gene_mask_all[k, i, :G_p] = False          # mark real genes

            if has_ge:
                vals = ge_row.reindex(sel, fill_value=0.0).values.astype(np.float32)
                gene_omics_all[k, i, :G_p, 0] = vals

            if has_cnv:
                vals = cnv_row.reindex(sel, fill_value=1.0).fillna(1.0).values.astype(np.float32)
                gene_omics_all[k, i, :G_p, 1] = np.abs(vals - 1.0)

            if has_mu:
                vals = mu_row.reindex(sel, fill_value=0.0).values.astype(np.float32)
                gene_omics_all[k, i, :G_p, 2] = vals

        if verbose:
            print(f"  {cl_name} ({ach_id}): done  "
                  f"[expr_mean={gene_omics_all[k,:,:,0].mean():.3f}, "
                  f"cnv_mean={gene_omics_all[k,:,:,1].mean():.4f}]")

    return {
        "gene_omics":      gene_omics_all,
        "gene_mask":       gene_mask_all,
        "cell_line_names": cell_line_names,
        "pathway_names":   pathway_names,
        "gene_names":      [gene_selections[pw] for pw in pathway_names],
    }


def load_geneset_omics_dict(
    npz_path: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Load gene-level pathway omics from .npz file (produced by preprocess script).

    Returns:
        geneset_dict  : {cell_line_name: (gene_omics [P,G_max,3], gene_mask [P,G_max])}
        pathway_names : ordered list of pathway names
    """
    data = np.load(npz_path, allow_pickle=True)
    cl_names      = data["cell_line_names"].tolist()
    gene_omics_all = data["gene_omics"]    # [N_cl, P, G_max, 3]
    gene_mask_all  = data["gene_mask"]     # [N_cl, P, G_max]
    pathway_names  = data["pathway_names"].tolist()

    geneset_dict = {
        cl: (gene_omics_all[i].astype(np.float32), gene_mask_all[i])
        for i, cl in enumerate(cl_names)
    }
    return geneset_dict, pathway_names


# ─────────────────────────────────────────────────────────────────────────────
# GE-only 6-stat pathway features
# ─────────────────────────────────────────────────────────────────────────────

def compute_ge_pathway_stats_selfatt(
    ach_id: str,
    ge_z: pd.DataFrame,
    pathways: dict[str, list[str]],
) -> np.ndarray:
    """Compute [P, 6] GE-only distribution stats for one cell line.

    Features per pathway:
        0: expr_mean    — mean z-score of pathway genes
        1: expr_std     — std of z-scores
        2: expr_q25     — 25th percentile of pathway gene z-scores
        3: expr_q75     — 75th percentile of pathway gene z-scores
        4: expr_frac_up — fraction of pathway genes above cell-line P75 (all genes)
        5: expr_frac_dn — fraction of pathway genes below cell-line P25 (all genes)

    Thresholds for frac_up/dn are cell-line-specific (computed from full ~17k-gene transcriptome).
    """
    P = len(pathways)
    stats = np.zeros((P, 6), dtype=np.float32)

    if ach_id not in ge_z.index:
        return stats

    ge_row = ge_z.loc[ach_id]
    cl_p25 = float(np.percentile(ge_row.values, 25))
    cl_p75 = float(np.percentile(ge_row.values, 75))

    for i, (pw_name, pw_genes) in enumerate(pathways.items()):
        overlap = [g for g in pw_genes if g in ge_row.index]
        if not overlap:
            continue
        vals = ge_row[overlap].values.astype(np.float32)
        stats[i, 0] = vals.mean()
        stats[i, 1] = vals.std()
        stats[i, 2] = float(np.percentile(vals, 25))
        stats[i, 3] = float(np.percentile(vals, 75))
        stats[i, 4] = float((vals > cl_p75).mean())
        stats[i, 5] = float((vals < cl_p25).mean())

    return stats


def compute_all_ge_pathway_stats_selfatt(
    cell_line_name_to_ach: dict[str, str],
    pathways: dict[str, list[str]],
    verbose: bool = True,
) -> dict:
    """Compute [P, 6] GE stats for all cell lines. Returns JSON-serialisable dict."""
    ge_raw, _ = load_ccle_ge(verbose=verbose)
    ge_z = zscore_ge(ge_raw)

    pathway_names = list(pathways.keys())
    result = {}
    for cl_name, ach_id in cell_line_name_to_ach.items():
        scores = compute_ge_pathway_stats_selfatt(ach_id, ge_z, pathways)
        result[cl_name] = {
            "ach_id": ach_id,
            "pathway_names": pathway_names,
            "scores": scores.tolist(),
        }
        if verbose:
            print(f"  {cl_name} ({ach_id}): mean={scores[:,0].mean():.3f}, "
                  f"frac_up_avg={scores[:,4].mean():.3f}")
    return result


def load_ge_pathway_dict_selfatt(json_path: str) -> dict[str, np.ndarray]:
    """Load ge_pathway_selfatt stats. Returns {cell_line: np.ndarray [P, 6]}."""
    with open(json_path) as f:
        data = json.load(f)
    return {cl: np.array(v["scores"], dtype=np.float32) for cl, v in data.items()}


# ─────────────────────────────────────────────────────────────────────────────
# pair_pathway_evidence mode: Reactome [P,8] activity + per-gene omics matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_reactome_activity(
    cell_line_name_to_ach: dict[str, str],
    pathways: dict[str, list[str]],
    verbose: bool = True,
) -> dict:
    """Per-cell-line [P, 8] pathway activity = 6 GE distribution stats + cnv_score + mut_score.

    Columns 0-5 reuse :func:`compute_ge_pathway_stats_selfatt`
    (expr_mean/std/q25/q75/frac_up/frac_dn); columns 6-7 reuse the cnv/mut channels
    of :func:`compute_pathway_scores`. CCLE is loaded once. Cell-line & pathway specific,
    gene-pair INDEPENDENT — this is the cross-attention "context" part of the token.
    """
    ge_raw, _ = load_ccle_ge(verbose=verbose)
    ge_z  = zscore_ge(ge_raw)
    cn_df = load_ccle_cnv(ge_raw.index, verbose=verbose)
    mu_df = load_ccle_mu(verbose=verbose)

    pathway_names = list(pathways.keys())
    result = {}
    for cl_name, ach_id in cell_line_name_to_ach.items():
        ge6 = compute_ge_pathway_stats_selfatt(ach_id, ge_z, pathways)          # [P,6]
        cnvmut = compute_pathway_scores(ach_id, ge_z, cn_df, mu_df, pathways)[:, 1:3]  # [P,2]
        scores = np.concatenate([ge6, cnvmut.astype(np.float32)], axis=1)        # [P,8]
        result[cl_name] = {
            "ach_id": ach_id,
            "pathway_names": pathway_names,
            "scores": scores.tolist(),
        }
        if verbose:
            print(f"  {cl_name} ({ach_id}): [P,8] expr_mean={scores[:,0].mean():.3f} "
                  f"cnv={scores[:,6].mean():.4f} mut={scores[:,7].mean():.3f}")
    return result


def compute_gene_omics_matrix(
    cell_line_name_to_ach: dict[str, str],
    symbols: list[str],
    verbose: bool = True,
) -> dict:
    """Per-gene omics [N_cl, n_sym, 3] for a fixed symbol list (channels: GE z, |CNV-1|, MU).

    Mirrors the per-gene channel definitions in :func:`compute_gene_level_pathway_omics`
    (reindex with fill GE=0 / CNV=1 / MU=0). Used to inject the SPECIFIC gene-pair's
    omics state into each evidence token (the pair-conditioning signal).

    Returns: {"omics": [N_cl, n_sym, 3] float32, "symbols": [str], "cell_line_names": [str]}
    """
    ge_raw, _ = load_ccle_ge(verbose=verbose)
    ge_z  = zscore_ge(ge_raw)
    cn_df = load_ccle_cnv(ge_raw.index, verbose=verbose)
    mu_df = load_ccle_mu(verbose=verbose)

    cell_line_names = list(cell_line_name_to_ach.keys())
    n_sym = len(symbols)
    omics = np.zeros((len(cell_line_names), n_sym, 3), dtype=np.float32)

    for k, (cl_name, ach_id) in enumerate(cell_line_name_to_ach.items()):
        if ach_id in ge_z.index:
            omics[k, :, 0] = ge_z.loc[ach_id].reindex(symbols, fill_value=0.0).values.astype(np.float32)
        if ach_id in cn_df.index:
            cnv = cn_df.loc[ach_id].reindex(symbols, fill_value=1.0).fillna(1.0).values.astype(np.float32)
            omics[k, :, 1] = np.abs(cnv - 1.0)
        if ach_id in mu_df.index:
            omics[k, :, 2] = mu_df.loc[ach_id].reindex(symbols, fill_value=0.0).values.astype(np.float32)
        if verbose:
            print(f"  {cl_name} ({ach_id}): gene omics filled "
                  f"[ge_mean={omics[k,:,0].mean():.3f} mut_frac={omics[k,:,2].mean():.3f}]")

    return {"omics": omics, "symbols": list(symbols), "cell_line_names": cell_line_names}
