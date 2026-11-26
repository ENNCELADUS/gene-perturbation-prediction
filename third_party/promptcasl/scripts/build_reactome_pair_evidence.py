#!/usr/bin/env python3
"""Precompute the Reactome pathway-evidence artifacts used by pair_pathway_evidence.

Run: python -m scripts.build_reactome_pair_evidence --device 0
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.pathway_utils import (
    load_reactome,
    load_ccle_ge,
    compute_all_reactome_activity,
    compute_gene_omics_matrix,
)
from src.pair_pathway import N_FEATURES
from src.config import BIOMEDBERT, MULTI_OMICS_DIR

BIOBERT = BIOMEDBERT

# ACH IDs for the paper cell lines.
CELL_LINE_ACH = {
    "A549":   "ACH-000681",
    "PK1":    "ACH-000307",
    "IPC298": "ACH-000915",
    "MEWO":   "ACH-000987",
    "GI1":    "ACH-000756",
    "HS936T": "ACH-000801",
}


def humanize(name: str) -> str:
    """REACTOME_INTERLEUKIN_6_SIGNALING -> 'interleukin 6 signaling'."""
    return name[len("REACTOME_"):].replace("_", " ").lower()


def build_desc_emb(pathway_names, device):
    """Full BiomedBERT (12-layer encoder) over humanized names, masked mean-pool,
    then mean-center the bank + L2-normalize (de-anisotropize so the frozen branch
    does not dominate early training)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BIOBERT)
    model = AutoModel.from_pretrained(BIOBERT).to(device).eval()
    texts = [humanize(n) for n in pathway_names]
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            hs = model(**enc).last_hidden_state            # [b,L,768] full forward
        m = enc.attention_mask.unsqueeze(-1).float()
        out.append(((hs * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    desc = torch.cat(out).numpy().astype(np.float32)        # [P,768]
    desc = desc - desc.mean(0, keepdims=True)               # center bank
    desc = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-8)
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--min_ccle_overlap", type=int, default=15)
    args = ap.parse_args()
    device = f"cuda:{args.device}"
    out_dir = MULTI_OMICS_DIR

    # ── Reactome gene sets filtered to CCLE genes ───────────────────────────────
    _, ccle_genes = load_ccle_ge(verbose=True)
    pathways = load_reactome(min_ccle_overlap=args.min_ccle_overlap, ccle_genes=ccle_genes)
    pathway_names = list(pathways.keys())
    P = len(pathway_names)
    pathway_sizes = [len(pathways[n]) for n in pathway_names]
    print(f"\nReactome: {P} pathways (min_overlap={args.min_ccle_overlap})")

    # gene -> sorted pathway indices (membership); symbols = union (all in CCLE)
    gene2pathways = {}
    for idx, name in enumerate(pathway_names):
        for g in pathways[name]:
            gene2pathways.setdefault(g, []).append(idx)
    symbols = sorted(gene2pathways)
    n_sym = len(symbols)
    print(f"genes covered (symbols) = {n_sym}")

    with open(os.path.join(out_dir, "reactome_pathways.json"), "w") as f:
        json.dump({"pathway_names": pathway_names,
                   "pathway_sizes": pathway_sizes,
                   "gene2pathways": gene2pathways}, f)
    print("saved reactome_pathways.json")

    # ── [P,8] activity per cell line ─────────────────────────────────────────
    print("\nComputing per-pathway activity [P,8] …")
    activity = compute_all_reactome_activity(CELL_LINE_ACH, pathways, verbose=True)
    with open(os.path.join(out_dir, "reactome_activity.json"), "w") as f:
        json.dump(activity, f)
    print("saved reactome_activity.json")

    # ── per-gene omics [N_cl, n_sym, 3] ──────────────────────────────────────
    print("\nComputing per-gene omics matrix …")
    gom = compute_gene_omics_matrix(CELL_LINE_ACH, symbols, verbose=True)
    np.savez_compressed(os.path.join(out_dir, "reactome_gene_omics.npz"),
                        symbols=np.array(gom["symbols"]),
                        cell_line_names=np.array(gom["cell_line_names"]),
                        omics=gom["omics"])
    print(f"saved reactome_gene_omics.npz  omics={gom['omics'].shape}")

    # ── frozen description embeddings [P,768] ────────────────────────────────
    print("\nEncoding pathway descriptions with BiomedBERT …")
    desc = build_desc_emb(pathway_names, device)
    np.save(os.path.join(out_dir, "reactome_desc_emb.npy"), desc)
    print(f"saved reactome_desc_emb.npy  {desc.shape}")

    # ── manifest ─────────────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "reactome_manifest.json"), "w") as f:
        json.dump({"P": P, "n_sym": n_sym, "n_features": N_FEATURES,
                   "min_ccle_overlap": args.min_ccle_overlap, "biobert": BIOBERT}, f, indent=2)
    print(f"\nDONE. P={P}, n_sym={n_sym}, N_FEATURES={N_FEATURES}")


if __name__ == "__main__":
    main()
