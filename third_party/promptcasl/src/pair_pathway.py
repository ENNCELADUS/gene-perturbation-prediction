"""Runtime builder for the `pair_pathway_evidence` omics mode.

Assembles PAIR-CONDITIONED Reactome evidence tokens on the fly from precomputed
pair-INDEPENDENT tables (built by scripts/build_reactome_pair_evidence.py). For a
gene pair (A,B) in cell line k it selects the top-K Reactome pathways biologically
connected to the pair (pw(A)∪pw(B)) and emits, per selected pathway p, a feature
vector O_{p,ij,k} so cross-attention reads pair-specific evidence instead of a
cell-line-constant pathway profile.

No learnable pathway-id embedding: the frozen description embedding (built offline)
is the generalizable pathway identity anchor.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import torch

# Per-token feature dimension. SINGLE source of truth — the encoder reads
# numeric.shape[-1] and the builder asserts against this; never hardcode 24 elsewhere.
N_FEATURES = 24

# Lineage of each cell line (for lineage-normalized pathway activity, feature 23).
LINEAGE = {
    "A549": "lung", "PK1": "pancreas", "IPC298": "melanoma", "MEWO": "melanoma",
    "HS936T": "melanoma", "GI1": "gbm",
}

_EMPTY = np.empty(0, dtype=np.int64)
_ZERO3 = np.zeros(3, dtype=np.float32)

_DEFAULT_ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "multi_omics"
)


class PairPathwayEvidenceBuilder:
    """Holds precomputed Reactome tables; builds pair-conditioned token batches.

    Tables (all pair-independent): pathway membership, per-cell-line [P,8] activity,
    per-cell-line per-gene omics, frozen description embeddings, lineage means.
    """

    def __init__(self, artifact_dir: str | None = None, k: int = 16):
        self.dir = artifact_dir or _DEFAULT_ARTIFACT_DIR
        self.k = int(k)
        self.n_features = N_FEATURES

        with open(os.path.join(self.dir, "reactome_pathways.json")) as f:
            pw = json.load(f)
        self.pathway_names: list[str] = pw["pathway_names"]
        self.P = len(self.pathway_names)
        self.gene2pws: dict[str, np.ndarray] = {
            g: np.asarray(sorted(idx), dtype=np.int64) for g, idx in pw["gene2pathways"].items()
        }

        with open(os.path.join(self.dir, "reactome_activity.json")) as f:
            act = json.load(f)
        self.activity: dict[str, np.ndarray] = {
            cl: np.asarray(v["scores"], dtype=np.float32) for cl, v in act.items()
        }  # cl -> [P,8]

        gom = np.load(os.path.join(self.dir, "reactome_gene_omics.npz"), allow_pickle=True)
        symbols = [str(s) for s in gom["symbols"].tolist()]
        self.gene_row = {s: i for i, s in enumerate(symbols)}
        cl_names = [str(c) for c in gom["cell_line_names"].tolist()]
        omics = gom["omics"].astype(np.float32)                                          # [N_cl,n_sym,3]
        self.gom_by_cl: dict[str, np.ndarray] = {cl: omics[i] for i, cl in enumerate(cl_names)}

        self.desc_emb = np.load(os.path.join(self.dir, "reactome_desc_emb.npy")).astype(np.float32)  # [P,768]
        self.desc_dim = self.desc_emb.shape[1]
        assert self.desc_emb.shape[0] == self.P, "desc_emb / pathway_names length mismatch"

        # lineage-mean of expr_mean (activity col 0), averaged over same-lineage cell lines
        self.cl2lineage = LINEAGE
        groups = defaultdict(list)
        for cl in self.activity:
            groups[LINEAGE.get(cl, cl)].append(cl)
        self.lineage_mean: dict[str, np.ndarray] = {
            lin: np.mean([self.activity[c][:, 0] for c in cls], axis=0).astype(np.float32)
            for lin, cls in groups.items()
        }

        # cross-check manifest if present
        man_path = os.path.join(self.dir, "reactome_manifest.json")
        if os.path.exists(man_path):
            with open(man_path) as f:
                man = json.load(f)
            assert man.get("n_features", N_FEATURES) == N_FEATURES, \
                f"manifest n_features={man.get('n_features')} != N_FEATURES={N_FEATURES}"

    def _gene_omics(self, cl: str, gene: str) -> np.ndarray:
        j = self.gene_row.get(gene)
        if j is None:
            return _ZERO3
        return self.gom_by_cl[cl][j]

    def _select_topk(self, union: np.ndarray, shared: np.ndarray, cl: str) -> np.ndarray:
        """Deterministic top-K: shared membership, then descending |mean expression|,
        with a fixed pathway index used only to break ties."""
        if len(union) <= self.k:
            return union
        expr = self.activity[cl][union, 0]
        is_shared = np.isin(union, shared)
        # np.lexsort: last key is primary -> (~shared), then -|expr|, then index.
        order = np.lexsort((union, -np.abs(expr), (~is_shared).astype(np.int8)))
        return union[order[:self.k]]

    def build_batch(self, gene_pair_list, cell_line_list, device):
        """-> (numeric [B,K,F] float32, desc_emb [B,K,768] float32, pad_mask [B,K] bool).

        pad_mask True = padded slot (ignored downstream). A pair with no Reactome
        pathway yields an all-padded row → text-only context fallback.
        """
        B, K, F = len(gene_pair_list), self.k, self.n_features
        numeric = np.zeros((B, K, F), dtype=np.float32)
        desc = np.zeros((B, K, self.desc_dim), dtype=np.float32)
        pad_mask = np.ones((B, K), dtype=bool)

        for i, (gp, cl) in enumerate(zip(gene_pair_list, cell_line_list)):
            g1, _, g2 = str(gp).partition("|")
            pa = self.gene2pws.get(g1, _EMPTY)
            pb = self.gene2pws.get(g2, _EMPTY)
            union = np.union1d(pa, pb)
            if union.size == 0:
                continue
            shared = np.intersect1d(pa, pb, assume_unique=True)
            sel = self._select_topk(union, shared, cl)
            n = len(sel)

            a_in = np.isin(sel, pa)
            b_in = np.isin(sel, pb)
            ga = self._gene_omics(cl, g1)
            gb = self._gene_omics(cl, g2)
            asel = self.activity[cl][sel]                                    # [n,8]

            blk = numeric[i, :n]
            blk[:, 0] = a_in
            blk[:, 1] = b_in
            blk[:, 2] = a_in & b_in                                          # both
            blk[:, 3] = a_in ^ b_in                                          # union-only
            blk[:, 4:7] = ga                                                 # gene A omics (broadcast)
            blk[:, 7:10] = gb                                                # gene B omics
            blk[:, 10] = min(ga[0], gb[0])                                   # pooled GE z min
            blk[:, 11] = max(ga[0], gb[0])                                   # pooled GE z max
            blk[:, 12] = 0.5 * (ga[0] + gb[0])                               # pooled GE z mean
            blk[:, 13] = max(ga[1], gb[1])                                   # pooled |CNV-1| max
            blk[:, 14] = max(ga[2], gb[2])                                   # pooled MU max
            blk[:, 15:21] = asel[:, 0:6]                                     # 6 GE pathway stats
            blk[:, 21] = asel[:, 6]                                          # cnv_score
            blk[:, 22] = asel[:, 7]                                          # mut_score
            blk[:, 23] = asel[:, 0] - self.lineage_mean[self.cl2lineage.get(cl, cl)][sel]

            desc[i, :n] = self.desc_emb[sel]
            pad_mask[i, :n] = False

        return (torch.from_numpy(numeric).to(device),
                torch.from_numpy(desc).to(device),
                torch.from_numpy(pad_mask).to(device))
