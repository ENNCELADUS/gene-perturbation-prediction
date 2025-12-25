"""
Data collator for Route B1 gene-level scoring.

Builds batches of perturbed cells with multi-hot gene targets derived
from condition labels.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

# Add scGPT to path
scgpt_path = Path(__file__).parent.parent.parent / "scGPT"
if str(scgpt_path) not in sys.path:
    sys.path.insert(0, str(scgpt_path))

from scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch

from ..evaluate.metrics import parse_condition_genes


class GeneScoreDataset(Dataset):
    """
    Dataset for gene-level scoring (Route B1).

    Each example is a perturbed cell with its condition-derived gene targets.
    """

    def __init__(
        self,
        adata,
        conditions: List[str],
        vocab: Dict[str, int],
        n_bins: int = 51,
        control_key: str = "control",
        condition_key: str = "condition",
        seed: int = 42,
    ):
        self.adata = adata
        self.conditions = set(conditions)
        self.vocab = vocab
        self.n_bins = n_bins
        self.control_key = control_key
        self.condition_key = condition_key
        self.rng = np.random.RandomState(seed)

        # Use 'gene_name' column if available (gene symbols),
        # otherwise fall back to var_names (may be Ensembl IDs)
        if "gene_name" in self.adata.var.columns:
            self.gene_names = self.adata.var["gene_name"].tolist()
        else:
            self.gene_names = self.adata.var_names.tolist()
        self.gene_name_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        self.gene_ids = np.array(
            [self.vocab.get(g, self.vocab.get("<pad>", 0)) for g in self.gene_names]
        )

        self.examples = []
        self.condition_to_indices: Dict[str, List[int]] = {}

        for idx, condition in enumerate(self.adata.obs[self.condition_key]):
            if condition == "ctrl":
                continue
            if condition not in self.conditions:
                continue
            if self.adata.obs[self.control_key].iloc[idx] == 1:
                continue
            self.examples.append((idx, condition))
            self.condition_to_indices.setdefault(condition, []).append(
                len(self.examples) - 1
            )

        self.condition_to_gene_indices = {
            condition: self._condition_gene_indices(condition)
            for condition in self.condition_to_indices
        }

        print(
            f"GeneScoreDataset: {len(self.examples)} perturbed cells across {len(self.condition_to_indices)} conditions"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        cell_idx, condition = self.examples[idx]

        expr = (
            self.adata.X[cell_idx].toarray().flatten()
            if hasattr(self.adata.X, "toarray")
            else self.adata.X[cell_idx]
        )
        expr_binned = self._bin_expression(expr)

        pert_gene_indices = self.condition_to_gene_indices.get(condition, [])

        return {
            "expr": expr_binned,
            "gene_ids": self.gene_ids,
            "condition": condition,
            "pert_gene_indices": pert_gene_indices,
        }

    def _bin_expression(self, expr: np.ndarray) -> np.ndarray:
        """Bin expression values to [0, n_bins-1]."""
        expr_clip = np.clip(expr, 0, None)
        max_val = expr_clip.max() if expr_clip.max() > 0 else 1.0
        binned = np.floor(expr_clip / max_val * (self.n_bins - 1)).astype(int)
        binned = np.clip(binned, 0, self.n_bins - 1)
        return binned

    def _condition_gene_indices(self, condition: str) -> List[int]:
        genes = parse_condition_genes(condition)
        indices = [
            self.gene_name_to_idx[g] for g in genes if g in self.gene_name_to_idx
        ]
        return indices


def collate_gene_score_batch(
    batch: List[Dict],
    vocab: Dict[str, int],
    n_genes: int,
    max_len: int = 1200,
    pad_token: str = "<pad>",
    pad_value: int = -2,
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of gene scoring examples.

    Returns:
        Dictionary with tokenized inputs and multi-hot gene targets.
    """
    exprs = np.stack([ex["expr"] for ex in batch])
    gene_ids = batch[0]["gene_ids"]

    # Safeguard: ensure no all-zero rows exist to avoid empty tensors.
    # PyTorch transformer's to_padded_tensor fails with empty sequences.
    for i in range(len(exprs)):
        if np.count_nonzero(exprs[i]) == 0:
            exprs[i, 0] = 1  # Set minimal placeholder value

    tokenized = tokenize_and_pad_batch(
        exprs,
        gene_ids,
        max_len=max_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=False,
        return_pt=True,
    )

    padding_mask = tokenized["genes"] == vocab[pad_token]

    targets = np.zeros((len(batch), n_genes), dtype=np.float32)
    for i, ex in enumerate(batch):
        if ex["pert_gene_indices"]:
            targets[i, ex["pert_gene_indices"]] = 1.0

    return {
        "genes": tokenized["genes"],
        "values": tokenized["values"],
        "padding_mask": padding_mask,
        "targets": torch.from_numpy(targets),
        "conditions": [ex["condition"] for ex in batch],
    }
