"""
Data collator for Route B1 gene-level scoring.

Builds batches of perturbed cells with multi-hot gene targets derived
from condition labels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
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
from .preprocess_utils import preprocess_counts_to_bins, scgpt_binning


class GeneScoreDataset(Dataset):
    """
    Dataset for gene-level scoring (Route B1).

    Each example is a perturbed cell with matched control cells for delta embeddings.
    """

    def __init__(
        self,
        adata,
        conditions: List[str],
        vocab: Dict[str, int],
        n_bins: int = 51,
        control_key: str = "control",
        condition_key: str = "condition",
        match_keys: Optional[Sequence[str]] = None,
        n_control_samples: int = 8,
        sample_weight_alpha: Optional[float] = None,
        sample_weight_cap: float = 10.0,
        sample_weight_eps: float = 1.0,
        seed: int = 42,
    ):
        self.adata = adata
        self.conditions = set(conditions)
        self.vocab = vocab
        self.n_bins = n_bins
        self.control_key = control_key
        self.condition_key = condition_key
        self.match_keys = (
            list(match_keys)
            if match_keys is not None
            else [
                "batch",
                "cell_type",
            ]
        )
        self.n_control_samples = n_control_samples
        self.sample_weight_alpha = sample_weight_alpha
        self.sample_weight_cap = sample_weight_cap
        self.sample_weight_eps = sample_weight_eps
        if self.n_control_samples <= 0:
            raise ValueError("n_control_samples must be a positive integer.")
        self.rng = np.random.RandomState(seed)
        self.binning_rng = np.random.RandomState(seed + 1)

        if "counts" in self.adata.layers:
            self.expr_layer = "counts"
        else:
            self.expr_layer = None
            print("Warning: counts layer missing; falling back to adata.X for inputs.")

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

        self.control_indices = np.where(self.adata.obs[self.control_key] == 1)[0]
        if len(self.control_indices) == 0:
            raise ValueError(
                f"No control cells found for control_key='{self.control_key}'. "
                "Delta-embedding training requires matched controls."
            )

        self.match_keys = [k for k in self.match_keys if k in self.adata.obs.columns]
        self.control_pool_by_key: Dict[Tuple, np.ndarray] = {}
        if self.match_keys:
            for idx in self.control_indices:
                key = self._get_match_key(idx)
                self.control_pool_by_key.setdefault(key, []).append(idx)
            for key, values in list(self.control_pool_by_key.items()):
                self.control_pool_by_key[key] = np.array(values)
        else:
            self.control_pool_by_key[()] = self.control_indices

        self.examples = []
        self.condition_to_indices: Dict[str, List[int]] = {}

        for idx, condition in enumerate(self.adata.obs[self.condition_key]):
            if condition == "ctrl":
                continue
            if condition not in self.conditions:
                continue
            if self.adata.obs[self.control_key].iloc[idx] == 1:
                continue
            match_key = self._get_match_key(idx)
            self.examples.append((idx, condition, match_key))
            self.condition_to_indices.setdefault(condition, []).append(
                len(self.examples) - 1
            )

        self.condition_to_gene_indices = {
            condition: self._condition_gene_indices(condition)
            for condition in self.condition_to_indices
        }
        self.example_gene_indices: List[List[int]] = []
        self.gene_counts = np.zeros(len(self.gene_names), dtype=np.int64)
        for _, condition, _ in self.examples:
            gene_indices = self.condition_to_gene_indices.get(condition, [])
            self.example_gene_indices.append(gene_indices)
            for gene_idx in gene_indices:
                self.gene_counts[gene_idx] += 1

        self.sample_weights: Optional[np.ndarray] = None
        if self.sample_weight_alpha is not None and self.sample_weight_alpha > 0:
            weights = []
            for gene_indices in self.example_gene_indices:
                if not gene_indices:
                    weights.append(1.0)
                    continue
                counts = self.gene_counts[gene_indices].astype(np.float64)
                inv = (
                    1.0 / (counts + self.sample_weight_eps)
                ) ** self.sample_weight_alpha
                weight = float(inv.mean())
                if self.sample_weight_cap is not None:
                    weight = min(weight, self.sample_weight_cap)
                weights.append(weight)
            self.sample_weights = np.asarray(weights, dtype=np.float64)

        print(
            f"GeneScoreDataset: {len(self.examples)} perturbed cells across {len(self.condition_to_indices)} conditions"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        cell_idx, condition, match_key = self.examples[idx]

        expr = self._get_expression(cell_idx)
        expr_binned = self._bin_expression(expr)

        pert_gene_indices = self.condition_to_gene_indices.get(condition, [])
        control_pool = self.control_pool_by_key.get(match_key)
        if control_pool is None or len(control_pool) == 0:
            control_pool = self.control_indices
        replace = len(control_pool) < self.n_control_samples
        control_indices = self.rng.choice(
            control_pool, size=self.n_control_samples, replace=replace
        )
        control_exprs = []
        for ctrl_idx in control_indices:
            ctrl_expr = self._get_expression(ctrl_idx)
            control_exprs.append(self._bin_expression(ctrl_expr))
        control_exprs = np.stack(control_exprs)

        return {
            "expr": expr_binned,
            "control_exprs": control_exprs,
            "gene_ids": self.gene_ids,
            "condition": condition,
            "pert_gene_indices": pert_gene_indices,
        }

    def _get_expression(self, idx: int) -> np.ndarray:
        if self.expr_layer is None:
            row = self.adata.X[idx]
        else:
            row = self.adata.layers[self.expr_layer][idx]
        return row.toarray().flatten() if hasattr(row, "toarray") else np.array(row)

    def _bin_expression(self, expr: np.ndarray) -> np.ndarray:
        """scGPT-aligned: counts -> normalize_total -> log1p -> quantile binning."""
        if self.expr_layer == "counts":
            return preprocess_counts_to_bins(
                expr, n_bins=self.n_bins, rng=self.binning_rng
            )
        return scgpt_binning(expr, n_bins=self.n_bins, rng=self.binning_rng)

    def _condition_gene_indices(self, condition: str) -> List[int]:
        genes = parse_condition_genes(condition)
        indices = [
            self.gene_name_to_idx[g] for g in genes if g in self.gene_name_to_idx
        ]
        return indices

    def _get_match_key(self, idx: int) -> Tuple:
        if not self.match_keys:
            return ()
        row = self.adata.obs.iloc[idx]
        return tuple(row[key] for key in self.match_keys)

    def get_gene_weights(
        self,
        alpha: float,
        cap: Optional[float] = None,
        eps: float = 1.0,
    ) -> np.ndarray:
        """Compute per-gene weights based on inverse frequency."""
        counts = self.gene_counts.astype(np.float64)
        weights = (1.0 / (counts + eps)) ** alpha
        if cap is not None:
            weights = np.minimum(weights, cap)
        return weights


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
    batch_size = len(batch)
    exprs = np.stack([ex["expr"] for ex in batch])
    control_exprs = np.stack([ex["control_exprs"] for ex in batch])
    gene_ids = batch[0]["gene_ids"]

    # Safeguard: ensure no all-zero rows exist to avoid empty tensors.
    # PyTorch transformer's to_padded_tensor fails with empty sequences.
    for i in range(batch_size):
        if np.count_nonzero(exprs[i]) == 0:
            exprs[i, 0] = 1  # Set minimal placeholder value
    control_exprs_flat = control_exprs.reshape(-1, control_exprs.shape[-1])
    for i in range(len(control_exprs_flat)):
        if np.count_nonzero(control_exprs_flat[i]) == 0:
            control_exprs_flat[i, 0] = 1

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

    control_tokenized = tokenize_and_pad_batch(
        control_exprs_flat,
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
    control_padding_mask = control_tokenized["genes"] == vocab[pad_token]

    targets = np.zeros((len(batch), n_genes), dtype=np.float32)
    for i, ex in enumerate(batch):
        if ex["pert_gene_indices"]:
            targets[i, ex["pert_gene_indices"]] = 1.0

    controls_per_sample = control_exprs.shape[1]
    return {
        "genes": tokenized["genes"],
        "values": tokenized["values"],
        "padding_mask": padding_mask,
        "control_genes": control_tokenized["genes"],
        "control_values": control_tokenized["values"],
        "control_padding_mask": control_padding_mask,
        "control_counts": controls_per_sample,
        "targets": torch.from_numpy(targets),
        "conditions": [ex["condition"] for ex in batch],
    }
