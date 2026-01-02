"""
Data collator for scGPT forward model training.

Handles batching of (control, condition) → perturbed examples.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Add scGPT to path
scgpt_path = Path(__file__).parent.parent.parent / "scGPT"
if str(scgpt_path) not in sys.path:
    sys.path.insert(0, str(scgpt_path))

from scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch

from .preprocess_utils import preprocess_counts_to_bins, scgpt_binning


class ForwardModelDataset(Dataset):
    """
    Dataset for forward model training: (control, condition) → perturbed.

    For each perturbation condition, we pair control cells with the perturbed cells
    to create training examples.
    """

    def __init__(
        self,
        adata,  # AnnData object
        conditions: List[str],  # List of condition names
        vocab: Dict[str, int],  # Gene name → ID mapping
        n_bins: int = 51,  # Number of bins for expression values
        control_key: str = "control",  # Column name for control indicator
        condition_key: str = "condition",  # Column name for condition labels
        max_control_per_condition: int = 100,  # Max control cells to sample per condition
        seed: int = 42,
    ):
        self.adata = adata
        self.conditions = conditions
        self.vocab = vocab
        self.n_bins = n_bins
        self.control_key = control_key
        self.condition_key = condition_key
        self.max_control_per_condition = max_control_per_condition
        self.rng = np.random.RandomState(seed)
        self.binning_rng = np.random.RandomState(seed + 1)

        if "counts" in self.adata.layers:
            self.expr_layer = "counts"
        else:
            self.expr_layer = None
            print("Warning: counts layer missing; falling back to adata.X for inputs.")

        # Get control cells
        self.control_mask = adata.obs[control_key] == 1
        self.control_indices = np.where(self.control_mask)[0]
        if len(self.control_indices) == 0:
            raise ValueError(
                f"No control cells found for control_key='{control_key}'. "
                "Forward training requires control cells to pair with perturbed examples."
            )

        # Build examples: (control_idx, perturbed_idx, condition_name)
        self.examples = []
        for condition in conditions:
            # Get perturbed cells for this condition
            condition_mask = adata.obs[condition_key] == condition
            perturbed_indices = np.where(condition_mask)[0]

            if len(perturbed_indices) == 0:
                continue

            # Sample control cells
            n_control = min(len(self.control_indices), self.max_control_per_condition)
            if n_control == 0:
                continue
            sampled_control = self.rng.choice(
                self.control_indices, size=n_control, replace=False
            )

            # Pair each perturbed cell with a random control
            for pert_idx in perturbed_indices:
                ctrl_idx = self.rng.choice(sampled_control)
                self.examples.append((ctrl_idx, pert_idx, condition))

        print(
            f"Created {len(self.examples)} training examples from {len(conditions)} conditions"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ctrl_idx, pert_idx, condition = self.examples[idx]

        ctrl_expr = self._get_expression(ctrl_idx)
        pert_expr = self._get_expression(pert_idx)

        ctrl_binned = self._bin_expression(ctrl_expr)
        pert_binned = self._bin_expression(pert_expr)

        # Get gene IDs - use 'gene_name' column if available (gene symbols),
        # otherwise fall back to var_names (may be Ensembl IDs)
        if "gene_name" in self.adata.var.columns:
            gene_names = self.adata.var["gene_name"].tolist()
        else:
            gene_names = self.adata.var_names.tolist()
        gene_ids = np.array(
            [self.vocab.get(g, self.vocab.get("<pad>", 0)) for g in gene_names]
        )

        # Parse perturbation genes from condition
        pert_genes = self._parse_condition(condition)
        pert_gene_ids = [self.vocab.get(g, -1) for g in pert_genes if g in self.vocab]

        return {
            "control_expr": ctrl_binned,
            "perturbed_expr": pert_binned,
            "gene_ids": gene_ids,
            "condition": condition,
            "pert_gene_ids": pert_gene_ids,
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

    def _parse_condition(self, condition: str) -> List[str]:
        """Parse condition string to extract gene names (e.g., 'GENE1+GENE2' → ['GENE1', 'GENE2'])."""
        if not condition or condition == "ctrl":
            return []
        return [
            g.strip() for g in condition.split("+") if g.strip() and g.strip() != "ctrl"
        ]


def collate_forward_batch(
    batch: List[Dict],
    vocab: Dict[str, int],
    max_len: int | None = None,
    pad_token: str = "<pad>",
    pad_value: int = -2,
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of forward model examples.

    Args:
        batch: List of examples from ForwardModelDataset
        vocab: Vocabulary mapping
        max_len: Maximum sequence length (None uses full gene count)
        pad_token: Padding token
        pad_value: Padding value for expression

    Returns:
        Dictionary with batched tensors
    """
    batch_size = len(batch)

    # Stack control and perturbed expressions
    control_exprs = np.stack([ex["control_expr"] for ex in batch])
    perturbed_exprs = np.stack([ex["perturbed_expr"] for ex in batch])
    gene_ids = batch[0]["gene_ids"]  # Same for all examples
    if max_len is None:
        max_len = len(gene_ids)

    # Tokenize and pad using scGPT's function
    # This handles non-zero gene selection and padding
    control_batch = tokenize_and_pad_batch(
        control_exprs,
        gene_ids,
        max_len=max_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=False,  # Don't append CLS token for forward modeling
        include_zero_gene=True,  # Keep all genes to align control/perturbed sequences
        return_pt=True,
    )

    perturbed_batch = tokenize_and_pad_batch(
        perturbed_exprs,
        gene_ids,
        max_len=max_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=False,
        include_zero_gene=True,
        return_pt=True,
    )

    # Create padding masks
    control_mask = control_batch["genes"] == vocab[pad_token]
    pert_mask = perturbed_batch["genes"] == vocab[pad_token]

    return {
        "control_genes": control_batch["genes"],  # (batch, seq_len)
        "control_values": control_batch["values"],  # (batch, seq_len)
        "control_padding_mask": control_mask,
        "perturbed_genes": perturbed_batch["genes"],
        "perturbed_values": perturbed_batch["values"],
        "perturbed_padding_mask": pert_mask,
        "conditions": [ex["condition"] for ex in batch],
        "pert_gene_ids": [ex["pert_gene_ids"] for ex in batch],
    }
