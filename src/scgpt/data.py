"""scGPT gene-score dataset and collator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.distributed import suppress_torchtext_deprecation_warning
from src.utils.metrics import normalize_condition, parse_condition_genes


class GeneScoreDataset(Dataset):
    """Perturbed-cell examples with matched control cells."""

    def __init__(
        self,
        adata: ad.AnnData,
        conditions: Sequence[str],
        vocab: dict[str, int],
        n_bins: int = 51,
        condition_key: str = "condition",
        control_key: str = "control",
        n_control_samples: int = 8,
        seed: int = 42,
    ) -> None:
        self.adata = adata
        self.conditions = {normalize_condition(condition) for condition in conditions}
        self.vocab = vocab
        self.n_bins = n_bins
        self.condition_key = condition_key
        self.control_key = control_key
        self.n_control_samples = n_control_samples
        self.rng = np.random.RandomState(seed)
        self.gene_names = _gene_names(adata)
        self.gene_name_to_idx = {gene: idx for idx, gene in enumerate(self.gene_names)}
        self.gene_ids = np.asarray(
            [vocab.get(gene, vocab.get("<pad>", 0)) for gene in self.gene_names]
        )
        self.control_indices = np.where(adata.obs[control_key].to_numpy() == 1)[0]
        if len(self.control_indices) == 0:
            raise ValueError("scGPT gene scoring requires control cells")
        self.examples = []
        for idx, condition in enumerate(adata.obs[condition_key].tolist()):
            normalized_condition = normalize_condition(str(condition))
            if (
                normalized_condition in self.conditions
                and idx not in self.control_indices
            ):
                self.examples.append((idx, normalized_condition))
        self.gene_counts = np.zeros(len(self.gene_names), dtype=np.int64)
        self.example_gene_indices = []
        for _, condition in self.examples:
            gene_indices = [
                self.gene_name_to_idx[gene]
                for gene in parse_condition_genes(condition)
                if gene in self.gene_name_to_idx
            ]
            self.example_gene_indices.append(gene_indices)
            for gene_idx in gene_indices:
                self.gene_counts[gene_idx] += 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        cell_idx, condition = self.examples[index]
        controls = self.rng.choice(
            self.control_indices,
            size=self.n_control_samples,
            replace=len(self.control_indices) < self.n_control_samples,
        )
        return {
            "expr": _bin_expression(_row_array(self.adata, cell_idx), self.n_bins),
            "control_exprs": np.stack(
                [
                    _bin_expression(_row_array(self.adata, idx), self.n_bins)
                    for idx in controls
                ]
            ),
            "gene_ids": self.gene_ids,
            "condition": condition,
            "pert_gene_indices": self.example_gene_indices[index],
        }


def collate_gene_score_batch(
    batch: list[dict],
    vocab: dict[str, int],
    n_genes: int,
    max_len: int = 1200,
) -> dict[str, torch.Tensor | list[str] | int]:
    """Tokenize scGPT examples and build multi-hot targets."""
    tokenize_and_pad_batch = _tokenizer()
    exprs = np.stack([item["expr"] for item in batch])
    control_exprs = np.stack([item["control_exprs"] for item in batch])
    gene_ids = batch[0]["gene_ids"]
    control_flat = control_exprs.reshape(-1, control_exprs.shape[-1])
    tokenized = tokenize_and_pad_batch(
        exprs,
        gene_ids,
        max_len=max_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
        append_cls=True,
        include_zero_gene=False,
        return_pt=True,
    )
    control_tokenized = tokenize_and_pad_batch(
        control_flat,
        gene_ids,
        max_len=max_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
        append_cls=True,
        include_zero_gene=False,
        return_pt=True,
    )
    targets = np.zeros((len(batch), n_genes), dtype=np.float32)
    for row_idx, item in enumerate(batch):
        targets[row_idx, item["pert_gene_indices"]] = 1.0
    return {
        "genes": tokenized["genes"],
        "values": tokenized["values"],
        "padding_mask": tokenized["genes"] == vocab["<pad>"],
        "control_genes": control_tokenized["genes"],
        "control_values": control_tokenized["values"],
        "control_padding_mask": control_tokenized["genes"] == vocab["<pad>"],
        "control_counts": control_exprs.shape[1],
        "targets": torch.from_numpy(targets),
        "conditions": [item["condition"] for item in batch],
    }


def _tokenizer():
    scgpt_path = Path(__file__).parents[2] / "scGPT"
    if str(scgpt_path) not in sys.path:
        sys.path.insert(0, str(scgpt_path))
    suppress_torchtext_deprecation_warning()
    from scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch

    return tokenize_and_pad_batch


def _gene_names(adata: ad.AnnData) -> list[str]:
    if "gene_name" in adata.var.columns:
        return [str(value) for value in adata.var["gene_name"].tolist()]
    return [str(value) for value in adata.var_names.tolist()]


def _row_array(adata: ad.AnnData, index: int) -> np.ndarray:
    layer = adata.layers["counts"] if "counts" in adata.layers else adata.X
    row = layer[index]
    return row.toarray().ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()


def _bin_expression(expr: np.ndarray, n_bins: int) -> np.ndarray:
    if np.count_nonzero(expr) == 0:
        values = np.zeros_like(expr, dtype=np.int64)
        values[0] = 1
        return values
    nonzero = expr[expr > 0]
    quantiles = np.quantile(nonzero, np.linspace(0, 1, n_bins - 1))
    return np.digitize(expr, quantiles, right=False).astype(np.int64)
