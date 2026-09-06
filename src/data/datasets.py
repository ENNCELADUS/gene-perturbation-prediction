"""Datasets and collators for prepared joint GeneEffect inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.batches import DependencyBatch, OnlineConditionBatch, ResponseBatch
from src.data.prepared import PreparedInputs


def _train_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("train", {})
    if not isinstance(value, Mapping):
        raise ValueError("config.train must be a mapping")
    return value


def _pooled_context(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    mean = tensor.mean(dim=0)
    variance = (tensor - mean).square().mean(dim=0)
    return torch.cat((mean, variance), dim=0)


class DependencyDataset(Dataset[int]):
    """Finite GeneEffect rows with a collator over opened prepared arrays."""

    def __init__(self, inputs: PreparedInputs, split: str) -> None:
        if split == "train":
            model_ids = set(inputs.split.supervised_train)
        elif split == "val":
            model_ids = set(inputs.split.val)
        elif split == "test":
            model_ids = set(inputs.split.test)
        else:
            raise ValueError("dependency split must be train, val, or test")
        rows = inputs.labels.loc[inputs.labels["model_id"].isin(model_ids)].copy()
        if rows.empty:
            raise ValueError(
                f"prepared inputs expose no finite dependency rows for split {split!r}"
            )
        if rows[["model_id", "gene_symbol"]].duplicated().any():
            raise ValueError("dependency labels contain duplicate ModelID/gene rows")
        unknown_lines = sorted(set(rows["model_id"]) - set(inputs.lines))
        if unknown_lines:
            raise ValueError(
                f"dependency rows have no opened line cache: {unknown_lines[:10]}"
            )
        self.inputs = inputs
        self.split = split
        self.rows = rows.reset_index(drop=True)
        self._gene_index = {gene: index for index, gene in enumerate(inputs.genes)}
        self._hvg_index = {gene: index for index, gene in enumerate(inputs.hvg_order)}
        self._esm2_index = {
            gene: index for index, gene in enumerate(inputs.esm2_symbols)
        }
        self._contexts = {
            model_id: _pooled_context(line.controls_tx1)
            for model_id, line in inputs.lines.items()
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> int:
        return int(index)

    def collate(self, indices: Sequence[int]) -> DependencyBatch:
        selected = self.rows.iloc[list(indices)]
        model_ids = tuple(selected["model_id"].astype(str))
        genes = tuple(selected["gene_symbol"].astype(str))
        gene_indices = [self._gene_index[gene] for gene in genes]
        q_values = np.stack(
            [
                self.inputs.lines[model_id].q_sc.values[gene_index]
                for model_id, gene_index in zip(model_ids, gene_indices, strict=True)
            ]
        )
        q_available = np.asarray(
            [
                self.inputs.lines[model_id].q_sc.available[gene_index]
                for model_id, gene_index in zip(model_ids, gene_indices, strict=True)
            ],
            dtype=bool,
        )
        hvg_indices = tuple(self._hvg_index.get(gene) for gene in genes)
        conditions = OnlineConditionBatch(
            controls_tx1=tuple(
                torch.from_numpy(self.inputs.lines[model_id].controls_tx1)
                for model_id in model_ids
            ),
            basal_hvg=tuple(
                torch.from_numpy(self.inputs.lines[model_id].basal_hvg)
                for model_id in model_ids
            ),
            genes=genes,
            model_ids=model_ids,
            q_sc=torch.from_numpy(np.nan_to_num(q_values, nan=0.0).astype(np.float32)),
            e_g=torch.from_numpy(
                np.stack(
                    [self.inputs.esm2_vectors[self._esm2_index[gene]] for gene in genes]
                ).astype(np.float32, copy=False)
            ),
            z_c=torch.stack([self._contexts[model_id] for model_id in model_ids]),
            q_sc_mask=torch.from_numpy(q_available),
            gene_in_hvg_panel=torch.tensor(
                [index is not None for index in hvg_indices], dtype=torch.bool
            ),
            own_gene_hvg_indices=hvg_indices,
            own_gene_shift_available=torch.tensor(
                [
                    index is not None and bool(available)
                    for index, available in zip(hvg_indices, q_available, strict=True)
                ],
                dtype=torch.bool,
            ),
        )
        batch = DependencyBatch(
            conditions=conditions,
            residual=torch.tensor(selected["residual"].to_numpy(), dtype=torch.float32),
            gene_mean=torch.tensor(
                [self.inputs.train_gene_means[gene] for gene in genes],
                dtype=torch.float32,
            ),
            valid=torch.ones(len(indices), dtype=torch.bool),
        )
        batch.validate()
        return batch


class ResponseDataset(Dataset[int]):
    """Prepared response conditions selected by the persisted holdout."""

    def __init__(self, inputs: PreparedInputs, *, holdout: bool) -> None:
        if inputs.response_targets is None:
            raise ValueError("prepared inputs have no opened response cache")
        self.inputs = inputs
        self.cache = inputs.response_targets
        self.indices = tuple(
            index
            for index, key in enumerate(self.cache.keys)
            if (key in inputs.response_holdout) is holdout
        )
        if not self.indices:
            kind = "holdout" if holdout else "training"
            raise ValueError(f"prepared response cache has no {kind} conditions")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> int:
        return self.indices[index]

    def collate(self, indices: Sequence[int]) -> ResponseBatch:
        model_ids = tuple(self.cache.model_ids[index] for index in indices)
        genes = tuple(self.cache.genes[index] for index in indices)
        observed = tuple(
            torch.from_numpy(
                np.array(self.cache.target_bag(index), dtype=np.float32, copy=True)
            )
            for index in indices
        )
        batch = ResponseBatch(
            model_ids=model_ids,
            genes=genes,
            controls_tx1=tuple(
                torch.from_numpy(self.inputs.lines[model_id].controls_tx1)
                for model_id in model_ids
            ),
            observed_hvg=observed,
            control_hvg=tuple(
                torch.from_numpy(self.inputs.lines[model_id].basal_hvg)
                for model_id in model_ids
            ),
        )
        batch.validate()
        return batch


def make_evaluation_loaders(
    inputs: PreparedInputs,
    config: Mapping[str, Any],
    split: str,
    accelerator: Any,
) -> tuple[DataLoader[DependencyBatch], DataLoader[ResponseBatch]]:
    """Build sequential tail-preserving dependency and response loaders."""
    train = _train_config(config)
    dependency_size = int(train.get("dependency_batch_size", 256))
    response_size = int(train.get("response_batch_size", 64))
    if dependency_size <= 0 or response_size <= 0:
        raise ValueError("evaluation batch sizes must be positive")
    dependency_dataset = DependencyDataset(inputs, split)
    response_dataset = ResponseDataset(inputs, holdout=True)
    dependency_loader = DataLoader(
        dependency_dataset,
        batch_size=dependency_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dependency_dataset.collate,
    )
    response_loader = DataLoader(
        response_dataset,
        batch_size=response_size,
        shuffle=False,
        drop_last=False,
        collate_fn=response_dataset.collate,
    )
    if accelerator is not None:
        dependency_loader = accelerator.prepare_data_loader(
            dependency_loader, device_placement=False
        )
        response_loader = accelerator.prepare_data_loader(
            response_loader, device_placement=False
        )
    return dependency_loader, response_loader


__all__ = [
    "DependencyDataset",
    "ResponseDataset",
    "make_evaluation_loaders",
]
