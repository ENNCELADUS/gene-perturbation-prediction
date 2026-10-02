"""GeneEffect rows and anchor response conditions over prepared inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.batches import DependencyBatch, OnlineConditionBatch, ResponseBatch
from src.data.prepared import PreparedInputs


def pooled_context(controls_tx1: np.ndarray) -> torch.Tensor:
    """Tx1 context ``z_c``: per-dimension mean and population variance of the cells."""
    tensor = torch.from_numpy(np.asarray(controls_tx1, dtype=np.float32))
    mean = tensor.mean(dim=0)
    return torch.cat((mean, (tensor - mean).square().mean(dim=0)))


def split_lines(inputs: PreparedInputs, split: str) -> tuple[str, ...]:
    """Lines scored for a split; training uses the supervised training lines."""
    if split == "train":
        return inputs.split.supervised_train
    return getattr(inputs.split, split)


class DependencyDataset(Dataset[int]):
    """Labelled (line, gene) GeneEffect rows of one split, with a collator.

    Each line's basal HVG cells are kept once on ``device`` and shared by every
    row of that line.
    """

    def __init__(
        self, inputs: PreparedInputs, split: str, *, device: torch.device | str = "cpu"
    ) -> None:
        lines = set(split_lines(inputs, split))
        self.inputs = inputs
        self.split = split
        self.rows = inputs.labels.loc[inputs.labels["model_id"].isin(lines)]
        self.rows = self.rows.reset_index(drop=True)
        self._hvg_index = {gene: index for index, gene in enumerate(inputs.hvg_order)}
        self._gene_index = {gene: index for index, gene in enumerate(inputs.genes)}
        self._esm2_index = {g: i for i, g in enumerate(inputs.esm2_symbols)}
        present = sorted(set(self.rows["model_id"]))
        self._contexts = {
            m: pooled_context(inputs.lines[m].controls_tx1) for m in present
        }
        self._basal = {
            m: torch.from_numpy(np.asarray(inputs.lines[m].basal_hvg)).to(device)
            for m in present
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> int:
        return int(index)

    def collate(self, indices: Sequence[int]) -> DependencyBatch:
        selected = self.rows.iloc[list(indices)]
        model_ids = tuple(selected["model_id"].astype(str))
        genes = tuple(selected["gene_symbol"].astype(str))
        positions = [self._gene_index[gene] for gene in genes]
        q_sc = [self.inputs.lines[m].q_sc for m in model_ids]
        q_values = np.stack([q.values[i] for q, i in zip(q_sc, positions, strict=True)])
        q_available = np.asarray(
            [q.available[i] for q, i in zip(q_sc, positions, strict=True)], dtype=bool
        )
        hvg_indices = tuple(self._hvg_index.get(gene) for gene in genes)
        conditions = OnlineConditionBatch(
            basal_hvg=tuple(self._basal[m] for m in model_ids),
            genes=genes,
            model_ids=model_ids,
            q_sc=torch.from_numpy(np.nan_to_num(q_values, nan=0.0).astype(np.float32)),
            e_g=torch.from_numpy(
                np.stack(
                    [self.inputs.esm2_vectors[self._esm2_index[gene]] for gene in genes]
                ).astype(np.float32)
            ),
            z_c=torch.stack([self._contexts[m] for m in model_ids]),
            q_sc_mask=torch.from_numpy(q_available),
            gene_in_hvg_panel=torch.tensor([i is not None for i in hvg_indices]),
            own_gene_hvg_indices=hvg_indices,
            own_gene_shift_available=torch.tensor(
                [
                    i is not None and bool(a)
                    for i, a in zip(hvg_indices, q_available, strict=True)
                ],
                dtype=torch.bool,
            ),
        )

        def column(name: str) -> torch.Tensor:
            return torch.tensor(selected[name].to_numpy(), dtype=torch.float32)

        return DependencyBatch(
            conditions=conditions,
            residual=column("residual"),
            gene_effect=column("gene_effect"),
            gene_mean=torch.tensor(
                [self.inputs.train_gene_means[gene] for gene in genes],
                dtype=torch.float32,
            ),
        )


class ResponseDataset(Dataset[int]):
    """Every prepared perturbation condition of the response anchors."""

    def __init__(
        self, inputs: PreparedInputs, *, device: torch.device | str = "cpu"
    ) -> None:
        self.inputs = inputs
        self.cache = inputs.response_targets
        self.keys = tuple(self.cache.keys)
        self._basal = {
            m: torch.from_numpy(np.asarray(inputs.lines[m].basal_hvg)).to(device)
            for m in sorted({model_id for model_id, _ in self.keys})
        }

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> int:
        return int(index)

    def collate(self, indices: Sequence[int]) -> ResponseBatch:
        keys = [self.keys[index] for index in indices]
        return ResponseBatch(
            model_ids=tuple(model_id for model_id, _ in keys),
            genes=tuple(gene for _, gene in keys),
            control_hvg=tuple(self._basal[model_id] for model_id, _ in keys),
            observed_hvg=tuple(
                torch.from_numpy(np.array(self.cache.target_bag(i), dtype=np.float32))
                for i in indices
            ),
        )


def make_evaluation_loader(
    inputs: PreparedInputs,
    config: Mapping[str, Any],
    split: str,
    accelerator: Any = None,
) -> DataLoader[DependencyBatch]:
    """Fixed-order GeneEffect rows of one split, sharded across ranks if launched."""
    device = "cpu" if accelerator is None else accelerator.device
    dataset = DependencyDataset(inputs, split, device=device)
    loader = DataLoader(
        dataset,
        batch_size=config["train"]["dependency_batch_size"],
        shuffle=False,
        collate_fn=dataset.collate,
    )
    if accelerator is not None:
        loader = accelerator.prepare_data_loader(loader, device_placement=False)
    return loader
