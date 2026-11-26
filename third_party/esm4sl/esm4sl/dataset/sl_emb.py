from pathlib import Path
import os
import random
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY
from .sl import SLDataset

__all__ = ["SLembDataset", "SLwholeembDataset"]


@DATASET_REGISTRY.register()
class SLembDataset(SLDataset):
    """
    Dataset for SL prediction.
    Each pair of SL genes generates: (cmb_gene_emb, label)
    """

    @configurable
    def __init__(self, cfg: DictConfig, stage: RunningStage) -> None:  #sl_root: Path, test_fold: int, cell_line: str | None, esm_root: Path
        super().__init__(cfg, stage)
        self.esm_root = cfg.DATASET.ESM_ROOT

    def common_getitem(self, index: int, key: str) -> tuple[torch.Tensor, torch.Tensor, float, int, int] | tuple[torch.Tensor, torch.Tensor, float, int, int, np.ndarray]:
        """
        key == "mean_rep": [1280]
        key == "rep": [L, 1280], L different across samples!
        """
        g1_idx = self.df['0'][index]
        g2_idx = self.df['1'][index]
        label = float(self.df['2'][index])

        gene1 = torch.load(os.path.join(self.esm_root, f"{g1_idx}.pt"), weights_only=True)#, allow_pickle=True  .item()[key]
        gene2 = torch.load(os.path.join(self.esm_root, f"{g2_idx}.pt"), weights_only=True)#, allow_pickle=True  .item()[key]
        if key == "mean_rep":
            gene1 = torch.mean(gene1, dim=0)
            gene2 = torch.mean(gene2, dim=0)

        if self.stage == RunningStage.TRAINING and random.uniform(0, 1) > 0.5:  # concat 1 and 2 in random order
            gene2, gene1 = gene1, gene2
            g2_idx, g1_idx = g1_idx, g2_idx
        
        if not self.cell_line:
            return gene1, gene2, label, g1_idx, g2_idx
        else:
            return gene1, gene2, label, g1_idx, g2_idx, self.cell_embedding

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        return self.common_getitem(index, "mean_rep")


class CollateBatch():
    def __init__(self, cell_line: str | None):
        self.cell_line = cell_line

    def padding(self, genes: list[torch.Tensor], max_len: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
        count_len = [g.shape[0] for g in genes]

        batched_sample = torch.zeros((len(genes), max_len, genes[0].shape[1]))
        for i, (gene, length) in enumerate(zip(genes, count_len)):
            if length <= max_len:
                batched_sample[i, :length, :] = gene
            else:
                batched_sample[i, :max_len, :] = gene[:max_len, :]
        batched_sample = batched_sample.float()

        prot_mask = torch.ones((len(genes), max_len), dtype=bool)
        for i, seq_len in enumerate(count_len):
            if seq_len <= max_len:
                prot_mask[i, :seq_len] = 0
            else:
                prot_mask[i, :] = 0

        return batched_sample, prot_mask

    def __call__(self, 
        batch: list[tuple[torch.Tensor, torch.Tensor, float, int, int]]
    ) -> tuple[torch.Tensor, ...]:
        if self.cell_line:
            g1, g2, labels, g1_indices, g2_indices, cell_lines = zip(*batch)
        else:
            g1, g2, labels, g1_indices, g2_indices = zip(*batch)

        batched_g1, g1_mask = self.padding(g1)
        batched_g2, g2_mask = self.padding(g2)

        if self.cell_line:
            return torch.tensor(g1_indices), batched_g1, g1_mask, \
                   torch.tensor(g2_indices), batched_g2, g2_mask, \
                   torch.tensor(labels), torch.from_numpy(np.array(cell_lines)).float()
        else:
            return torch.tensor(g1_indices), batched_g1, g1_mask, \
                   torch.tensor(g2_indices), batched_g2, g2_mask, \
                   torch.tensor(labels)


@DATASET_REGISTRY.register()
class SLwholeembDataset(SLembDataset):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        return self.common_getitem(index, "rep")

    @property
    def collate_fn(self) -> Any:
        return CollateBatch(self.cell_line)
