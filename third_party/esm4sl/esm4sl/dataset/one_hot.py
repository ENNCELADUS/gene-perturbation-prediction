from typing import Any
import random
import pickle

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY
from .sl import SLDataset
from .sl_emb import CollateBatch


amino_acids = "ACDEFGHIKLMNPQRSTVWYU"
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}


def amino_acid_to_one_hot(sequence: str) -> np.ndarray:
    one_hot = np.zeros((len(sequence), len(amino_acids)), dtype=np.float32)

    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            one_hot[i, aa_to_index[aa]] = 1
        else:
            raise ValueError(f"Invalid amino acid '{aa}' in sequence.")

    return one_hot


@DATASET_REGISTRY.register()
class SLonehotDataset(SLDataset):
    """
    Dataset for SL prediction.
    Each pair of SL genes generates: (cmb_gene_emb, label)
    """
    @configurable
    def __init__(self, stage: RunningStage, cfg: DictConfig) -> None:
        super().__init__(stage, cfg)
        self.esm_root = cfg.DATASET.ESM_ROOT

        with open('/sharedata/home/daihzh/protein/ESM4SL/data/SLKB/all_id_seq.pkl', 'rb') as f:
            prot_seq_list = pickle.load(f)
        self.prot_seqs = dict(prot_seq_list)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, float, int, int] | tuple[torch.Tensor, torch.Tensor, float, int, int, np.ndarray]:
        g1_idx = self.df['0'][index]
        g2_idx = self.df['1'][index]
        label = float(self.df['2'][index])

        gene1 = amino_acid_to_one_hot(self.prot_seqs[g1_idx])
        gene1 = torch.from_numpy(gene1)
        gene2 = amino_acid_to_one_hot(self.prot_seqs[g2_idx])
        gene2 = torch.from_numpy(gene2)

        if self.stage == RunningStage.TRAINING and random.uniform(0, 1) > 0.5:  # swap 1 and 2 in random order
            gene2, gene1 = gene1, gene2
            g2_idx, g1_idx = g1_idx, g2_idx
        
        if not self.cell_line:
            return gene1, gene2, label, g1_idx, g2_idx
        else:
            return gene1, gene2, label, g1_idx, g2_idx, self.cell_embedding

    @property
    def collate_fn(self) -> Any:
        return CollateBatch(self.cell_line)


@DATASET_REGISTRY.register()
class SLonehotmeanDataset(SLonehotDataset):
    def __getitem__(self, index: int):
        g1_idx = self.df['0'][index]
        g2_idx = self.df['1'][index]
        label = float(self.df['2'][index])

        gene1 = amino_acid_to_one_hot(self.prot_seqs[g1_idx])  # [L, 21]
        gene1 = torch.from_numpy(gene1).mean(dim=0)  # [21]
        gene2 = amino_acid_to_one_hot(self.prot_seqs[g2_idx])  # [L, 21]
        gene2 = torch.from_numpy(gene2).mean(dim=0)  # [21]

        if self.stage == RunningStage.TRAINING and random.uniform(0, 1) > 0.5:  # swap 1 and 2 in random order
            gene2, gene1 = gene1, gene2
            g2_idx, g1_idx = g1_idx, g2_idx
        
        if not self.cell_line:
            return gene1, gene2, label, g1_idx, g2_idx
        else:
            return gene1, gene2, label, g1_idx, g2_idx, self.cell_embedding

    @property
    def collate_fn(self) -> Any:
        return None
