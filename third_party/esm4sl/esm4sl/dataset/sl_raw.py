import random
import pickle

import torch
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY
from .sl import SLDataset

__all__ = ["SLrawDataset"]


class CollateBatch():
    def __init__(self):
        pass

    def __call__(self, batch: list[tuple[str, str, float, str, str]]
    ) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        g1_seqs, g2_seqs, labels, g1_indices, g2_indices = zip(*batch)
        g1_list = [(str(g1_idx), g1_seq) for g1_idx, g1_seq in zip(g1_indices, g1_seqs)]
        g2_list = [(str(g2_idx), g2_seq) for g2_idx, g2_seq in zip(g2_indices, g2_seqs)]
        return g1_list, g2_list, torch.tensor(labels), torch.tensor(g1_indices), torch.tensor(g2_indices)


@DATASET_REGISTRY.register()
class SLrawDataset(SLDataset):
    """
    Dataset for SL prediction.
    Each pair of SL genes generates: (cmb_gene_emb, label)
    """

    @configurable
    def __init__(self, cfg: DictConfig, stage: RunningStage) -> None:
        super().__init__(cfg, stage)
        with open(cfg.DATASET.GENE_SEQ_PATH, 'rb') as f:
            self.dic = pickle.load(f)
        self.dic = dict(self.dic)

    def __getitem__(self, index: int) -> tuple[tuple[str, str], tuple[str, str], float]:
        gene1 = self.df['0'][index]
        gene2 = self.df['1'][index]
        label = float(self.df['2'][index])

        seq1 = self.dic[gene1]
        seq2 = self.dic[gene2]

        if self.stage == RunningStage.TRAINING and random.uniform(0, 1) > 0.5:  # swap 1 and 2 in randomly
            seq1, seq2 = seq2, seq1
            gene1, gene2 = gene2, gene1

        return seq1, seq2, label, gene1, gene2

    @property
    def collate_fn(self):
        return CollateBatch()
