from pathlib import Path
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.sampler import WeightedRandomSampler
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY

__all__ = ["SLDataset"]


@DATASET_REGISTRY.register()
class SLDataset(Dataset):
    """
    Dataset for SL prediction.
    Each pair of SL genes generates: (cmb_gene_emb, label)
    """
    @configurable
    def __init__(self, cfg: DictConfig, stage: RunningStage) -> None: # sl_root: Path, test_fold: int, cell_line: str | None | float, np_ratio: Optional[int] = None
        self.stage = stage
        self.cell_line = cfg.DATASET.CELL_LINE

        if stage == RunningStage.TRAINING:
            self.df = pd.read_csv(cfg.DATASET.TRAIN_FILE)
            self.df = self.df.sample(frac=1).reset_index(drop=True)  # shuffle
        elif stage == RunningStage.VALIDATING:
            self.df = pd.read_csv(cfg.DATASET.VAL_FILE)
        else:
            self.df = pd.read_csv(cfg.DATASET.TEST_FILE)
        
        self.sample = cfg.DATALOADER.SAMPLE

        if self.cell_line is not None:
            cell_embeddings = np.load('/home/qingyuyang/ESM4SL/data/mapping/clname2embed.npy', allow_pickle=True).item()
            self.cell_embedding = cell_embeddings[self.cell_line]  # [4079, 6]

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
        return {
            "cfg": cfg,
            "stage": stage,
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self):
        """ Subclass should overwrite this method """
        pass

    def pos_neg_ratio(self) -> float:
        pos_indices = self.df[self.df['2'] == 1].index.tolist()
        neg_indices = self.df[self.df['2'] == 0].index.tolist()
        return len(neg_indices) / len(pos_indices)
    
    def compute_weights(self) -> np.ndarray[float]:
        sl_pairs = [(row['0'], row['1'], row['2']) for _, row in self.df.iterrows()]
        sl_pairs = np.array(sl_pairs, dtype=np.int32)

        labels = sl_pairs[:, 2]
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight = 1. / class_sample_count
        return np.array([weight[t] for t in labels])

    @property
    def sampler(self) -> Sampler | None:
        if self.sample:
            class_weights = self.compute_weights()
            return WeightedRandomSampler(class_weights, len(class_weights)) if self.stage == RunningStage.TRAINING else None
        else:
            return None

    @property
    def collate_fn(self) -> Any:
        return None  # Subclass should overwrite this variable if needed
