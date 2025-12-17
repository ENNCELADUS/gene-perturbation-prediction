"""
Norman Perturb-seq dataset loading using GEARS.

This module wraps the GEARS PertData class to load and manage
the Norman et al. 2019 CRISPRa perturbation dataset.
"""

from pathlib import Path
from typing import Optional

import anndata as ad
from gears import PertData


class NormanDataset:
    """Wrapper for Norman Perturb-seq dataset via GEARS."""

    def __init__(
        self,
        data_dir: str = "data/",
        split: str = "simulation",
        seed: int = 1,
        batch_size: int = 64,
    ):
        """
        Initialize Norman dataset.

        Args:
            data_dir: Base directory containing norman/ subdirectory
            split: Split type ('simulation', 'combo_seen0/1/2')
            seed: Random seed for split
            batch_size: Batch size for dataloaders
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.seed = seed
        self.batch_size = batch_size

        self.pert_data: Optional[PertData] = None
        self.adata: Optional[ad.AnnData] = None

    def load(self) -> "NormanDataset":
        """Load the dataset using GEARS."""
        self.pert_data = PertData(str(self.data_dir))
        self.pert_data.load(data_name="norman")
        self.adata = self.pert_data.adata
        return self

    def prepare_split(self) -> "NormanDataset":
        """Prepare train/val/test split."""
        if self.pert_data is None:
            raise RuntimeError("Must call load() before prepare_split()")

        self.pert_data.prepare_split(split=self.split, seed=self.seed)
        self.pert_data.get_dataloader(
            batch_size=self.batch_size,
            test_batch_size=self.batch_size,
        )
        return self

    @property
    def train_loader(self):
        """Get train dataloader."""
        return self.pert_data.train_loader if self.pert_data else None

    @property
    def val_loader(self):
        """Get validation dataloader."""
        return self.pert_data.val_loader if self.pert_data else None

    @property
    def test_loader(self):
        """Get test dataloader."""
        return self.pert_data.test_loader if self.pert_data else None

    @property
    def conditions(self) -> dict:
        """Get conditions per split."""
        if self.pert_data and hasattr(self.pert_data, "set2conditions"):
            return self.pert_data.set2conditions
        return {}

    def get_control_cells(self) -> ad.AnnData:
        """Get control cells from the dataset."""
        if self.adata is None:
            raise RuntimeError("Must call load() first")
        return self.adata[self.adata.obs["control"] == 1].copy()

    def get_perturbed_cells(self, condition: str) -> ad.AnnData:
        """Get cells for a specific perturbation condition."""
        if self.adata is None:
            raise RuntimeError("Must call load() first")
        return self.adata[self.adata.obs["condition"] == condition].copy()


def load_norman_data(
    data_dir: str = "data/",
    split: str = "simulation",
    seed: int = 1,
    batch_size: int = 64,
) -> NormanDataset:
    """
    Convenience function to load Norman dataset.

    Args:
        data_dir: Base directory containing norman/ subdirectory
        split: Split type
        seed: Random seed
        batch_size: Batch size

    Returns:
        Loaded and split NormanDataset
    """
    dataset = NormanDataset(
        data_dir=data_dir,
        split=split,
        seed=seed,
        batch_size=batch_size,
    )
    dataset.load()
    dataset.prepare_split()
    return dataset
