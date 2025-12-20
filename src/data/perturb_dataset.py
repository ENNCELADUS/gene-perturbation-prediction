"""
Perturbation dataset for within-condition cell-level evaluation.

Direct h5ad loading without GEARS dependency, supporting the
within-condition split protocol where cells are split into
reference and query sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import anndata as ad
import scanpy as sc
import numpy as np

from .splits import CellSplit, CellSplitter
from .condition_splits import ConditionSplit


class PerturbDataset:
    """
    Dataset container for perturbation retrieval evaluation.

    Loads h5ad directly and applies within-condition cell-level splits.
    """

    def __init__(
        self,
        h5ad_path: str | Path,
        condition_col: str = "condition",
        control_col: str = "control",
    ):
        """
        Initialize dataset.

        Args:
            h5ad_path: Path to preprocessed h5ad file
            condition_col: Column name for condition labels
            control_col: Column name for control indicator
        """
        self.h5ad_path = Path(h5ad_path)
        self.condition_col = condition_col
        self.control_col = control_col

        self.adata: Optional[ad.AnnData] = None
        self.split: Optional[CellSplit] = None
        self.condition_split: Optional[ConditionSplit] = None

    def load(self) -> "PerturbDataset":
        """Load h5ad file."""
        if not self.h5ad_path.exists():
            raise FileNotFoundError(f"H5AD file not found: {self.h5ad_path}")

        self.adata = sc.read_h5ad(self.h5ad_path)
        return self

    def create_split(
        self,
        min_cells_per_condition: int = 50,
        query_fraction: float = 0.2,
        min_query_cells: int = 10,
        seed: int = 42,
    ) -> CellSplit:
        """
        Create a new within-condition cell-level split.

        Args:
            min_cells_per_condition: Minimum cells per condition
            query_fraction: Fraction for query set
            min_query_cells: Minimum query cells
            seed: Random seed

        Returns:
            CellSplit with ref/query cell indices
        """
        if self.adata is None:
            raise RuntimeError("Must call load() before create_split()")

        splitter = CellSplitter(
            min_cells_per_condition=min_cells_per_condition,
            query_fraction=query_fraction,
            min_query_cells=min_query_cells,
            seed=seed,
            condition_col=self.condition_col,
            control_col=self.control_col,
        )

        self.split = splitter.split(self.adata)
        return self.split

    def load_split(self, split_path: str | Path) -> CellSplit:
        """Load an existing split from JSON."""
        self.split = CellSplit.load(split_path)
        return self.split

    def apply_split(self, split: CellSplit) -> None:
        """Apply an external split to this dataset."""
        self.split = split

    def load_condition_split(self, split_path: str | Path) -> ConditionSplit:
        """Load an existing condition split from JSON."""
        self.condition_split = ConditionSplit.load(split_path)
        return self.condition_split

    def apply_condition_split(self, split: ConditionSplit) -> None:
        """Apply an external condition split to this dataset."""
        self.condition_split = split

    @property
    def ref_adata(self) -> ad.AnnData:
        """Get reference cells as AnnData subset."""
        if self.adata is None:
            raise RuntimeError("Must call load() first")
        if self.split is None:
            raise RuntimeError("Must create or load split first")

        indices = self.split.all_ref_indices
        return self.adata[indices].copy()

    @property
    def query_adata(self) -> ad.AnnData:
        """Get query cells as AnnData subset."""
        if self.adata is None:
            raise RuntimeError("Must call load() first")
        if self.split is None:
            raise RuntimeError("Must create or load split first")

        indices = self.split.all_query_indices
        return self.adata[indices].copy()

    @property
    def all_conditions(self) -> List[str]:
        """Get all valid conditions (in candidate pool)."""
        if self.split is None:
            # Return all non-control conditions from adata
            if self.adata is None:
                return []
            conditions = self.adata.obs[self.condition_col].unique()
            return [c for c in conditions if c != "ctrl"]
        return self.split.conditions

    def get_ref_adata_for_conditions(self, conditions: List[str]) -> ad.AnnData:
        """Get reference cells for a list of conditions."""
        ref = self.ref_adata
        mask = ref.obs[self.condition_col].isin(conditions)
        return ref[mask].copy()

    def get_query_adata_for_conditions(self, conditions: List[str]) -> ad.AnnData:
        """Get query cells for a list of conditions."""
        query = self.query_adata
        mask = query.obs[self.condition_col].isin(conditions)
        return query[mask].copy()

    def get_condition_sets(self) -> dict:
        """
        Get condition sets for train/val/test.

        Returns:
            Dict with keys: train, val, test, all
        """
        if self.condition_split is None:
            conditions = self.all_conditions
            return {
                "train": conditions,
                "val": [],
                "test": conditions,
                "all": conditions,
            }

        return {
            "train": self.condition_split.train_conditions,
            "val": self.condition_split.val_conditions,
            "test": self.condition_split.test_conditions,
            "all": self.condition_split.all_conditions,
        }

    @property
    def control_adata(self) -> ad.AnnData:
        """Get control cells as AnnData subset."""
        if self.adata is None:
            raise RuntimeError("Must call load() first")

        mask = self.adata.obs[self.control_col] == 1
        return self.adata[mask].copy()

    def get_ref_cells_for_condition(self, condition: str) -> ad.AnnData:
        """Get reference cells for a specific condition."""
        if self.adata is None or self.split is None:
            raise RuntimeError("Must load data and split first")

        if condition not in self.split.ref_cells:
            raise ValueError(f"Condition '{condition}' not in split")

        indices = self.split.ref_cells[condition]
        return self.adata[indices].copy()

    def get_query_cells_for_condition(self, condition: str) -> ad.AnnData:
        """Get query cells for a specific condition."""
        if self.adata is None or self.split is None:
            raise RuntimeError("Must load data and split first")

        if condition not in self.split.query_cells:
            raise ValueError(f"Condition '{condition}' not in split")

        indices = self.split.query_cells[condition]
        return self.adata[indices].copy()

    def summary(self) -> dict:
        """Get dataset summary statistics."""
        stats = {
            "h5ad_path": str(self.h5ad_path),
            "loaded": self.adata is not None,
            "has_split": self.split is not None,
        }

        if self.adata is not None:
            stats["n_cells"] = self.adata.n_obs
            stats["n_genes"] = self.adata.n_vars
            n_control = (self.adata.obs[self.control_col] == 1).sum()
            stats["n_control_cells"] = int(n_control)
            stats["n_perturbed_cells"] = int(self.adata.n_obs - n_control)
            stats["n_conditions"] = len(self.all_conditions)

        if self.split is not None:
            stats["n_ref_cells"] = len(self.split.all_ref_indices)
            stats["n_query_cells"] = len(self.split.all_query_indices)
            stats["n_valid_conditions"] = len(self.split.conditions)
            stats["n_dropped_conditions"] = len(self.split.dropped_conditions)
            stats["split_seed"] = self.split.seed

        if self.condition_split is not None:
            stats["condition_track"] = self.condition_split.track
            stats["n_train_conditions"] = len(self.condition_split.train_conditions)
            stats["n_val_conditions"] = len(self.condition_split.val_conditions)
            stats["n_test_conditions"] = len(self.condition_split.test_conditions)

        return stats


def load_perturb_data(
    h5ad_path: str | Path,
    split_path: Optional[str | Path] = None,
    min_cells_per_condition: int = 50,
    query_fraction: float = 0.2,
    min_query_cells: int = 10,
    seed: int = 42,
) -> PerturbDataset:
    """
    Convenience function to load perturbation dataset with split.

    Args:
        h5ad_path: Path to h5ad file
        split_path: Path to existing split JSON (if None, creates new split)
        min_cells_per_condition: Minimum cells per condition
        query_fraction: Fraction for query set
        min_query_cells: Minimum query cells
        seed: Random seed

    Returns:
        Loaded PerturbDataset with split applied
    """
    dataset = PerturbDataset(h5ad_path)
    dataset.load()

    if split_path and Path(split_path).exists():
        dataset.load_split(split_path)
    else:
        dataset.create_split(
            min_cells_per_condition=min_cells_per_condition,
            query_fraction=query_fraction,
            min_query_cells=min_query_cells,
            seed=seed,
        )

    return dataset
