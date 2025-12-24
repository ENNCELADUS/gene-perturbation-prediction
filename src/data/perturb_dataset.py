"""
Perturbation dataset for condition-level evaluation.

Direct h5ad loading without GEARS dependency, supporting the
Norman GEARS-style condition-level split protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict

import anndata as ad
import scanpy as sc
import numpy as np

from .condition_splits import ConditionSplit, NormanConditionSplitter


class PerturbDataset:
    """
    Dataset container for perturbation prediction evaluation.

    Loads h5ad directly and applies condition-level splits following
    the Norman GEARS-style protocol.
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
        self.condition_split: Optional[ConditionSplit] = None

    def load(self) -> "PerturbDataset":
        """Load h5ad file."""
        if not self.h5ad_path.exists():
            raise FileNotFoundError(f"H5AD file not found: {self.h5ad_path}")

        self.adata = sc.read_h5ad(self.h5ad_path)
        return self

    def create_condition_split(
        self,
        unseen_gene_fraction: float = 0.25,
        seen_single_train_ratio: float = 0.9,
        combo_seen2_train_ratio: float = 0.7,
        combo_seen2_val_ratio: float = 0.15,
        min_cells_per_condition: int = 50,
        seed: int = 42,
    ) -> ConditionSplit:
        """
        Create a Norman GEARS-style condition-level split.

        Args:
            unseen_gene_fraction: Fraction of single-genes to designate as unseen
            seen_single_train_ratio: Train ratio for seen single-gene conditions
            combo_seen2_train_ratio: Train ratio for 2/2-seen double-gene conditions
            combo_seen2_val_ratio: Val ratio for 2/2-seen double-gene conditions
            min_cells_per_condition: Minimum cells per condition (filter threshold)
            seed: Random seed

        Returns:
            ConditionSplit with train/val/test conditions and test strata
        """
        if self.adata is None:
            raise RuntimeError("Must call load() before create_condition_split()")

        # Get all non-control conditions
        conditions = self._get_valid_conditions(min_cells_per_condition)

        # Create splitter and split
        splitter = NormanConditionSplitter(
            unseen_gene_fraction=unseen_gene_fraction,
            seen_single_train_ratio=seen_single_train_ratio,
            combo_seen2_train_ratio=combo_seen2_train_ratio,
            combo_seen2_val_ratio=combo_seen2_val_ratio,
            seed=seed,
        )

        self.condition_split = splitter.split(conditions)
        return self.condition_split

    def _get_valid_conditions(self, min_cells: int = 50) -> List[str]:
        """
        Get conditions with at least min_cells cells.

        Excludes control conditions.
        """
        if self.adata is None:
            return []

        # Count cells per condition
        condition_counts = self.adata.obs[self.condition_col].value_counts()

        # Filter by min cells and exclude control
        valid = []
        for cond, count in condition_counts.items():
            if count >= min_cells and cond != "ctrl":
                valid.append(cond)

        return valid

    def load_condition_split(self, split_path: str | Path) -> ConditionSplit:
        """Load an existing condition split from JSON."""
        self.condition_split = ConditionSplit.load(split_path)
        return self.condition_split

    def apply_condition_split(self, split: ConditionSplit) -> None:
        """Apply an external condition split to this dataset."""
        self.condition_split = split

    @property
    def all_conditions(self) -> List[str]:
        """Get all valid conditions."""
        if self.condition_split is not None:
            return self.condition_split.all_conditions
        # Fallback: return all non-control conditions from adata
        if self.adata is None:
            return []
        conditions = self.adata.obs[self.condition_col].unique()
        return [c for c in conditions if c != "ctrl"]

    @property
    def train_conditions(self) -> List[str]:
        """Get training conditions."""
        if self.condition_split is None:
            return self.all_conditions
        return self.condition_split.train_conditions

    @property
    def val_conditions(self) -> List[str]:
        """Get validation conditions."""
        if self.condition_split is None:
            return []
        return self.condition_split.val_conditions

    @property
    def test_conditions(self) -> List[str]:
        """Get test conditions."""
        if self.condition_split is None:
            return self.all_conditions
        return self.condition_split.test_conditions

    @property
    def test_strata(self) -> Dict[str, List[str]]:
        """Get test condition strata for tier-wise evaluation."""
        if self.condition_split is None:
            return {}
        return self.condition_split.test_strata

    def get_adata_for_conditions(self, conditions: List[str]) -> ad.AnnData:
        """Get cells for a list of conditions."""
        if self.adata is None:
            raise RuntimeError("Must call load() first")
        mask = self.adata.obs[self.condition_col].isin(conditions)
        return self.adata[mask].copy()

    @property
    def train_adata(self) -> ad.AnnData:
        """Get training cells as AnnData subset."""
        return self.get_adata_for_conditions(self.train_conditions)

    @property
    def val_adata(self) -> ad.AnnData:
        """Get validation cells as AnnData subset."""
        return self.get_adata_for_conditions(self.val_conditions)

    @property
    def test_adata(self) -> ad.AnnData:
        """Get test cells as AnnData subset."""
        return self.get_adata_for_conditions(self.test_conditions)

    @property
    def control_adata(self) -> ad.AnnData:
        """Get control cells as AnnData subset."""
        if self.adata is None:
            raise RuntimeError("Must call load() first")
        mask = self.adata.obs[self.control_col] == 1
        return self.adata[mask].copy()

    def get_condition_sets(self) -> dict:
        """
        Get condition sets for train/val/test.

        Returns:
            Dict with keys: train, val, test, all, strata
        """
        return {
            "train": self.train_conditions,
            "val": self.val_conditions,
            "test": self.test_conditions,
            "all": self.all_conditions,
            "strata": self.test_strata,
        }

    def summary(self) -> dict:
        """Get dataset summary statistics."""
        stats = {
            "h5ad_path": str(self.h5ad_path),
            "loaded": self.adata is not None,
            "has_condition_split": self.condition_split is not None,
        }

        if self.adata is not None:
            stats["n_cells"] = self.adata.n_obs
            stats["n_genes"] = self.adata.n_vars
            n_control = (self.adata.obs[self.control_col] == 1).sum()
            stats["n_control_cells"] = int(n_control)
            stats["n_perturbed_cells"] = int(self.adata.n_obs - n_control)
            stats["n_conditions"] = len(self.all_conditions)

        if self.condition_split is not None:
            stats["n_train_conditions"] = len(self.condition_split.train_conditions)
            stats["n_val_conditions"] = len(self.condition_split.val_conditions)
            stats["n_test_conditions"] = len(self.condition_split.test_conditions)
            stats["n_unseen_genes"] = len(self.condition_split.unseen_genes)
            stats["test_strata_counts"] = {
                k: len(v) for k, v in self.condition_split.test_strata.items()
            }
            stats["split_seed"] = self.condition_split.seed

        return stats


def load_perturb_data(
    h5ad_path: str | Path,
    condition_split_path: Optional[str | Path] = None,
    unseen_gene_fraction: float = 0.25,
    seen_single_train_ratio: float = 0.9,
    combo_seen2_train_ratio: float = 0.7,
    combo_seen2_val_ratio: float = 0.15,
    min_cells_per_condition: int = 50,
    seed: int = 42,
) -> PerturbDataset:
    """
    Convenience function to load perturbation dataset with condition split.

    Args:
        h5ad_path: Path to h5ad file
        condition_split_path: Path to existing condition split JSON (if None, creates new)
        unseen_gene_fraction: Fraction of single-genes to designate as unseen
        seen_single_train_ratio: Train ratio for seen single-gene conditions
        combo_seen2_train_ratio: Train ratio for 2/2-seen double-gene conditions
        combo_seen2_val_ratio: Val ratio for 2/2-seen double-gene conditions
        min_cells_per_condition: Minimum cells per condition
        seed: Random seed

    Returns:
        Loaded PerturbDataset with condition split applied
    """
    dataset = PerturbDataset(h5ad_path)
    dataset.load()

    if condition_split_path and Path(condition_split_path).exists():
        try:
            dataset.load_condition_split(condition_split_path)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Split file exists but is not valid JSON: {condition_split_path}"
            ) from exc
    else:
        dataset.create_condition_split(
            unseen_gene_fraction=unseen_gene_fraction,
            seen_single_train_ratio=seen_single_train_ratio,
            combo_seen2_train_ratio=combo_seen2_train_ratio,
            combo_seen2_val_ratio=combo_seen2_val_ratio,
            min_cells_per_condition=min_cells_per_condition,
            seed=seed,
        )

    return dataset
