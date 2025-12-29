"""
Tahoe dataset loader with drug-based condition splits.

Reference: docs/roadmap/10_tahoe_data_preprocessing.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import scanpy as sc
import pandas as pd

try:
    from .drug_condition_splits import TahoeConditionSplit, TahoeDrugSplitter
except ImportError:  # Allow running as a script.
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[3]))
    from src.data.tahoe.drug_condition_splits import (  # type: ignore[no-redef]
        TahoeConditionSplit,
        TahoeDrugSplitter,
    )


class TahoeDataset:
    """
    Dataset container for Tahoe perturbation prediction.

    Loads preprocessed H5AD and applies drug-based condition splits.
    """

    def __init__(
        self,
        h5ad_path: str | Path,
        condition_col: str = "condition",
        drug_col: str = "drug",
        control_col: str = "control",
    ):
        """
        Initialize dataset.

        Args:
            h5ad_path: Path to preprocessed Tahoe H5AD
            condition_col: Column name for condition labels
            drug_col: Column name for drug labels
            control_col: Column name for control indicator
        """
        self.h5ad_path = Path(h5ad_path)
        self.condition_col = condition_col
        self.drug_col = drug_col
        self.control_col = control_col

        self.adata: Optional[ad.AnnData] = None
        self.condition_split: Optional[TahoeConditionSplit] = None

    def load(self, backed: bool = False) -> "TahoeDataset":
        """
        Load H5AD file.

        Args:
            backed: If True, load in backed mode (memory-efficient)
        """
        if not self.h5ad_path.exists():
            raise FileNotFoundError(f"H5AD file not found: {self.h5ad_path}")

        if backed:
            self.adata = sc.read_h5ad(self.h5ad_path, backed="r")
        else:
            self.adata = sc.read_h5ad(self.h5ad_path)
        return self

    def create_drug_split(
        self,
        unseen_single_gene_fraction: float = 0.25,
        n_unseen_single_genes: Optional[int] = None,
        single_seen_val_ratio: float = 0.1,
        multi_train_ratio: float = 0.65,
        multi_val_ratio: float = 0.1,
        min_cells_per_condition: int = 5,
        seed: int = 42,
        save_path: Optional[str | Path] = None,
    ) -> TahoeConditionSplit:
        """
        Create a drug-based condition-level split.

        Args:
            unseen_single_gene_fraction: Fraction of unique single-target genes to hold out
            n_unseen_single_genes: Optional explicit count of unseen single-target genes
            single_seen_val_ratio: Validation ratio for seen single-target drugs
            multi_train_ratio: Train ratio for seen multi-target drugs
            multi_val_ratio: Validation ratio for seen multi-target drugs
            min_cells_per_condition: Minimum cells to keep a condition
            seed: Random seed
            save_path: Optional path to save split JSON

        Returns:
            TahoeConditionSplit with train/val/test conditions
        """
        if self.adata is None:
            raise RuntimeError("Must call load() before create_drug_split()")

        splitter = TahoeDrugSplitter(
            unseen_single_gene_fraction=unseen_single_gene_fraction,
            n_unseen_single_genes=n_unseen_single_genes,
            single_seen_val_ratio=single_seen_val_ratio,
            multi_train_ratio=multi_train_ratio,
            multi_val_ratio=multi_val_ratio,
            min_cells_per_condition=min_cells_per_condition,
            seed=seed,
        )

        self.condition_split = splitter.split(
            self.adata.obs,
            condition_col=self.condition_col,
            drug_col=self.drug_col,
            target_gene_col="target_gene",
        )

        if save_path:
            self.condition_split.save(save_path)

        return self.condition_split

    def load_split(self, path: str | Path) -> TahoeConditionSplit:
        """Load an existing condition split from JSON."""
        self.condition_split = TahoeConditionSplit.load(path)
        return self.condition_split

    def apply_split(self, split: TahoeConditionSplit) -> None:
        """Apply an external condition split."""
        self.condition_split = split

    # Properties for accessing conditions
    @property
    def train_conditions(self) -> List[str]:
        if self.condition_split is None:
            return []
        return self.condition_split.train_conditions

    @property
    def val_conditions(self) -> List[str]:
        if self.condition_split is None:
            return []
        return self.condition_split.val_conditions

    @property
    def test_conditions(self) -> List[str]:
        if self.condition_split is None:
            return []
        return self.condition_split.test_conditions

    @property
    def test_strata(self) -> Dict[str, List[str]]:
        if self.condition_split is None:
            return {}
        return self.condition_split.test_strata

    # Properties for accessing AnnData subsets
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

    def summary(self) -> dict:
        """Get dataset summary statistics."""
        stats = {
            "h5ad_path": str(self.h5ad_path),
            "loaded": self.adata is not None,
            "has_split": self.condition_split is not None,
        }

        if self.adata is not None:
            stats["n_cells"] = self.adata.n_obs
            stats["n_genes"] = self.adata.n_vars
            stats["n_drugs"] = self.adata.obs[self.drug_col].nunique()

        if self.condition_split is not None:
            stats["n_train_conditions"] = len(self.train_conditions)
            stats["n_val_conditions"] = len(self.val_conditions)
            stats["n_test_conditions"] = len(self.test_conditions)
            stats["n_unseen_genes"] = len(self.condition_split.unseen_genes)
            stats["test_strata_counts"] = {
                k: len(v) for k, v in self.test_strata.items()
            }

        return stats


def load_tahoe_data(
    h5ad_path: str | Path,
    split_path: Optional[str | Path] = None,
    **split_kwargs,
) -> TahoeDataset:
    """
    Convenience function to load Tahoe dataset with condition split.

    Args:
        h5ad_path: Path to preprocessed Tahoe H5AD
        split_path: Path to existing split JSON (if None, creates new)
        **split_kwargs: Arguments for create_drug_split()

    Returns:
        Loaded TahoeDataset with condition split applied
    """
    dataset = TahoeDataset(h5ad_path)
    dataset.load()

    if split_path and Path(split_path).exists():
        dataset.load_split(split_path)
    else:
        dataset.create_drug_split(**split_kwargs)

    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Tahoe drug-based train/val/test split JSON.",
    )
    parser.add_argument(
        "--input",
        default="data/processed/tahoe/tahoe_log1p.h5ad",
        help="Path to preprocessed Tahoe H5AD.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/tahoe/splits/tahoe_drug_split_seed42.json",
        help="Path to write split JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split.",
    )
    parser.add_argument(
        "--unseen-single-gene-fraction",
        type=float,
        default=0.25,
        help="Fraction of single-target genes held out as unseen.",
    )
    parser.add_argument(
        "--n-unseen-single-genes",
        type=int,
        default=None,
        help="Optional explicit count of unseen single-target genes.",
    )
    parser.add_argument(
        "--single-seen-val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio for seen single-target drugs.",
    )
    parser.add_argument(
        "--multi-train-ratio",
        type=float,
        default=0.65,
        help="Train ratio for seen multi-target drugs.",
    )
    parser.add_argument(
        "--multi-val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio for seen multi-target drugs.",
    )
    parser.add_argument(
        "--min-cells-per-condition",
        type=int,
        default=5,
        help="Minimum cells required to keep a condition.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TahoeDataset(args.input)
    dataset.load()
    dataset.create_drug_split(
        unseen_single_gene_fraction=args.unseen_single_gene_fraction,
        n_unseen_single_genes=args.n_unseen_single_genes,
        single_seen_val_ratio=args.single_seen_val_ratio,
        multi_train_ratio=args.multi_train_ratio,
        multi_val_ratio=args.multi_val_ratio,
        min_cells_per_condition=args.min_cells_per_condition,
        seed=args.seed,
        save_path=args.output,
    )
    print(dataset.summary())


if __name__ == "__main__":
    main()
