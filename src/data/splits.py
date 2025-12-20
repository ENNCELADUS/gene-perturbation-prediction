"""
Cell-level split for within-condition evaluation.

This module implements the within-condition cell-level split protocol:
- Split cells within each condition into reference (80%) and query (20%)
- Ensure all conditions remain in candidate pool for exact retrieval
- Support deterministic, reproducible splits via seed
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import anndata as ad


@dataclass
class CellSplit:
    """Container for cell-level split results."""

    # Per-condition cell indices (as lists for JSON serialization)
    ref_cells: Dict[str, List[int]] = field(default_factory=dict)
    query_cells: Dict[str, List[int]] = field(default_factory=dict)

    # Dropped conditions (insufficient cells)
    dropped_conditions: List[str] = field(default_factory=list)

    # Split parameters
    min_cells_per_condition: int = 50
    query_fraction: float = 0.2
    min_query_cells: int = 10
    seed: int = 42

    @property
    def all_ref_indices(self) -> np.ndarray:
        """Get all reference cell indices."""
        indices = []
        for idx_list in self.ref_cells.values():
            indices.extend(idx_list)
        return np.array(indices, dtype=int)

    @property
    def all_query_indices(self) -> np.ndarray:
        """Get all query cell indices."""
        indices = []
        for idx_list in self.query_cells.values():
            indices.extend(idx_list)
        return np.array(indices, dtype=int)

    @property
    def conditions(self) -> List[str]:
        """Get all valid conditions (in candidate pool)."""
        return list(self.ref_cells.keys())

    def save(self, path: Path | str) -> None:
        """Save split to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "CellSplit":
        """Load split from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class CellSplitter:
    """
    Within-condition cell-level splitter.

    Splits perturbed cells for each condition into reference and query sets,
    ensuring the ground truth condition is always in the candidate pool.
    """

    def __init__(
        self,
        min_cells_per_condition: int = 50,
        query_fraction: float = 0.2,
        min_query_cells: int = 10,
        seed: int = 42,
        condition_col: str = "condition",
        control_col: str = "control",
    ):
        """
        Initialize splitter.

        Args:
            min_cells_per_condition: Minimum perturbed cells required per condition
            query_fraction: Fraction of cells to use as queries (default 0.2 = 20%)
            min_query_cells: Minimum number of query cells required
            seed: Random seed for reproducibility
            condition_col: Column name for condition labels
            control_col: Column name for control indicator
        """
        self.min_cells_per_condition = min_cells_per_condition
        self.query_fraction = query_fraction
        self.min_query_cells = min_query_cells
        self.seed = seed
        self.condition_col = condition_col
        self.control_col = control_col

    def split(self, adata: ad.AnnData) -> CellSplit:
        """
        Split cells within each condition into reference and query sets.

        Args:
            adata: AnnData object with condition and control columns

        Returns:
            CellSplit containing per-condition ref/query cell indices
        """
        rng = np.random.default_rng(self.seed)

        result = CellSplit(
            min_cells_per_condition=self.min_cells_per_condition,
            query_fraction=self.query_fraction,
            min_query_cells=self.min_query_cells,
            seed=self.seed,
        )

        # Get all unique conditions (excluding control)
        conditions = adata.obs[self.condition_col].unique()

        for cond in conditions:
            # Skip control condition
            if cond == "ctrl":
                continue

            # Get perturbed cells for this condition
            mask = (adata.obs[self.condition_col] == cond) & (
                adata.obs[self.control_col] == 0
            )
            cell_indices = np.where(mask)[0]
            n_cells = len(cell_indices)

            # Check minimum cell threshold
            if n_cells < self.min_cells_per_condition:
                result.dropped_conditions.append(cond)
                continue

            # Shuffle indices deterministically
            shuffled = rng.permutation(cell_indices)

            # Calculate split sizes
            n_query = max(self.min_query_cells, int(n_cells * self.query_fraction))
            n_ref = n_cells - n_query

            # Ensure we have enough cells for both splits
            if n_ref < 1 or n_query < self.min_query_cells:
                result.dropped_conditions.append(cond)
                continue

            # Split: first n_ref for reference, rest for query
            ref_indices = shuffled[:n_ref].tolist()
            query_indices = shuffled[n_ref:].tolist()

            result.ref_cells[cond] = ref_indices
            result.query_cells[cond] = query_indices

        return result

    def validate(self, split: CellSplit) -> Dict[str, any]:
        """
        Validate split properties.

        Args:
            split: CellSplit to validate

        Returns:
            Dictionary with validation results
        """
        validation = {
            "n_conditions": len(split.conditions),
            "n_dropped": len(split.dropped_conditions),
            "n_ref_cells": len(split.all_ref_indices),
            "n_query_cells": len(split.all_query_indices),
            "overlap_count": 0,
            "is_valid": True,
        }

        # Check for overlap between ref and query cells
        ref_set = set(split.all_ref_indices)
        query_set = set(split.all_query_indices)
        overlap = ref_set & query_set
        validation["overlap_count"] = len(overlap)

        if overlap:
            validation["is_valid"] = False
            validation["error"] = f"Found {len(overlap)} overlapping cells"

        # Check all query conditions exist in ref
        ref_conditions = set(split.ref_cells.keys())
        query_conditions = set(split.query_cells.keys())
        missing = query_conditions - ref_conditions

        if missing:
            validation["is_valid"] = False
            validation["error"] = f"Query conditions missing from ref: {missing}"

        return validation
