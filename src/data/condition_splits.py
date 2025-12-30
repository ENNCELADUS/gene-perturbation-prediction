"""
Norman condition-level splits for compositional generalization evaluation.

Implements a compositional split without unseen single-gene holdout:
- Single-gene conditions: all used for train/val (no single-gene test holdout)
- Double-gene conditions: stratified sampling by gene frequency and cell count

Reference: docs/roadmap/07_norman_data_split.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np


@dataclass
class ConditionSplit:
    """
    Stores condition-level train/val/test split with test strata metadata.

    Attributes:
        train_conditions: Conditions for training (all singles + selected combos)
        val_conditions: Conditions for validation (all singles + selected combos)
        test_conditions: All test conditions (subset of double-gene conditions)
        test_strata: Dict mapping tier names to condition lists:
            - 'double_test': all double-gene conditions in test
            - 'double_f{bin}_c{bin}': stratified test subsets by freq/cell bins
        unseen_genes: Reserved for backward compatibility (unused)
        seed: Random seed used for splitting
    """

    train_conditions: List[str]
    val_conditions: List[str]
    test_conditions: List[str]
    test_strata: Dict[str, List[str]] = field(default_factory=dict)
    unseen_genes: List[str] = field(default_factory=list)
    seed: int = 42

    @property
    def all_conditions(self) -> List[str]:
        """Get all conditions across splits."""
        return self.train_conditions + self.val_conditions + self.test_conditions

    def save(self, path: str | Path) -> None:
        """Save split to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ConditionSplit":
        """Load split from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class NormanConditionSplitter:
    """
    GEARS-style condition-level splitter for Norman dataset.

    Implements compositional stratification:
    - Single-gene conditions: all → train/val
    - Double-gene conditions: stratified by gene frequency and cell counts

    See docs/roadmap/07_norman_data_split.md for full specification.
    """

    def __init__(
        self,
        seen_single_train_ratio: float = 0.9,
        double_train_ratio: float = 0.7,
        double_val_ratio: float = 0.15,
        double_freq_bins: int = 3,
        double_count_bins: int = 3,
        seed: int = 42,
    ):
        """
        Initialize splitter.

        Args:
            seen_single_train_ratio: Train ratio for seen single-gene conditions (remaining → val)
            double_train_ratio: Train ratio for double-gene conditions
            double_val_ratio: Val ratio for double-gene conditions (remaining → test)
            double_freq_bins: Number of bins for per-gene double frequency
            double_count_bins: Number of bins for per-condition cell counts
            seed: Random seed for reproducibility
        """
        self.seen_single_train_ratio = seen_single_train_ratio
        self.double_train_ratio = double_train_ratio
        self.double_val_ratio = double_val_ratio
        self.double_freq_bins = double_freq_bins
        self.double_count_bins = double_count_bins
        self.seed = seed

    @staticmethod
    def normalize_condition(condition: str) -> str:
        """
        Normalize condition to canonical format.

        - Control: 'ctrl' → 'ctrl'
        - Single-gene: 'X+ctrl' or 'ctrl+X' → 'X' (strip ctrl)
        - Double-gene: enforce lexicographic ordering 'A+B' where A < B
        """
        condition = condition.strip()

        # Handle control
        if condition == "ctrl":
            return condition

        # Split by +
        if "+" in condition:
            genes = [g.strip() for g in condition.split("+")]
            # Remove 'ctrl' from the list
            genes = [g for g in genes if g != "ctrl"]

            # If only one gene left, it's a single perturbation
            if len(genes) == 1:
                return genes[0]

            # Multiple genes: sort for canonical order
            genes = sorted(genes)
            return "+".join(genes)

        return condition

    @staticmethod
    def is_double_perturbation(condition: str) -> bool:
        """
        Check if condition is a double-gene perturbation.

        Single: 'ctrl', 'GENE', 'GENE+ctrl', 'ctrl+GENE'
        Double: 'GENE1+GENE2' (both not ctrl)
        """
        if condition == "ctrl":
            return False

        if "+" not in condition:
            return False

        # Split and filter out ctrl
        genes = [g.strip() for g in condition.split("+") if g.strip() != "ctrl"]

        # Double perturbation has 2+ non-control genes
        return len(genes) >= 2

    @staticmethod
    def get_genes_from_condition(condition: str) -> List[str]:
        """
        Extract gene names from condition string (excluding ctrl).

        Returns:
            List of gene names (empty list for 'ctrl')
        """
        if condition == "ctrl":
            return []

        if "+" in condition:
            genes = [g.strip() for g in condition.split("+") if g.strip() != "ctrl"]
            return genes

        return [condition.strip()]

    def split(
        self,
        conditions: List[str],
        condition_counts: Optional[Dict[str, int]] = None,
    ) -> ConditionSplit:
        """
        Create compositional condition-level split.

        Args:
            conditions: List of condition names (single and double perturbations)
                        Should NOT include control conditions.
            condition_counts: Optional mapping of condition -> cell count

        Returns:
            ConditionSplit with train/val/test conditions and test strata
        """
        rng = np.random.default_rng(self.seed)

        # Step 1: Normalize all conditions
        conditions = [self.normalize_condition(c) for c in conditions]
        conditions = list(set(conditions))  # Remove duplicates

        # Step 2: Separate single-gene and double-gene conditions
        single_conditions = [
            c for c in conditions if not self.is_double_perturbation(c)
        ]
        double_conditions = [c for c in conditions if self.is_double_perturbation(c)]

        # Step 3: Split single-gene conditions (all to train/val)
        all_singles = list(single_conditions)
        rng.shuffle(all_singles)
        n_train_single = int(len(all_singles) * self.seen_single_train_ratio)
        train_single = all_singles[:n_train_single]
        val_single = all_singles[n_train_single:]

        # Step 4: Stratified split for double-gene conditions
        condition_counts = condition_counts or {}
        gene_double_counts: Dict[str, int] = {}
        for cond in double_conditions:
            for gene in self.get_genes_from_condition(cond):
                gene_double_counts[gene] = gene_double_counts.get(gene, 0) + 1

        freq_values = []
        count_values = []
        cond_freq_score: Dict[str, float] = {}
        cond_count_score: Dict[str, int] = {}
        for cond in double_conditions:
            genes = self.get_genes_from_condition(cond)
            if genes:
                freq_score = float(
                    np.mean([gene_double_counts.get(g, 0) for g in genes])
                )
            else:
                freq_score = 0.0
            count_score = int(condition_counts.get(cond, 0))
            cond_freq_score[cond] = freq_score
            cond_count_score[cond] = count_score
            freq_values.append(freq_score)
            count_values.append(count_score)

        freq_bins = self._make_bins(freq_values, self.double_freq_bins)
        count_bins = self._make_bins(count_values, self.double_count_bins)

        strata: Dict[str, List[str]] = {}
        for cond in double_conditions:
            f_bin = self._assign_bin(cond_freq_score[cond], freq_bins)
            c_bin = self._assign_bin(cond_count_score[cond], count_bins)
            key = f"double_f{f_bin}_c{c_bin}"
            strata.setdefault(key, []).append(cond)

        train_double: List[str] = []
        val_double: List[str] = []
        test_double: List[str] = []
        test_strata: Dict[str, List[str]] = {}
        for key, group in strata.items():
            rng.shuffle(group)
            n_train = int(len(group) * self.double_train_ratio)
            n_val = int(len(group) * self.double_val_ratio)
            train_double.extend(group[:n_train])
            val_double.extend(group[n_train : n_train + n_val])
            test_group = group[n_train + n_val :]
            test_double.extend(test_group)
            test_strata[key] = test_group

        # Step 7: Assemble final splits
        train_conditions = train_single + train_double
        val_conditions = val_single + val_double
        test_conditions = test_double

        # Shuffle for good measure
        rng.shuffle(train_conditions)
        rng.shuffle(val_conditions)
        rng.shuffle(test_conditions)

        # Step 8: Create test strata for evaluation
        test_strata["double_test"] = test_double

        return ConditionSplit(
            train_conditions=train_conditions,
            val_conditions=val_conditions,
            test_conditions=test_conditions,
            test_strata=test_strata,
            unseen_genes=[],
            seed=self.seed,
        )

    def summary(self, split: ConditionSplit) -> Dict:
        """Generate summary statistics for a split."""
        return {
            "n_train": len(split.train_conditions),
            "n_val": len(split.val_conditions),
            "n_test": len(split.test_conditions),
            "test_strata": {k: len(v) for k, v in split.test_strata.items()},
            "seed": split.seed,
        }

    @staticmethod
    def _make_bins(values: List[float], n_bins: int) -> List[float]:
        if not values or n_bins <= 1:
            return []
        unique_vals = np.unique(values)
        if len(unique_vals) <= 1:
            return []
        quantiles = np.linspace(0, 1, num=n_bins + 1)[1:-1]
        edges = np.quantile(values, quantiles)
        return sorted(set(edges.tolist()))

    @staticmethod
    def _assign_bin(value: float, edges: List[float]) -> int:
        if not edges:
            return 0
        for idx, edge in enumerate(edges):
            if value <= edge:
                return idx
        return len(edges)
