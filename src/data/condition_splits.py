"""
Norman condition-level splits for GEARS-style generalization evaluation.

Implements the GEARS/community-standard gene-visibility-based stratification:
- Single-gene conditions: seen (train/val) vs unseen (test-only)
- Double-gene conditions: stratified by how many genes are seen (0/2, 1/2, 2/2)

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
        train_conditions: Conditions for training (seen singles + 2/2-seen combos)
        val_conditions: Conditions for validation (seen singles + 2/2-seen combos)
        test_conditions: All test conditions (unseen singles + all combo tiers)
        test_strata: Dict mapping tier names to condition lists:
            - 'single_unseen': unseen single-gene conditions
            - 'combo_seen2': 2/2-seen double-gene conditions in test
            - 'combo_seen1': 1/2-seen double-gene conditions
            - 'combo_seen0': 0/2-seen double-gene conditions
        unseen_genes: Set of genes designated as unseen (test-only)
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

    Implements gene-visibility-based stratification:
    - Unseen genes: selected fraction of single-gene conditions → test only
    - Single-gene conditions: seen → train/val, unseen → test
    - Double-gene conditions: stratified by 0/1/2-seen based on unseen_genes

    See docs/roadmap/07_norman_data_split.md for full specification.
    """

    def __init__(
        self,
        unseen_gene_fraction: float = 0.25,
        seen_single_train_ratio: float = 0.9,
        combo_seen2_train_ratio: float = 0.7,
        combo_seen2_val_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Initialize splitter.

        Args:
            unseen_gene_fraction: Fraction of single-genes to designate as unseen (0.2-0.3 recommended)
            seen_single_train_ratio: Train ratio for seen single-gene conditions (remaining → val)
            combo_seen2_train_ratio: Train ratio for 2/2-seen double-gene conditions
            combo_seen2_val_ratio: Val ratio for 2/2-seen double-gene conditions (remaining → test)
            seed: Random seed for reproducibility
        """
        self.unseen_gene_fraction = unseen_gene_fraction
        self.seen_single_train_ratio = seen_single_train_ratio
        self.combo_seen2_train_ratio = combo_seen2_train_ratio
        self.combo_seen2_val_ratio = combo_seen2_val_ratio
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

    def split(self, conditions: List[str]) -> ConditionSplit:
        """
        Create GEARS-style condition-level split.

        Args:
            conditions: List of condition names (single and double perturbations)
                        Should NOT include control conditions.

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

        # Step 3: Select unseen genes (fraction of single-gene conditions → test only)
        n_unseen = max(1, int(len(single_conditions) * self.unseen_gene_fraction))
        shuffled_singles = rng.permutation(single_conditions).tolist()
        unseen_genes = set(shuffled_singles[:n_unseen])
        seen_genes = set(shuffled_singles[n_unseen:])

        # Step 4: Split single-gene conditions
        # Unseen → test only
        test_single_unseen = list(unseen_genes)

        # Seen → train/val split
        seen_singles = list(seen_genes)
        rng.shuffle(seen_singles)
        n_train_single = int(len(seen_singles) * self.seen_single_train_ratio)
        train_single = seen_singles[:n_train_single]
        val_single = seen_singles[n_train_single:]

        # Step 5: Classify double-gene conditions by seen tier
        combo_seen0 = []  # Both genes unseen
        combo_seen1 = []  # One gene unseen
        combo_seen2 = []  # Both genes seen

        for cond in double_conditions:
            genes = self.get_genes_from_condition(cond)
            n_seen = sum(1 for g in genes if g in seen_genes)

            if n_seen == 0:
                combo_seen0.append(cond)
            elif n_seen == 1:
                combo_seen1.append(cond)
            else:  # n_seen == 2
                combo_seen2.append(cond)

        # Step 6: Split combo_seen2 into train/val/test
        # Only combo_seen2 may enter training (Rule 2 from spec)
        rng.shuffle(combo_seen2)
        n_train_combo = int(len(combo_seen2) * self.combo_seen2_train_ratio)
        n_val_combo = int(len(combo_seen2) * self.combo_seen2_val_ratio)

        train_double = combo_seen2[:n_train_combo]
        val_double = combo_seen2[n_train_combo : n_train_combo + n_val_combo]
        test_double_seen2 = combo_seen2[n_train_combo + n_val_combo :]

        # combo_seen0 and combo_seen1 go entirely to test (Rule 1 from spec)

        # Step 7: Assemble final splits
        train_conditions = train_single + train_double
        val_conditions = val_single + val_double
        test_conditions = (
            test_single_unseen + test_double_seen2 + combo_seen1 + combo_seen0
        )

        # Shuffle for good measure
        rng.shuffle(train_conditions)
        rng.shuffle(val_conditions)
        rng.shuffle(test_conditions)

        # Step 8: Create test strata for tier-wise evaluation
        test_strata = {
            "single_unseen": test_single_unseen,
            "combo_seen2": test_double_seen2,
            "combo_seen1": combo_seen1,
            "combo_seen0": combo_seen0,
        }

        return ConditionSplit(
            train_conditions=train_conditions,
            val_conditions=val_conditions,
            test_conditions=test_conditions,
            test_strata=test_strata,
            unseen_genes=list(unseen_genes),
            seed=self.seed,
        )

    def summary(self, split: ConditionSplit) -> Dict:
        """Generate summary statistics for a split."""
        return {
            "n_train": len(split.train_conditions),
            "n_val": len(split.val_conditions),
            "n_test": len(split.test_conditions),
            "n_unseen_genes": len(split.unseen_genes),
            "test_strata": {k: len(v) for k, v in split.test_strata.items()},
            "seed": split.seed,
        }
