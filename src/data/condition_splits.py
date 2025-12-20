"""
Condition-level splits for generalization evaluation.

Implements multiple evaluation tracks:
- In-distribution: held-out conditions with seen genes
- Unseen-combination: held-out gene pairs where singles are seen
- Unseen-gene: held-out genes and all conditions containing them
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np


def parse_genes_from_condition(condition: str) -> Set[str]:
    """Extract gene names from condition string."""
    if not condition or condition == "ctrl":
        return set()
    genes = condition.split("+")
    return {g.strip() for g in genes if g.strip() and g.strip() != "ctrl"}


def is_single_gene_condition(condition: str) -> bool:
    """Check if condition is single-gene perturbation."""
    genes = parse_genes_from_condition(condition)
    return len(genes) == 1


def is_pair_condition(condition: str) -> bool:
    """Check if condition is gene-pair perturbation."""
    genes = parse_genes_from_condition(condition)
    return len(genes) == 2


@dataclass
class ConditionSplit:
    """Container for condition-level split results."""

    train_conditions: List[str] = field(default_factory=list)
    val_conditions: List[str] = field(default_factory=list)
    test_conditions: List[str] = field(default_factory=list)

    # Track type
    track: str = "in_dist"  # "in_dist", "unseen_combo", "unseen_gene"

    # Split metadata
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    @property
    def all_conditions(self) -> List[str]:
        """Get all conditions in split."""
        return self.train_conditions + self.val_conditions + self.test_conditions

    def save(self, path: Path | str) -> None:
        """Save split to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "ConditionSplit":
        """Load split from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class ConditionSplitter:
    """
    Condition-level splitter for generalization evaluation.

    Supports multiple generalization tracks:
    - in_dist: Random condition holdout (all genes seen in training)
    - unseen_combo: Hold out gene pairs where individual genes are in training
    - unseen_gene: Hold out genes and all conditions containing them
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 42,
    ):
        """
        Initialize splitter.

        Args:
            train_ratio: Fraction of conditions for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    def split_in_distribution(self, conditions: List[str]) -> ConditionSplit:
        """
        Random split at condition level.

        All genes appear in training; only specific conditions are held out.
        """
        rng = np.random.default_rng(self.seed)
        conditions = [c for c in conditions if c != "ctrl"]
        shuffled = rng.permutation(conditions).tolist()

        n_train = int(len(shuffled) * self.train_ratio)
        n_val = int(len(shuffled) * self.val_ratio)

        return ConditionSplit(
            train_conditions=shuffled[:n_train],
            val_conditions=shuffled[n_train : n_train + n_val],
            test_conditions=shuffled[n_train + n_val :],
            track="in_dist",
            seed=self.seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )

    def split_unseen_combination(self, conditions: List[str]) -> ConditionSplit:
        """
        Hold out gene pairs where individual genes are seen.

        - Train: all single-gene conditions + some pairs
        - Test: pairs where both genes appear in train singles
        """
        rng = np.random.default_rng(self.seed)
        conditions = [c for c in conditions if c != "ctrl"]

        singles = [c for c in conditions if is_single_gene_condition(c)]
        pairs = [c for c in conditions if is_pair_condition(c)]

        # All singles go to train
        train_genes = set()
        for s in singles:
            train_genes.update(parse_genes_from_condition(s))

        # Split pairs: those with both genes in singles can be test
        testable_pairs = []
        train_only_pairs = []

        for p in pairs:
            genes = parse_genes_from_condition(p)
            if genes.issubset(train_genes):
                testable_pairs.append(p)
            else:
                train_only_pairs.append(p)

        # Shuffle testable pairs
        rng.shuffle(testable_pairs)
        n_test = int(
            len(testable_pairs)
            * self.test_ratio
            / (self.val_ratio + self.test_ratio + 0.001)
        )
        n_val = int(
            len(testable_pairs)
            * self.val_ratio
            / (self.val_ratio + self.test_ratio + 0.001)
        )

        test_pairs = testable_pairs[:n_test]
        val_pairs = testable_pairs[n_test : n_test + n_val]
        train_pairs = testable_pairs[n_test + n_val :] + train_only_pairs

        return ConditionSplit(
            train_conditions=singles + train_pairs,
            val_conditions=val_pairs,
            test_conditions=test_pairs,
            track="unseen_combo",
            seed=self.seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )

    def split_unseen_gene(
        self, conditions: List[str], n_holdout_genes: int = 5
    ) -> ConditionSplit:
        """
        Hold out genes and all conditions containing them.

        - Test: all conditions containing held-out genes
        - Train: only conditions with seen genes
        """
        rng = np.random.default_rng(self.seed)
        conditions = [c for c in conditions if c != "ctrl"]

        # Collect all genes
        all_genes = set()
        for c in conditions:
            all_genes.update(parse_genes_from_condition(c))

        all_genes = list(all_genes)
        rng.shuffle(all_genes)

        # Hold out genes
        holdout_genes = set(all_genes[:n_holdout_genes])
        train_genes = set(all_genes[n_holdout_genes:])

        # Split conditions
        test_conditions = []
        trainable_conditions = []

        for c in conditions:
            genes = parse_genes_from_condition(c)
            if genes & holdout_genes:  # Any overlap with holdout
                test_conditions.append(c)
            else:
                trainable_conditions.append(c)

        # Split trainable into train/val
        rng.shuffle(trainable_conditions)
        n_val = int(
            len(trainable_conditions)
            * self.val_ratio
            / (self.train_ratio + self.val_ratio)
        )

        return ConditionSplit(
            train_conditions=trainable_conditions[n_val:],
            val_conditions=trainable_conditions[:n_val],
            test_conditions=test_conditions,
            track="unseen_gene",
            seed=self.seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )

    def split(self, conditions: List[str], track: str = "in_dist") -> ConditionSplit:
        """
        Create split for specified track.

        Args:
            conditions: List of condition names
            track: "in_dist", "unseen_combo", or "unseen_gene"

        Returns:
            ConditionSplit for the specified track
        """
        if track == "in_dist":
            return self.split_in_distribution(conditions)
        elif track == "unseen_combo":
            return self.split_unseen_combination(conditions)
        elif track == "unseen_gene":
            return self.split_unseen_gene(conditions)
        else:
            raise ValueError(
                f"Unknown track: {track}. Use 'in_dist', 'unseen_combo', or 'unseen_gene'"
            )
