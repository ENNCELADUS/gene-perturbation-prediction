"""
Norman condition-level splits for compositional generalization evaluation.

Implements a hard split with unseen single-gene holdout:
- Single-gene conditions: split into seen (train/val) and unseen (test-only)
- Double-gene conditions: seen2 can enter train/val/test; seen1/seen0 test-only

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
        unseen_genes: Single-gene conditions held out as unseen (test-only)
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
    - Single-gene conditions: seen → train/val, unseen → test-only
    - Double-gene conditions: seen2 stratified by gene frequency and cell counts
      (seen1/seen0 → test-only)

    See docs/roadmap/07_norman_data_split.md for full specification.
    """

    def __init__(
        self,
        unseen_gene_fraction: float = 0.15,
        seen_single_train_ratio: float = 0.9,
        combo_seen2_train_ratio: float = 0.7,
        combo_seen2_val_ratio: float = 0.15,
        double_freq_bins: int = 3,
        double_count_bins: int = 3,
        unseen_gene_count_bins: int = 3,
        seed: int = 42,
    ):
        """
        Initialize splitter.

        Args:
            unseen_gene_fraction: Fraction of single-gene conditions held out as unseen test genes
            seen_single_train_ratio: Train ratio for seen single-gene conditions (remaining → val)
            combo_seen2_train_ratio: Train ratio for 2/2-seen double-gene conditions
            combo_seen2_val_ratio: Val ratio for 2/2-seen double-gene conditions (remaining → test)
            double_freq_bins: Number of bins for per-gene double frequency
            double_count_bins: Number of bins for per-condition cell counts
            unseen_gene_count_bins: Bins for stratifying unseen single genes by cell count
            seed: Random seed for reproducibility
        """
        self.unseen_gene_fraction = unseen_gene_fraction
        self.seen_single_train_ratio = seen_single_train_ratio
        self.combo_seen2_train_ratio = combo_seen2_train_ratio
        self.combo_seen2_val_ratio = combo_seen2_val_ratio
        self.double_freq_bins = double_freq_bins
        self.double_count_bins = double_count_bins
        self.unseen_gene_count_bins = unseen_gene_count_bins
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

        # Step 3: Select unseen single genes (test-only)
        condition_counts = condition_counts or {}
        all_singles = sorted(single_conditions)
        n_unseen = int(round(len(all_singles) * self.unseen_gene_fraction))
        if self.unseen_gene_fraction > 0:
            n_unseen = max(1, n_unseen)
        n_unseen = min(n_unseen, max(len(all_singles) - 1, 0))
        unseen_genes: List[str] = []
        if n_unseen > 0:
            single_counts = {g: int(condition_counts.get(g, 0)) for g in all_singles}
            count_values = [single_counts[g] for g in all_singles]
            count_bins = self._make_bins(count_values, self.unseen_gene_count_bins)
            bin_to_genes: Dict[int, List[str]] = {}
            for gene in all_singles:
                bin_id = self._assign_bin(single_counts[gene], count_bins)
                bin_to_genes.setdefault(bin_id, []).append(gene)

            total = len(all_singles)
            targets = {}
            fractions = {}
            for bin_id, genes in bin_to_genes.items():
                frac = (len(genes) / total) if total > 0 else 0.0
                raw = n_unseen * frac
                targets[bin_id] = int(np.floor(raw))
                fractions[bin_id] = raw - targets[bin_id]

            remainder = n_unseen - sum(targets.values())
            for bin_id, _ in sorted(
                fractions.items(), key=lambda x: x[1], reverse=True
            ):
                if remainder <= 0:
                    break
                targets[bin_id] += 1
                remainder -= 1

            for bin_id, genes in bin_to_genes.items():
                if not genes:
                    continue
                rng.shuffle(genes)
                k = min(targets.get(bin_id, 0), len(genes))
                unseen_genes.extend(genes[:k])

        unseen_set = set(unseen_genes)

        # Step 4: Split single-gene conditions (seen -> train/val, unseen -> test)
        seen_singles = [g for g in all_singles if g not in unseen_set]
        rng.shuffle(seen_singles)
        n_train_single = int(len(seen_singles) * self.seen_single_train_ratio)
        train_single = seen_singles[:n_train_single]
        val_single = seen_singles[n_train_single:]
        test_single_unseen = sorted(unseen_set)

        # Step 5: Assign double-gene conditions by unseen tier
        double_seen2: List[str] = []
        double_seen1: List[str] = []
        double_seen0: List[str] = []
        for cond in double_conditions:
            genes = self.get_genes_from_condition(cond)
            if not genes:
                continue
            unseen_count = sum(1 for g in genes if g in unseen_set)
            if unseen_count == 0:
                double_seen2.append(cond)
            elif unseen_count == 1:
                double_seen1.append(cond)
            else:
                double_seen0.append(cond)

        # Step 6: Stratified split for seen2 double-gene conditions only
        gene_double_counts: Dict[str, int] = {}
        for cond in double_seen2:
            for gene in self.get_genes_from_condition(cond):
                gene_double_counts[gene] = gene_double_counts.get(gene, 0) + 1

        freq_values = []
        count_values = []
        cond_freq_score: Dict[str, float] = {}
        cond_count_score: Dict[str, int] = {}
        for cond in double_seen2:
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
        for cond in double_seen2:
            f_bin = self._assign_bin(cond_freq_score[cond], freq_bins)
            c_bin = self._assign_bin(cond_count_score[cond], count_bins)
            key = f"double_seen2_f{f_bin}_c{c_bin}"
            strata.setdefault(key, []).append(cond)

        train_double: List[str] = []
        val_double: List[str] = []
        test_double_seen2: List[str] = []
        test_strata: Dict[str, List[str]] = {}
        for key, group in strata.items():
            rng.shuffle(group)
            n_train = int(len(group) * self.combo_seen2_train_ratio)
            n_val = int(len(group) * self.combo_seen2_val_ratio)
            train_double.extend(group[:n_train])
            val_double.extend(group[n_train : n_train + n_val])
            test_group = group[n_train + n_val :]
            test_double_seen2.extend(test_group)
            test_strata[key] = test_group

        # Step 7: Assemble final splits
        train_conditions = train_single + train_double
        val_conditions = val_single + val_double
        test_conditions = (
            test_single_unseen + test_double_seen2 + double_seen1 + double_seen0
        )

        # Shuffle for good measure
        rng.shuffle(train_conditions)
        rng.shuffle(val_conditions)
        rng.shuffle(test_conditions)

        # Step 8: Create test strata for evaluation
        test_strata["single_unseen"] = test_single_unseen
        test_strata["double_seen2"] = test_double_seen2
        test_strata["double_seen1"] = double_seen1
        test_strata["double_seen0"] = double_seen0
        test_strata["double_test"] = test_double_seen2 + double_seen1 + double_seen0

        return ConditionSplit(
            train_conditions=train_conditions,
            val_conditions=val_conditions,
            test_conditions=test_conditions,
            test_strata=test_strata,
            unseen_genes=sorted(unseen_set),
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
