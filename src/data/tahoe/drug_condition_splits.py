"""
Drug-based condition splits for Tahoe dataset.

Implements a split strategy that keeps test targets seen in training
while using drug-based conditions.

Reference: docs/roadmap/10_tahoe_data_preprocessing.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TahoeConditionSplit:
    """
    Stores condition-level train/val/test split with test strata metadata.

    Attributes:
        train_conditions: Conditions for training
        val_conditions: Conditions for validation
        test_conditions: All test conditions
        test_strata: Dict mapping tier names to condition lists:
            - 'single_holdout': single-target drugs held out for drug OOD (optional)
            - 'multi_seen_holdout': multi-target drugs held out with seen targets
        unseen_genes: List of target genes that appear only in test (should be empty)
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
    def load(cls, path: str | Path) -> "TahoeConditionSplit":
        """Load split from JSON file."""
        with open(path) as f:
            data = json.load(f)
        if "unseen_genes" not in data and "unseen_drugs" in data:
            data["unseen_genes"] = []
        allowed = set(cls.__dataclass_fields__.keys())
        data = {k: v for k, v in data.items() if k in allowed}
        return cls(**data)


class TahoeDrugSplitter:
    """
    Drug-based condition splitter for Tahoe dataset.

    Keeps test targets seen in training by:
    - Sending single-target drugs to train/val by default.
    - Optionally holding out a few single-target drugs per gene for drug OOD.
    - Allowing multi-target drugs into test only when all targets are seen in train.
    - Limiting multi-target test drugs to low target counts (prefer 2, fallback 3).
    - Keeping high target-count multi-target drugs in train.
    """

    def __init__(
        self,
        test_ratio: float = 0.15,
        single_val_ratio: float = 0.1,
        multi_val_ratio: float = 0.1,
        single_target_test_per_gene: int = 0,
        multi_test_max_targets: int = 2,
        multi_test_fallback_max_targets: int = 3,
        min_cells_per_condition: int = 5,
        seed: int = 42,
    ):
        """
        Initialize splitter.

        Args:
            test_ratio: Fraction of drugs held out as test
            single_val_ratio: Validation ratio for single-target drugs (after holdouts)
            multi_val_ratio: Validation ratio for remaining multi-target drugs
            single_target_test_per_gene: Hold out up to this many single-target drugs per gene
            multi_test_max_targets: Max target count allowed in multi-target test (preferred)
            multi_test_fallback_max_targets: Fallback max target count if not enough tests
            min_cells_per_condition: Minimum cells to keep a condition
            seed: Random seed for reproducibility
        """
        if not 0 <= test_ratio <= 1:
            raise ValueError("test_ratio must be between 0 and 1.")
        if not 0 <= single_val_ratio <= 1:
            raise ValueError("single_val_ratio must be between 0 and 1.")
        if not 0 <= multi_val_ratio <= 1:
            raise ValueError("multi_val_ratio must be between 0 and 1.")
        if single_target_test_per_gene < 0:
            raise ValueError("single_target_test_per_gene must be >= 0.")
        if multi_test_max_targets < 2:
            raise ValueError("multi_test_max_targets must be >= 2.")
        if multi_test_fallback_max_targets < multi_test_max_targets:
            raise ValueError(
                "multi_test_fallback_max_targets must be >= multi_test_max_targets."
            )

        self.test_ratio = test_ratio
        self.single_val_ratio = single_val_ratio
        self.multi_val_ratio = multi_val_ratio
        self.single_target_test_per_gene = single_target_test_per_gene
        self.multi_test_max_targets = multi_test_max_targets
        self.multi_test_fallback_max_targets = multi_test_fallback_max_targets
        self.min_cells_per_condition = min_cells_per_condition
        self.seed = seed

    def split(
        self,
        obs_df: pd.DataFrame,
        condition_col: str = "condition",
        drug_col: str = "drug",
        target_gene_col: str = "target_gene",
    ) -> TahoeConditionSplit:
        """
        Create drug-based condition-level split.

        Args:
            obs_df: DataFrame with condition, drug, and target-gene columns
            condition_col: Column name for conditions
            drug_col: Column name for drugs
            target_gene_col: Column name for target genes

        Returns:
            TahoeConditionSplit with train/val/test conditions and test strata
        """
        rng = np.random.default_rng(self.seed)

        # Step 1: Filter low-coverage conditions
        condition_counts = obs_df[condition_col].value_counts()
        valid_conditions = condition_counts[
            condition_counts >= self.min_cells_per_condition
        ].index.tolist()

        # Get unique condition-drug mapping (condition can be multi-gene per drug)
        cond_drug = (
            obs_df[[condition_col, drug_col]]
            .drop_duplicates()
            .set_index(condition_col)[[drug_col]]
        )
        cond_drug = cond_drug.loc[cond_drug.index.isin(valid_conditions)]

        filtered_obs = obs_df[obs_df[condition_col].isin(valid_conditions)]

        # Step 2: Build drug -> target genes mapping (from obs)
        drug_targets = filtered_obs.groupby(drug_col)[target_gene_col].apply(
            lambda genes: sorted(
                {g for combo in genes for g in self._parse_gene_combo(combo)}
            )
        )
        single_target_drugs = [
            drug for drug, targets in drug_targets.items() if len(targets) == 1
        ]
        multi_target_drugs = [
            drug for drug, targets in drug_targets.items() if len(targets) > 1
        ]
        single_target_gene_for_drug = {
            drug: drug_targets[drug][0] for drug in single_target_drugs
        }

        # Step 3: Assign single-target drugs (train/val + optional test holdout)
        single_train_drugs, single_val_drugs, single_test_drugs = (
            self._split_single_target_drugs(
                single_target_drugs,
                single_target_gene_for_drug,
                rng,
            )
        )

        # Step 4: Determine test counts
        n_total_test = self._resolve_test_count(len(drug_targets))
        n_multi_test_target = max(0, n_total_test - len(single_test_drugs))

        # Step 5: Select eligible multi-target test drugs
        multi_test_drugs, gene_train_support = self._select_multi_test_drugs(
            multi_target_drugs,
            drug_targets,
            single_train_drugs,
            single_target_gene_for_drug,
            n_multi_test_target,
            rng,
        )

        # Step 6: Split remaining multi-target drugs into train/val
        multi_test_set = set(multi_test_drugs)
        remaining_multi_drugs = [
            drug for drug in multi_target_drugs if drug not in multi_test_set
        ]
        required_genes = self._collect_test_genes(
            single_test_drugs,
            multi_test_drugs,
            drug_targets,
            single_target_gene_for_drug,
        )
        multi_val_drugs = self._select_multi_val_drugs(
            remaining_multi_drugs,
            drug_targets,
            gene_train_support,
            required_genes,
            rng,
        )
        multi_val_set = set(multi_val_drugs)
        multi_train_drugs = [
            drug for drug in remaining_multi_drugs if drug not in multi_val_set
        ]

        # Step 6: Assign conditions to splits based on drug
        train_drugs = set(single_train_drugs + multi_train_drugs)
        val_drugs = set(single_val_drugs + multi_val_drugs)
        test_drugs = set(single_test_drugs + multi_test_drugs)

        train_conditions = [
            c for c, row in cond_drug.iterrows() if row[drug_col] in train_drugs
        ]
        val_conditions = [
            c for c, row in cond_drug.iterrows() if row[drug_col] in val_drugs
        ]
        test_conditions = [
            c for c, row in cond_drug.iterrows() if row[drug_col] in test_drugs
        ]

        # Shuffle for good measure
        rng.shuffle(train_conditions)
        rng.shuffle(val_conditions)
        rng.shuffle(test_conditions)

        # Step 7: Define test strata
        test_strata = {
            "single_holdout": [
                c
                for c, row in cond_drug.iterrows()
                if row[drug_col] in single_test_drugs
            ],
            "multi_seen_holdout": [
                c
                for c, row in cond_drug.iterrows()
                if row[drug_col] in multi_test_drugs
            ],
        }

        train_genes = {single_target_gene_for_drug[drug] for drug in single_train_drugs}
        train_genes.update(
            self._collect_genes_for_drugs(multi_train_drugs, drug_targets)
        )
        test_genes = required_genes
        unseen_genes = sorted(test_genes - train_genes)

        return TahoeConditionSplit(
            train_conditions=train_conditions,
            val_conditions=val_conditions,
            test_conditions=test_conditions,
            test_strata=test_strata,
            unseen_genes=unseen_genes,
            seed=self.seed,
        )

    def _resolve_test_count(self, n_drugs: int) -> int:
        """Resolve test count based on total drugs and ratio."""
        n_test = int(round(n_drugs * self.test_ratio))
        return max(0, min(n_test, n_drugs))

    def _split_single_target_drugs(
        self,
        single_target_drugs: List[str],
        single_target_gene_for_drug: Dict[str, str],
        rng: np.random.Generator,
    ) -> tuple[List[str], List[str], List[str]]:
        """Split single-target drugs into train/val (+ optional test holdout)."""
        gene_to_drugs: Dict[str, List[str]] = {}
        for drug in single_target_drugs:
            gene = single_target_gene_for_drug[drug]
            gene_to_drugs.setdefault(gene, []).append(drug)

        train: List[str] = []
        val: List[str] = []
        test: List[str] = []

        for drugs in gene_to_drugs.values():
            rng.shuffle(drugs)
            n_test = 0
            if self.single_target_test_per_gene > 0 and len(drugs) > 1:
                n_test = min(self.single_target_test_per_gene, len(drugs) - 1)
            test.extend(drugs[:n_test])
            remaining = drugs[n_test:]
            if not remaining:
                continue
            n_val = int(round(len(remaining) * self.single_val_ratio))
            n_val = min(n_val, len(remaining) - 1)
            val.extend(remaining[:n_val])
            train.extend(remaining[n_val:])

        return train, val, test

    @staticmethod
    def _parse_gene_combo(combo: str) -> List[str]:
        """Parse a target gene string into a sorted list of genes."""
        if combo is None:
            return []
        genes = [
            g.strip()
            for g in str(combo).split("+")
            if g.strip() and g.strip() != "ctrl"
        ]
        return sorted(set(genes))

    @staticmethod
    def _collect_genes_for_drugs(
        drugs: List[str],
        drug_targets: pd.Series,
    ) -> set[str]:
        genes: set[str] = set()
        for drug in drugs:
            genes.update(drug_targets[drug])
        return genes

    @staticmethod
    def _collect_test_genes(
        single_test_drugs: List[str],
        multi_test_drugs: List[str],
        drug_targets: pd.Series,
        single_target_gene_for_drug: Dict[str, str],
    ) -> set[str]:
        genes: set[str] = set()
        for drug in single_test_drugs:
            genes.add(single_target_gene_for_drug[drug])
        genes.update(
            TahoeDrugSplitter._collect_genes_for_drugs(
                multi_test_drugs,
                drug_targets,
            )
        )
        return genes

    @staticmethod
    def _build_gene_support(
        single_train_drugs: List[str],
        multi_train_drugs: List[str],
        single_target_gene_for_drug: Dict[str, str],
        drug_targets: pd.Series,
    ) -> Dict[str, int]:
        support: Dict[str, int] = {}
        for drug in single_train_drugs:
            gene = single_target_gene_for_drug[drug]
            support[gene] = support.get(gene, 0) + 1
        for drug in multi_train_drugs:
            for gene in drug_targets[drug]:
                support[gene] = support.get(gene, 0) + 1
        return support

    def _ordered_multi_test_candidates(
        self,
        multi_target_drugs: List[str],
        drug_targets: pd.Series,
        rng: np.random.Generator,
    ) -> List[str]:
        buckets: Dict[int, List[str]] = {}
        for drug in multi_target_drugs:
            count = len(drug_targets[drug])
            if count > self.multi_test_fallback_max_targets:
                continue
            buckets.setdefault(count, []).append(drug)

        for bucket in buckets.values():
            rng.shuffle(bucket)

        primary_counts = [
            count
            for count in sorted(buckets.keys())
            if count <= self.multi_test_max_targets
        ]
        fallback_counts = [
            count
            for count in sorted(buckets.keys())
            if self.multi_test_max_targets
            < count
            <= self.multi_test_fallback_max_targets
        ]

        candidates: List[str] = []
        for count in primary_counts + fallback_counts:
            candidates.extend(buckets[count])
        return candidates

    @staticmethod
    def _can_remove_from_train(
        drug: str,
        drug_targets: pd.Series,
        gene_support: Dict[str, int],
        required_genes: Optional[set[str]] = None,
    ) -> bool:
        targets = drug_targets[drug]
        if required_genes is not None:
            targets = [g for g in targets if g in required_genes]
        return all(gene_support.get(gene, 0) >= 2 for gene in targets)

    def _select_multi_test_drugs(
        self,
        multi_target_drugs: List[str],
        drug_targets: pd.Series,
        single_train_drugs: List[str],
        single_target_gene_for_drug: Dict[str, str],
        n_multi_test_target: int,
        rng: np.random.Generator,
    ) -> tuple[List[str], Dict[str, int]]:
        gene_support = self._build_gene_support(
            single_train_drugs,
            multi_target_drugs,
            single_target_gene_for_drug,
            drug_targets,
        )
        if n_multi_test_target <= 0 or not multi_target_drugs:
            return [], gene_support

        candidates = self._ordered_multi_test_candidates(
            multi_target_drugs,
            drug_targets,
            rng,
        )

        test: List[str] = []
        for drug in candidates:
            if len(test) >= n_multi_test_target:
                break
            if not self._can_remove_from_train(drug, drug_targets, gene_support):
                continue
            test.append(drug)
            for gene in drug_targets[drug]:
                gene_support[gene] -= 1

        return test, gene_support

    def _select_multi_val_drugs(
        self,
        remaining_multi_drugs: List[str],
        drug_targets: pd.Series,
        gene_support: Dict[str, int],
        required_genes: set[str],
        rng: np.random.Generator,
    ) -> List[str]:
        if not remaining_multi_drugs:
            return []

        n_val = int(round(len(remaining_multi_drugs) * self.multi_val_ratio))
        n_val = min(n_val, len(remaining_multi_drugs))
        if n_val == 0:
            return []

        candidates = [
            drug
            for drug in remaining_multi_drugs
            if len(drug_targets[drug]) <= self.multi_test_fallback_max_targets
        ]
        n_val = min(n_val, len(candidates))
        if n_val == 0:
            return []

        rng.shuffle(candidates)
        val: List[str] = []

        for drug in candidates:
            if len(val) >= n_val:
                break
            if required_genes and not self._can_remove_from_train(
                drug,
                drug_targets,
                gene_support,
                required_genes,
            ):
                continue
            val.append(drug)
            for gene in drug_targets[drug]:
                gene_support[gene] -= 1

        return val

    def summary(self, split: TahoeConditionSplit) -> Dict:
        """Generate summary statistics for a split."""
        return {
            "n_train": len(split.train_conditions),
            "n_val": len(split.val_conditions),
            "n_test": len(split.test_conditions),
            "n_unseen_genes": len(split.unseen_genes),
            "test_strata": {k: len(v) for k, v in split.test_strata.items()},
            "seed": split.seed,
        }
