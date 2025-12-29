"""
Drug-based condition splits for Tahoe dataset.

Implements a split strategy that mirrors Norman's gene-visibility
logic while keeping Tahoe's drug-based conditions.

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
            - 'single_unseen': single-target drugs whose target genes are unseen
            - 'multi_unseen': multi-target drugs containing unseen single-target genes
            - 'multi_seen_holdout': multi-target drugs seen but held out for test
        unseen_genes: List of unseen single-target genes (test-only)
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

    Aligns with Norman-style visibility logic by:
    - Sampling unseen genes from single-target drugs.
    - Sending all single-target drugs for unseen genes to test.
    - Sending multi-target drugs with unseen genes to test.
    - Splitting remaining multi-target drugs into train/val/test.
    """

    def __init__(
        self,
        unseen_single_gene_fraction: float = 0.25,
        n_unseen_single_genes: Optional[int] = None,
        single_seen_val_ratio: float = 0.1,
        multi_train_ratio: float = 0.65,
        multi_val_ratio: float = 0.1,
        min_cells_per_condition: int = 5,
        seed: int = 42,
    ):
        """
        Initialize splitter.

        Args:
            unseen_single_gene_fraction: Fraction of unique single-target genes to hold out
            n_unseen_single_genes: Optional explicit count of unseen single-target genes
            single_seen_val_ratio: Validation ratio for seen single-target drugs
            multi_train_ratio: Train ratio for seen multi-target drugs
            multi_val_ratio: Validation ratio for seen multi-target drugs
            min_cells_per_condition: Minimum cells to keep a condition
            seed: Random seed for reproducibility
        """
        self.unseen_single_gene_fraction = unseen_single_gene_fraction
        self.n_unseen_single_genes = n_unseen_single_genes
        self.single_seen_val_ratio = single_seen_val_ratio
        self.multi_train_ratio = multi_train_ratio
        self.multi_val_ratio = multi_val_ratio
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

        # Step 3: Select unseen genes from unique single-target genes
        single_target_genes = sorted(set(single_target_gene_for_drug.values()))
        if not single_target_genes:
            raise ValueError("No single-target genes found for Tahoe split logic.")

        single_obs = filtered_obs[filtered_obs[drug_col].isin(single_target_drugs)]
        gene_sizes = (
            single_obs[target_gene_col]
            .apply(self._parse_gene_combo)
            .explode()
            .value_counts()
        )
        n_unseen = self._resolve_unseen_gene_count(
            len(single_target_genes),
        )
        unseen_genes = self._stratified_gene_sample(
            gene_sizes,
            single_target_genes,
            n_unseen,
            rng,
        )

        # Step 4: Assign single-target drugs (unseen -> test, seen -> train/val)
        single_unseen_drugs = [
            drug
            for drug in single_target_drugs
            if single_target_gene_for_drug[drug] in unseen_genes
        ]
        single_seen_drugs = [
            drug for drug in single_target_drugs if drug not in single_unseen_drugs
        ]

        rng.shuffle(single_seen_drugs)
        n_single_val = int(round(len(single_seen_drugs) * self.single_seen_val_ratio))
        n_single_val = min(n_single_val, len(single_seen_drugs))
        single_val_drugs = single_seen_drugs[:n_single_val]
        single_train_drugs = single_seen_drugs[n_single_val:]

        # Step 5: Assign multi-target drugs based on unseen genes
        multi_unseen_drugs = [
            drug
            for drug in multi_target_drugs
            if self._has_unseen_gene(drug_targets[drug], unseen_genes)
        ]
        multi_seen_drugs = [
            drug for drug in multi_target_drugs if drug not in multi_unseen_drugs
        ]

        multi_train_drugs, multi_val_drugs, multi_test_drugs = (
            self._stratified_multi_split(
                multi_seen_drugs,
                drug_targets,
                rng,
            )
        )

        # Step 6: Assign conditions to splits based on drug
        train_drugs = set(single_train_drugs + multi_train_drugs)
        val_drugs = set(single_val_drugs + multi_val_drugs)
        test_drugs = set(single_unseen_drugs + multi_unseen_drugs + multi_test_drugs)

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
            "single_unseen": [
                c
                for c, row in cond_drug.iterrows()
                if row[drug_col] in single_unseen_drugs
            ],
            "multi_unseen": [
                c
                for c, row in cond_drug.iterrows()
                if row[drug_col] in multi_unseen_drugs
            ],
            "multi_seen_holdout": [
                c
                for c, row in cond_drug.iterrows()
                if row[drug_col] in multi_test_drugs
            ],
        }

        return TahoeConditionSplit(
            train_conditions=train_conditions,
            val_conditions=val_conditions,
            test_conditions=test_conditions,
            test_strata=test_strata,
            unseen_genes=sorted(unseen_genes),
            seed=self.seed,
        )

    def _resolve_unseen_gene_count(self, n_genes: int) -> int:
        """Resolve the unseen gene count based on fraction or explicit value."""
        if self.n_unseen_single_genes is not None:
            n_unseen = self.n_unseen_single_genes
        else:
            n_unseen = int(n_genes * self.unseen_single_gene_fraction)

        n_unseen = max(1, min(n_unseen, n_genes))
        return n_unseen

    def _stratified_gene_sample(
        self,
        gene_sizes: pd.Series,
        genes: List[str],
        n_sample: int,
        rng: np.random.Generator,
    ) -> List[str]:
        """
        Sample genes stratified by size quartiles.

        Ensures unseen genes are representative of size ranges.
        """
        if len(genes) <= 3:
            return rng.choice(genes, size=n_sample, replace=False).tolist()

        sizes = gene_sizes.reindex(genes).fillna(0)
        try:
            quartiles = pd.qcut(sizes, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        except ValueError:
            return rng.choice(genes, size=n_sample, replace=False).tolist()

        quartile_genes = {q: [] for q in ["Q1", "Q2", "Q3", "Q4"]}
        for gene, q in quartiles.items():
            quartile_genes[q].append(gene)

        sampled = []
        n_per_quartile = max(1, n_sample // 4)

        for q in ["Q1", "Q2", "Q3", "Q4"]:
            bucket = quartile_genes[q]
            if not bucket:
                continue
            n_take = min(len(bucket), n_per_quartile)
            sampled.extend(rng.choice(bucket, size=n_take, replace=False).tolist())

        remaining = [g for g in genes if g not in sampled]
        if len(sampled) < n_sample and remaining:
            extra = min(n_sample - len(sampled), len(remaining))
            sampled.extend(rng.choice(remaining, size=extra, replace=False).tolist())

        return sampled[:n_sample]

    @staticmethod
    def _has_unseen_gene(targets: List[str], unseen_genes: List[str]) -> bool:
        """Check if any target is in unseen genes."""
        unseen_set = set(unseen_genes)
        return any(t in unseen_set for t in targets)

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

    def _stratified_multi_split(
        self,
        drugs: List[str],
        drug_targets: pd.Series,
        rng: np.random.Generator,
    ) -> tuple[List[str], List[str], List[str]]:
        """Split multi-target drugs by target count buckets."""
        buckets: Dict[int, List[str]] = {}
        for drug in drugs:
            count = len(drug_targets[drug])
            buckets.setdefault(count, []).append(drug)

        train: List[str] = []
        val: List[str] = []
        test: List[str] = []

        for count in sorted(buckets.keys()):
            bucket = buckets[count]
            rng.shuffle(bucket)
            n_train, n_val = self._split_counts(len(bucket))
            train.extend(bucket[:n_train])
            val.extend(bucket[n_train : n_train + n_val])
            test.extend(bucket[n_train + n_val :])

        return train, val, test

    def _split_counts(self, n_items: int) -> tuple[int, int]:
        """Get train/val counts for a bucket."""
        n_train = int(round(n_items * self.multi_train_ratio))
        n_val = int(round(n_items * self.multi_val_ratio))
        if n_train + n_val > n_items:
            n_val = max(0, n_items - n_train)
        n_train = min(n_train, n_items - n_val)
        return n_train, n_val

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
