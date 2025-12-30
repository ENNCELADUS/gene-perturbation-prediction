"""
Evaluation script for TGA (Target-Gene Activation) baseline.

Evaluates the TGA heuristic on test conditions and reports:
- Exact hit@K (exact condition match in top-K)
- Relevant hit@K (one-gene overlap in top-K)
- Stratified results by seen2/seen1/seen0

For CRISPRa datasets like Norman where target genes are upregulated.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from tqdm import tqdm
from scipy import sparse

from ..data import load_perturb_data
from ..model.tga import TGA
from .metrics import (
    compute_all_metrics,
    parse_condition_genes,
    top_k_accuracy,
    one_gene_overlap_hit_at_k,
    mrr,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TGA baseline")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument(
        "--output",
        type=str,
        default="results/tgd/eval_results.json",
        help="Output path for results JSON",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_cells_for_condition(dataset, condition: str):
    """
    Get cells for a condition, handling format variations.

    The split uses normalized conditions (e.g., 'ATL1') but adata may use
    original format (e.g., 'ATL1+ctrl' or 'ctrl+ATL1').

    Args:
        dataset: PerturbDataset
        condition: Normalized condition string

    Returns:
        AnnData subset for this condition
    """
    adata = dataset.adata
    cond_col = dataset.condition_col

    # Try exact match first
    mask = adata.obs[cond_col] == condition
    if mask.sum() > 0:
        return adata[mask].copy()

    # Parse genes from condition
    genes = parse_condition_genes(condition)
    if not genes:
        return adata[mask].copy()  # Return empty

    # Build possible format variations
    possible_formats = [condition]

    if len(genes) == 1:
        # Single gene: try GENE+ctrl and ctrl+GENE
        gene = list(genes)[0]
        possible_formats.extend(
            [
                f"{gene}+ctrl",
                f"ctrl+{gene}",
            ]
        )
    elif len(genes) == 2:
        # Double gene: try both orderings
        g1, g2 = sorted(genes)
        possible_formats.extend(
            [
                f"{g1}+{g2}",
                f"{g2}+{g1}",
            ]
        )

    # Try each format
    for fmt in possible_formats:
        mask = adata.obs[cond_col] == fmt
        if mask.sum() > 0:
            return adata[mask].copy()

    # Return empty if nothing found
    return adata[adata.obs[cond_col] == "__NOMATCH__"].copy()


def evaluate_tga(
    dataset,
    tga_model: TGA,
    candidate_conditions: List[str],
    top_k_values: List[int] = [1, 5, 8, 10],
    mask_k: int = 0,
    target_gene_pool: List[str] | None = None,
) -> Dict:
    """
    Evaluate TGD baseline on test conditions.

    Args:
        dataset: PerturbDataset with condition split
        tgd_model: Fitted TGD model
        candidate_conditions: List of candidate conditions
        top_k_values: K values for hit@K metrics

    Returns:
        Dictionary with evaluation results
    """
    test_conditions = dataset.test_conditions
    test_strata = dataset.test_strata

    # Collect predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    strata_predictions = defaultdict(list)
    strata_ground_truth = defaultdict(list)

    # Build strata mapping for efficient lookup
    condition_to_stratum = {}
    for stratum, conditions in test_strata.items():
        for cond in conditions:
            condition_to_stratum[cond] = stratum

    print(f"\nEvaluating on {len(test_conditions)} test conditions...")

    for condition in tqdm(test_conditions, desc="Evaluating"):
        # Get cells for this condition (handles format variations)
        try:
            query_adata = get_cells_for_condition(dataset, condition)
        except Exception as e:
            print(f"  Warning: Could not get cells for {condition}: {e}")
            continue

        if query_adata.n_obs == 0:
            continue

        if mask_k > 0:
            query_adata = apply_target_gene_mask(
                query_adata,
                condition,
                tga_model,
                mask_k=mask_k,
                target_gene_pool=target_gene_pool,
            )

        # Predict top-K (use max K value)
        max_k = max(top_k_values)
        predictions = tga_model.predict(
            query_adata,
            candidate_conditions,
            top_k=max_k,
            use_pseudobulk=True,
        )

        all_predictions.append(predictions)
        all_ground_truth.append(condition)

        # Track by stratum
        stratum = condition_to_stratum.get(condition, "unknown")
        strata_predictions[stratum].append(predictions)
        strata_ground_truth[stratum].append(condition)

    # Compute overall metrics
    print("\nComputing metrics...")
    overall_metrics = compute_all_metrics(
        all_predictions,
        all_ground_truth,
        top_k_values=top_k_values,
        candidate_pool=candidate_conditions,
    )

    # Compute stratified metrics
    strata_metrics = {}
    for stratum in sorted(strata_predictions.keys()):
        preds = strata_predictions[stratum]
        truth = strata_ground_truth[stratum]

        if len(truth) == 0:
            continue

        strata_metrics[stratum] = compute_all_metrics(
            preds,
            truth,
            top_k_values=top_k_values,
            candidate_pool=candidate_conditions,
        )

    results = {
        "overall": overall_metrics,
        "by_stratum": strata_metrics,
        "n_test_conditions": len(test_conditions),
        "n_evaluated": len(all_ground_truth),
        "n_candidates": len(candidate_conditions),
    }

    return results


def apply_target_gene_mask(
    query_adata,
    condition: str,
    tga_model: TGA,
    mask_k: int,
    target_gene_pool: List[str] | None,
):
    genes = sorted(parse_condition_genes(condition))
    if not genes:
        return query_adata

    if tga_model.gene_name_to_idx is None:
        return query_adata

    if mask_k < len(genes):
        mask_k = len(genes)

    extra_genes = select_extra_mask_genes(
        condition=condition,
        target_genes=genes,
        target_gene_pool=target_gene_pool or [],
        extra_k=mask_k - len(genes),
    )
    mask_genes = genes + extra_genes

    X = query_adata.X
    if sparse.issparse(X):
        X = X.toarray()

    X = np.array(X, copy=True)
    for gene in mask_genes:
        idx = tga_model.gene_name_to_idx.get(gene)
        if idx is None:
            continue
        X[:, idx] = 0.0

    query_adata.X = X
    return query_adata


def select_extra_mask_genes(
    condition: str,
    target_genes: List[str],
    target_gene_pool: List[str],
    extra_k: int,
) -> List[str]:
    if extra_k <= 0:
        return []

    pool = [g for g in target_gene_pool if g not in target_genes]
    if not pool:
        return []

    seed = int(hashlib.md5(condition.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    sample_k = min(extra_k, len(pool))
    return rng.choice(pool, size=sample_k, replace=False).tolist()


def build_target_gene_pool(conditions: List[str]) -> List[str]:
    pool = set()
    for cond in conditions:
        pool.update(parse_condition_genes(cond))
    return sorted(pool)


def main():
    args = parse_args()
    config = load_config(args.config)

    print("=" * 60)
    print("TGA Baseline Evaluation (Target-Gene Activation)")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading dataset...")
    cond_split_config = config.get("condition_split", {})
    dataset = load_perturb_data(
        h5ad_path=config["data"]["h5ad_path"],
        condition_split_path=cond_split_config.get("output_path"),
        **{k: v for k, v in cond_split_config.items() if k != "output_path"},
    )

    print(f"  - Total cells: {dataset.adata.n_obs}")
    print(f"  - Train conditions: {len(dataset.train_conditions)}")
    print(f"  - Test conditions: {len(dataset.test_conditions)}")
    print(f"  - Test strata: {list(dataset.test_strata.keys())}")

    # Get control cells from train split only (avoid test data leakage)
    print("\n[2/4] Fitting TGA model on train control cells...")

    # Get all control cells
    control_adata = dataset.control_adata
    print(f"  - Control cells: {control_adata.n_obs}")

    # Create and fit TGA model
    baseline_config = config.get("baseline", {})
    tga = TGA(epsilon=baseline_config.get("epsilon", 1e-6))
    tga.fit(control_adata)

    # Build candidate condition set
    print("\n[3/4] Building candidate condition set...")
    # Use all conditions from the split (train + val + test)
    candidate_conditions = dataset.all_conditions
    print(f"  - Candidate conditions: {len(candidate_conditions)}")

    # Evaluate
    print("\n[4/4] Running evaluation...")
    eval_config = config.get("evaluation", {})
    top_k_values = eval_config.get("top_k_values", [1, 5, 8, 10])
    mask_k = eval_config.get("mask", 0)
    if isinstance(mask_k, bool):
        mask_k = int(mask_k)
    if mask_k > 0:
        print(f"  - Masking {mask_k} genes per query (anti-cheat) enabled")
    target_gene_pool = build_target_gene_pool(dataset.all_conditions)

    results = evaluate_tga(
        dataset,
        tga,
        candidate_conditions,
        top_k_values=top_k_values,
        mask_k=mask_k,
        target_gene_pool=target_gene_pool,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    print("\nOverall Metrics:")
    for k in top_k_values:
        exact = results["overall"].get(f"exact_hit@{k}", 0)
        relevant = results["overall"].get(f"relevant_hit@{k}", 0)
        print(f"  Hit@{k}: exact={exact:.3f}, relevant={relevant:.3f}")

    print(f"  MRR: {results['overall'].get('mrr', 0):.3f}")

    print("\nBy Stratum:")
    for stratum in sorted(results["by_stratum"].keys()):
        metrics = results["by_stratum"][stratum]
        n = metrics.get("n_queries", 0)
        exact_1 = metrics.get("exact_hit@1", 0)
        exact_5 = metrics.get("exact_hit@5", 0)
        print(f"  {stratum} (n={n}): hit@1={exact_1:.3f}, hit@5={exact_5:.3f}")

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
