"""
Evaluation script for DE (Differential Expression) baseline.

Uses Wilcoxon rank-sum test to find top upregulated genes and predicts
condition by matching them to candidate conditions.

For CRISPRa datasets like Norman where target genes are upregulated.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from tqdm import tqdm

from ..data import load_perturb_data
from ..model.de_baseline import DEBaseline
from .metrics import compute_all_metrics, parse_condition_genes


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DE baseline")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument(
        "--output",
        type=str,
        default="results/de/eval_results.json",
        help="Output path for results JSON",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_cells_for_condition(dataset, condition: str):
    """
    Get cells for a condition, handling format variations.
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
        return adata[mask].copy()

    # Build possible format variations
    possible_formats = [condition]

    if len(genes) == 1:
        gene = list(genes)[0]
        possible_formats.extend([f"{gene}+ctrl", f"ctrl+{gene}"])
    elif len(genes) == 2:
        g1, g2 = sorted(genes)
        possible_formats.extend([f"{g1}+{g2}", f"{g2}+{g1}"])

    for fmt in possible_formats:
        mask = adata.obs[cond_col] == fmt
        if mask.sum() > 0:
            return adata[mask].copy()

    return adata[adata.obs[cond_col] == "__NOMATCH__"].copy()


def evaluate_de(
    dataset,
    de_model: DEBaseline,
    candidate_conditions: List[str],
    top_k_values: List[int] = [1, 5, 8, 10],
) -> Dict:
    """
    Evaluate DE baseline on test conditions.
    """
    test_conditions = dataset.test_conditions
    test_strata = dataset.test_strata

    all_predictions = []
    all_ground_truth = []
    strata_predictions = defaultdict(list)
    strata_ground_truth = defaultdict(list)

    condition_to_stratum = {}
    for stratum, conditions in test_strata.items():
        for cond in conditions:
            condition_to_stratum[cond] = stratum

    print(f"\nEvaluating on {len(test_conditions)} test conditions...")

    for condition in tqdm(test_conditions, desc="Evaluating"):
        try:
            query_adata = get_cells_for_condition(dataset, condition)
        except Exception as e:
            print(f"  Warning: Could not get cells for {condition}: {e}")
            continue

        if query_adata.n_obs == 0:
            continue

        max_k = max(top_k_values)
        predictions = de_model.predict(query_adata, candidate_conditions, top_k=max_k)

        all_predictions.append(predictions)
        all_ground_truth.append(condition)

        stratum = condition_to_stratum.get(condition, "unknown")
        strata_predictions[stratum].append(predictions)
        strata_ground_truth[stratum].append(condition)

    print("\nComputing metrics...")
    overall_metrics = compute_all_metrics(
        all_predictions,
        all_ground_truth,
        top_k_values=top_k_values,
        candidate_pool=candidate_conditions,
    )

    strata_metrics = {}
    for stratum in sorted(strata_predictions.keys()):
        preds = strata_predictions[stratum]
        truth = strata_ground_truth[stratum]

        if len(truth) == 0:
            continue

        strata_metrics[stratum] = compute_all_metrics(
            preds, truth, top_k_values=top_k_values, candidate_pool=candidate_conditions
        )

    return {
        "overall": overall_metrics,
        "by_stratum": strata_metrics,
        "n_test_conditions": len(test_conditions),
        "n_evaluated": len(all_ground_truth),
        "n_candidates": len(candidate_conditions),
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    print("=" * 60)
    print("DE Baseline Evaluation (Differential Expression)")
    print("=" * 60)

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

    print("\n[2/4] Fitting DE model on train control cells...")
    control_adata = dataset.control_adata
    print(f"  - Control cells: {control_adata.n_obs}")

    baseline_config = config.get("baseline", {})
    de = DEBaseline(top_n_de_genes=baseline_config.get("top_n_de_genes", 2))
    de.fit(control_adata)

    print("\n[3/4] Building candidate condition set...")
    candidate_conditions = dataset.all_conditions
    print(f"  - Candidate conditions: {len(candidate_conditions)}")

    print("\n[4/4] Running evaluation...")
    eval_config = config.get("evaluation", {})
    top_k_values = eval_config.get("top_k_values", [1, 5, 8, 10])

    results = evaluate_de(dataset, de, candidate_conditions, top_k_values=top_k_values)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

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
