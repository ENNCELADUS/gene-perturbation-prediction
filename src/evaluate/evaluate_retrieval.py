"""
Retrieval-based evaluation for forward model.

Query test cells against reference database and evaluate with Hit@K metrics.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import pickle

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import defaultdict

from ..data import load_perturb_data
from ..evaluate.metrics import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument(
        "--reference_db", type=str, required=True, help="Reference database path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/forward/eval_results.json",
        help="Output path",
    )
    parser.add_argument(
        "--top_k", type=int, nargs="+", default=[1, 5, 10, 20], help="Top-K values"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def retrieve_nearest_conditions(
    query_profiles: np.ndarray,  # (n_queries, n_features)
    reference_db: Dict,
    metric: str = "cosine",
    k: int = 20,
) -> List[List[str]]:
    """
    Retrieve top-K nearest conditions for each query.

    Args:
        query_profiles: Query expression profiles
        reference_db: Reference database with predictions per condition
        metric: Similarity metric (cosine or euclidean)
        k: Number of neighbors to retrieve

    Returns:
        List of top-K condition names for each query
    """
    # Flatten reference database
    all_predictions = []
    condition_labels = []

    for condition, preds in reference_db["predictions"].items():
        all_predictions.append(preds)
        condition_labels.extend([condition] * len(preds))

    all_predictions = np.vstack(all_predictions)  # (total_samples, n_features)

    print(
        f"Reference database: {all_predictions.shape[0]} profiles from {len(reference_db['predictions'])} conditions"
    )
    print(f"Query: {query_profiles.shape[0]} profiles")

    # Compute similarities
    if metric == "cosine":
        similarities = cosine_similarity(query_profiles, all_predictions)
    elif metric == "euclidean":
        from scipy.spatial.distance import cdist

        distances = cdist(query_profiles, all_predictions, metric="euclidean")
        similarities = -distances  # Convert to similarity (higher is better)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Get top-K indices for each query
    top_k_indices = np.argsort(-similarities, axis=1)[:, :k]  # (n_queries, k)

    # Convert to condition names
    results = []
    for query_idx in range(len(query_profiles)):
        top_k_conds = [condition_labels[idx] for idx in top_k_indices[query_idx]]
        results.append(top_k_conds)

    return results


def aggregate_by_voting(
    per_cell_rankings: List[List[str]],
    k: int = 20,
) -> List[str]:
    """
    Aggregate per-cell rankings by voting to get a single ranking.

    Args:
        per_cell_rankings: List of top-K condition lists for each cell
        k: Number of conditions to return

    Returns:
        Top-K conditions after voting
    """
    # Count votes for each condition
    vote_counts = defaultdict(int)

    for ranking in per_cell_rankings:
        # Weight votes by rank (top-1 gets more weight than top-K)
        for rank, condition in enumerate(ranking):
            weight = len(ranking) - rank  # Linear decay
            vote_counts[condition] += weight

    # Sort by votes
    sorted_conditions = sorted(vote_counts.items(), key=lambda x: -x[1])
    top_k_conditions = [cond for cond, _ in sorted_conditions[:k]]

    return top_k_conditions


def evaluate_retrieval(
    dataset,
    reference_db: Dict,
    metric: str = "cosine",
    top_k_values: List[int] = [1, 5, 10, 20],
) -> Dict:
    """
    Evaluate retrieval performance on test set.

    Returns:
        Dictionary with metrics
    """
    print("\nEvaluating on test set...")

    test_adata = dataset.test_adata
    test_conditions = dataset.test_conditions

    print(f"  Test cells: {len(test_adata)}")
    print(f"  Test conditions: {len(test_conditions)}")

    # Group test cells by condition
    condition_to_cells = defaultdict(list)
    for i, condition in enumerate(test_adata.obs[dataset.condition_col]):
        condition_to_cells[condition].append(i)

    # Get test cell expressions
    test_exprs = (
        test_adata.X.toarray() if hasattr(test_adata.X, "toarray") else test_adata.X
    )

    # Retrieve for each condition
    all_predictions = []
    all_ground_truth = []

    for condition in tqdm(test_conditions, desc="Retrieving"):
        cell_indices = condition_to_cells[condition]
        if len(cell_indices) == 0:
            continue

        # Get query profiles for this condition
        query_profiles = test_exprs[cell_indices]

        # Retrieve top-K for each cell
        per_cell_rankings = retrieve_nearest_conditions(
            query_profiles,
            reference_db,
            metric=metric,
            k=max(top_k_values),
        )

        # Aggregate by voting
        aggregated_ranking = aggregate_by_voting(per_cell_rankings, k=max(top_k_values))

        # Store results
        all_predictions.append(aggregated_ranking)
        all_ground_truth.append(condition)

    print(f"\nEvaluated {len(all_predictions)} test conditions")

    # Compute metrics
    metrics = compute_all_metrics(
        predictions=all_predictions,
        ground_truth=all_ground_truth,
        top_k_values=top_k_values,
        candidate_pool=None,  # All conditions are in the pool
        include_macro=True,
    )

    return metrics


def main():
    args = parse_args()
    config = load_config(args.config)

    print("=" * 60)
    print("Retrieval-based Evaluation")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading dataset...")
    cond_split_config = config.get("condition_split", {})
    dataset = load_perturb_data(
        h5ad_path=config["data"]["h5ad_path"],
        condition_split_path=cond_split_config.get("output_path"),
        **{k: v for k, v in cond_split_config.items() if k != "output_path"},
    )

    # Load reference database
    print("\n[2/3] Loading reference database...")
    with open(args.reference_db, "rb") as f:
        reference_db = pickle.load(f)

    print(f"  Conditions in database: {len(reference_db['predictions'])}")

    # Evaluate
    print("\n[3/3] Evaluating...")
    metrics = evaluate_retrieval(
        dataset,
        reference_db,
        metric=config.get("retrieval", {}).get("metric", "cosine"),
        top_k_values=args.top_k,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
