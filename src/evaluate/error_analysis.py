"""
Error analysis for retrieval failures.

Provides:
- Confusion matrix construction
- Commonly confused condition pairs
- DE gene overlap explanations
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

import numpy as np


def parse_condition_genes(condition: str) -> Set[str]:
    """Extract gene names from condition string."""
    if not condition or condition == "ctrl":
        return set()
    genes = condition.split("+")
    return {g.strip() for g in genes if g.strip() and g.strip() != "ctrl"}


def build_confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
    conditions: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build confusion matrix from predictions.

    Args:
        predictions: Top-1 predictions per query
        ground_truth: True labels per query
        conditions: Ordered list of conditions (optional, inferred if None)

    Returns:
        Tuple of (confusion_matrix, condition_labels)
        Matrix shape: (n_conditions, n_conditions)
        Matrix[i,j] = count of queries with true=i, pred=j
    """
    if conditions is None:
        conditions = sorted(set(ground_truth) | set(predictions))

    cond_to_idx = {c: i for i, c in enumerate(conditions)}
    n_conds = len(conditions)

    matrix = np.zeros((n_conds, n_conds), dtype=int)

    for pred, true in zip(predictions, ground_truth):
        if true in cond_to_idx and pred in cond_to_idx:
            i = cond_to_idx[true]
            j = cond_to_idx[pred]
            matrix[i, j] += 1

    return matrix, conditions


def find_commonly_confused_pairs(
    confusion_matrix: np.ndarray,
    conditions: List[str],
    k: int = 10,
) -> List[Tuple[str, str, int]]:
    """
    Find most commonly confused condition pairs.

    Args:
        confusion_matrix: From build_confusion_matrix
        conditions: Condition labels
        k: Number of top pairs to return

    Returns:
        List of (true_condition, predicted_condition, count) tuples
    """
    n = len(conditions)
    pairs = []

    for i in range(n):
        for j in range(n):
            if i != j and confusion_matrix[i, j] > 0:
                pairs.append((conditions[i], conditions[j], confusion_matrix[i, j]))

    # Sort by count descending
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def explain_confusion_by_gene_overlap(
    true_condition: str,
    predicted_condition: str,
) -> Dict[str, any]:
    """
    Explain why two conditions might be confused based on gene overlap.

    Args:
        true_condition: Ground truth condition
        predicted_condition: Wrongly predicted condition

    Returns:
        Dictionary with overlap analysis
    """
    true_genes = parse_condition_genes(true_condition)
    pred_genes = parse_condition_genes(predicted_condition)

    overlap = true_genes & pred_genes
    only_true = true_genes - pred_genes
    only_pred = pred_genes - true_genes

    return {
        "true_condition": true_condition,
        "predicted_condition": predicted_condition,
        "true_genes": list(true_genes),
        "predicted_genes": list(pred_genes),
        "overlapping_genes": list(overlap),
        "only_in_true": list(only_true),
        "only_in_pred": list(only_pred),
        "jaccard_similarity": len(overlap) / len(true_genes | pred_genes)
        if (true_genes | pred_genes)
        else 0.0,
    }


def compute_per_condition_accuracy(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Compute accuracy per condition.

    Args:
        predictions: Top-K predictions per query
        ground_truth: True labels
        k: K for Hit@K

    Returns:
        Dict mapping condition -> {hits, total, accuracy}
    """
    stats = defaultdict(lambda: {"hits": 0, "total": 0})

    for preds, true in zip(predictions, ground_truth):
        stats[true]["total"] += 1
        if true in preds[:k]:
            stats[true]["hits"] += 1

    # Compute accuracy
    for cond in stats:
        total = stats[cond]["total"]
        stats[cond]["accuracy"] = stats[cond]["hits"] / total if total > 0 else 0.0

    return dict(stats)


def find_hardest_conditions(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int = 1,
    n_hardest: int = 10,
) -> List[Tuple[str, float, int]]:
    """
    Find conditions with lowest accuracy.

    Args:
        predictions: Top-K predictions per query
        ground_truth: True labels
        k: K for Hit@K
        n_hardest: Number of hardest conditions to return

    Returns:
        List of (condition, accuracy, n_queries) sorted by accuracy ascending
    """
    per_cond = compute_per_condition_accuracy(predictions, ground_truth, k)

    results = [
        (cond, stats["accuracy"], stats["total"]) for cond, stats in per_cond.items()
    ]

    # Sort by accuracy ascending (hardest first)
    results.sort(key=lambda x: x[1])
    return results[:n_hardest]


def generate_error_report(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int = 1,
    n_confused_pairs: int = 10,
    n_hardest: int = 10,
) -> Dict[str, any]:
    """
    Generate comprehensive error analysis report.

    Args:
        predictions: Top-K predictions per query
        ground_truth: True labels
        k: K for Hit@K evaluation
        n_confused_pairs: Number of confused pairs to report
        n_hardest: Number of hardest conditions to report

    Returns:
        Dictionary with error analysis results
    """
    # Get top-1 predictions
    top1_preds = [preds[0] if preds else "" for preds in predictions]

    # Build confusion matrix
    conf_matrix, conditions = build_confusion_matrix(top1_preds, ground_truth)

    # Find confused pairs
    confused_pairs = find_commonly_confused_pairs(
        conf_matrix, conditions, n_confused_pairs
    )

    # Explain confusions
    explanations = [
        explain_confusion_by_gene_overlap(pair[0], pair[1]) for pair in confused_pairs
    ]

    # Find hardest conditions
    hardest = find_hardest_conditions(predictions, ground_truth, k, n_hardest)

    # Overall stats
    n_correct = sum(1 for p, t in zip(top1_preds, ground_truth) if p == t)
    accuracy = n_correct / len(ground_truth) if ground_truth else 0.0

    return {
        "overall_accuracy": accuracy,
        "n_queries": len(ground_truth),
        "n_conditions": len(conditions),
        "confused_pairs": [
            {"true": p[0], "predicted": p[1], "count": p[2]} for p in confused_pairs
        ],
        "confusion_explanations": explanations,
        "hardest_conditions": [
            {"condition": h[0], "accuracy": h[1], "n_queries": h[2]} for h in hardest
        ],
    }
