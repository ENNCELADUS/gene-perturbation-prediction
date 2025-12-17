"""
Retrieval metrics for reverse perturbation prediction.

Implements:
- Top-K accuracy: fraction of queries where true label is in top-K
- MRR (Mean Reciprocal Rank): average of 1/rank of true label
- NDCG (Normalized Discounted Cumulative Gain): ranking quality
"""

from typing import List, Dict, Union
import numpy as np


def top_k_accuracy(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int,
) -> float:
    """
    Compute Top-K accuracy.

    Args:
        predictions: List of top-K predictions per query
        ground_truth: List of true labels
        k: K value

    Returns:
        Fraction of queries where true label is in top-K
    """
    correct = 0
    for preds, true in zip(predictions, ground_truth):
        if true in preds[:k]:
            correct += 1
    return correct / len(ground_truth) if ground_truth else 0.0


def mrr(
    predictions: List[List[str]],
    ground_truth: List[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        predictions: List of ranked predictions per query
        ground_truth: List of true labels

    Returns:
        MRR score
    """
    reciprocal_ranks = []
    for preds, true in zip(predictions, ground_truth):
        try:
            rank = preds.index(true) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def ndcg(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).

    For single-label retrieval, NDCG@K = 1/log2(rank+1) if found in top-K, else 0.

    Args:
        predictions: List of ranked predictions per query
        ground_truth: List of true labels
        k: Cutoff (None = use full list)

    Returns:
        NDCG score
    """
    dcg_scores = []
    for preds, true in zip(predictions, ground_truth):
        if k:
            preds = preds[:k]
        try:
            rank = preds.index(true) + 1
            dcg = 1.0 / np.log2(rank + 1)
        except ValueError:
            dcg = 0.0
        dcg_scores.append(dcg)

    # Ideal DCG is 1/log2(2) = 1.0 for single relevant item at rank 1
    idcg = 1.0
    return np.mean(dcg_scores) / idcg if dcg_scores else 0.0


def compute_all_metrics(
    predictions: List[List[str]],
    ground_truth: List[str],
    top_k_values: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """
    Compute all retrieval metrics.

    Args:
        predictions: List of ranked predictions per query
        ground_truth: List of true labels
        top_k_values: K values for Top-K accuracy

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}

    # Top-K accuracy
    for k in top_k_values:
        metrics[f"top_{k}_accuracy"] = top_k_accuracy(predictions, ground_truth, k)

    # MRR
    metrics["mrr"] = mrr(predictions, ground_truth)

    # NDCG
    for k in top_k_values:
        metrics[f"ndcg@{k}"] = ndcg(predictions, ground_truth, k)

    return metrics
