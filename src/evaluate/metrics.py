"""
Retrieval metrics for reverse perturbation prediction.

Implements:
- Top-K accuracy (exact match): fraction of queries where true label is in top-K
- One-gene-overlap Hit@K (relevant retrieval): fraction where any prediction shares ≥1 gene
- MRR (Mean Reciprocal Rank): average of 1/rank of true label
- NDCG (Normalized Discounted Cumulative Gain): ranking quality
- DE gene overlap (Jaccard/F1): overlap of DE gene sets
"""

from typing import List, Dict, Union, Set
import numpy as np


def parse_condition_genes(condition: str) -> Set[str]:
    """
    Extract gene names from a perturbation condition string.

    Args:
        condition: Condition string like 'GENE1+GENE2' or 'GENE+ctrl'

    Returns:
        Set of gene names (excluding 'ctrl')

    Examples:
        >>> parse_condition_genes('CNN1+MAPK1')
        {'CNN1', 'MAPK1'}
        >>> parse_condition_genes('FOSB+ctrl')
        {'FOSB'}
        >>> parse_condition_genes('ctrl')
        set()
    """
    if not condition or condition == "ctrl":
        return set()

    genes = condition.split("+")
    return {g.strip() for g in genes if g.strip() and g.strip() != "ctrl"}


def top_k_accuracy(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int,
) -> float:
    """
    Compute Top-K accuracy (exact match).

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


def one_gene_overlap_hit_at_k(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int,
) -> float:
    """
    Compute one-gene-overlap Hit@K (scGPT "relevant retrieval").

    A hit is scored if any of the top-K predictions shares at least
    one gene with the ground truth condition.

    Args:
        predictions: List of top-K predictions per query
        ground_truth: List of true labels
        k: K value

    Returns:
        Fraction of queries with relevant retrieval in top-K
    """
    hits = 0
    for preds, true in zip(predictions, ground_truth):
        true_genes = parse_condition_genes(true)
        if not true_genes:
            continue

        # Check if any prediction shares a gene
        for pred in preds[:k]:
            pred_genes = parse_condition_genes(pred)
            if pred_genes & true_genes:  # Non-empty intersection
                hits += 1
                break

    valid_queries = sum(1 for t in ground_truth if parse_condition_genes(t))
    return hits / valid_queries if valid_queries > 0 else 0.0


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


def de_gene_overlap(
    predicted_de_genes: List[Set[str]],
    observed_de_genes: List[Set[str]],
    metric: str = "jaccard",
) -> float:
    """
    Compute overlap between predicted and observed DE gene sets.

    Args:
        predicted_de_genes: List of predicted DE gene sets per query
        observed_de_genes: List of observed DE gene sets per query
        metric: 'jaccard' or 'f1'

    Returns:
        Mean overlap score across queries
    """
    scores = []
    for pred, obs in zip(predicted_de_genes, observed_de_genes):
        if not pred and not obs:
            scores.append(1.0)  # Both empty = perfect match
            continue
        if not pred or not obs:
            scores.append(0.0)
            continue

        intersection = len(pred & obs)
        union = len(pred | obs)

        if metric == "jaccard":
            score = intersection / union if union > 0 else 0.0
        elif metric == "f1":
            precision = intersection / len(pred) if pred else 0.0
            recall = intersection / len(obs) if obs else 0.0
            score = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'jaccard' or 'f1'.")

        scores.append(score)

    return np.mean(scores) if scores else 0.0


def compute_all_metrics(
    predictions: List[List[str]],
    ground_truth: List[str],
    top_k_values: List[int] = [1, 5, 8, 10],
    candidate_pool: List[str] = None,
) -> Dict[str, float]:
    """
    Compute all retrieval metrics.

    Args:
        predictions: List of ranked predictions per query
        ground_truth: List of true labels
        top_k_values: K values for Top-K metrics
        candidate_pool: If provided, exact metrics only computed for queries
                        whose ground truth is in this pool

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}

    # Determine which queries have ground truth in candidate pool
    if candidate_pool is not None:
        pool_set = set(candidate_pool)
        in_pool_mask = [true in pool_set for true in ground_truth]
        in_pool_preds = [p for p, m in zip(predictions, in_pool_mask) if m]
        in_pool_truth = [t for t, m in zip(ground_truth, in_pool_mask) if m]
    else:
        in_pool_preds = predictions
        in_pool_truth = ground_truth

    # Exact match metrics (only for queries where ground truth is in candidate pool)
    for k in top_k_values:
        metrics[f"exact_hit@{k}"] = top_k_accuracy(in_pool_preds, in_pool_truth, k)

    # MRR (exact match, in-pool only)
    metrics["mrr"] = mrr(in_pool_preds, in_pool_truth)

    # Relevant retrieval metrics (one-gene overlap) - computed on all queries
    for k in top_k_values:
        metrics[f"relevant_hit@{k}"] = one_gene_overlap_hit_at_k(
            predictions, ground_truth, k
        )

    # NDCG (in-pool only)
    for k in top_k_values:
        metrics[f"ndcg@{k}"] = ndcg(in_pool_preds, in_pool_truth, k)

    # Report counts
    metrics["n_queries"] = len(ground_truth)
    metrics["n_in_pool"] = len(in_pool_truth)

    return metrics
