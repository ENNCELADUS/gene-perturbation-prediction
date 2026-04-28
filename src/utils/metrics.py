"""Metrics and label helpers for inverse perturbation retrieval."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def parse_condition_genes(condition: str) -> set[str]:
    """Extract non-control gene names from a perturbation condition."""
    if not condition or condition == "ctrl":
        return set()
    return {
        gene.strip()
        for gene in condition.split("+")
        if gene.strip() and gene.strip() != "ctrl"
    }


def normalize_condition(condition: str) -> str:
    """Normalize equivalent condition labels into sorted gene tokens."""
    genes = sorted(parse_condition_genes(condition))
    return "+".join(genes) if genes else "ctrl"


def compute_gene_metrics(
    scores: Sequence[np.ndarray],
    targets: Sequence[Sequence[int]],
    top_k_values: Sequence[int],
) -> dict[str, float | int]:
    """Compute gene-ranking metrics for multi-label target genes."""
    metrics: dict[str, float | int] = {
        f"relevant_hit@{k}": 0.0 for k in top_k_values
    }
    metrics.update({f"exact_hit@{k}": 0.0 for k in top_k_values})
    metrics.update({f"recall@{k}": 0.0 for k in top_k_values})
    metrics.update({f"ndcg@{k}": 0.0 for k in top_k_values})
    reciprocal_rank_sum = 0.0
    n_queries = 0

    for score_vec, target_indices in zip(scores, targets):
        target_set = {int(idx) for idx in target_indices}
        if not target_set:
            continue
        n_queries += 1
        ranking = np.argsort(-score_vec)
        positions = np.empty_like(ranking)
        positions[ranking] = np.arange(1, len(ranking) + 1)
        reciprocal_rank_sum += 1.0 / int(positions[list(target_set)].min())

        for k in top_k_values:
            topk = ranking[:k]
            topk_set = set(topk.tolist())
            overlap = len(topk_set & target_set)
            metrics[f"relevant_hit@{k}"] += float(overlap > 0)
            metrics[f"exact_hit@{k}"] += float(target_set.issubset(topk_set))
            metrics[f"recall@{k}"] += overlap / len(target_set)
            metrics[f"ndcg@{k}"] += _ndcg_for_topk(topk, target_set)

    if n_queries == 0:
        for key in list(metrics):
            metrics[key] = 0.0
        metrics["mrr"] = 0.0
        metrics["n_queries"] = 0
        return metrics

    for key in list(metrics):
        metrics[key] = float(metrics[key]) / n_queries
    metrics["mrr"] = reciprocal_rank_sum / n_queries
    metrics["n_queries"] = n_queries
    return metrics


def compute_combo_metrics(
    predictions: Sequence[Sequence[str]],
    ground_truth: Sequence[str],
    top_k_values: Sequence[int],
) -> dict[str, float | int]:
    """Compute exact and one-gene-overlap metrics for condition rankings."""
    metrics: dict[str, float | int] = {
        f"exact_hit@{k}": 0.0 for k in top_k_values
    }
    metrics.update({f"relevant_hit@{k}": 0.0 for k in top_k_values})
    reciprocal_rank_sum = 0.0
    n_queries = 0

    for ranked_conditions, truth in zip(predictions, ground_truth):
        truth_norm = normalize_condition(truth)
        truth_genes = parse_condition_genes(truth_norm)
        if not truth_genes:
            continue
        n_queries += 1
        normalized_preds = [normalize_condition(pred) for pred in ranked_conditions]
        if truth_norm in normalized_preds:
            reciprocal_rank_sum += 1.0 / (normalized_preds.index(truth_norm) + 1)

        for k in top_k_values:
            topk = normalized_preds[:k]
            metrics[f"exact_hit@{k}"] += float(truth_norm in topk)
            metrics[f"relevant_hit@{k}"] += float(
                any(parse_condition_genes(pred) & truth_genes for pred in topk)
            )

    if n_queries == 0:
        for key in list(metrics):
            metrics[key] = 0.0
        metrics["mrr"] = 0.0
        metrics["n_queries"] = 0
        return metrics

    for key in list(metrics):
        metrics[key] = float(metrics[key]) / n_queries
    metrics["mrr"] = reciprocal_rank_sum / n_queries
    metrics["n_queries"] = n_queries
    return metrics


def build_label_matrix(
    conditions: Sequence[str],
    gene_name_to_idx: dict[str, int],
    n_genes: int,
) -> np.ndarray:
    """Build a multi-hot condition-by-gene target matrix."""
    labels = np.zeros((len(conditions), n_genes), dtype=np.float32)
    for row_idx, condition in enumerate(conditions):
        for gene in parse_condition_genes(condition):
            gene_idx = gene_name_to_idx.get(gene)
            if gene_idx is not None:
                labels[row_idx, gene_idx] = 1.0
    return labels


def target_indices_for_conditions(
    conditions: Iterable[str],
    gene_name_to_idx: dict[str, int],
) -> list[list[int]]:
    """Map condition labels to target gene indices."""
    targets = []
    for condition in conditions:
        indices = [
            gene_name_to_idx[gene]
            for gene in parse_condition_genes(condition)
            if gene in gene_name_to_idx
        ]
        targets.append(indices)
    return targets


def _ndcg_for_topk(topk: np.ndarray, target_set: set[int]) -> float:
    dcg = 0.0
    for rank, gene_idx in enumerate(topk, start=1):
        if int(gene_idx) in target_set:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(target_set), len(topk))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg
