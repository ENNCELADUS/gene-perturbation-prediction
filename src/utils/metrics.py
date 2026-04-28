"""Metrics and label helpers for inverse perturbation retrieval."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

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


def build_gene_ranking_diagnostics(
    scores: Sequence[np.ndarray],
    targets: Sequence[Sequence[int]],
    gene_names: Sequence[str],
    conditions: Sequence[str],
    top_k_values: Sequence[int],
    top_n_predictions: int = 10,
    split_genes: Mapping[str, Sequence[str]] | None = None,
    nearest_neighbors: Sequence[Sequence[Mapping[str, object]]] | None = None,
) -> dict[str, object]:
    """Build per-query ranking diagnostics for gene retrieval outputs."""
    top_n = max(1, min(int(top_n_predictions), len(gene_names)))
    split_lookup = _build_split_lookup(split_genes)
    per_query: list[dict[str, object]] = []
    best_ranks: list[int] = []
    rank_bins = Counter[str]()
    query_type_counts = Counter[str]()
    split_group_counts = Counter[str]()
    split_group_hit_sums: dict[str, dict[int, float]] = defaultdict(
        lambda: {int(k): 0.0 for k in top_k_values}
    )

    for query_idx, (score_vec, target_indices) in enumerate(zip(scores, targets)):
        target_set = {int(idx) for idx in target_indices}
        if not target_set:
            continue

        ranking = np.argsort(-score_vec)
        positions = np.empty_like(ranking)
        positions[ranking] = np.arange(1, len(ranking) + 1)
        target_genes = [gene_names[idx] for idx in sorted(target_set)]
        target_ranks = {
            gene_names[idx]: int(positions[idx]) for idx in sorted(target_set)
        }
        target_scores = {
            gene_names[idx]: _score_value(score_vec[idx]) for idx in sorted(target_set)
        }
        best_target_idx = min(target_set, key=lambda idx: int(positions[idx]))
        best_rank = int(positions[best_target_idx])
        hit_at: dict[str, bool] = {}
        recall_at: dict[str, float] = {}
        for k_value in top_k_values:
            k = int(k_value)
            topk_set = set(ranking[:k].tolist())
            overlap = len(topk_set & target_set)
            hit_at[str(k)] = overlap > 0
            recall_at[str(k)] = overlap / len(target_set)

        query_type = "single_gene" if len(target_set) == 1 else "combo"
        query_type_counts[query_type] += 1
        rank_bins[_rank_bin(best_rank)] += 1
        best_ranks.append(best_rank)

        query_payload: dict[str, object] = {
            "condition": conditions[query_idx],
            "target_genes": target_genes,
            "target_ranks": target_ranks,
            "target_scores": target_scores,
            "best_target_gene": gene_names[best_target_idx],
            "best_target_rank": best_rank,
            "hit_at": hit_at,
            "recall_at": recall_at,
            "top_predictions": [
                {
                    "gene": gene_names[int(gene_idx)],
                    "rank": rank,
                    "score": _score_value(score_vec[int(gene_idx)]),
                    "is_target": int(gene_idx) in target_set,
                }
                for rank, gene_idx in enumerate(ranking[:top_n], start=1)
            ],
        }

        if split_lookup:
            membership = {
                gene: split_lookup.get(gene, "unknown") for gene in target_genes
            }
            split_group = _split_group(membership.values())
            query_payload["target_split_membership"] = membership
            query_payload["target_split_group"] = split_group
            split_group_counts[split_group] += 1
            for k_value in top_k_values:
                split_group_hit_sums[split_group][int(k_value)] += float(
                    hit_at[str(int(k_value))]
                )

        if nearest_neighbors is not None:
            query_payload["nearest_neighbors"] = [
                dict(neighbor) for neighbor in nearest_neighbors[query_idx]
            ]

        per_query.append(query_payload)

    summary: dict[str, object] = {
        "n_queries": len(per_query),
        "n_candidate_genes": len(gene_names),
        "top_n_predictions": top_n,
        "best_target_rank": _rank_summary(best_ranks),
        "target_rank_bins": {
            "<=1": rank_bins["<=1"],
            "<=5": rank_bins["<=5"],
            "<=10": rank_bins["<=10"],
            "<=20": rank_bins["<=20"],
            "<=40": rank_bins["<=40"],
            ">40": rank_bins[">40"],
        },
        "query_type_counts": dict(query_type_counts),
    }
    if split_lookup:
        summary["target_split_group_metrics"] = {
            group: {
                "n_queries": count,
                **{
                    f"hit@{int(k_value)}": split_group_hit_sums[group][int(k_value)]
                    / count
                    for k_value in top_k_values
                },
            }
            for group, count in sorted(split_group_counts.items())
        }

    return {"summary": summary, "per_query": per_query}


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


def _build_split_lookup(
    split_genes: Mapping[str, Sequence[str]] | None,
) -> dict[str, str]:
    if not split_genes:
        return {}
    lookup = {}
    for split_name, genes in split_genes.items():
        for gene in genes:
            lookup[str(gene)] = str(split_name)
    return lookup


def _rank_summary(ranks: Sequence[int]) -> dict[str, float | int | None]:
    if not ranks:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(ranks),
        "median": median(ranks),
        "min": min(ranks),
        "max": max(ranks),
    }


def _rank_bin(rank: int) -> str:
    for upper_bound in (1, 5, 10, 20, 40):
        if rank <= upper_bound:
            return f"<={upper_bound}"
    return ">40"


def _score_value(value: float | np.floating) -> float:
    return round(float(value), 6)


def _split_group(split_names: Iterable[str]) -> str:
    unique_names = sorted(set(split_names))
    if len(unique_names) == 1:
        return f"{unique_names[0]}-only"
    return "+".join(unique_names)


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
