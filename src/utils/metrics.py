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
    metrics = _compute_gene_metrics_core(scores, targets, top_k_values)
    metrics.update(_compute_stratified_gene_metrics(scores, targets, top_k_values))
    return metrics


def compute_cardinality_metrics(
    cardinality_logits: Sequence[np.ndarray],
    targets: Sequence[Sequence[int]],
) -> dict[str, float | int]:
    """Compute target-set cardinality metrics from cardinality logits."""
    if len(cardinality_logits) != len(targets):
        raise ValueError(
            "cardinality_logits and targets must have the same number of queries: "
            f"{len(cardinality_logits)} != {len(targets)}"
        )
    if len(cardinality_logits) == 0:
        return {"cardinality_accuracy": 0.0, "cardinality_mae": 0.0}
    predicted = [int(np.argmax(logits)) for logits in cardinality_logits]
    truth = [len(target) for target in targets]
    errors = [abs(pred - true) for pred, true in zip(predicted, truth)]
    metrics: dict[str, float | int] = {
        "cardinality_accuracy": mean(
            float(pred == true) for pred, true in zip(predicted, truth)
        ),
        "cardinality_mae": mean(errors),
    }
    for cardinality, count in sorted(Counter(predicted).items()):
        metrics[f"predicted_cardinality_{cardinality}"] = int(count)
    return metrics


def _compute_gene_metrics_core(
    scores: Sequence[np.ndarray],
    targets: Sequence[Sequence[int]],
    top_k_values: Sequence[int],
) -> dict[str, float | int]:
    """Compute unstratified gene-ranking metrics."""
    _validate_scores_and_targets(scores, targets)
    metrics: dict[str, float | int] = {f"relevant_hit@{k}": 0.0 for k in top_k_values}
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


def _compute_stratified_gene_metrics(
    scores: Sequence[np.ndarray],
    targets: Sequence[Sequence[int]],
    top_k_values: Sequence[int],
) -> dict[str, float | int]:
    stratified: dict[str, float | int] = {}
    groups = {
        "single_gene": [idx for idx, target in enumerate(targets) if len(target) == 1],
        "combo": [idx for idx, target in enumerate(targets) if len(target) > 1],
    }
    for prefix, indices in groups.items():
        if not indices:
            stratified[f"{prefix}_n_queries"] = 0
            continue
        group_scores = [scores[idx] for idx in indices]
        group_targets = [targets[idx] for idx in indices]
        group_metrics = _compute_gene_metrics_core(
            group_scores,
            group_targets,
            top_k_values,
        )
        for key, value in group_metrics.items():
            stratified[f"{prefix}_{key}"] = value
    return stratified


def compute_combo_metrics(
    predictions: Sequence[Sequence[str]],
    ground_truth: Sequence[str],
    top_k_values: Sequence[int],
) -> dict[str, float | int]:
    """Compute exact and one-gene-overlap metrics for condition rankings."""
    metrics: dict[str, float | int] = {f"exact_hit@{k}": 0.0 for k in top_k_values}
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
    query_ids: Sequence[int] | None = None,
    split_genes: Mapping[str, Sequence[str]] | None = None,
    nearest_neighbors: Sequence[Sequence[Mapping[str, object]]] | None = None,
) -> dict[str, object]:
    """Build per-query ranking diagnostics for gene retrieval outputs."""
    _validate_diagnostic_inputs(
        scores=scores,
        targets=targets,
        gene_names=gene_names,
        conditions=conditions,
        query_ids=query_ids,
        nearest_neighbors=nearest_neighbors,
    )
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
        if query_ids is not None:
            query_payload["cell_index"] = int(query_ids[query_idx])

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


def _validate_scores_and_targets(
    scores: Sequence[np.ndarray],
    targets: Sequence[Sequence[int]],
) -> None:
    if len(scores) != len(targets):
        raise ValueError(
            "scores and targets must have the same number of queries: "
            f"{len(scores)} != {len(targets)}"
        )
    for query_idx, (score_vec, target_indices) in enumerate(zip(scores, targets)):
        score_array = np.asarray(score_vec)
        if score_array.ndim != 1:
            raise ValueError(f"score vector at query {query_idx} must be 1D")
        for target_idx in target_indices:
            if int(target_idx) < 0 or int(target_idx) >= score_array.shape[0]:
                raise ValueError(
                    f"target index {target_idx} at query {query_idx} is outside "
                    f"score vector width {score_array.shape[0]}"
                )


def _validate_diagnostic_inputs(
    scores: Sequence[np.ndarray],
    targets: Sequence[Sequence[int]],
    gene_names: Sequence[str],
    conditions: Sequence[str],
    query_ids: Sequence[int] | None,
    nearest_neighbors: Sequence[Sequence[Mapping[str, object]]] | None,
) -> None:
    _validate_scores_and_targets(scores, targets)
    if len(scores) != len(conditions):
        raise ValueError(
            "scores and conditions must have the same number of queries: "
            f"{len(scores)} != {len(conditions)}"
        )
    if query_ids is not None and len(scores) != len(query_ids):
        raise ValueError(
            "scores and query_ids must have the same number of queries: "
            f"{len(scores)} != {len(query_ids)}"
        )
    if nearest_neighbors is not None and len(scores) != len(nearest_neighbors):
        raise ValueError(
            "scores and nearest_neighbors must have the same number of queries: "
            f"{len(scores)} != {len(nearest_neighbors)}"
        )
    for query_idx, score_vec in enumerate(scores):
        if np.asarray(score_vec).shape[0] != len(gene_names):
            raise ValueError(
                "score vector width must match gene_names length at query "
                f"{query_idx}: {np.asarray(score_vec).shape[0]} != {len(gene_names)}"
            )


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
