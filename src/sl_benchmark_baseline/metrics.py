"""Official classification and ranking metrics for SL-pair scoring."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    ndcg_score,
    precision_recall_curve,
    roc_auc_score,
)


def official_classification_metrics(
    score_matrix: np.ndarray,
    pos_index: np.ndarray,
    neg_index: np.ndarray,
) -> dict[str, float]:
    """Compute official SL_benchmark classification metrics from a score matrix."""
    score_matrix = np.asarray(score_matrix, dtype=float)
    pos_index = np.asarray(pos_index, dtype=int)
    neg_index = np.asarray(neg_index, dtype=int)
    pos_scores = score_matrix[pos_index[:, 0], pos_index[:, 1]]
    neg_scores = score_matrix[neg_index[:, 0], neg_index[:, 1]]
    y_score = np.hstack([pos_scores, neg_scores])
    y_true = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1_values = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "aupr": float(auc(recall, precision)),
        "f1": float(f1_values.max()),
    }


def _average_precision_at_k(relevance: np.ndarray, k: int) -> float:
    relevant_docs = 0
    precision_sum = 0.0
    for rank, value in enumerate(relevance[:k], start=1):
        if int(value) == 1:
            relevant_docs += 1
            precision_sum += relevant_docs / rank
    return precision_sum / relevant_docs if relevant_docs > 0 else 0.0


def official_ranking_metrics(
    score_matrix: np.ndarray,
    pos_index: np.ndarray,
    seen_index: np.ndarray | None,
    ks: tuple[int, ...],
) -> dict[str, float]:
    """Compute official per-anchor SL_benchmark ranking metrics."""
    score_matrix = np.asarray(score_matrix, dtype=float)
    pos_index = np.asarray(pos_index, dtype=int)
    if seen_index is not None:
        seen_index = np.asarray(seen_index, dtype=int)

    ranking_scores = score_matrix.copy()
    n_gene = ranking_scores.shape[0]
    ranking_scores[np.arange(n_gene), np.arange(n_gene)] = 0.0
    if seen_index is not None and len(seen_index) > 0:
        ranking_scores[seen_index[:, 0], seen_index[:, 1]] = -999999.0
        ranking_scores[seen_index[:, 1], seen_index[:, 0]] = -999999.0

    positives_by_anchor: dict[int, set[int]] = {}
    for gene_a, gene_b in pos_index:
        positives_by_anchor.setdefault(int(gene_a), set()).add(int(gene_b))
        positives_by_anchor.setdefault(int(gene_b), set()).add(int(gene_a))

    max_k = max(ks)
    relevance_rows = []
    score_rows = []
    positive_counts = []
    for anchor in sorted(positives_by_anchor):
        positives = positives_by_anchor[anchor]
        if not positives:
            continue
        order = np.argsort(ranking_scores[anchor, :])[::-1][:max_k]
        relevance = np.array(
            [1 if int(candidate) in positives else 0 for candidate in order]
        )
        relevance_rows.append(relevance)
        score_rows.append(ranking_scores[anchor, order])
        positive_counts.append(len(positives))

    if not relevance_rows:
        return {
            metric: 0.0
            for k in ks
            for metric in (f"ndcg@{k}", f"recall@{k}", f"precision@{k}", f"map@{k}")
        }

    relevance_array = np.asarray(relevance_rows)
    score_array = np.asarray(score_rows)
    positive_count_array = np.asarray(positive_counts)
    out: dict[str, float] = {}
    for k in ks:
        hits = relevance_array[:, :k].sum(axis=1)
        out[f"ndcg@{k}"] = float(ndcg_score(relevance_array, score_array, k=k))
        out[f"recall@{k}"] = float((hits / positive_count_array).mean())
        out[f"precision@{k}"] = float(
            (hits / np.minimum(positive_count_array, k)).mean()
        )
        out[f"map@{k}"] = float(
            np.mean([_average_precision_at_k(row, k) for row in relevance_array])
        )
    return out
