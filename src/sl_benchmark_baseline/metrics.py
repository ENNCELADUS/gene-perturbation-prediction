"""Classification and pair-level ranking metrics for SL-pair scoring."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Compute AUROC, AUPR, and F1 at threshold 0.5.

    Args:
        y_true: Binary labels, shape ``(n,)``.
        scores: Predicted probabilities in ``[0, 1]``, shape ``(n,)``.

    Returns:
        Mapping with keys ``auroc``, ``aupr``, ``f1@0.5``.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "aupr": float(average_precision_score(y_true, scores)),
        "f1@0.5": float(f1_score(y_true, preds, zero_division=0)),
    }


def _ranked_relevance(
    y_true: np.ndarray, scores: np.ndarray, pair_ids: list[str]
) -> list[int]:
    """Order items by descending score, breaking ties by ascending pair_id."""
    order = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), str(pair_ids[i])),
    )
    return [int(y_true[i]) for i in order]


def ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    pair_ids: list[str],
    ks: tuple[int, ...],
) -> dict[str, float]:
    """Compute pair-level NDCG/Recall/Precision@k over the flat test list.

    Items are ranked by descending score with ties broken deterministically by
    ascending ``pair_id``. Positives are the relevant items.

    Args:
        y_true: Binary labels, shape ``(n,)``.
        scores: Predicted scores (any monotonic scale), shape ``(n,)``.
        pair_ids: Stable identifiers used for tie-breaking, length ``n``.
        ks: Rank cutoffs.

    Returns:
        Mapping with ``ndcg@k``, ``recall@k``, ``precision@k`` for each ``k``.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    sorted_rel = _ranked_relevance(y_true, scores, pair_ids)
    total_pos = int(sum(sorted_rel))
    ideal_rel = sorted(sorted_rel, reverse=True)
    out: dict[str, float] = {}
    for k in ks:
        topk = sorted_rel[:k]
        hits = sum(topk)
        out[f"precision@{k}"] = hits / k if k > 0 else 0.0
        out[f"recall@{k}"] = hits / total_pos if total_pos > 0 else 0.0
        dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(topk))
        idcg = sum(
            rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_rel[:k])
        )
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    return out
