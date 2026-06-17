from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, precision_recall_curve


def test_official_classification_metrics_match_cal_metrics_definitions() -> None:
    from sl_benchmark_baseline.metrics import official_classification_metrics

    score_matrix = np.array(
        [
            [0.0, 0.9, 0.6, 0.2],
            [0.9, 0.0, 0.4, 0.8],
            [0.6, 0.4, 0.0, 0.3],
            [0.2, 0.8, 0.3, 0.0],
        ]
    )
    pos_index = np.array([[0, 1], [1, 3]])
    neg_index = np.array([[0, 2], [2, 3]])

    out = official_classification_metrics(score_matrix, pos_index, neg_index)
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.6, 0.3])
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    assert set(out) == {"auroc", "aupr", "f1"}
    assert out["auroc"] == 1.0
    assert out["aupr"] == auc(recall, precision)
    assert out["f1"] == f1_scores.max()


def test_official_ranking_metrics_rank_candidate_partners_per_anchor() -> None:
    from sl_benchmark_baseline.metrics import official_ranking_metrics

    score_matrix = np.array(
        [
            [0.0, 0.9, 0.8, 0.1],
            [0.9, 0.0, 0.2, 0.7],
            [0.8, 0.2, 0.0, 0.3],
            [0.1, 0.7, 0.3, 0.0],
        ]
    )
    pos_index = np.array([[0, 1], [0, 2], [1, 3]])

    out = official_ranking_metrics(score_matrix, pos_index, seen_index=None, ks=(2,))

    assert out == {
        "ndcg@2": 1.0,
        "recall@2": 1.0,
        "precision@2": 1.0,
        "map@2": 1.0,
    }


def test_official_ranking_metrics_mask_seen_train_pairs() -> None:
    from sl_benchmark_baseline.metrics import official_ranking_metrics

    score_matrix = np.array(
        [
            [0.0, 0.95, 0.8, 0.1],
            [0.95, 0.0, 0.1, 0.2],
            [0.8, 0.1, 0.0, 0.3],
            [0.1, 0.2, 0.3, 0.0],
        ]
    )
    pos_index = np.array([[0, 2]])
    seen_index = np.array([[0, 1]])

    out = official_ranking_metrics(score_matrix, pos_index, seen_index, ks=(2,))

    assert out["precision@2"] == 1.0
    assert out["recall@2"] == 1.0
    assert out["map@2"] == 1.0
