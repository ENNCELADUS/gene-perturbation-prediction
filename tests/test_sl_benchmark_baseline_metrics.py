from __future__ import annotations

import numpy as np


def test_classification_metrics_perfect_and_keys() -> None:
    from sl_benchmark_baseline.metrics import classification_metrics

    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    out = classification_metrics(y_true, scores)
    assert set(out) == {"auroc", "aupr", "f1@0.5"}
    assert out["auroc"] == 1.0
    assert out["aupr"] == 1.0
    assert out["f1@0.5"] == 1.0


def test_ranking_metrics_keys_and_topk() -> None:
    from sl_benchmark_baseline.metrics import ranking_metrics

    y_true = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    pair_ids = ["P0", "P1", "P2", "P3", "P4"]
    out = ranking_metrics(y_true, scores, pair_ids, ks=(2, 5))
    assert "ndcg@2" in out and "recall@2" in out and "precision@2" in out
    assert out["precision@2"] == 1.0
    assert out["recall@2"] == 1.0
    assert out["ndcg@2"] == 1.0
    assert out["recall@5"] == 1.0
    assert out["precision@5"] == 2 / 5


def test_ranking_metrics_breaks_ties_by_pair_id() -> None:
    from sl_benchmark_baseline.metrics import ranking_metrics

    y_true = np.array([1, 0, 0])
    scores = np.array([0.5, 0.5, 0.5])
    pair_ids = ["A", "B", "C"]
    out = ranking_metrics(y_true, scores, pair_ids, ks=(1,))
    assert out["precision@1"] == 1.0
