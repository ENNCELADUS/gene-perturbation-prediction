from __future__ import annotations

import numpy as np

from src.utils.metrics import (
    compute_combo_metrics,
    compute_gene_metrics,
    parse_condition_genes,
)


def test_parse_condition_genes_normalizes_control_and_order() -> None:
    assert parse_condition_genes("ctrl") == set()
    assert parse_condition_genes("MAPK1+ctrl") == {"MAPK1"}
    assert parse_condition_genes("CNN1+MAPK1") == {"CNN1", "MAPK1"}


def test_compute_gene_metrics_reports_relevant_exact_recall_ndcg() -> None:
    scores = [
        np.array([0.9, 0.2, 0.8, 0.1], dtype=np.float32),
        np.array([0.1, 0.8, 0.7, 0.2], dtype=np.float32),
    ]
    targets = [[0, 2], [1]]

    metrics = compute_gene_metrics(scores, targets, top_k_values=[1, 2])

    assert metrics["relevant_hit@1"] == 1.0
    assert metrics["exact_hit@1"] == 0.5
    assert metrics["exact_hit@2"] == 1.0
    assert metrics["recall@1"] == 0.75
    assert metrics["recall@2"] == 1.0
    assert metrics["ndcg@2"] == 1.0
    assert metrics["n_queries"] == 2


def test_compute_combo_metrics_uses_exact_and_gene_overlap_semantics() -> None:
    predictions = [
        ["CNN1+MAPK1", "FOSB+UBASH3B"],
        ["CNN1+FOSB", "MAPK1+UBASH3B"],
    ]
    truth = ["CNN1+MAPK1", "FOSB+UBASH3B"]

    metrics = compute_combo_metrics(predictions, truth, top_k_values=[1, 2])

    assert metrics["exact_hit@1"] == 0.5
    assert metrics["exact_hit@2"] == 0.5
    assert metrics["relevant_hit@1"] == 1.0
    assert metrics["relevant_hit@2"] == 1.0
    assert metrics["mrr"] == 0.5
