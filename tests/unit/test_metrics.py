from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import (
    build_gene_ranking_diagnostics,
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


def test_compute_gene_metrics_rejects_misaligned_query_counts() -> None:
    scores = [np.array([0.9, 0.1], dtype=np.float32)]
    targets = [[0], [1]]

    with pytest.raises(ValueError, match="scores and targets"):
        compute_gene_metrics(scores, targets, top_k_values=[1])


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


def test_build_gene_ranking_diagnostics_reports_target_ranks_and_hits() -> None:
    scores = [
        np.array([0.9, 0.2, 0.8, 0.1], dtype=np.float32),
        np.array([0.1, 0.8, 0.7, 0.2], dtype=np.float32),
    ]
    diagnostics = build_gene_ranking_diagnostics(
        scores=scores,
        targets=[[0, 2], [1]],
        gene_names=["A", "B", "C", "D"],
        conditions=["A+C", "B"],
        top_k_values=[1, 2],
        top_n_predictions=2,
        query_ids=[10, 11],
        split_genes={
            "train": ["A"],
            "validation": ["B"],
            "test": ["C"],
        },
    )

    assert diagnostics["summary"]["n_queries"] == 2
    assert diagnostics["summary"]["n_candidate_genes"] == 4
    assert diagnostics["summary"]["target_rank_bins"]["<=1"] == 2
    assert diagnostics["summary"]["query_type_counts"] == {
        "combo": 1,
        "single_gene": 1,
    }
    assert diagnostics["summary"]["target_split_group_metrics"]["test+train"][
        "hit@2"
    ] == 1.0

    first_query = diagnostics["per_query"][0]
    assert first_query["condition"] == "A+C"
    assert first_query["cell_index"] == 10
    assert first_query["target_ranks"] == {"A": 1, "C": 2}
    assert first_query["target_scores"] == {"A": 0.9, "C": 0.8}
    assert first_query["best_target_gene"] == "A"
    assert first_query["best_target_rank"] == 1
    assert first_query["hit_at"] == {"1": True, "2": True}
    assert first_query["recall_at"] == {"1": 0.5, "2": 1.0}
    assert first_query["target_split_membership"] == {
        "A": "train",
        "C": "test",
    }
    assert first_query["target_split_group"] == "test+train"
    assert first_query["top_predictions"] == [
        {"gene": "A", "rank": 1, "score": 0.9, "is_target": True},
        {"gene": "C", "rank": 2, "score": 0.8, "is_target": True},
    ]


def test_build_gene_ranking_diagnostics_records_low_rank_targets() -> None:
    diagnostics = build_gene_ranking_diagnostics(
        scores=[np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)],
        targets=[[3]],
        gene_names=["A", "B", "C", "D"],
        conditions=["D"],
        top_k_values=[1, 2],
        top_n_predictions=2,
    )

    query = diagnostics["per_query"][0]
    assert query["target_ranks"] == {"D": 4}
    assert query["best_target_rank"] == 4
    assert query["hit_at"] == {"1": False, "2": False}
    assert query["top_predictions"] == [
        {"gene": "A", "rank": 1, "score": 0.9, "is_target": False},
        {"gene": "B", "rank": 2, "score": 0.8, "is_target": False},
    ]


def test_build_gene_ranking_diagnostics_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="scores and conditions"):
        build_gene_ranking_diagnostics(
            scores=[np.array([0.9, 0.1], dtype=np.float32)],
            targets=[[0]],
            gene_names=["A", "B"],
            conditions=["A", "B"],
            top_k_values=[1],
        )

    with pytest.raises(ValueError, match="score vector width"):
        build_gene_ranking_diagnostics(
            scores=[np.array([0.9], dtype=np.float32)],
            targets=[[0]],
            gene_names=["A", "B"],
            conditions=["A"],
            top_k_values=[1],
        )

    with pytest.raises(ValueError, match="scores and query_ids"):
        build_gene_ranking_diagnostics(
            scores=[np.array([0.9, 0.1], dtype=np.float32)],
            targets=[[0]],
            gene_names=["A", "B"],
            conditions=["A"],
            top_k_values=[1],
            query_ids=[10, 11],
        )
