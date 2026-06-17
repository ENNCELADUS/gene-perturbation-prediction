# tests/sl_dl_model/test_scoring.py
"""Tests for sl_dl_model.scoring — per-fold producer scoring."""

from __future__ import annotations

import pandas as pd

from sl_dl_model.config import SLDLConfig
from sl_dl_model.evaluate import ZeroEmbeddingProducer
from sl_dl_model.scoring import run_fold_with_producer, train_symbols_for_fold


def _toy_frame() -> pd.DataFrame:
    rows = []
    genes = [f"G{i}" for i in range(6)]
    eff = {g: float(i) - 3 for i, g in enumerate(genes)}
    pid = 0
    for role in ("train", "test"):
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                rows.append(
                    {
                        "pair_id": f"p{pid}",
                        "fold_id": 0,
                        "split_type": "CV2",
                        "split_role": role,
                        "sl_label": (i + j) % 2,
                        "gene_a_symbol": genes[i],
                        "gene_b_symbol": genes[j],
                        "gene_a_k562_gene_effect": eff[genes[i]],
                        "gene_b_k562_gene_effect": eff[genes[j]],
                    }
                )
                pid += 1
    return pd.DataFrame(rows)


def test_train_symbols_excludes_test_only_genes() -> None:
    df = pd.DataFrame(
        {
            "split_role": ["train", "train", "test"],
            "gene_a_symbol": ["A", "B", "E"],
            "gene_b_symbol": ["B", "C", "F"],
        }
    )
    train_only = df[df["split_role"] == "train"]
    syms = train_symbols_for_fold(train_only)
    assert syms == {"A", "B", "C"}
    assert "E" not in syms and "F" not in syms


def test_run_fold_emits_metric_rows() -> None:
    cfg = SLDLConfig(include_coverage_flag=False, ranking_k=(10,))
    rows = run_fold_with_producer(_toy_frame(), "CV2", 0, cfg, ZeroEmbeddingProducer())
    assert rows
    metrics = {r["metric"] for r in rows}
    assert "auroc" in metrics and "ndcg@10" in metrics
    assert all(r["split_type"] == "CV2" for r in rows)
