# tests/test_ddgcn_scoring.py
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from ddgcn.config import DdgcnConfig


def _fast_config() -> DdgcnConfig:
    return dataclasses.replace(
        DdgcnConfig(),
        hidden1=8,
        hidden2=4,
        max_epochs=15,
        tolerance_epoch=2,
        eval_interval=5,
    )


def _toy_frame() -> pd.DataFrame:
    # 6 genes, one CV1 fold, train+test pos/neg rows.
    rows = [
        # split_type, fold_id, role, label, a_id, b_id, a_sym, b_sym, a_eff, b_eff
        ("CV1", 0, "train", 1, 0, 1, "G0", "G1", -0.5, -0.4),
        ("CV1", 0, "train", 1, 2, 3, "G2", "G3", -0.6, -0.3),
        ("CV1", 0, "train", 0, 0, 4, "G0", "G4", -0.5, 0.1),
        ("CV1", 0, "train", 0, 1, 5, "G1", "G5", -0.4, 0.2),
        ("CV1", 0, "test", 1, 0, 2, "G0", "G2", -0.5, -0.6),
        ("CV1", 0, "test", 0, 3, 5, "G3", "G5", -0.3, 0.2),
    ]
    cols = [
        "split_type",
        "fold_id",
        "split_role",
        "sl_label",
        "gene_a_unified_id",
        "gene_b_unified_id",
        "gene_a_symbol",
        "gene_b_symbol",
        "gene_a_k562_gene_effect",
        "gene_b_k562_gene_effect",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_producer_score_matrix_shape_and_diag() -> None:
    from ddgcn.scoring import DdgcnProducer

    prod = DdgcnProducer(_fast_config())
    pos_index = np.array([[0, 1], [2, 3]])
    neg_index = np.array([[0, 4], [1, 5]])
    sm = prod.score_matrix_for_fold(pos_index, neg_index, n_gene=6)
    assert sm.shape == (6, 6)
    assert np.allclose(np.diag(sm), 0.0)


def test_run_fold_ddgcn_emits_official_metric_rows() -> None:
    from sl_benchmark_baseline.evaluate import _build_gene_universe

    from ddgcn.scoring import run_fold_ddgcn

    frame = _toy_frame()
    universe = _build_gene_universe(frame)
    rows = run_fold_ddgcn(frame, "CV1", 0, _fast_config(), universe)
    assert len(rows) > 0
    assert {r["model"] for r in rows} == {"ddgcn"}
    assert {r["split_type"] for r in rows} == {"CV1"}
    assert {r["slice"] for r in rows} == {"full_universe"}
    metrics = {r["metric"] for r in rows}
    assert "auroc" in metrics
    assert "ndcg@10" in metrics
    for r in rows:
        assert np.isfinite(r["value"])
