# tests/sl_dl_model/test_evaluate_parity.py
"""Tests for sl_dl_model.evaluate — embedding-producer protocol and CV runner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sl_dl_model.config import SLDLConfig
from sl_dl_model.evaluate import ZeroEmbeddingProducer, run_cv


def _toy_frame() -> pd.DataFrame:
    rows = []
    genes = [f"G{i}" for i in range(8)]
    rng = np.random.default_rng(0)
    eff = {g: float(rng.normal()) for g in genes}
    pid = 0
    for split in ("CV2",):
        for fold in (0, 1):
            for role in ("train", "test"):
                for i in range(len(genes)):
                    for j in range(i + 1, len(genes)):
                        a, b = genes[i], genes[j]
                        rows.append(
                            {
                                "pair_id": f"p{pid}",
                                "fold_id": fold,
                                "split_type": split,
                                "split_role": role,
                                "sl_label": (i + j) % 2,
                                "gene_a_symbol": a,
                                "gene_b_symbol": b,
                                "gene_a_k562_gene_effect": eff[a],
                                "gene_b_k562_gene_effect": eff[b],
                            }
                        )
                        pid += 1
    return pd.DataFrame(rows)


def test_zero_producer_runs_and_emits_full_universe_metrics(tmp_path) -> None:
    csv = tmp_path / "toy.csv"
    _toy_frame().to_csv(csv, index=False)
    cfg = SLDLConfig(
        input_csv=csv,
        output_dir=tmp_path / "run",
        split_types=("CV2",),
        folds=(0, 1),
        include_coverage_flag=False,
    )
    summary = run_cv(cfg, ZeroEmbeddingProducer())
    assert (tmp_path / "run" / "fold_metrics.csv").exists()
    assert (tmp_path / "run" / "manifest.json").exists()
    metrics = set(summary["metric"])
    assert "auroc" in metrics and "ndcg@10" in metrics
    assert set(summary["slice"]) >= {"full_universe"}
