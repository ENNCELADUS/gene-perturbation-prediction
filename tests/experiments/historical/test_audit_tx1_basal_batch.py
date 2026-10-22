from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.historical.audit_tx1_basal_batch import (
    audit_metrics,
    tahoe_pseudobulk,
)


def test_tahoe_pseudobulk_ignores_special_and_nonpositive_values() -> None:
    metadata = pd.DataFrame(
        {
            "gene_symbol": ["A", "B"],
            "ensembl_id": ["ENSG1", "ENSG2"],
            "token_id": [3, 4],
        }
    )
    reservoirs = {
        "CVCL_X": [
            (np.array([1, 3, 4]), np.array([-2.0, 2.0, 1.0])),
            (np.array([3, 4]), np.array([2.0, 0.0])),
        ]
    }
    result = tahoe_pseudobulk(reservoirs, metadata)
    assert list(result.columns) == ["A", "B"]
    assert result.loc["CVCL_X", "A"] > result.loc["CVCL_X", "B"]


def test_audit_metrics_detects_matched_cross_source_lines() -> None:
    rng = np.random.default_rng(7)
    ids = [f"ACH-{index:06d}" for index in range(10)]
    genes = [f"G{index}" for index in range(30)]
    base = rng.normal(size=(10, 30))
    tahoe = pd.DataFrame(
        base + rng.normal(scale=0.01, size=base.shape), index=ids, columns=genes
    )
    ccle = pd.DataFrame(
        base + rng.normal(scale=0.01, size=base.shape), index=ids, columns=genes
    )
    lineages = pd.Series({model_id: "Lung" for model_id in ids})
    result = audit_metrics(tahoe, ccle, lineages, ids[:8], ids)
    assert result["matched_line_top1_accuracy_all"] == 1.0
    assert result["matched_line_top1_accuracy_heldout"] == 1.0
