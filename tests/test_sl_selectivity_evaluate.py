"""Selectivity-mode integration tests for the SL benchmark evaluate module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.evaluate import GeneUniverse, _selectivity_raw
from sl_benchmark_baseline.selectivity import UniverseSelectivity


def _toy_universe_with_selectivity() -> GeneUniverse:
    symbols = np.array(["GA", "GB"])
    sel = UniverseSelectivity(
        sel_matrix=np.array([[0.0, -1.7], [0.0, 0.0]]),
        pan_essential=np.array([-0.5, -0.93]),
        coverage_flag=np.array([1, 1]),
        essential_fraction=np.array([0.2, 0.8]),
    )
    return GeneUniverse(
        keys=(10, 20),
        symbols=symbols,
        gene_effects=np.array([-0.5, -1.0]),
        index_by_key={10: 0, 20: 1},
        entrez=np.array([10, 20]),
        selectivity=sel,
    )


def test_selectivity_raw_block_width_and_values():
    universe = _toy_universe_with_selectivity()
    frame = pd.DataFrame(
        {
            "gene_a_unified_id": [10],
            "gene_b_unified_id": [20],
            "gene_a_symbol": ["GA"],
            "gene_b_symbol": ["GB"],
            "gene_a_k562_gene_effect": [-0.5],
            "gene_b_k562_gene_effect": [-1.0],
            "sl_label": [1],
        }
    )
    cfg = SLBaselineConfig()
    raw = _selectivity_raw(frame, universe, cfg)
    assert raw.shape == (1, 8)
    # selectivity block (cols 5,6,7): sel_mean=(-1.7+0.0)/2=-0.85,
    # absdiff=1.7, pan_min=min(-0.5,-0.93)=-0.93
    np.testing.assert_allclose(raw[0, 5:], [-0.85, 1.7, -0.93], atol=1e-6)
