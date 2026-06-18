"""Tests for exp09 selectivity models and factory."""

from __future__ import annotations

from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.models import (
    LogRegSelectivityModel,
    XGBSelectivityModel,
    build_selectivity_models,
)


def test_selectivity_models_names_and_factory():
    cfg = SLBaselineConfig()
    assert LogRegSelectivityModel(cfg).name == "A_xcl"
    assert XGBSelectivityModel(cfg).name == "B_xcl"
    names = [m.name for m in build_selectivity_models(cfg)]
    assert names == ["A", "B", "C", "A_xcl", "B_xcl"]
