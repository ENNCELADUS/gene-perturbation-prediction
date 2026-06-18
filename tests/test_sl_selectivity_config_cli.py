"""Tests for exp09 selectivity config fields, Rand-only guard, and CLI flags."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sl_benchmark_baseline.__main__ import _parse_args
from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.data import assert_rand_only


def test_config_selectivity_property():
    assert SLBaselineConfig().selectivity is False
    cfg = SLBaselineConfig(depmap_dir=Path("data/sl_dependency_v0/raw/depmap"))
    assert cfg.selectivity is True
    assert cfg.cn_loss_thr == 0.8
    assert cfg.sel_n_min == 20


def test_assert_rand_only():
    rand = pd.DataFrame({"negative_sampling_method": ["Rand", "Rand"]})
    assert_rand_only(rand)  # no raise
    no_col = pd.DataFrame({"x": [1]})
    assert_rand_only(no_col)  # no raise
    dep = pd.DataFrame({"negative_sampling_method": ["Rand", "Dep"]})
    with pytest.raises(ValueError, match="Rand"):
        assert_rand_only(dep)


def test_cli_parses_selectivity_flags():
    args = _parse_args(
        [
            "--depmap-dir",
            "data/sl_dependency_v0/raw/depmap",
            "--cn-loss-thr",
            "0.7",
            "--expr-low-quantile",
            "0.15",
            "--sel-n-min",
            "25",
            "--sel-lambda",
            "0.5",
        ]
    )
    assert str(args.depmap_dir) == "data/sl_dependency_v0/raw/depmap"
    assert args.cn_loss_thr == 0.7
    assert args.expr_low_quantile == 0.15
    assert args.sel_n_min == 25
    assert args.sel_lambda == 0.5
