from __future__ import annotations

from pathlib import Path


def test_config_defaults_and_override() -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig

    config = SLBaselineConfig()
    assert config.input_csv == Path("data/k562_SL_benchmark_minimal.csv")
    assert config.split_types is None
    assert config.folds == (0, 1, 2, 3, 4)
    assert config.ranking_k == (10, 20, 50)
    assert config.seed == 17

    overridden = SLBaselineConfig(seed=99, split_types=("CV2",), folds=(0, 1))
    assert overridden.seed == 99
    assert overridden.split_types == ("CV2",)
    assert overridden.folds == (0, 1)
    try:
        overridden.seed = 5  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("SLBaselineConfig must be frozen")
