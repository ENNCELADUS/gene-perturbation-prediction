from __future__ import annotations

from pathlib import Path

from sl_benchmark_baseline.config import SLBaselineConfig


def test_config_defaults_and_override() -> None:
    config = SLBaselineConfig()
    assert config.input_csv == Path(
        "data/SL_benchmark/derived/k562_depmap_rand_1to1/"
        "all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv"
    )
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


def test_config_augmentation_defaults_preserve_exp06():
    config = SLBaselineConfig()
    assert config.bags_npz is None
    assert config.augmented is False
    assert config.embedding_method == "pca_delta_meanpool"
    assert config.fallback_strategy == "zero"
    assert config.include_coverage_flag is True


def test_config_augmented_property_true_when_bags_set(tmp_path):
    config = SLBaselineConfig(bags_npz=tmp_path / "bags.npz")
    assert config.augmented is True
