import numpy as np
import pandas as pd
import pytest

from src.experiments.sl import paris as paris_module
from src.experiments.sl.paris import COLUMNS, janitza_selected, paris

MODES = ("raw", "Ginicorrected")


def world(seed: int = 0, n: int = 150, p: int = 40):
    """Dependency of `dep` = +2 f0 - 1.5 f1 + noise; the other features are noise."""
    rng = np.random.default_rng(seed)
    lines = [f"ACH-{i:04d}" for i in range(n)]
    expression = pd.DataFrame(
        rng.normal(size=(n, p)), index=lines, columns=[f"f{j}" for j in range(p)]
    )
    y = 2 * expression["f0"] - 1.5 * expression["f1"] + 0.3 * rng.normal(size=n)
    return pd.DataFrame({"dep": y}), expression


@pytest.fixture(autouse=True)
def few_trees(monkeypatch):
    monkeypatch.setattr(paris_module, "TREES", 100)


def test_boruta_confirms_the_planted_drivers_and_no_noise_with_the_right_signs():
    dependency, expression = world()
    pairs = paris(dependency, expression, "raw", seed=43, n_jobs=1)
    assert list(pairs.columns) == COLUMNS
    assert set(pairs["feature_gene"]) == {"f0", "f1"}
    by = pairs.set_index("feature_gene")
    assert by.loc["f0", "r"] > 0.5 and by.loc["f1", "r"] < -0.4
    assert (by["r_pvalue"] < 1e-6).all() and (by["importance"] > 0).all()
    assert (pairs["dependency_gene"] == "dep").all() and (pairs["mode"] == "raw").all()


def test_corrected_impurity_selects_the_drivers_and_most_noise_stays_out():
    # Janitza's null is the mirrored negative importances, so it needs many features;
    # chance associations in the sample still let a few percent of noise through.
    dependency, expression = world(p=300)
    pairs = paris(dependency, expression, "Ginicorrected", seed=43, n_jobs=1)
    by = pairs.set_index("feature_gene")
    assert {"f0", "f1"} <= set(by.index) and len(by) < 30
    assert by.loc["f0", "r"] > 0.5 and by.loc["f1", "r"] < -0.4
    assert (pairs["mode"] == "Ginicorrected").all()


@pytest.mark.parametrize("mode", MODES)
def test_a_gene_never_predicts_itself_and_nan_lines_are_dropped(mode):
    dependency, expression = world(p=300 if mode == "Ginicorrected" else 40)
    expression["dep"] = dependency["dep"]  # a perfect predictor under its own name
    dependency.iloc[:5, 0] = np.nan
    pairs = paris(dependency, expression, mode, seed=43, n_jobs=1)
    assert "dep" not in set(pairs["feature_gene"])
    assert {"f0", "f1"} <= set(pairs["feature_gene"])


def test_janitza_keeps_importances_above_the_mirrored_negatives():
    selected = janitza_selected(np.array([-0.2, -0.1, 0.0, 0.1, 0.3]))
    assert selected.tolist() == [False, False, False, False, True]
    assert janitza_selected(np.array([0.0, 0.1])) is None


def test_genes_run_in_parallel_workers():
    dependency, expression = world(p=300)
    dependency["dep2"] = -expression["f5"] + 0.2 * np.random.default_rng(1).normal(
        size=len(expression)
    )
    pairs = paris(dependency, expression, "Ginicorrected", seed=43, n_jobs=2)
    chosen = pairs.groupby("dependency_gene")["feature_gene"].agg(set)
    assert {"f0", "f1"} <= chosen["dep"] and "f5" in chosen["dep2"]
