import numpy as np
import pandas as pd
import pytest

from src.experiments.sl.stage_two import (
    context_weights,
    dyadic_difference,
    effect_features,
    fit_head,
    head_input,
    paris_features,
    pair_identity,
    ranking_metrics,
    varying_pairs,
)

HEAD = {
    "hidden": 16,
    "learning_rate": 1e-2,
    "weight_decay": 0.0,
    "batch_size": 64,
    "max_epochs": 15,
}


def test_head_input_ignores_the_pairs_order():
    rng = np.random.default_rng(0)
    v_a, v_b = rng.normal(size=(5, 32)), rng.normal(size=(5, 32))
    screens = pd.Series(["ito_2021", "harle_2025", "klingbeil_2024", "ito_2021", "x"])
    out = head_input(v_a, v_b, screens)
    np.testing.assert_array_equal(out, head_input(v_b, v_a, screens))
    assert out.shape == (5, 67)
    assert out[4, 64:].sum() == 0  # an unknown screen sets no indicator


def test_fit_head_learns_a_planted_rule_and_keeps_the_best_validation_epoch():
    rng = np.random.default_rng(0)

    def world(n: int) -> tuple[np.ndarray, np.ndarray]:
        v_a, v_b = rng.normal(size=(n, 32)), rng.normal(size=(n, 32))
        label = ((v_a[:, 0] + v_b[:, 0]) < -1.0).astype(int)  # both genes "dependent"
        return head_input(v_a, v_b, pd.Series(["ito_2021"] * n)), label

    x, y = world(2000)
    validation = [world(400), world(400)]
    score, record = fit_head((x, y, np.ones(len(y))), validation, HEAD, seed=0)
    assert record["validation_lift"] == max(record["history"])
    assert record["epoch"] == 1 + int(np.argmax(record["history"]))
    test_x, test_y = world(400)
    assert ranking_metrics(test_y, score(test_x), 20)["auroc"] > 0.9


def test_context_weights_balance_contexts_and_classes():
    contexts = pd.Series(["a"] * 10 + ["b"] * 4)
    labels = pd.Series([1] * 2 + [0] * 8 + [1] * 2 + [0] * 2)
    weights = context_weights(contexts, labels)
    frame = pd.DataFrame({"c": contexts, "y": labels, "w": weights})
    assert frame.groupby("c")["w"].sum().tolist() == pytest.approx([1.0, 1.0])
    assert frame.groupby(["c", "y"])["w"].sum().tolist() == pytest.approx([0.5] * 4)


def test_pair_identity_and_varying_read_the_same_screen_only():
    training = pd.DataFrame(
        {
            "screen": ["ito_2021", "ito_2021", "harle_2025"],
            "gene_a": ["G0", "G0", "G0"],
            "gene_b": ["G1", "G1", "G1"],
            "label": [1, 0, 1],
        }
    )
    rows = pd.DataFrame(
        {
            "screen": ["ito_2021", "harle_2025", "klingbeil_2024"],
            "gene_a": ["G0"] * 3,
            "gene_b": ["G1"] * 3,
        }
    )
    np.testing.assert_allclose(pair_identity(training, rows), [0.5, 1.0, np.nan])
    assert varying_pairs(training, rows).tolist() == [True, False, False]


def test_dyadic_difference_is_zero_for_one_score_and_brackets_a_real_gap():
    rng = np.random.default_rng(0)
    n = 400
    gene_a = rng.integers(0, 60, n).astype(str)
    gene_b = rng.integers(60, 120, n).astype(str)
    label = rng.random(n) < 0.15
    good = label + rng.normal(0, 0.5, n)
    noise = rng.normal(0, 1, n)
    same = dyadic_difference(label, noise, noise, gene_a, gene_b, 50, rng)
    assert same["difference"] == 0 and same["interval"] == [0.0, 0.0]
    gap = dyadic_difference(label, good, noise, gene_a, gene_b, 200, rng)
    low, high = gap["interval"]
    assert 0 < low <= gap["difference"] <= high


def test_paris_features_ignore_the_pairs_order_and_read_unselected_as_zero():
    selected = pd.DataFrame(
        {
            "dependency_gene": ["B", "A", "C"],
            "feature_gene": ["A", "B", "D"],
            "importance": [2.0, 0.5, 1.0],
            "r": [0.4, -0.1, 0.3],
        }
    )
    rows = pd.DataFrame(
        {"gene_a": ["A", "B", "C", "E"], "gene_b": ["B", "A", "D", "F"]}
    )
    out = paris_features(selected, rows)
    np.testing.assert_allclose(out[0], [2.0, 2.5, 0.3, 0.5, 2])
    np.testing.assert_allclose(out[1], out[0])
    np.testing.assert_allclose(out[2], [1.0, 1.0, 0.3, 0.3, 1])  # D's for C only
    np.testing.assert_allclose(out[3], [0, 0, 0, 0, 0])


def test_effect_features_read_the_rows_line():
    effects = pd.DataFrame(
        [[-1.0, -0.2], [0.1, -0.5]], index=["L0", "L1"], columns=["A", "B"]
    )
    rows = pd.DataFrame({"model_id": ["L1"], "gene_a": ["B"], "gene_b": ["A"]})
    np.testing.assert_allclose(effect_features(effects, rows), [[-0.4, 0.6, -0.5]])
