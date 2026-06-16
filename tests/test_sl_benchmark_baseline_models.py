from __future__ import annotations

import numpy as np
import pandas as pd


def _make_fold_data(labels: np.ndarray, features: np.ndarray, df: pd.DataFrame):
    from sl_benchmark_baseline.models import FoldData

    return FoldData(df=df, features=features, labels=labels)


def test_models_emit_probabilities_in_unit_interval() -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.models import build_models

    rng = np.random.default_rng(1)
    n_train, n_test = 40, 12
    train_labels = np.array([1] * 20 + [0] * 20)
    train_feats = np.where(train_labels[:, None] == 1, -1.0, 0.5) + rng.normal(
        0, 0.05, size=(n_train, 5)
    )
    train_df = pd.DataFrame(
        {
            "pair_id": [f"T{i}" for i in range(n_train)],
            "sl_label": train_labels,
            "gene_a_symbol": ["GA"] * n_train,
            "gene_b_symbol": ["GB"] * n_train,
        }
    )
    test_labels = np.array([1] * 6 + [0] * 6)
    test_feats = np.where(test_labels[:, None] == 1, -1.0, 0.5) + rng.normal(
        0, 0.05, size=(n_test, 5)
    )
    test_df = pd.DataFrame(
        {
            "pair_id": [f"S{i}" for i in range(n_test)],
            "sl_label": test_labels,
            "gene_a_symbol": ["GA"] * n_test,
            "gene_b_symbol": ["GB"] * n_test,
        }
    )
    train = _make_fold_data(train_labels, train_feats, train_df)
    test = _make_fold_data(test_labels, test_feats, test_df)

    models = build_models(SLBaselineConfig())
    assert {m.name for m in models} == {"A", "B", "C"}
    for model in models:
        model.fit(train)
        scores = model.predict_proba(test)
        assert scores.shape == (n_test,)
        assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_frequency_probe_uses_train_positive_degree() -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.models import FoldData, build_models

    train_df = pd.DataFrame(
        {
            "pair_id": [f"T{i}" for i in range(4)],
            "sl_label": [1, 1, 1, 0],
            "gene_a_symbol": ["HUB", "HUB", "HUB", "RARE"],
            "gene_b_symbol": ["X1", "X2", "X3", "X4"],
        }
    )
    test_df = pd.DataFrame(
        {
            "pair_id": ["S0", "S1"],
            "sl_label": [1, 0],
            "gene_a_symbol": ["HUB", "RARE"],
            "gene_b_symbol": ["HUB", "RARE"],
        }
    )
    dummy_feats_train = np.zeros((4, 5))
    dummy_feats_test = np.zeros((2, 5))
    train = FoldData(
        df=train_df, features=dummy_feats_train, labels=np.array([1, 1, 1, 0])
    )
    test = FoldData(df=test_df, features=dummy_feats_test, labels=np.array([1, 0]))

    probe = next(m for m in build_models(SLBaselineConfig()) if m.name == "C")
    probe.fit(train)
    scores = probe.predict_proba(test)
    assert scores[0] > scores[1]
