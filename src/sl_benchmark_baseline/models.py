"""Baseline models A (logreg), B (xgboost), C (frequency probe)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sl_benchmark_baseline.config import SLBaselineConfig


@dataclass
class FoldData:
    """Per-fold inputs shared across models.

    Attributes:
        df: Fold rows, including ``pair_id``, ``sl_label``, gene symbols.
        features: Standardized symmetric features, shape ``(n, 5)``.
        labels: Binary labels aligned with ``features``, shape ``(n,)``.
    """

    df: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray


class LogRegModel:
    """Model A: symmetric logistic regression on the 5 standardized features."""

    name = "A"

    def __init__(self, config: SLBaselineConfig) -> None:
        self._model = LogisticRegression(
            C=config.logreg_c,
            max_iter=config.logreg_max_iter,
            random_state=config.seed,
        )

    def fit(self, train: FoldData) -> None:
        self._model.fit(train.features, train.labels)

    def predict_proba(self, test: FoldData) -> np.ndarray:
        return self._model.predict_proba(test.features)[:, 1]


class XGBModel:
    """Model B: gradient-boosted trees on the same 5 features."""

    name = "B"

    def __init__(self, config: SLBaselineConfig) -> None:
        from xgboost import XGBClassifier

        self._model = XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            random_state=config.seed,
            eval_metric="logloss",
        )

    def fit(self, train: FoldData) -> None:
        self._model.fit(train.features, train.labels)

    def predict_proba(self, test: FoldData) -> np.ndarray:
        return self._model.predict_proba(test.features)[:, 1]


class LogRegTranscriptModel(LogRegModel):
    """Model A_transcript: logistic regression on augmented features."""

    name = "A_transcript"


class XGBTranscriptModel(XGBModel):
    """Model B_transcript: XGBoost on augmented features."""

    name = "B_transcript"


class FrequencyProbeModel:
    """Model C: preferential-attachment probe from train-positive degree.

    Scores a test pair by ``pos_degree[a] * pos_degree[b]`` using only training
    positives. ``predict_proba`` preserves the old bounded-score interface for
    labeled pair scoring, while official matrix evaluation uses raw degree
    products via ``predict_score_matrix``.
    """

    name = "C"

    def __init__(self, config: SLBaselineConfig) -> None:
        self._pos_degree: Counter[str] = Counter()

    def fit(self, train: FoldData) -> None:
        positives = train.df[train.df["sl_label"] == 1]
        self._pos_degree = Counter()
        for symbol in positives["gene_a_symbol"]:
            self._pos_degree[str(symbol)] += 1
        for symbol in positives["gene_b_symbol"]:
            self._pos_degree[str(symbol)] += 1

    def predict_proba(self, test: FoldData) -> np.ndarray:
        raw = np.array(
            [
                self._pos_degree[str(a)] * self._pos_degree[str(b)]
                for a, b in zip(
                    test.df["gene_a_symbol"],
                    test.df["gene_b_symbol"],
                    strict=True,
                )
            ],
            dtype=float,
        )
        span = raw.max() - raw.min()
        if span == 0.0:
            return np.zeros_like(raw)
        return (raw - raw.min()) / span

    def predict_score_matrix(self, gene_symbols: np.ndarray) -> np.ndarray:
        """Score all candidate gene pairs by train-positive degree product."""
        degrees = np.array(
            [self._pos_degree[str(symbol)] for symbol in gene_symbols],
            dtype=float,
        )
        return np.outer(degrees, degrees)


def build_models(config: SLBaselineConfig) -> list:
    """Construct the three baseline models in canonical order A, B, C."""
    return [
        LogRegModel(config),
        XGBModel(config),
        FrequencyProbeModel(config),
    ]


def build_augmented_models(config: SLBaselineConfig) -> list:
    """Construct exp07 augmented-mode models: baseline A/B + transcript A/B."""
    return [
        LogRegModel(config),
        XGBModel(config),
        LogRegTranscriptModel(config),
        XGBTranscriptModel(config),
    ]
