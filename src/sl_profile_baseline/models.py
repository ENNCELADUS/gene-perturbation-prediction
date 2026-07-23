"""Models for the strict two-GeneEffect-profile experiment."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sl_profile_baseline.config import ProfileBaselineConfig


class ProbabilityModel(Protocol):
    """Minimal fitted-model interface used by the evaluator."""

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return two-class probabilities."""


def fit_summary_logreg(
    features: np.ndarray, labels: np.ndarray, config: ProfileBaselineConfig
) -> ProbabilityModel:
    """Fit the inspectable L2 logistic profile-summary baseline."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.logreg_c,
            max_iter=config.logreg_max_iter,
            random_state=config.seed,
        ),
    )
    model.fit(features, labels)
    return model


def fit_summary_xgb(
    features: np.ndarray, labels: np.ndarray, config: ProfileBaselineConfig
) -> ProbabilityModel:
    """Fit the nonlinear compact-feature upper-bound model."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.seed,
        n_jobs=-1,
    )
    model.fit(features, labels)
    return model


def fit_raw_sgd(
    features: np.ndarray, labels: np.ndarray, config: ProfileBaselineConfig
) -> ProbabilityModel:
    """Fit elastic-net logistic SGD on ``[sum, absolute difference]`` profiles."""
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=config.raw_alpha,
            l1_ratio=config.raw_l1_ratio,
            max_iter=config.raw_max_iter,
            random_state=config.seed,
            average=True,
            n_jobs=-1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        ),
    )
    model.fit(features, labels)
    return model


def positive_probability(model: ProbabilityModel, features: np.ndarray) -> np.ndarray:
    """Return the fitted probability of the sampled Feng positive label."""
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
