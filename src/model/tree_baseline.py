"""
Tree-based baselines for gene-level scoring.

Trains regressors on pseudobulk condition profiles to predict per-gene scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class TreeGeneScorer:
    """Predict gene-level scores from pseudobulk condition profiles."""

    model_type: str
    n_genes: int
    target_gene_indices: Optional[Sequence[int]] = None
    standardize: bool = False
    random_state: int = 42
    rf_params: Optional[Dict[str, object]] = None
    xgb_params: Optional[Dict[str, object]] = None

    scaler: Optional[StandardScaler] = None
    model: Optional[object] = None

    def __post_init__(self) -> None:
        if self.target_gene_indices is not None:
            unique = sorted({int(i) for i in self.target_gene_indices})
            self.target_gene_indices = np.array(unique, dtype=int)
        self.model_type = self.model_type.lower()

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "TreeGeneScorer":
        X_proc = self._fit_transform(X_train)
        Y_target = self._select_targets(Y_train)
        self.model = self._build_model()
        self.model.fit(X_proc, Y_target)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        X_proc = self._transform(X)
        scores = self.model.predict(X_proc)
        if scores.ndim == 1:
            scores = scores[:, None]

        if self.target_gene_indices is None:
            return scores

        full_scores = np.zeros((scores.shape[0], self.n_genes), dtype=np.float32)
        full_scores[:, self.target_gene_indices] = scores
        return full_scores

    def _select_targets(self, Y: np.ndarray) -> np.ndarray:
        if self.target_gene_indices is None:
            return Y
        return Y[:, self.target_gene_indices]

    def _fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self.standardize:
            self.scaler = StandardScaler(with_mean=True, with_std=True)
            return self.scaler.fit_transform(X)
        self.scaler = None
        return X

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X
        return self.scaler.transform(X)

    def _build_model(self) -> object:
        if self.model_type in {"random_forest", "rf", "randomforest"}:
            params = {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
                "n_jobs": -1,
            }
            if self.rf_params:
                params.update(self.rf_params)
            return RandomForestRegressor(random_state=self.random_state, **params)

        if self.model_type in {"xgboost", "xgb"}:
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError(
                    "xgboost is required for the xgboost baseline; "
                    "install it with 'pip install xgboost'."
                ) from exc

            params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "objective": "reg:squarederror",
                "tree_method": "auto",
                "n_jobs": -1,
                "verbosity": 0,
            }
            if self.xgb_params:
                params.update(self.xgb_params)
            base = XGBRegressor(random_state=self.random_state, **params)
            return MultiOutputRegressor(base)

        raise ValueError(f"Unknown model_type: {self.model_type}")
