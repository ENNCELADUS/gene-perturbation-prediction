"""
PCA + kNN gene-scoring baseline for reverse perturbation prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PcaKnnGeneScorer:
    """Fit PCA on condition profiles and score genes via kNN label aggregation."""

    n_components: int
    standardize: bool = False
    random_state: int = 42

    pca: Optional[PCA] = None
    scaler: Optional[StandardScaler] = None
    train_embeddings: Optional[np.ndarray] = None
    train_labels: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "PcaKnnGeneScorer":
        if self.standardize:
            self.scaler = StandardScaler(with_mean=True, with_std=True)
            X_train = self.scaler.fit_transform(X_train)
        else:
            self.scaler = None

        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.train_embeddings = self.pca.fit_transform(X_train)
        self.train_labels = Y_train.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.pca is None:
            raise RuntimeError("PCA model is not fitted.")
        return self.pca.transform(X)
