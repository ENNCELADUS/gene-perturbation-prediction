"""
PCA-based expression encoder (baseline).
"""

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from .base import BaseEncoder


class PCAEncoder(BaseEncoder):
    """PCA-based expression encoder (baseline)."""

    def __init__(self, n_components: int = 50):
        """
        Initialize PCA encoder.

        Args:
            n_components: Number of principal components
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self._fitted = False

    def fit(self, X: np.ndarray) -> "PCAEncoder":
        """
        Fit PCA on expression data.

        Args:
            X: Expression matrix (n_samples, n_genes)

        Returns:
            self
        """
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.pca.fit(X)
        self._fitted = True
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode expression to PCA space.

        Args:
            X: Expression matrix (n_samples, n_genes)

        Returns:
            Embeddings (n_samples, n_components)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before encode()")
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self.pca.transform(X)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio per component."""
        return self.pca.explained_variance_ratio_
