"""
Logistic Regression classifier baseline for reverse perturbation prediction.

Treats the retrieval task as multi-class classification:
- Each condition is a class
- Model outputs probability scores for top-K retrieval
"""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .base import BaseEncoder


class LogRegClassifier(BaseEncoder):
    """
    Logistic Regression classifier for condition prediction.

    Uses multinomial logistic regression to predict perturbation conditions.
    Returns class probabilities for top-K retrieval ranking.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        n_jobs: int = -1,
    ):
        """
        Initialize logistic regression classifier.

        Args:
            C: Inverse of regularization strength
            max_iter: Maximum iterations for optimization
            solver: Solver algorithm ('lbfgs', 'saga', etc.)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.n_jobs = n_jobs

        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            n_jobs=n_jobs,
            multi_class="multinomial",
        )
        self.label_encoder = LabelEncoder()
        self._fitted = False
        self._classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LogRegClassifier":
        """
        Fit classifier on (expression, condition) pairs.

        Args:
            X: Expression matrix (n_samples, n_genes)
            y: Condition labels (n_samples,)

        Returns:
            self
        """
        if y is None:
            raise ValueError("LogRegClassifier.fit() requires labels y")

        if hasattr(X, "toarray"):
            X = X.toarray()

        # Encode string labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        self._classes = self.label_encoder.classes_

        self.model.fit(X, y_encoded)
        self._fitted = True
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability scores for all classes.

        Args:
            X: Expression matrix (n_samples, n_genes)

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before encode()")
        if hasattr(X, "toarray"):
            X = X.toarray()

        return self.model.predict_proba(X)

    def predict_topk(
        self, X: np.ndarray, k: int = 10
    ) -> Tuple[List[List[str]], np.ndarray]:
        """
        Predict top-K conditions with scores.

        Args:
            X: Expression matrix (n_samples, n_genes)
            k: Number of top conditions to return

        Returns:
            Tuple of:
            - List of top-K condition names per sample
            - Score matrix (n_samples, k)
        """
        probs = self.encode(X)
        n_samples = probs.shape[0]
        n_classes = probs.shape[1]
        k = min(k, n_classes)

        # Get top-K indices and scores
        top_indices = np.argsort(probs, axis=1)[:, ::-1][:, :k]
        top_scores = np.take_along_axis(probs, top_indices, axis=1)

        # Convert indices to condition names
        top_conditions = []
        for i in range(n_samples):
            conditions = self.label_encoder.inverse_transform(top_indices[i])
            top_conditions.append(list(conditions))

        return top_conditions, top_scores

    @property
    def classes(self) -> np.ndarray:
        """Get all class labels."""
        if self._classes is None:
            raise RuntimeError("Must call fit() first")
        return self._classes

    @property
    def n_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes)
