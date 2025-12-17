"""
Retrieval module for reverse perturbation prediction.

Given a query expression profile, retrieve the most similar
perturbation conditions from a reference library.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def compute_similarity(
    query: np.ndarray,
    library: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Compute similarity between query and library profiles.

    Args:
        query: Query embeddings (n_queries, n_features)
        library: Reference library (n_references, n_features)
        metric: 'cosine' or 'euclidean'

    Returns:
        Similarity scores (n_queries, n_references)
        Higher = more similar for cosine, lower for euclidean
    """
    if query.ndim == 1:
        query = query.reshape(1, -1)

    if metric == "cosine":
        return cosine_similarity(query, library)
    elif metric == "euclidean":
        # Return negative distance so higher = more similar
        return -euclidean_distances(query, library)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")


class RetrievalHead:
    """Retrieval head for perturbation prediction."""

    def __init__(
        self,
        metric: str = "cosine",
        top_k: List[int] = [1, 5, 10, 20],
    ):
        """
        Initialize retrieval head.

        Args:
            metric: Similarity metric
            top_k: List of K values for Top-K retrieval
        """
        self.metric = metric
        self.top_k = top_k

        # Reference library
        self.library_embeddings: np.ndarray = None
        self.library_conditions: List[str] = None

    def build_library(
        self,
        embeddings: np.ndarray,
        conditions: List[str],
    ) -> "RetrievalHead":
        """
        Build reference library from perturbation embeddings.

        Args:
            embeddings: Embedding matrix (n_conditions, n_features)
            conditions: List of condition names

        Returns:
            self
        """
        self.library_embeddings = embeddings
        self.library_conditions = conditions
        return self

    def retrieve(
        self,
        query: np.ndarray,
        k: int = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-K similar conditions for a single query.

        Args:
            query: Query embedding (n_features,)
            k: Number of results (default: max(self.top_k))

        Returns:
            List of (condition, score) tuples, sorted by score descending
        """
        if self.library_embeddings is None:
            raise RuntimeError("Must call build_library() first")

        k = k or max(self.top_k)

        # Compute similarities
        scores = compute_similarity(query, self.library_embeddings, self.metric)
        scores = scores.flatten()

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:k]

        return [
            (self.library_conditions[i], float(scores[i]))
            for i in top_indices
        ]

    def batch_retrieve(
        self,
        queries: np.ndarray,
        k: int = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Retrieve top-K for multiple queries.

        Args:
            queries: Query embeddings (n_queries, n_features)
            k: Number of results per query

        Returns:
            List of retrieval results per query
        """
        return [self.retrieve(q, k) for q in queries]

    def predict(
        self,
        query: np.ndarray,
    ) -> str:
        """
        Predict the most likely perturbation condition.

        Args:
            query: Query embedding

        Returns:
            Predicted condition name
        """
        results = self.retrieve(query, k=1)
        return results[0][0] if results else None
