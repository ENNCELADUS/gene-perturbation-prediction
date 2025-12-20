"""
Confidence scoring for retrieval reliability analysis.

Implements:
- Margin-based confidence (gap between top-1 and top-2 scores)
- Entropy-based confidence (inverse entropy of score distribution)
- Prototype agreement (consensus across multiple prototypes)
- Coverage vs accuracy curves for selective retrieval
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional

import numpy as np


def compute_margin_confidence(scores: np.ndarray) -> float:
    """
    Compute margin-based confidence.

    Margin = score of top-1 minus score of top-2.
    Higher margin indicates more confident prediction.

    Args:
        scores: Score array for all candidates (1D)

    Returns:
        Margin confidence value
    """
    if len(scores) < 2:
        return 1.0

    sorted_scores = np.sort(scores)[::-1]
    return float(sorted_scores[0] - sorted_scores[1])


def compute_entropy_confidence(probs: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute entropy-based confidence (1 - normalized entropy).

    Low entropy = high confidence, high entropy = low confidence.

    Args:
        probs: Probability distribution over classes (must sum to 1)
        eps: Small value to avoid log(0)

    Returns:
        Confidence value in [0, 1]
    """
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum()

    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))

    if max_entropy == 0:
        return 1.0

    normalized_entropy = entropy / max_entropy
    return float(1.0 - normalized_entropy)


def compute_prototype_agreement(
    scores_per_prototype: np.ndarray,
    top_k: int = 1,
) -> float:
    """
    Compute agreement across multiple prototypes.

    Measures how consistently prototypes agree on top-K predictions.

    Args:
        scores_per_prototype: Shape (n_prototypes, n_conditions)
        top_k: Number of top predictions to consider

    Returns:
        Agreement score in [0, 1]
    """
    if scores_per_prototype.ndim != 2:
        raise ValueError("Expected 2D array (n_prototypes, n_conditions)")

    n_prototypes = scores_per_prototype.shape[0]

    # Get top-K predictions for each prototype
    top_predictions = []
    for i in range(n_prototypes):
        top_k_indices = np.argsort(scores_per_prototype[i])[::-1][:top_k]
        top_predictions.append(set(top_k_indices))

    # Compute pairwise Jaccard similarity
    similarities = []
    for i in range(n_prototypes):
        for j in range(i + 1, n_prototypes):
            intersection = len(top_predictions[i] & top_predictions[j])
            union = len(top_predictions[i] | top_predictions[j])
            if union > 0:
                similarities.append(intersection / union)

    return float(np.mean(similarities)) if similarities else 1.0


def coverage_accuracy_curve(
    confidences: np.ndarray,
    is_correct: np.ndarray,
    n_points: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coverage vs accuracy curve for selective retrieval.

    At each coverage level (fraction of queries answered), compute accuracy
    on the most confident queries.

    Args:
        confidences: Confidence score per query
        is_correct: Boolean array indicating correct predictions
        n_points: Number of points on the curve

    Returns:
        Tuple of (coverage_levels, accuracies)
    """
    confidences = np.asarray(confidences)
    is_correct = np.asarray(is_correct, dtype=bool)

    # Sort by confidence (descending)
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_correct = is_correct[sorted_indices]

    n_total = len(confidences)
    coverages = np.linspace(0.05, 1.0, n_points)
    accuracies = []

    for cov in coverages:
        n_selected = max(1, int(cov * n_total))
        acc = np.mean(sorted_correct[:n_selected])
        accuracies.append(acc)

    return coverages, np.array(accuracies)


def compute_auc_coverage_accuracy(
    confidences: np.ndarray,
    is_correct: np.ndarray,
) -> float:
    """
    Compute area under coverage-accuracy curve.

    Higher is better (more accurate on confident predictions).

    Args:
        confidences: Confidence score per query
        is_correct: Boolean array indicating correct predictions

    Returns:
        AUC value in [0, 1]
    """
    coverages, accuracies = coverage_accuracy_curve(confidences, is_correct)
    # Trapezoidal integration
    return float(np.trapz(accuracies, coverages))


class ConfidenceScorer:
    """
    Wrapper class for computing confidence scores during evaluation.
    """

    def __init__(
        self,
        method: str = "margin",
        top_k_agreement: int = 1,
    ):
        """
        Initialize confidence scorer.

        Args:
            method: "margin", "entropy", or "agreement"
            top_k_agreement: K for prototype agreement
        """
        self.method = method
        self.top_k_agreement = top_k_agreement

    def score(self, scores: np.ndarray) -> float:
        """
        Compute confidence for a single query.

        Args:
            scores: Scores/probabilities over candidates

        Returns:
            Confidence value
        """
        if self.method == "margin":
            return compute_margin_confidence(scores)
        elif self.method == "entropy":
            # Normalize to probabilities
            probs = np.exp(scores - np.max(scores))
            probs = probs / probs.sum()
            return compute_entropy_confidence(probs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def score_batch(self, all_scores: np.ndarray) -> np.ndarray:
        """
        Compute confidence for batch of queries.

        Args:
            all_scores: Shape (n_queries, n_candidates)

        Returns:
            Confidence array (n_queries,)
        """
        return np.array([self.score(s) for s in all_scores])
