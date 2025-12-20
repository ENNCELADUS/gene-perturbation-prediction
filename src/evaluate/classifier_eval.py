"""
Classifier-based evaluator for reverse perturbation retrieval.

Implements classifier-style evaluation protocol:
- Direct multi-class classification
- Probability-based ranking for top-K retrieval
- Compatible with LogRegClassifier and similar models
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data import PerturbDataset, mask_perturbed_genes, create_pseudo_bulk
from src.model import LogRegClassifier
from .metrics import compute_all_metrics
from .confidence import (
    ConfidenceScorer,
    coverage_accuracy_curve,
    compute_auc_coverage_accuracy,
)


class ClassifierEvaluator:
    """
    Classifier-based evaluator for discriminative baselines.

    Key features:
    - Trains classifier on (ref cells, condition labels)
    - Uses class probabilities for top-K ranking
    - No prototype library needed
    """

    def __init__(
        self,
        classifier_kwargs: Optional[dict] = None,
        top_k: List[int] = [1, 5, 8, 10],
        mask_perturbed: bool = True,
        query_mode: str = "cell",
        pseudo_bulk_config: Optional[dict] = None,
        candidate_source: str = "all",
        query_split: str = "test",
    ):
        """
        Initialize classifier evaluator.

        Args:
            classifier_kwargs: Arguments for LogRegClassifier
            top_k: K values for evaluation
            mask_perturbed: Whether to mask perturbed gene expression
        """
        self.classifier_kwargs = classifier_kwargs or {}
        self.top_k = top_k
        self.mask_perturbed = mask_perturbed
        self.query_mode = query_mode
        self.pseudo_bulk_config = pseudo_bulk_config or {}
        self.candidate_source = candidate_source
        self.query_split = query_split

        # Components (initialized during setup)
        self.classifier: Optional[LogRegClassifier] = None
        self.candidate_conditions: Optional[List[str]] = None

    def setup(self, dataset: PerturbDataset) -> "ClassifierEvaluator":
        """
        Train classifier on reference cells.

        Args:
            dataset: Loaded PerturbDataset with split applied

        Returns:
            self
        """
        if dataset.split is None:
            raise RuntimeError("Dataset must have split applied")

        condition_sets = dataset.get_condition_sets()
        train_conditions = condition_sets["train"]

        # Get reference cells
        ref_adata = dataset.get_ref_adata_for_conditions(train_conditions)

        # Apply masking if requested
        if self.mask_perturbed:
            ref_adata = mask_perturbed_genes(ref_adata)

        # Prepare training data
        X_ref = ref_adata.X
        if hasattr(X_ref, "toarray"):
            X_ref = X_ref.toarray()
        y_ref = ref_adata.obs["condition"].values

        # Train classifier
        self.classifier = LogRegClassifier(**self.classifier_kwargs)
        self.classifier.fit(X_ref, y_ref)
        self.candidate_conditions = self._resolve_candidate_conditions(condition_sets)

        return self

    def _resolve_candidate_conditions(self, condition_sets: dict) -> List[str]:
        """Resolve candidate conditions for retrieval."""
        if self.candidate_source == "train":
            return condition_sets["train"]
        if self.candidate_source == "train_val":
            return condition_sets["train"] + condition_sets["val"]
        return condition_sets["all"]

    def _resolve_query_conditions(self, condition_sets: dict) -> List[str]:
        """Resolve query conditions for evaluation."""
        if self.query_split == "val":
            return condition_sets["val"]
        if self.query_split == "train":
            return condition_sets["train"]
        return condition_sets["test"]

    def _prepare_query_data(
        self, dataset: PerturbDataset
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare query data for evaluation."""
        condition_sets = dataset.get_condition_sets()
        query_conditions = self._resolve_query_conditions(condition_sets)
        query_adata = dataset.get_query_adata_for_conditions(query_conditions)

        if self.mask_perturbed:
            query_adata = mask_perturbed_genes(query_adata)

        if self.query_mode == "pseudobulk":
            cells_per_bulk = self.pseudo_bulk_config.get("cells_per_bulk", 50)
            n_bulks = self.pseudo_bulk_config.get("n_bulks", 1)
            seed = self.pseudo_bulk_config.get("seed", 42)
            bulks, labels = create_pseudo_bulk(
                query_adata,
                cells_per_bulk=cells_per_bulk,
                n_bulks=n_bulks,
                condition_col="condition",
                seed=seed,
            )
            return bulks, labels

        X_query = query_adata.X
        if hasattr(X_query, "toarray"):
            X_query = X_query.toarray()
        ground_truth = query_adata.obs["condition"].tolist()
        return X_query, ground_truth

    def evaluate(
        self,
        dataset: PerturbDataset,
        confidence_config: Optional[dict] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on query cells.

        Args:
            dataset: PerturbDataset with split applied

        Returns:
            Dictionary of metrics
        """
        if self.classifier is None:
            raise RuntimeError("Must call setup() before evaluate()")

        X_query, ground_truth = self._prepare_query_data(dataset)
        if len(ground_truth) == 0:
            return {}

        # Get predictions using classifier
        max_k = max(self.top_k)
        predictions, scores = self.classifier.predict_topk(X_query, k=max_k)

        # Compute metrics
        candidate_pool = self.candidate_conditions or list(self.classifier.classes)
        metrics = compute_all_metrics(
            predictions,
            ground_truth,
            self.top_k,
            candidate_pool=candidate_pool,
        )

        if confidence_config and confidence_config.get("enable", False):
            scorer = ConfidenceScorer(
                method=confidence_config.get("method", "margin"),
                top_k_agreement=confidence_config.get("top_k_agreement", 1),
            )
            confidences = scorer.score_batch(scores)
            is_correct = np.array(
                [preds[0] == true for preds, true in zip(predictions, ground_truth)]
            )
            n_points = confidence_config.get("coverage_points", 20)
            coverages, accuracies = coverage_accuracy_curve(
                confidences, is_correct, n_points=n_points
            )
            metrics["confidence_auc"] = compute_auc_coverage_accuracy(
                confidences, is_correct
            )
            metrics["coverage_accuracy_curve"] = {
                "coverage": coverages.tolist(),
                "accuracy": accuracies.tolist(),
            }

        return metrics

    def evaluate_with_details(
        self,
        dataset: PerturbDataset,
        confidence_config: Optional[dict] = None,
    ) -> Tuple[Dict[str, float], Dict[str, list]]:
        """Evaluate and return predictions for downstream analysis."""
        X_query, ground_truth = self._prepare_query_data(dataset)
        if len(ground_truth) == 0:
            return {}, {"predictions": [], "ground_truth": []}

        max_k = max(self.top_k)
        predictions, scores = self.classifier.predict_topk(X_query, k=max_k)

        candidate_pool = self.candidate_conditions or list(self.classifier.classes)
        metrics = compute_all_metrics(
            predictions,
            ground_truth,
            self.top_k,
            candidate_pool=candidate_pool,
        )

        if confidence_config and confidence_config.get("enable", False):
            scorer = ConfidenceScorer(
                method=confidence_config.get("method", "margin"),
                top_k_agreement=confidence_config.get("top_k_agreement", 1),
            )
            confidences = scorer.score_batch(scores)
            is_correct = np.array(
                [preds[0] == true for preds, true in zip(predictions, ground_truth)]
            )
            n_points = confidence_config.get("coverage_points", 20)
            coverages, accuracies = coverage_accuracy_curve(
                confidences, is_correct, n_points=n_points
            )
            metrics["confidence_auc"] = compute_auc_coverage_accuracy(
                confidences, is_correct
            )
            metrics["coverage_accuracy_curve"] = {
                "coverage": coverages.tolist(),
                "accuracy": accuracies.tolist(),
            }

        details = {
            "predictions": predictions,
            "ground_truth": ground_truth,
        }
        return metrics, details

    def evaluate_pseudo_bulk_curve(
        self,
        dataset: PerturbDataset,
        cells_per_bulk_values: List[int],
        n_bulks: int = 5,
        seed: int = 42,
        confidence_config: Optional[dict] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate performance across pseudo-bulk sizes."""
        if not cells_per_bulk_values:
            return {}

        curve = {}
        original_config = dict(self.pseudo_bulk_config)
        original_mode = self.query_mode
        self.query_mode = "pseudobulk"
        for i, size in enumerate(cells_per_bulk_values):
            self.pseudo_bulk_config = {
                "cells_per_bulk": size,
                "n_bulks": n_bulks,
                "seed": seed + i,
            }
            metrics = self.evaluate(
                dataset, confidence_config=confidence_config or {"enable": False}
            )
            curve[size] = metrics
        self.pseudo_bulk_config = original_config
        self.query_mode = original_mode
        return curve

    def save_results(
        self,
        metrics: Dict[str, float],
        output_dir: str | Path,
        experiment_name: str,
    ) -> Path:
        """Save evaluation results to JSON."""
        output_path = Path(output_dir) / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)

        metrics_file = output_path / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics_file
