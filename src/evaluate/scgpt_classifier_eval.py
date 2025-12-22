"""
scGPT classifier-head evaluator for reverse perturbation retrieval.

Uses the fine-tuned classification head to produce top-K predictions,
aligned with the logreg baseline protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np

from src.data import PerturbDataset, mask_perturbed_genes, create_pseudo_bulk
from src.model import get_encoder
from .metrics import compute_all_metrics
from .confidence import (
    ConfidenceScorer,
    coverage_accuracy_curve,
    compute_auc_coverage_accuracy,
)


class ScGPTClassifierEvaluator:
    """
    Classifier-style evaluator for scGPT fine-tuned with classification loss.

    Uses the classification head to rank conditions, mirroring the
    logreg baseline evaluation (top-K probability ranking).
    """

    def __init__(
        self,
        encoder_kwargs: Optional[dict] = None,
        top_k: List[int] = [1, 5, 8, 10],
        mask_perturbed: bool = True,
        mask_layer_key: Optional[str] = None,
        query_mode: str = "cell",
        pseudo_bulk_config: Optional[dict] = None,
        candidate_source: str = "all",
        query_split: str = "test",
    ):
        self.encoder_kwargs = encoder_kwargs or {}
        self.top_k = top_k
        self.mask_perturbed = mask_perturbed
        self.mask_layer_key = mask_layer_key
        self.query_mode = query_mode
        self.pseudo_bulk_config = pseudo_bulk_config or {}
        self.candidate_source = candidate_source
        self.query_split = query_split

        self.encoder = None
        self.condition_order: Optional[List[str]] = None
        self.candidate_conditions: Optional[List[str]] = None

    def setup(self, dataset: PerturbDataset) -> "ScGPTClassifierEvaluator":
        """Load scGPT encoder + classification head."""
        if dataset.split is None:
            raise RuntimeError("Dataset must have split applied")

        self.encoder = get_encoder("scgpt", **self.encoder_kwargs)
        self.encoder.fit(np.empty((0, 0), dtype=np.float32))

        has_head = getattr(self.encoder, "has_classifier_head", None)
        if not has_head or not self.encoder.has_classifier_head():
            raise RuntimeError(
                "scGPT classifier head not loaded. "
                "Ensure finetune_apply_classifier is enabled and checkpoint is classification-trained."
            )

        self.condition_order = (
            self.encoder.get_condition_order() or dataset.all_conditions
        )
        condition_sets = dataset.get_condition_sets()
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

    def _prepare_query_adata(
        self, dataset: PerturbDataset
    ) -> Tuple[Optional[ad.AnnData], List[str]]:
        """Prepare query AnnData for evaluation."""
        condition_sets = dataset.get_condition_sets()
        query_conditions = self._resolve_query_conditions(condition_sets)
        query_adata = dataset.get_query_adata_for_conditions(query_conditions)

        if self.mask_perturbed:
            query_adata = mask_perturbed_genes(
                query_adata,
                condition_col=dataset.condition_col,
                layer=self.mask_layer_key,
            )

        if self.query_mode == "pseudobulk":
            cells_per_bulk = self.pseudo_bulk_config.get("cells_per_bulk", 50)
            n_bulks = self.pseudo_bulk_config.get("n_bulks", 1)
            seed = self.pseudo_bulk_config.get("seed", 42)
            bulk_source = query_adata
            if self.mask_layer_key and self.mask_layer_key != "X":
                bulk_source = query_adata.copy()
                bulk_source.X = bulk_source.layers[self.mask_layer_key]

            bulks, labels = create_pseudo_bulk(
                bulk_source,
                cells_per_bulk=cells_per_bulk,
                n_bulks=n_bulks,
                condition_col=dataset.condition_col,
                seed=seed,
            )
            if len(labels) == 0:
                return None, []

            bulk_adata = ad.AnnData(
                X=bulks,
                obs={dataset.condition_col: labels},
                var=query_adata.var.copy(),
            )
            if self.mask_layer_key and self.mask_layer_key != "X":
                bulk_adata.layers[self.mask_layer_key] = bulk_adata.X.copy()
            return bulk_adata, labels

        ground_truth = query_adata.obs[dataset.condition_col].tolist()
        return query_adata, ground_truth

    def evaluate(
        self,
        dataset: PerturbDataset,
        confidence_config: Optional[dict] = None,
    ) -> Dict[str, float]:
        """Evaluate on query cells using classification head."""
        if self.encoder is None:
            raise RuntimeError("Must call setup() before evaluate()")

        query_adata, ground_truth = self._prepare_query_adata(dataset)
        if not ground_truth:
            return {}

        max_k = min(max(self.top_k), len(self.condition_order))
        predictions, scores = self.encoder.predict_topk_adata(
            query_adata,
            k=max_k,
            condition_order=self.condition_order,
        )

        metrics = compute_all_metrics(
            predictions,
            ground_truth,
            self.top_k,
            candidate_pool=self.candidate_conditions,
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
        query_adata, ground_truth = self._prepare_query_adata(dataset)
        if not ground_truth:
            return {}, {"predictions": [], "ground_truth": []}

        max_k = min(max(self.top_k), len(self.condition_order))
        predictions, scores = self.encoder.predict_topk_adata(
            query_adata,
            k=max_k,
            condition_order=self.condition_order,
        )

        metrics = compute_all_metrics(
            predictions,
            ground_truth,
            self.top_k,
            candidate_pool=self.candidate_conditions,
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
