"""
Cell-level evaluator for reverse perturbation retrieval.

Implements cell-level evaluation protocol:
- Cell-level queries (each query cell → retrieve condition)
- Multi-prototype reference library from ref cells only
- Condition-level aggregation via max similarity
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import anndata as ad

from src.data import (
    PerturbDataset,
    mask_perturbed_genes,
    build_prototype_library,
    create_pseudo_bulk,
)
from src.model import get_encoder
from .metrics import compute_all_metrics
from .confidence import (
    ConfidenceScorer,
    coverage_accuracy_curve,
    compute_auc_coverage_accuracy,
)


class CellRetrievalEvaluator:
    """
    Cell-level evaluator for within-condition split.

    Key features:
    - Uses cell-level queries (not pseudo-bulk)
    - Builds library from ref cells only
    - Supports multi-prototype library with condition aggregation
    """

    def __init__(
        self,
        encoder_type: str = "pca",
        encoder_kwargs: Optional[dict] = None,
        metric: str = "cosine",
        top_k: List[int] = [1, 5, 8, 10],
        mask_perturbed: bool = True,
        mask_layer_key: Optional[str] = None,
        library_type: str = "bootstrap",
        n_prototypes: int = 30,
        m_cells_per_prototype: int = 50,
        library_seed: int = 42,
        query_mode: str = "cell",
        pseudo_bulk_config: Optional[dict] = None,
        candidate_source: str = "all",
        query_split: str = "test",
    ):
        """
        Initialize evaluator.

        Args:
            encoder_type: Type of encoder ('pca', 'scgpt')
            encoder_kwargs: Encoder-specific arguments
            metric: Similarity metric ('cosine', 'euclidean')
            top_k: K values for evaluation
            mask_perturbed: Whether to mask perturbed gene expression
            library_type: 'bootstrap', 'mean', or 'raw_cell'
            n_prototypes: Number of prototypes per condition (for bootstrap)
            m_cells_per_prototype: Cells sampled per prototype
            library_seed: Seed for library construction
        """
        self.encoder_type = encoder_type
        self.encoder_kwargs = encoder_kwargs or {}
        self.metric = metric
        self.top_k = top_k
        self.mask_perturbed = mask_perturbed
        self.mask_layer_key = mask_layer_key
        self.library_type = library_type
        self.n_prototypes = n_prototypes
        self.m_cells_per_prototype = m_cells_per_prototype
        self.library_seed = library_seed
        self.query_mode = query_mode
        self.pseudo_bulk_config = pseudo_bulk_config or {}
        self.candidate_source = candidate_source
        self.query_split = query_split

        # Components (initialized during setup)
        self.encoder = None
        self.library_vectors: Optional[np.ndarray] = None
        self.library_labels: Optional[List[str]] = None
        self.candidate_conditions: Optional[List[str]] = None
        self._label_to_idx: Optional[Dict[str, int]] = None

    def setup(self, dataset: PerturbDataset) -> "CellRetrievalEvaluator":
        """
        Set up encoder and build reference library from ref cells.

        Args:
            dataset: Loaded PerturbDataset with split applied

        Returns:
            self
        """
        if dataset.split is None:
            raise RuntimeError("Dataset must have split applied")

        condition_sets = dataset.get_condition_sets()
        train_conditions = condition_sets["train"]

        # Get reference cells for training
        ref_adata = dataset.get_ref_adata_for_conditions(train_conditions)

        # Apply masking if requested
        if self.mask_perturbed:
            ref_adata = mask_perturbed_genes(ref_adata, layer=self.mask_layer_key)

        # Fit encoder on reference data
        self.encoder = get_encoder(self.encoder_type, **self.encoder_kwargs)
        X_ref = ref_adata.X
        if hasattr(X_ref, "toarray"):
            X_ref = X_ref.toarray()
        self.encoder.fit(X_ref)

        # Build reference library
        candidate_conditions = self._resolve_candidate_conditions(condition_sets)
        self._build_library(ref_adata, dataset, candidate_conditions)

        return self

    def _build_library(
        self,
        ref_adata,
        dataset: PerturbDataset,
        conditions: List[str],
    ) -> None:
        """Build reference library from encoded ref cells."""
        ref_for_library = dataset.get_ref_adata_for_conditions(conditions)
        if self.mask_perturbed:
            ref_for_library = mask_perturbed_genes(
                ref_for_library, layer=self.mask_layer_key
            )

        use_embedding_space = self.encoder_type == "scgpt" and hasattr(
            self.encoder, "encode_adata"
        )

        if use_embedding_space:
            cell_embeddings = self.encoder.encode_adata(ref_for_library)
            cell_labels = np.array(ref_for_library.obs["condition"].tolist())

            if self.library_type == "raw_cell":
                self.library_vectors = cell_embeddings
                self.library_labels = cell_labels.tolist()

            elif self.library_type in {"bootstrap", "mean"}:
                profiles = []
                labels = []
                rng = np.random.default_rng(self.library_seed)

                for cond in conditions:
                    if cond == "ctrl":
                        continue
                    cond_mask = cell_labels == cond
                    if not np.any(cond_mask):
                        continue
                    cond_embeddings = cell_embeddings[cond_mask]
                    n_cells = len(cond_embeddings)

                    if self.library_type == "mean":
                        profiles.append(cond_embeddings.mean(axis=0))
                        labels.append(cond)
                        continue

                    sample_size = self.m_cells_per_prototype or n_cells
                    for _ in range(self.n_prototypes):
                        indices = rng.choice(n_cells, size=sample_size, replace=True)
                        prototype = cond_embeddings[indices].mean(axis=0)
                        profiles.append(prototype)
                        labels.append(cond)

                if profiles:
                    self.library_vectors = np.vstack(profiles)
                    self.library_labels = labels

            else:
                raise ValueError(f"Unknown library_type: {self.library_type}")

        else:
            X = ref_for_library.X
            if hasattr(X, "toarray"):
                X = X.toarray()

            if self.library_type == "bootstrap":
                profiles, labels = build_prototype_library(
                    adata=ref_for_library,
                    conditions=conditions,
                    n_prototypes=self.n_prototypes,
                    m_cells_per_prototype=self.m_cells_per_prototype,
                    method="bootstrap",
                    seed=self.library_seed,
                )
                self.library_vectors = self._encode_profiles(profiles, ref_for_library)
                self.library_labels = labels

            elif self.library_type == "mean":
                profiles = []
                labels = []
                for cond in conditions:
                    mask = ref_for_library.obs["condition"] == cond
                    if mask.sum() == 0:
                        continue
                    cond_X = X[mask]
                    mean_profile = np.mean(cond_X, axis=0)
                    profiles.append(mean_profile)
                    labels.append(cond)

                if profiles:
                    profiles = np.vstack(profiles)
                    self.library_vectors = self._encode_profiles(
                        profiles, ref_for_library
                    )
                    self.library_labels = labels

            elif self.library_type == "raw_cell":
                self.library_vectors = self._encode_profiles(X, ref_for_library)
                self.library_labels = ref_for_library.obs["condition"].tolist()

            else:
                raise ValueError(f"Unknown library_type: {self.library_type}")

        if not self.library_labels:
            raise RuntimeError("Reference library is empty; check conditions/split.")

        self.candidate_conditions = sorted(set(self.library_labels))
        self._label_to_idx = {c: i for i, c in enumerate(self.candidate_conditions)}

        # Normalize for cosine similarity
        if self.metric == "cosine" and self.library_vectors is not None:
            norms = np.linalg.norm(self.library_vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self.library_vectors = self.library_vectors / norms

    def _score_candidates(self, query_embedding: np.ndarray) -> np.ndarray:
        """Score all candidate conditions for a single query embedding."""
        if self.library_vectors is None or self._label_to_idx is None:
            raise RuntimeError("Library not built. Call setup() first.")

        # Normalize query for cosine
        if self.metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            if norm > 1e-8:
                query_embedding = query_embedding / norm

        # Compute similarities
        if self.metric == "cosine":
            similarities = self.library_vectors @ query_embedding
        else:  # euclidean
            dists = np.linalg.norm(self.library_vectors - query_embedding, axis=1)
            similarities = -dists  # Negate for ranking

        # Aggregate by condition (max similarity)
        scores = np.full(len(self._label_to_idx), -np.inf)
        for label, score in zip(self.library_labels, similarities):
            idx = self._label_to_idx[label]
            if score > scores[idx]:
                scores[idx] = score

        return scores

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-K conditions for a single query.

        Uses max-similarity aggregation across prototypes.
        """
        scores = self._score_candidates(query_embedding)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.candidate_conditions[i], scores[i]) for i in top_indices]

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

    def _encode_profiles(
        self,
        profiles: np.ndarray,
        template_adata: ad.AnnData,
        layer_key: Optional[str] = None,
    ) -> np.ndarray:
        """Encode profiles with encoder, supporting scGPT AnnData inputs."""
        if hasattr(self.encoder, "encode_adata"):
            adata = ad.AnnData(X=profiles, var=template_adata.var.copy())
            if layer_key and layer_key != "X":
                adata.layers[layer_key] = profiles
            return self.encoder.encode_adata(adata)
        return self.encoder.encode(profiles)

    def _prepare_query_data(
        self, dataset: PerturbDataset, query_mode: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str], ad.AnnData]:
        """Prepare query embeddings and labels."""
        mode = query_mode or self.query_mode
        condition_sets = dataset.get_condition_sets()
        query_conditions = self._resolve_query_conditions(condition_sets)
        query_adata = dataset.get_query_adata_for_conditions(query_conditions)

        if self.mask_perturbed:
            query_adata = mask_perturbed_genes(query_adata, layer=self.mask_layer_key)

        if mode == "pseudobulk":
            cells_per_bulk = self.pseudo_bulk_config.get("cells_per_bulk", 50)
            n_bulks = self.pseudo_bulk_config.get("n_bulks", 1)
            seed = self.pseudo_bulk_config.get("seed", 42)
            layer_key = (
                self.mask_layer_key if hasattr(self.encoder, "encode_adata") else None
            )
            bulks, labels = create_pseudo_bulk(
                query_adata,
                cells_per_bulk=cells_per_bulk,
                n_bulks=n_bulks,
                condition_col="condition",
                seed=seed,
                layer=layer_key,
            )
            if bulks.size == 0:
                return np.array([]), [], query_adata
            embeddings = self._encode_profiles(bulks, query_adata, layer_key=layer_key)
            return embeddings, labels, query_adata

        if hasattr(self.encoder, "encode_adata"):
            embeddings = self.encoder.encode_adata(query_adata)
            ground_truth = query_adata.obs["condition"].tolist()
            return embeddings, ground_truth, query_adata

        X_query = query_adata.X
        if hasattr(X_query, "toarray"):
            X_query = X_query.toarray()
        embeddings = self._encode_profiles(X_query, query_adata)
        ground_truth = query_adata.obs["condition"].tolist()
        return embeddings, ground_truth, query_adata

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
        if self.encoder is None or self.library_vectors is None:
            raise RuntimeError("Must call setup() before evaluate()")

        query_embeddings, ground_truth, _ = self._prepare_query_data(dataset)
        if len(ground_truth) == 0:
            return {}

        # Retrieve for each query
        max_k = max(self.top_k)
        predictions = []
        all_scores = []
        for emb in query_embeddings:
            scores = self._score_candidates(emb)
            all_scores.append(scores)
            top_indices = np.argsort(scores)[::-1][:max_k]
            predictions.append([self.candidate_conditions[i] for i in top_indices])

        # Compute metrics
        metrics = compute_all_metrics(
            predictions,
            ground_truth,
            self.top_k,
            candidate_pool=self.candidate_conditions,
        )

        # Confidence reporting
        if confidence_config and confidence_config.get("enable", False):
            scorer = ConfidenceScorer(
                method=confidence_config.get("method", "margin"),
                top_k_agreement=confidence_config.get("top_k_agreement", 1),
            )
            all_scores_arr = np.vstack(all_scores)
            confidences = scorer.score_batch(all_scores_arr)
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
        query_embeddings, ground_truth, _ = self._prepare_query_data(dataset)
        if len(ground_truth) == 0:
            return {}, {"predictions": [], "ground_truth": []}

        max_k = max(self.top_k)
        predictions = []
        all_scores = []
        for emb in query_embeddings:
            scores = self._score_candidates(emb)
            all_scores.append(scores)
            top_indices = np.argsort(scores)[::-1][:max_k]
            predictions.append([self.candidate_conditions[i] for i in top_indices])

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
            all_scores_arr = np.vstack(all_scores)
            confidences = scorer.score_batch(all_scores_arr)
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
