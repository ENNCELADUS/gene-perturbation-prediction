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

from src.data import PerturbDataset, mask_perturbed_genes, build_prototype_library
from src.model import get_encoder
from .metrics import compute_all_metrics


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
        library_type: str = "bootstrap",
        n_prototypes: int = 30,
        m_cells_per_prototype: int = 50,
        library_seed: int = 42,
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
        self.library_type = library_type
        self.n_prototypes = n_prototypes
        self.m_cells_per_prototype = m_cells_per_prototype
        self.library_seed = library_seed

        # Components (initialized during setup)
        self.encoder = None
        self.library_vectors: Optional[np.ndarray] = None
        self.library_labels: Optional[List[str]] = None

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

        # Get reference cells
        ref_adata = dataset.ref_adata

        # Apply masking if requested
        if self.mask_perturbed:
            ref_adata = mask_perturbed_genes(ref_adata)

        # Fit encoder on reference data
        self.encoder = get_encoder(self.encoder_type, **self.encoder_kwargs)
        X_ref = ref_adata.X
        if hasattr(X_ref, "toarray"):
            X_ref = X_ref.toarray()
        self.encoder.fit(X_ref)

        # Build reference library
        self._build_library(ref_adata, dataset.split.conditions)

        return self

    def _build_library(
        self,
        ref_adata,
        conditions: List[str],
    ) -> None:
        """Build reference library from encoded ref cells."""
        X = ref_adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        if self.library_type == "bootstrap":
            # Build bootstrap prototypes
            profiles, labels = build_prototype_library(
                adata=ref_adata,
                conditions=conditions,
                n_prototypes=self.n_prototypes,
                method="bootstrap",
                seed=self.library_seed,
            )
            # Encode prototypes
            self.library_vectors = self.encoder.encode(profiles)
            self.library_labels = labels

        elif self.library_type == "mean":
            # One mean prototype per condition
            profiles = []
            labels = []
            for cond in conditions:
                mask = ref_adata.obs["condition"] == cond
                if mask.sum() == 0:
                    continue
                cond_X = X[mask]
                mean_profile = np.mean(cond_X, axis=0)
                profiles.append(mean_profile)
                labels.append(cond)

            if profiles:
                profiles = np.vstack(profiles)
                self.library_vectors = self.encoder.encode(profiles)
                self.library_labels = labels

        elif self.library_type == "raw_cell":
            # Use all ref cell embeddings directly
            self.library_vectors = self.encoder.encode(X)
            self.library_labels = ref_adata.obs["condition"].tolist()

        else:
            raise ValueError(f"Unknown library_type: {self.library_type}")

        # Normalize for cosine similarity
        if self.metric == "cosine" and self.library_vectors is not None:
            norms = np.linalg.norm(self.library_vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self.library_vectors = self.library_vectors / norms

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-K conditions for a single query.

        Uses max-similarity aggregation across prototypes.

        Args:
            query_embedding: Single query embedding (1D)
            k: Number of conditions to return

        Returns:
            List of (condition, score) tuples
        """
        if self.library_vectors is None:
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
        condition_scores = defaultdict(float)
        for label, score in zip(self.library_labels, similarities):
            condition_scores[label] = max(condition_scores[label], score)

        # Sort and return top-K
        sorted_conditions = sorted(
            condition_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_conditions[:k]

    def evaluate(self, dataset: PerturbDataset) -> Dict[str, float]:
        """
        Evaluate on query cells.

        Args:
            dataset: PerturbDataset with split applied

        Returns:
            Dictionary of metrics
        """
        if self.encoder is None or self.library_vectors is None:
            raise RuntimeError("Must call setup() before evaluate()")

        # Get query cells
        query_adata = dataset.query_adata

        # Apply masking if requested
        if self.mask_perturbed:
            query_adata = mask_perturbed_genes(query_adata)

        # Encode queries
        X_query = query_adata.X
        if hasattr(X_query, "toarray"):
            X_query = X_query.toarray()

        query_embeddings = self.encoder.encode(X_query)
        ground_truth = query_adata.obs["condition"].tolist()

        # Retrieve for each query
        max_k = max(self.top_k)
        predictions = []
        for emb in query_embeddings:
            results = self.retrieve(emb, k=max_k)
            predictions.append([cond for cond, _ in results])

        # Compute metrics
        candidate_pool = list(set(self.library_labels))
        metrics = compute_all_metrics(
            predictions,
            ground_truth,
            self.top_k,
            candidate_pool=candidate_pool,
        )

        return metrics

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
