"""
Retrieval evaluation pipeline for reverse perturbation prediction.

End-to-end evaluation: data → encoder → retrieval → metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.data import NormanDataset, create_pseudo_bulk, mask_perturbed_genes
from src.model import get_encoder, RetrievalHead
from src.evaluate.metrics import compute_all_metrics


class RetrievalEvaluator:
    """End-to-end evaluation pipeline for perturbation retrieval."""

    def __init__(
        self,
        encoder_type: str = "pca",
        encoder_kwargs: Optional[dict] = None,
        metric: str = "cosine",
        top_k: List[int] = [1, 5, 10, 20],
        mask_perturbed: bool = True,
        pseudo_bulk: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            encoder_type: Type of encoder ('pca', 'scgpt')
            encoder_kwargs: Encoder-specific arguments
            metric: Similarity metric
            top_k: K values for evaluation
            mask_perturbed: Whether to mask perturbed gene expression
            pseudo_bulk: Whether to use pseudo-bulk aggregation
        """
        self.encoder_type = encoder_type
        self.encoder_kwargs = encoder_kwargs or {}
        self.metric = metric
        self.top_k = top_k
        self.mask_perturbed = mask_perturbed
        self.pseudo_bulk = pseudo_bulk

        # Components
        self.encoder = None
        self.retrieval = None

    def setup(self, dataset: NormanDataset) -> "RetrievalEvaluator":
        """
        Set up encoder and retrieval head using training data.

        Args:
            dataset: Loaded NormanDataset

        Returns:
            self
        """
        adata = dataset.adata

        # Get training conditions
        train_conditions = dataset.conditions.get("train", [])

        # Filter to training data
        train_mask = adata.obs["condition"].isin(train_conditions)
        train_data = adata[train_mask].copy()

        # Apply preprocessing
        if self.mask_perturbed:
            train_data = mask_perturbed_genes(train_data)

        if self.pseudo_bulk:
            train_data = create_pseudo_bulk(train_data)

        # Fit encoder
        self.encoder = get_encoder(self.encoder_type, **self.encoder_kwargs)
        X_train = train_data.X
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        self.encoder.fit(X_train)

        # Build reference library
        embeddings = self.encoder.encode(X_train)
        conditions = train_data.obs["condition"].tolist()

        self.retrieval = RetrievalHead(metric=self.metric, top_k=self.top_k)
        self.retrieval.build_library(embeddings, conditions)

        return self

    def evaluate(
        self,
        dataset: NormanDataset,
        split: str = "test",
    ) -> Dict[str, float]:
        """
        Evaluate on a dataset split.

        Args:
            dataset: NormanDataset
            split: Split to evaluate ('test', 'val')

        Returns:
            Dictionary of metrics
        """
        if self.encoder is None or self.retrieval is None:
            raise RuntimeError("Must call setup() before evaluate()")

        adata = dataset.adata

        # Get split conditions
        split_conditions = dataset.conditions.get(split, [])

        # Prepare test data
        test_mask = adata.obs["condition"].isin(split_conditions)
        test_data = adata[test_mask].copy()

        if self.mask_perturbed:
            test_data = mask_perturbed_genes(test_data)

        if self.pseudo_bulk:
            test_data = create_pseudo_bulk(test_data)

        # Encode and retrieve
        X_test = test_data.X
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()

        embeddings = self.encoder.encode(X_test)
        ground_truth = test_data.obs["condition"].tolist()

        # Get predictions
        max_k = max(self.top_k)
        predictions = []
        for emb in embeddings:
            results = self.retrieval.retrieve(emb, k=max_k)
            predictions.append([cond for cond, _ in results])

        # Compute metrics
        metrics = compute_all_metrics(predictions, ground_truth, self.top_k)

        return metrics

    def save_results(
        self,
        metrics: Dict[str, float],
        output_dir: str,
        experiment_name: str,
    ) -> Path:
        """
        Save evaluation results.

        Args:
            metrics: Metrics dictionary
            output_dir: Output directory
            experiment_name: Experiment name

        Returns:
            Path to saved metrics file
        """
        output_path = Path(output_dir) / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)

        metrics_file = output_path / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics_file
