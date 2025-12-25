"""
Target-Gene Activation (TGA) baseline for reverse perturbation prediction.

A training-free heuristic that uses the signal that activated genes
show increased expression to predict perturbation conditions.
(For CRISPRa datasets like Norman where target genes are upregulated)

Algorithm:
1. Compute gene-level statistics (mean, std) from control cells
2. For query cells, compute z-score activation: (x_query - μ_ctrl) / (σ_ctrl + ε)
3. Score conditions: single = s(g), double = s(g1) + s(g2)
4. Rank conditions by score and return top-K
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy import sparse


def parse_condition_genes(condition: str) -> Set[str]:
    """Extract gene names from condition string (e.g., 'GENE1+GENE2' -> {'GENE1', 'GENE2'})."""
    if not condition or condition == "ctrl":
        return set()
    genes = condition.split("+")
    return {g.strip() for g in genes if g.strip() and g.strip() != "ctrl"}


class TGA:
    """
    Target-Gene Activation (TGA) baseline for reverse perturbation prediction.

    This is a training-free heuristic baseline that predicts perturbation
    conditions based on the observation that activated genes (CRISPRa)
    show increased expression.
    """

    def __init__(self, epsilon: float = 1e-6):
        """
        Initialize TGA baseline.

        Args:
            epsilon: Small constant for numerical stability in z-score computation
        """
        self.epsilon = epsilon
        self.gene_means: Optional[np.ndarray] = None
        self.gene_stds: Optional[np.ndarray] = None
        self.gene_names: Optional[List[str]] = None
        self.gene_name_to_idx: Optional[Dict[str, int]] = None
        self._fitted = False

    def fit(self, control_adata) -> "TGA":
        """
        Compute gene-level statistics from control cells.

        Args:
            control_adata: AnnData object containing control cells
                          (X should be log-normalized expression)

        Returns:
            self
        """
        # Extract expression matrix
        X = control_adata.X
        if sparse.issparse(X):
            X = X.toarray()

        # Compute per-gene statistics
        self.gene_means = np.mean(X, axis=0)
        self.gene_stds = np.std(X, axis=0)

        # Store gene names for mapping
        if "gene_name" in control_adata.var.columns:
            self.gene_names = control_adata.var["gene_name"].tolist()
        else:
            self.gene_names = control_adata.var_names.tolist()

        self.gene_name_to_idx = {g: i for i, g in enumerate(self.gene_names)}

        self._fitted = True
        print(
            f"TGA fitted on {control_adata.n_obs} control cells, {len(self.gene_names)} genes"
        )

        return self

    def score_genes(self, query_expr: np.ndarray) -> np.ndarray:
        """
        Compute z-score based activation scores for all genes.

        Args:
            query_expr: Expression values for query, shape (n_genes,)
                       Should be pseudobulk (averaged) for multiple cells

        Returns:
            Array of activation scores, shape (n_genes,)
            Higher score = more upregulated = more likely to be a target gene
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before score_genes()")

        # z-score activation: (x_query - μ_ctrl) / (σ_ctrl + ε)
        # Positive = upregulated, Negative = downregulated
        scores = (query_expr - self.gene_means) / (self.gene_stds + self.epsilon)
        return scores

    def score_conditions(
        self,
        gene_scores: np.ndarray,
        candidate_conditions: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Score candidate conditions based on gene-level activation scores.

        Args:
            gene_scores: Activation scores for all genes
            candidate_conditions: List of condition strings to score

        Returns:
            List of (condition, score) tuples, sorted by score descending
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before score_conditions()")

        condition_scores = []

        for condition in candidate_conditions:
            genes = parse_condition_genes(condition)
            if not genes:
                continue

            # Sum scores for all genes in the condition
            score = 0.0
            valid_genes = 0
            for gene in genes:
                if gene in self.gene_name_to_idx:
                    idx = self.gene_name_to_idx[gene]
                    score += gene_scores[idx]
                    valid_genes += 1

            # Only include if at least one gene found
            if valid_genes > 0:
                condition_scores.append((condition, score))

        # Sort by score descending
        condition_scores.sort(key=lambda x: x[1], reverse=True)
        return condition_scores

    def predict(
        self,
        query_adata,
        candidate_conditions: List[str],
        top_k: int = 10,
        use_pseudobulk: bool = True,
    ) -> List[str]:
        """
        Predict top-K conditions for query cells.

        Args:
            query_adata: AnnData object containing query cells
            candidate_conditions: List of candidate condition strings
            top_k: Number of top conditions to return
            use_pseudobulk: Whether to average query cells before scoring

        Returns:
            List of top-K predicted condition strings
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        # Extract query expression
        X = query_adata.X
        if sparse.issparse(X):
            X = X.toarray()

        # Compute pseudobulk if requested
        if use_pseudobulk:
            query_expr = np.mean(X, axis=0)
        else:
            # Use first cell (for single-cell prediction)
            query_expr = X[0]

        # Score genes
        gene_scores = self.score_genes(query_expr)

        # Score and rank conditions
        ranked_conditions = self.score_conditions(gene_scores, candidate_conditions)

        # Return top-K
        return [cond for cond, _ in ranked_conditions[:top_k]]

    def predict_batch(
        self,
        query_adatas: Dict[str, any],  # condition -> adata
        candidate_conditions: List[str],
        top_k: int = 10,
        use_pseudobulk: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Predict top-K conditions for multiple query groups.

        Args:
            query_adatas: Dict mapping condition name to AnnData of query cells
            candidate_conditions: List of candidate condition strings
            top_k: Number of top conditions to return
            use_pseudobulk: Whether to average query cells before scoring

        Returns:
            Dict mapping condition name to list of top-K predictions
        """
        predictions = {}
        for condition, adata in query_adatas.items():
            predictions[condition] = self.predict(
                adata,
                candidate_conditions,
                top_k=top_k,
                use_pseudobulk=use_pseudobulk,
            )
        return predictions

    def get_gene_rankings(
        self, query_adata, top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get top-K genes by activation score (for debugging/interpretation).

        Args:
            query_adata: AnnData object containing query cells
            top_k: Number of top genes to return

        Returns:
            List of (gene_name, score) tuples
        """
        X = query_adata.X
        if sparse.issparse(X):
            X = X.toarray()

        query_expr = np.mean(X, axis=0)
        gene_scores = self.score_genes(query_expr)

        # Get top-K indices
        top_indices = np.argsort(gene_scores)[::-1][:top_k]

        return [(self.gene_names[i], gene_scores[i]) for i in top_indices]
