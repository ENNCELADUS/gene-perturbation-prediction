"""
Differential-Expression (DE) baseline for reverse perturbation prediction.

A training-free heuristic that uses Wilcoxon rank-sum test to find
differentially expressed genes between query cells and control cells,
then predicts the condition based on top upregulated genes.

For CRISPRa datasets like Norman, target genes are typically the most upregulated.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy import sparse
from scipy.stats import ranksums
import warnings


def parse_condition_genes(condition: str) -> Set[str]:
    """Extract gene names from condition string."""
    if not condition or condition == "ctrl":
        return set()
    genes = condition.split("+")
    return {g.strip() for g in genes if g.strip() and g.strip() != "ctrl"}


class DEBaseline:
    """
    Differential-Expression baseline for reverse perturbation prediction.

    For each query, performs Wilcoxon rank-sum test vs control cells,
    identifies top-K most upregulated genes, and predicts condition
    based on which candidate has the highest overlap with those genes.
    """

    def __init__(self, top_n_de_genes: int = 2):
        """
        Initialize DE baseline.

        Args:
            top_n_de_genes: Number of top upregulated genes to use for prediction
                           (1 for single-gene, 2 for double-gene conditions)
        """
        self.top_n_de_genes = top_n_de_genes
        self.control_expr: Optional[np.ndarray] = None
        self.gene_names: Optional[List[str]] = None
        self.gene_name_to_idx: Optional[Dict[str, int]] = None
        self._fitted = False

    def fit(self, control_adata) -> "DEBaseline":
        """
        Store control cell expression for DE testing.

        Args:
            control_adata: AnnData object containing control cells
                          (X should be log-normalized expression)

        Returns:
            self
        """
        X = control_adata.X
        if sparse.issparse(X):
            X = X.toarray()

        self.control_expr = X

        # Store gene names for mapping
        if "gene_name" in control_adata.var.columns:
            self.gene_names = control_adata.var["gene_name"].tolist()
        else:
            self.gene_names = control_adata.var_names.tolist()

        self.gene_name_to_idx = {g: i for i, g in enumerate(self.gene_names)}

        self._fitted = True
        print(
            f"DEBaseline fitted on {control_adata.n_obs} control cells, {len(self.gene_names)} genes"
        )

        return self

    def compute_de_scores(
        self, query_expr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute differential expression using Wilcoxon rank-sum test.

        Args:
            query_expr: Expression matrix for query cells, shape (n_cells, n_genes)

        Returns:
            Tuple of (log2_fc, pvalues) arrays, shape (n_genes,)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before compute_de_scores()")

        n_genes = self.control_expr.shape[1]
        pvalues = np.ones(n_genes)
        log2_fc = np.zeros(n_genes)

        # Compute log2 fold change (query mean / control mean)
        query_mean = np.mean(query_expr, axis=0)
        ctrl_mean = np.mean(self.control_expr, axis=0)

        # Avoid division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log2_fc = query_mean - ctrl_mean  # Already log-transformed

        # Wilcoxon test for each gene
        for i in range(n_genes):
            ctrl_vals = self.control_expr[:, i]
            query_vals = query_expr[:, i]

            # Skip if all values are the same
            if np.std(ctrl_vals) < 1e-10 and np.std(query_vals) < 1e-10:
                continue

            try:
                _, p = ranksums(query_vals, ctrl_vals, alternative="greater")
                pvalues[i] = p
            except Exception:
                pass

        return log2_fc, pvalues

    def get_top_upregulated_genes(
        self,
        query_expr: np.ndarray,
        top_k: int = 2,
    ) -> List[Tuple[str, float, float]]:
        """
        Get top-K most upregulated genes.

        Args:
            query_expr: Expression matrix for query cells
            top_k: Number of top genes to return

        Returns:
            List of (gene_name, log2_fc, pvalue) tuples
        """
        log2_fc, pvalues = self.compute_de_scores(query_expr)

        # Rank by log2_fc (most upregulated first)
        top_indices = np.argsort(log2_fc)[::-1][:top_k]

        return [(self.gene_names[i], log2_fc[i], pvalues[i]) for i in top_indices]

    def predict(
        self,
        query_adata,
        candidate_conditions: List[str],
        top_k: int = 10,
    ) -> List[str]:
        """
        Predict top-K conditions for query cells.

        Strategy:
        1. Find top-N most upregulated genes via DE analysis
        2. For each candidate condition, compute overlap with top DE genes
        3. Rank candidates by overlap score

        Args:
            query_adata: AnnData object containing query cells
            candidate_conditions: List of candidate condition strings
            top_k: Number of top conditions to return

        Returns:
            List of top-K predicted condition strings
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        # Get query expression
        X = query_adata.X
        if sparse.issparse(X):
            X = X.toarray()

        # Get top upregulated genes
        top_de = self.get_top_upregulated_genes(X, top_k=self.top_n_de_genes)
        top_de_genes = {gene for gene, _, _ in top_de}

        # Score conditions by overlap with DE genes
        condition_scores = []
        for condition in candidate_conditions:
            cond_genes = parse_condition_genes(condition)
            if not cond_genes:
                continue

            # Score = number of overlapping genes
            overlap = len(cond_genes & top_de_genes)

            # Tie-breaker: sum of log2_fc for condition genes
            fc_sum = 0.0
            for gene in cond_genes:
                if gene in self.gene_name_to_idx:
                    idx = self.gene_name_to_idx[gene]
                    fc_sum += np.mean(X[:, idx]) - np.mean(self.control_expr[:, idx])

            condition_scores.append((condition, overlap, fc_sum))

        # Sort by overlap (primary), then by fc_sum (secondary)
        condition_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

        return [cond for cond, _, _ in condition_scores[:top_k]]

    def predict_with_de_genes(
        self,
        query_adata,
        candidate_conditions: List[str],
        top_k: int = 10,
    ) -> Tuple[List[str], List[Tuple[str, float, float]]]:
        """
        Predict conditions and also return the DE genes used.

        Returns:
            Tuple of (predictions, de_genes)
        """
        X = query_adata.X
        if sparse.issparse(X):
            X = X.toarray()

        top_de = self.get_top_upregulated_genes(X, top_k=self.top_n_de_genes)
        predictions = self.predict(query_adata, candidate_conditions, top_k)

        return predictions, top_de
