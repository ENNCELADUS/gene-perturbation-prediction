"""Per-gene view weights from gene embeddings, learned on out-of-fold stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.context_prior.view_weights import fit_view_weights


def test_view_weights_follow_the_embedding():
    rng = np.random.default_rng(0)
    lines = [f"L{i}" for i in range(200)]
    genes = [f"G{i}" for i in range(40)]
    first = pd.DataFrame(rng.normal(size=(200, 40)), index=lines, columns=genes)
    second = pd.DataFrame(rng.normal(size=(200, 40)), index=lines, columns=genes)
    kind = np.array([i % 2 for i in range(40)])  # even genes follow view one
    truth = np.where(kind == 0, 2 * first, 2 * second)
    residual = pd.DataFrame(truth, index=lines, columns=genes)
    embeddings = {
        g: np.array([1.0, 0.0]) if k == 0 else np.array([0.0, 1.0])
        for g, k in zip(genes, kind, strict=True)
    }
    weights = fit_view_weights(
        {"expression_components": first, "pathway_scores": second},
        residual,
        embeddings,
        ("expression_components", "pathway_scores"),
    )
    assert weights.weights.loc["G0", "expression_components"] > 1.5
    assert weights.weights.loc["G1", "pathway_scores"] > 1.5
    combined = weights.combine(
        {"expression_components": first, "pathway_scores": second}
    )
    assert combined.shape == first.shape
