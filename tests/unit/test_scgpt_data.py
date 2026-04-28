from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from src.scgpt.data import GeneScoreDataset


def test_gene_score_dataset_matches_normalized_norman_single_gene_conditions() -> None:
    """Norman stores equivalent single-gene perturbations in both token orders."""
    adata = ad.AnnData(
        X=np.asarray(
            [
                [1.0, 1.0],
                [5.0, 1.0],
                [4.8, 1.2],
                [1.0, 5.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(
            {
                "condition": ["ctrl", "A+ctrl", "ctrl+A", "B+ctrl"],
                "control": [1, 0, 0, 0],
            },
            index=["cell_ctrl", "cell_a_first", "cell_ctrl_first", "cell_b"],
        ),
        var=pd.DataFrame({"gene_name": ["A", "B"]}, index=["gene_a", "gene_b"]),
    )
    vocab = {"<pad>": 0, "A": 1, "B": 2}

    dataset = GeneScoreDataset(
        adata=adata,
        conditions=["A"],
        vocab=vocab,
        n_control_samples=1,
    )

    assert len(dataset) == 2
    assert [condition for _, condition in dataset.examples] == ["A", "A"]
    assert dataset.example_gene_indices == [[0], [0]]
