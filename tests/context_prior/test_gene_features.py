"""Own, paralog, complex and low-partner features; data-selected genes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.context_prior.gene_features import (
    knowledge_features,
    partner_index,
    select_genes,
)
from src.context_prior.reference import Reference

REFERENCE = Reference(
    paralogs=pd.DataFrame(
        {
            "gene": ["A", "A", "B"],
            "paralog": ["B", "C", "A"],
            "identity": [60.0, 40.0, 60.0],
        }
    ),
    complexes=pd.DataFrame({"complex_id": [1, 1, 1], "gene": ["A", "C", "D"]}),
    hallmark=pd.DataFrame(columns=["gene_set", "gene"]),
    progeny=pd.DataFrame(columns=["pathway", "gene", "weight"]),
    drivers=pd.DataFrame(),
    msi=pd.Series(dtype=float),
)


def test_knowledge_features_and_an_unmeasured_gene():
    space = ["A", "B", "C", "D"]
    genes = ["A", "Q"]  # Q has no expression column and no partners
    index = partner_index(genes, space, REFERENCE)
    expression = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 1.0]])
    features = knowledge_features(expression, index, low_threshold=np.zeros(4))
    a = features[:, 0]
    assert a[0].tolist() == [1.0, 2.0, 5.0, 2.0, 3.5, 0.0]
    assert np.isclose(a[1, 5], 1.0 / 3.0)  # B (paralog) is at its threshold; C, D not
    assert np.isnan(features[:, 1]).all()


def test_selected_genes_exclude_the_gene_itself_and_survive_constant_columns():
    rng = np.random.default_rng(0)
    expression = rng.normal(size=(50, 6))
    expression[:, 5] = 2.0  # constant column
    residual = np.column_stack(
        [expression[:, 0] + 0.1 * rng.normal(size=50), expression[:, 3]]
    )
    selection = select_genes(expression, residual, 2, own=np.array([0, 3]))
    assert 0 not in selection[0] and 3 not in selection[1]
    assert np.isfinite(selection).all()
