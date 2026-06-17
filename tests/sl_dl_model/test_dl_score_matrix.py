# tests/sl_dl_model/test_dl_score_matrix.py
"""Test StateDlProducer.score_matrix shape and diagonal-zero constraint."""

import numpy as np

from sl_dl_model.bags import GwpsBags
from sl_dl_model.config import SLDLConfig
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
from sl_dl_model.train import StateDlProducer


def test_score_matrix_diag_zero_and_shape():
    symbols = np.array(["A", "B", "C"], dtype=object)
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={
            s: np.random.randn(8).astype("float32") for s in ["A", "B", "C"]
        },
    )
    bags = GwpsBags(
        control_template=np.random.randn(8, 6).astype("float32"),
        bags_by_symbol={},
        input_dim=6,
    )
    cfg = SLDLConfig(
        max_epochs=1,
        warmup_epochs=0,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        state_backend="linear_mock",
    )
    pairs = [("A", "B", 1, -1.0, -0.5), ("B", "C", 0, -0.5, 0.2)]
    producer = StateDlProducer(
        cfg,
        esm=esm,
        bags=bags,
        train_pairs=pairs,
        input_dim=6,
        output_dim=6,
    )
    gene_effects = np.array([-1.0, -0.5, 0.2])
    sm = producer.score_matrix(symbols, gene_effects)
    assert sm.shape == (3, 3)
    assert np.allclose(np.diag(sm), 0.0)
