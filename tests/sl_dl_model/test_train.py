"""Tests for SlDlModel and StateDlProducer (Task 2.3).

All tests use state_backend="linear_mock" — no checkpoint required.
PYTORCH_ENABLE_MPS_FALLBACK is set at module import time so that the energy-
distance backward (aten::_cdist_backward) falls back to CPU on MPS-only Macs.
"""

from __future__ import annotations

import os

# Force MPS ops that lack a native kernel to fall back to CPU.
# Must be set before importing torch (or at least before any MPS dispatch).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from sl_dl_model.config import SLDLConfig
from sl_dl_model.model import SlDlModel


def _model(esm_dim: int = 8, input_dim: int = 6) -> SlDlModel:
    return SlDlModel(
        backend="linear_mock",
        checkpoint=None,
        esm_dim=esm_dim,
        adapter_hidden=16,
        pert_dim=5,
        input_dim=input_dim,
        output_dim=input_dim,
        pooling="mean_std",
        pair_hidden=(16,),
        include_coverage_flag=False,
    )


def test_embed_gene_shape() -> None:
    """embed_gene returns (2*output_dim,) via mean_std pooling."""
    model = _model().eval()
    e_g = model.embed_gene(torch.randn(8), torch.randn(10, 6))
    assert e_g.shape == (12,), f"expected (12,), got {e_g.shape}"


def test_score_pairs_shape_and_backprop() -> None:
    """score_pairs returns (B,) logits and gradients flow back through e_a."""
    model = _model()
    e_a = torch.randn(4, 12, requires_grad=True)
    e_b = torch.randn(4, 12)
    ge = torch.randn(4, 5)
    logits = model.score_pairs(e_a, e_b, ge)
    assert logits.shape == (4,), f"expected (4,), got {logits.shape}"
    logits.sum().backward()
    assert e_a.grad is not None, "gradient did not flow back through e_a"


def test_producer_emits_universe_table(tmp_path) -> None:
    """StateDlProducer.produce returns (n_gene, emb_dim) embeddings + mask."""
    from sl_dl_model.bags import GwpsBags
    from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
    from sl_dl_model.train import StateDlProducer

    rng = np.random.default_rng(0)
    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={
            s: rng.standard_normal(8).astype("float32") for s in ["A", "B", "C", "D"]
        },
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={
            "A": rng.standard_normal((8, 6)).astype("float32"),
            "B": rng.standard_normal((8, 6)).astype("float32"),
        },
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        max_epochs=1,
        warmup_epochs=1,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        state_backend="linear_mock",
    )
    # 5-tuples: (gene_a, gene_b, label, ea, eb)
    pairs: list[tuple[str, str, int, float, float]] = [
        ("A", "B", 1, -1.0, -0.5),
        ("C", "D", 0, 0.1, 0.2),
        ("A", "C", 0, -1.0, 0.1),
    ]
    producer = StateDlProducer(
        cfg,
        esm=esm,
        bags=bags,
        train_pairs=pairs,
        input_dim=6,
        output_dim=6,
    )
    emb, mask = producer.produce(symbols, {"A", "B", "C", "D"})
    assert emb.shape[0] == 4, f"expected 4 rows, got {emb.shape[0]}"
    assert mask.shape == (4,), f"expected mask shape (4,), got {mask.shape}"
    # All 4 genes have ESM2 vectors so mask should be all-ones
    assert mask.sum() == 4, f"expected all 4 covered, got mask={mask}"
