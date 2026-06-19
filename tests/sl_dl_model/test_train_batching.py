"""Tests for gradient-accumulation batching in StateDlProducer._train."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from accelerate import PartialState

from sl_dl_model.config import SLDLConfig
from sl_dl_model.bags import GwpsBags
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
from sl_dl_model.train import StateDlProducer


def _producer(n_pairs: int, batch_pairs: int, max_epochs: int = 1) -> StateDlProducer:
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(n_pairs + 1)]
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={g: rng.standard_normal(8).astype("float32") for g in genes},
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={},
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        state_backend="linear_mock",
        max_epochs=max_epochs,
        warmup_epochs=0,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
        lambda_bag=0.0,
        batch_pairs=batch_pairs,
    )
    pairs = [
        (genes[i], genes[i + 1], i % 2, float(rng.normal()), float(rng.normal()))
        for i in range(n_pairs)
    ]
    return StateDlProducer(
        cfg, esm=esm, bags=bags, train_pairs=pairs, input_dim=6, output_dim=6
    )


def test_one_step_per_batch_not_per_pair():
    """With 10 pairs and batch_pairs=4, expect ceil(10/4)=3 steps in 1 epoch."""
    producer = _producer(n_pairs=10, batch_pairs=4, max_epochs=1)
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)

    steps = {"n": 0}
    real_step = torch.optim.Adam.step

    class CountingAdam(torch.optim.Adam):
        def step(self, *a, **k):
            steps["n"] += 1
            return real_step(self, *a, **k)

    opt = CountingAdam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    producer._train(model, opt, state, {"G0", "G1"})
    assert steps["n"] == 3, f"expected 3 optimizer steps, got {steps['n']}"


def test_tqdm_advances_once_per_batch(monkeypatch):
    """The epoch progress bar ticks once per optimizer step, not per pair.

    10 pairs, batch_pairs=4, 1 epoch -> ceil(10/4)=3 batches -> total=3 and
    exactly 3 .update(1) calls (not 10).
    """
    import sl_dl_model.train as train_mod

    constructed: list[dict] = []

    class _FakeBar:
        def __init__(self, *args, total=None, **kwargs):
            self.updates = 0
            constructed.append({"total": total, "bar": self})

        def update(self, n: int = 1) -> None:
            self.updates += n

        def close(self) -> None:
            pass

    monkeypatch.setattr(train_mod, "tqdm", _FakeBar)

    producer = _producer(n_pairs=10, batch_pairs=4, max_epochs=1)
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)
    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )
    producer._train(model, opt, state, {"G0", "G1"})

    # Exactly one bar constructed for the single training epoch.
    assert len(constructed) == 1, f"expected 1 bar, got {len(constructed)}"
    assert constructed[0]["total"] == 3, f"expected total=3, got {constructed[0]['total']}"
    assert constructed[0]["bar"].updates == 3, (
        f"expected 3 per-batch updates, got {constructed[0]['bar'].updates}"
    )
