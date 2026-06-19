"""NaN-defense guards for exp08 Phase 3 training (H1-H4)."""

from __future__ import annotations

import torch
from torch import nn, optim

from sl_dl_model.train import safe_optimizer_step


def test_safe_optimizer_step_applies_finite_step():
    model = nn.Linear(3, 1)
    opt = optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    loss = (model(torch.ones(2, 3)) - 2.0).pow(2).mean()
    applied = safe_optimizer_step(model, opt, loss, max_grad_norm=1.0)
    assert applied is True
    assert not torch.equal(before, model.weight), "finite step must update weights"
    assert torch.isfinite(model.weight).all()


def test_safe_optimizer_step_skips_nonfinite_loss():
    model = nn.Linear(3, 1)
    opt = optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    # NaN loss that still carries grad_fn back to the params.
    loss = (model(torch.ones(2, 3)) * float("nan")).mean()
    applied = safe_optimizer_step(model, opt, loss, max_grad_norm=1.0)
    assert applied is False, "non-finite loss must skip the step"
    assert torch.equal(before, model.weight), "weights must be unchanged on skip"
    assert torch.isfinite(model.weight).all(), "weights must stay finite"


def test_train_with_bag_supervision_keeps_params_finite():
    import numpy as np

    from sl_dl_model.bags import GwpsBags
    from sl_dl_model.config import SLDLConfig
    from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
    from sl_dl_model.train import StateDlProducer

    rng = np.random.default_rng(0)
    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={
            s: rng.standard_normal(8).astype("float32") for s in symbols
        },
    )
    # Identical pred/real-style bags + a constant-feature bag exercise the H1
    # self-distance and H3 zero-variance paths inside a real training loop.
    bags = GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={
            "A": np.zeros((8, 6), dtype="float32"),
            "B": rng.standard_normal((8, 6)).astype("float32"),
        },
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        max_epochs=2,
        warmup_epochs=1,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        state_backend="linear_mock",
        lambda_bag=1.0,
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
    )
    pairs = [
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
        val_pairs=pairs,
    )
    emb, _mask = producer.produce(symbols, {"A", "B", "C", "D"})
    assert np.isfinite(emb).all(), "produced embeddings must be finite"
    model = producer._model
    assert model is not None
    assert all(torch.isfinite(p).all() for p in model.parameters()), (
        "all trained params must remain finite"
    )
    assert all(np.isfinite(row["mean_train_loss"]) for row in producer.epoch_metrics), (
        "recorded train losses must be finite"
    )


def test_validate_auroc_returns_none_on_nonfinite_scores(monkeypatch):
    import numpy as np

    from sl_dl_model.bags import GwpsBags
    from sl_dl_model.config import SLDLConfig
    from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
    from sl_dl_model.train import StateDlProducer

    rng = np.random.default_rng(0)
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={
            s: rng.standard_normal(8).astype("float32") for s in ["A", "B"]
        },
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={"A": rng.standard_normal((8, 6)).astype("float32")},
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        state_backend="linear_mock",
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
    )
    val = [("A", "B", 1, -1.0, -0.5), ("B", "A", 0, 0.1, 0.2)]
    producer = StateDlProducer(
        cfg,
        esm=esm,
        bags=bags,
        train_pairs=val,
        input_dim=6,
        output_dim=6,
        val_pairs=val,
    )
    model = producer._build_model()
    control = torch.zeros(8, 6)
    # Force the pair head to emit a NaN logit so sigmoid -> NaN score.
    monkeypatch.setattr(
        model,
        "score_pairs",
        lambda *a, **k: torch.tensor([float("nan")]),
    )
    assert producer._validate_auroc(model, "cpu", control) is None
