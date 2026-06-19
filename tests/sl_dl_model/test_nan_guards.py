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


def _bag_producer(max_epochs=1, warmup_epochs=0, **cfg_overrides):
    """Build a linear_mock producer with bag supervision on, for guard tests."""
    import numpy as np

    from sl_dl_model.bags import GwpsBags
    from sl_dl_model.config import SLDLConfig
    from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
    from sl_dl_model.train import StateDlProducer

    rng = np.random.default_rng(0)
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={
            s: rng.standard_normal(8).astype("float32") for s in ["A", "B", "C", "D"]
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
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        lambda_bag=1.0,
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
        **cfg_overrides,
    )
    pairs = [("A", "B", 1, -1.0, -0.5), ("C", "D", 0, 0.1, 0.2)]
    return StateDlProducer(
        cfg, esm=esm, bags=bags, train_pairs=pairs, input_dim=6, output_dim=6,
        val_pairs=pairs,
    )


def test_epoch_records_optimizer_step_counts():
    """Finding 1: epoch metrics expose applied/skipped optimizer-step counts."""
    import numpy as np

    producer = _bag_producer(max_epochs=1, warmup_epochs=1)
    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    producer.produce(symbols, {"A", "B", "C", "D"})
    row = producer.epoch_metrics[0]
    assert "optimizer_steps_applied" in row
    assert "optimizer_steps_skipped" in row
    assert row["optimizer_steps_applied"] >= 1, "at least one step must apply"


def test_train_raises_when_all_steps_skipped(monkeypatch):
    """Finding 1: an epoch with zero applied steps must fail the fold."""
    import numpy as np

    import sl_dl_model.train as train_mod

    producer = _bag_producer(max_epochs=1, warmup_epochs=1)
    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    # Force every optimizer step to be skipped (simulates persistent non-finite).
    monkeypatch.setattr(train_mod, "safe_optimizer_step", lambda *a, **k: False)
    try:
        producer.produce(symbols, {"A", "B", "C", "D"})
    except RuntimeError as exc:
        assert "no optimizer step" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError when all steps skipped")


def test_validate_auroc_status_distinguishes_nonfinite_from_insufficient(monkeypatch):
    """Finding 2: non-finite validation is a distinct status, not 'no labels'."""
    producer = _bag_producer()
    model = producer._build_model()
    control = torch.zeros(8, 6)

    # Single-class val -> insufficient_labels, value None.
    producer.val_pairs = [("A", "B", 1, -1.0, -0.5), ("B", "A", 1, 0.1, 0.2)]
    value, status = producer._validate_auroc_with_status(model, "cpu", control)
    assert value is None and status == "insufficient_labels"

    # Non-finite scores -> non_finite, value None.
    producer.val_pairs = [("A", "B", 1, -1.0, -0.5), ("B", "A", 0, 0.1, 0.2)]
    monkeypatch.setattr(
        model, "score_pairs", lambda *a, **k: torch.tensor([float("nan")])
    )
    value, status = producer._validate_auroc_with_status(model, "cpu", control)
    assert value is None and status == "non_finite"


def test_train_raises_on_persistent_nonfinite_validation(monkeypatch):
    """Finding 2: a fold whose post-warmup validation is non-finite must fail."""
    import numpy as np

    producer = _bag_producer(max_epochs=2, warmup_epochs=0)
    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    # Force validation to always report a non-finite status.
    monkeypatch.setattr(
        producer,
        "_validate_auroc_with_status",
        lambda *a, **k: (None, "non_finite"),
    )
    try:
        producer.produce(symbols, {"A", "B", "C", "D"})
    except RuntimeError as exc:
        assert "non-finite" in str(exc).lower() or "non_finite" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError on persistent non-finite val")

