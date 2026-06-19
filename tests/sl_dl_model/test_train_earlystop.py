"""Tests for per-epoch validation + patience-based best-epoch selection."""

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


def _producer(max_epochs: int, patience: int, warmup: int = 0) -> StateDlProducer:
    rng = np.random.default_rng(1)
    genes = [f"G{i}" for i in range(6)]
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
        warmup_epochs=warmup,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
        lambda_bag=0.0,
        batch_pairs=4,
        early_stop_patience=patience,
    )
    train_pairs = [
        ("G0", "G1", 1, -1.0, -0.9),
        ("G2", "G3", 0, 0.8, 0.7),
        ("G0", "G3", 0, -1.0, 0.7),
        ("G1", "G2", 1, -0.9, 0.8),
    ]
    val_pairs = [
        ("G4", "G5", 1, -1.0, -0.8),
        ("G4", "G0", 0, -1.0, -1.0),
    ]
    return StateDlProducer(
        cfg,
        esm=esm,
        bags=bags,
        train_pairs=train_pairs,
        input_dim=6,
        output_dim=6,
        val_pairs=val_pairs,
    )


def test_stopped_epoch_recorded_and_within_bounds():
    producer = _producer(max_epochs=5, patience=2)
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    producer._train(model, opt, state, {"G0", "G1", "G2", "G3"})
    assert producer.stopped_epoch is not None
    assert 0 <= producer.stopped_epoch < 5
    assert len(producer.epoch_metrics) >= 1
    assert "val_pair_auroc" in producer.epoch_metrics[0]


def test_patience_triggers_early_stop_and_restores_best(monkeypatch):
    """If patience=1 and val never improves after epoch 0, stop early."""
    producer = _producer(max_epochs=10, patience=1)
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    val_scores = iter([0.9, 0.8])
    best_param: dict[str, torch.Tensor] = {}

    def fake_validate(*_args):
        score = next(val_scores)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if score == 0.9:
                best_param[name] = param.detach().cpu().clone()
            else:
                with torch.no_grad():
                    param.add_(10.0)
            break
        return score

    monkeypatch.setattr(producer, "_validate_auroc", fake_validate)
    producer._train(model, opt, state, {"G0", "G1", "G2", "G3"})
    assert len(producer.epoch_metrics) == 2
    assert producer.stopped_epoch == 0
    name, expected = next(iter(best_param.items()))
    torch.testing.assert_close(
        dict(model.named_parameters())[name].detach().cpu(),
        expected,
    )


def test_warmup_epochs_do_not_select_best_epoch(monkeypatch):
    """Validation before warmup does not drive patience or best-epoch restore."""
    producer = _producer(max_epochs=5, patience=1, warmup=2)
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    val_scores = iter([0.99, 0.1, 0.2, 0.1])

    monkeypatch.setattr(producer, "_validate_auroc", lambda *_args: next(val_scores))
    producer._train(model, opt, state, {"G0", "G1", "G2", "G3"})
    assert len(producer.epoch_metrics) == 4
    assert producer.stopped_epoch == 2


def test_no_val_pairs_uses_final_epoch():
    """With val_pairs=None, no early stopping; stopped_epoch is the last epoch."""
    producer = _producer(max_epochs=3, patience=2)
    producer.val_pairs = None
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    producer._train(model, opt, state, {"G0", "G1", "G2", "G3"})
    assert producer.stopped_epoch == 2


def test_validation_uses_esm_fallback_for_missing_gene():
    """Validation should score fallback-covered genes like final scoring does."""
    producer = _producer(max_epochs=1, patience=1)
    del producer.esm.vectors_by_symbol["G5"]
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)
    control = torch.tensor(producer.bags.control_template, device=state.device)
    assert producer._validate_auroc(model, state.device, control) is not None
