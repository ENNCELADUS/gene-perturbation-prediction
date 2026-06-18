"""Tests for PertAdapter and frozen-STATE StateEncoder (Task 1.3)."""

from pathlib import Path

import torch
from torch import nn

from sl_dl_model.encoder import PertAdapter, StateEncoder


class _StubStateModel(nn.Module):
    """Faithful stand-in for a loaded STATE checkpoint.

    Mirrors the real ``StateTransitionPerturbationModel`` interface the exp08
    adapter depends on: a raw ``pert_dim`` distinct from the encoder's
    ``hidden_dim``, a ``pert_encoder`` mapping raw pert space to hidden, and a
    ``predict_step`` that reshapes ``pert_emb`` to ``(1, -1, pert_dim)`` exactly
    as the real ``forward`` does (the line that crashes on a width mismatch).
    """

    def __init__(
        self, *, pert_dim: int, input_dim: int, output_dim: int, hidden: int
    ) -> None:
        super().__init__()
        self.pert_dim = int(pert_dim)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden)
        self.pert_encoder = nn.Linear(self.pert_dim, self.hidden_dim)
        self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim)
        self.batch_encoder = None

    def predict_step(self, batch, batch_idx: int = 0, padded: bool = False):  # noqa: ANN001
        del batch_idx, padded
        pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
        basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)
        hidden = self.pert_encoder(pert) + self.basal_encoder(basal)
        return self.decoder(hidden)


def _patch_loader(monkeypatch, model: nn.Module) -> None:
    """Force ``StateEncoder`` to load ``model`` instead of a real checkpoint."""
    import sl_dl_model.encoder as encoder_mod

    monkeypatch.setattr(encoder_mod, "load_state_model", lambda **_: model)


def test_adapter_width_inferred_from_checkpoint_pert_dim(monkeypatch) -> None:
    """Adapter output width follows the checkpoint's pert_dim, not the config.

    The config carries pert_dim=5 (the misleading default) but the checkpoint
    expects raw 7-wide pert vectors; the adapter must emit width 7.
    """
    stub = _StubStateModel(pert_dim=7, input_dim=6, output_dim=6, hidden=3)
    _patch_loader(monkeypatch, stub)
    enc = StateEncoder(
        backend="state_checkpoint",
        checkpoint=Path("/fake/checkpoints/final.ckpt"),
        esm_dim=8,
        adapter_hidden=16,
        pert_dim=5,
        input_dim=6,
        output_dim=6,
    )
    assert enc.pert_dim == 7, f"expected inferred pert_dim 7, got {enc.pert_dim}"
    out = enc.adapter(torch.randn(2, 8))
    assert out.shape == (2, 7), f"adapter must emit checkpoint width, got {out.shape}"


def test_encoder_forward_uses_checkpoint_pert_dim(monkeypatch) -> None:
    """forward feeds a checkpoint-width pert through predict_step without error.

    With the config width (5) the expanded pert is (10, 5) = 50 elements, which
    cannot reshape to (1, -1, 7); the fix must size the adapter to 7 so the bag
    comes back as (10, output_dim).
    """
    stub = _StubStateModel(pert_dim=7, input_dim=6, output_dim=6, hidden=3)
    _patch_loader(monkeypatch, stub)
    enc = StateEncoder(
        backend="state_checkpoint",
        checkpoint=Path("/fake/checkpoints/final.ckpt"),
        esm_dim=8,
        adapter_hidden=16,
        pert_dim=5,
        input_dim=6,
        output_dim=6,
    )
    bag = enc(torch.randn(8), torch.randn(10, 6))
    assert bag.shape == (10, 6), f"expected (10, 6), got {bag.shape}"


def test_config_pert_dim_mismatch_warns(monkeypatch, caplog) -> None:
    """A config pert_dim disagreeing with the checkpoint is logged, not silent."""
    import logging

    stub = _StubStateModel(pert_dim=7, input_dim=6, output_dim=6, hidden=3)
    _patch_loader(monkeypatch, stub)
    with caplog.at_level(logging.WARNING):
        StateEncoder(
            backend="state_checkpoint",
            checkpoint=Path("/fake/checkpoints/final.ckpt"),
            esm_dim=8,
            adapter_hidden=16,
            pert_dim=5,
            input_dim=6,
            output_dim=6,
        )
    assert any("pert_dim" in rec.message for rec in caplog.records), (
        "expected a warning about the config/checkpoint pert_dim mismatch"
    )


def test_pert_adapter_shapes():
    adapter = PertAdapter(esm_dim=8, hidden=16, pert_dim=5)
    out = adapter(torch.randn(3, 8))
    assert out.shape == (3, 5)


def test_state_encoder_forward_with_mock_backend():
    enc = StateEncoder(
        backend="linear_mock",
        checkpoint=None,
        esm_dim=8,
        adapter_hidden=16,
        pert_dim=5,
        input_dim=6,
        output_dim=6,
    )
    esm_vec = torch.randn(8)
    control = torch.randn(10, 6)
    bag = enc(esm_vec, control)
    assert bag.shape == (10, 6)


def test_backbone_frozen_adapter_trainable():
    enc = StateEncoder(
        backend="linear_mock",
        checkpoint=None,
        esm_dim=8,
        adapter_hidden=16,
        pert_dim=5,
        input_dim=6,
        output_dim=6,
    )
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert all(n.startswith("adapter") for n in trainable)
    assert any(n.startswith("adapter") for n in trainable)
