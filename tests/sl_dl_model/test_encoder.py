"""Tests for PertAdapter and frozen-STATE StateEncoder (Task 1.3)."""

import torch

from sl_dl_model.encoder import PertAdapter, StateEncoder


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
