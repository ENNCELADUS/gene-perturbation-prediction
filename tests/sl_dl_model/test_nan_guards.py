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
