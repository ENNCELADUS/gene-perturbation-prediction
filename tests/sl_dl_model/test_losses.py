"""Tests for the three-part loss assembly (Task 2.2)."""

import torch

from sl_dl_model.losses import bag_loss, combine, distill_loss, sl_bce_loss


def test_sl_bce_decreases_with_correct_logits():
    labels = torch.tensor([1.0, 0.0, 1.0])
    good = sl_bce_loss(torch.tensor([5.0, -5.0, 5.0]), labels)
    bad = sl_bce_loss(torch.tensor([-5.0, 5.0, -5.0]), labels)
    assert good < bad


def test_distill_zero_when_equal():
    t = torch.randn(4, 8)
    assert distill_loss(t, t.clone()).item() < 1e-8


def test_bag_loss_nonnegative_and_zero_for_identical():
    bag = torch.randn(16, 6)
    assert bag_loss(bag, bag.clone()).item() < 1e-4


def test_bag_loss_grad_finite_on_identical_bags():
    # H1b: cdist(x, x) self-distance has a 0/0 backward; identical pred/real
    # makes every cross- and self-distance zero, the exact NaN-grad trigger.
    pred = torch.randn(8, 6, requires_grad=True)
    real = pred.detach().clone()
    loss = bag_loss(pred, real)
    assert torch.isfinite(loss).all(), "bag_loss value must be finite"
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), (
        "bag_loss gradient must be finite on identical bags"
    )


def test_bag_loss_grad_finite_with_duplicate_rows():
    # Duplicate rows create zero pairwise distances inside a single bag.
    pred = torch.zeros(5, 4, requires_grad=True)
    real = torch.randn(7, 4)
    loss = bag_loss(pred, real)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_combine_weights():
    parts = {"sl": torch.tensor(2.0), "distill": torch.tensor(4.0)}
    weights = {"sl": 1.0, "distill": 0.5}
    total = combine(parts, weights)
    assert abs(total.item() - 4.0) < 1e-6


def test_combine_zero_weight_ignores_nonfinite():
    # Warmup: SL weight 0 while a (hypothetically) non-finite SL term exists.
    parts = {"sl": torch.tensor(float("nan")), "bag": torch.tensor(2.0)}
    weights = {"sl": 0.0, "bag": 1.0}
    total = combine(parts, weights)
    assert torch.isfinite(total).all()
    assert abs(total.item() - 2.0) < 1e-6
