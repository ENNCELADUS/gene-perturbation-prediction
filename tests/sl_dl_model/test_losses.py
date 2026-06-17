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


def test_combine_weights():
    parts = {"sl": torch.tensor(2.0), "distill": torch.tensor(4.0)}
    weights = {"sl": 1.0, "distill": 0.5}
    total = combine(parts, weights)
    assert abs(total.item() - 4.0) < 1e-6
