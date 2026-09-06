"""Cell-bag transfer preserves ragged boundaries, values and gradients."""

import pytest
import torch

from src.data.batches import ResponseForwardBatch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ragged_transfer_values_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    bags = tuple(torch.randn(n, 5, requires_grad=True) for n in (3, 8, 2))
    batch = ResponseForwardBatch(bags, ("A", "B", "C"))
    moved = batch.to(device)
    assert moved.genes == batch.genes
    for source, actual in zip(bags, moved.controls_tx1, strict=True):
        assert actual.shape == source.shape
        torch.testing.assert_close(actual.cpu(), source)
    sum(value.square().sum() for value in moved.controls_tx1).backward()
    for source in bags:
        torch.testing.assert_close(source.grad, 2 * source.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_ragged_bags_use_one_host_to_device_transfer(monkeypatch):
    batch = ResponseForwardBatch(
        tuple(torch.randn(n, 5) for n in (3, 8, 2)), ("A", "B", "C")
    )
    original = torch.Tensor.to
    copied = []

    def counted(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if value.device.type == "cpu" and result.device.type == "cuda":
            copied.append(value.numel())
        return result

    monkeypatch.setattr(torch.Tensor, "to", counted)
    moved = batch.to("cuda")
    assert copied == [sum(value.numel() for value in batch.controls_tx1)]
    # Repeated .to calls by evaluator/wrappers should allocate nothing.
    again = moved.to("cuda")
    for first, second in zip(moved.controls_tx1, again.controls_tx1, strict=True):
        assert first is second
