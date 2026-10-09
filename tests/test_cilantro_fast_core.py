"""Pure-logic tests of the Cilantro harness's fast paths (hpc/cilantro/fast_core.py):
the sliced training loader reproduces the released per-item DataLoader's batches and
RNG consumption, and the line selection and deletion match the released rules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

spec = importlib.util.spec_from_file_location(
    "fast_core",
    Path(__file__).resolve().parents[1] / "hpc" / "cilantro" / "fast_core.py",
)
fast_core = importlib.util.module_from_spec(spec)
sys.modules["fast_core"] = fast_core
spec.loader.exec_module(fast_core)


class PerItem(Dataset):
    """The released dataset_classifier's access pattern."""

    def __init__(self, features, labels):
        self.features, self.labels = features, labels

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx, :]

    def __len__(self):
        return len(self.features)


def _epochs(loader, n):
    torch.manual_seed(0)
    batches = []
    for _ in range(n):
        batches.append([(x.clone(), y.clone()) for x, y in loader])
    return batches, torch.get_rng_state()


@pytest.mark.parametrize("rows", [1000, 1003])
def test_sliced_loader_matches_per_item_dataloader(rows):
    rng = np.random.default_rng(0)
    features = torch.from_numpy(rng.standard_normal((rows, 8)))
    labels = torch.from_numpy(rng.integers(0, 2, (rows, 1)).astype(float))
    released = DataLoader(PerItem(features, labels), batch_size=128, shuffle=True)
    fast = fast_core.sliced_loader(features, labels, 128)
    assert len(released) == len(fast)
    ref, ref_state = _epochs(released, 2)
    out, out_state = _epochs(fast, 2)
    for epoch_ref, epoch_out in zip(ref, out):
        assert len(epoch_ref) == len(epoch_out)
        for (x_ref, y_ref), (x_out, y_out) in zip(epoch_ref, epoch_out):
            assert torch.equal(x_ref, x_out)
            assert torch.equal(y_ref, y_out)
    assert torch.equal(ref_state, out_state)
    # The two epochs are different permutations (the sampler reseeds per epoch).
    assert not torch.equal(out[0][0][0], out[1][0][0])


def test_predict_rows_concatenates_each_output():
    x = torch.arange(10.0).reshape(5, 2)
    (doubled, summed) = fast_core.predict_rows(
        lambda i: (x[i : i + 1] * 2, x[i : i + 1].sum(dim=1)), 5
    )
    assert torch.equal(doubled, x * 2)
    assert torch.equal(summed, x.sum(dim=1))
    assert fast_core.predict_rows(lambda i: (x[i],), 0) == []


def test_line_order_is_stable_longest_first():
    assert fast_core.line_order([5, 7, 5, 7, 3]) == [1, 3, 0, 2, 4]


def test_lines_with_token_keeps_order():
    order = [1, 3, 0, 2, 4]
    token_sets = [{1, 2}, {2}, {1}, {1, 2}, {3}]
    assert fast_core.lines_with_token(order, token_sets, 1) == [3, 0, 2]
    assert fast_core.lines_with_token(order, token_sets, 9) == []


def test_delete_token_removes_first_occurrence_only():
    assert fast_core.delete_token([4, 8, 15, 8], 8) == [4, 15, 8]
    with pytest.raises(ValueError):
        fast_core.delete_token([4, 8], 9)
