"""Pure helpers of the harness's fast paths: no released module, only torch and numpy.

They reproduce, bit for bit, what the released code computes through per-item
Datasets, default DataLoaders and batch-size-1 inference loops, by slicing resident
tensors instead (docs/specs/2026-10-08-cilantro-sl-reproduction-design.md §4.4):

- ``sliced_loader`` is ``DataLoader(dataset, batch_size, shuffle=True)`` with the
  per-item ``__getitem__`` plus ``default_collate`` replaced by one tensor gather per
  batch. The permutation comes from the same ``RandomSampler`` under the same
  ``DataLoader`` iterator, so the global torch RNG is consumed exactly as before and
  every batch holds the same rows in the same order.
- ``predict_rows`` runs a network on one row at a time, as the released loops do with
  ``batch_size=1`` loaders. Batched forward passes of these nets are not bitwise equal
  to batch-size-1 passes on the GPU (cuBLAS picks kernels by shape; measured max abs
  difference 6e-8 on the classifier's logits, 3.6e-7 on stage one's penultimate
  layer), so batching is not used.
- ``line_order`` and ``delete_token`` replicate the released perturber's line
  selection (filter by token, stable sort by length, longest first) and deletion (the
  token's first occurrence).
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler


class _Sliced(Dataset):
    """Indexed by a list of row indices; returns the gathered rows."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)


def sliced_loader(
    features: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> DataLoader:
    """The released shuffled training loader, fetching each batch by one gather."""
    data = _Sliced(features, labels)
    return DataLoader(
        data,
        batch_size=None,
        sampler=BatchSampler(RandomSampler(data), batch_size, drop_last=False),
    )


def consume_loader_seed() -> None:
    """Draw what a released ``DataLoader`` draws from the global torch RNG when it is
    iterated: every ``_BaseDataLoaderIter`` takes one int64 base seed, shuffled or not.
    The fast paths iterate no loader over test, calibration or conformal rows, so they
    call this once per released loop to leave the RNG stream where the released code
    leaves it (the next fold's network is initialised from it)."""
    torch.empty((), dtype=torch.int64).random_()


def predict_rows(
    forward: Callable[[int], Sequence[torch.Tensor]], n: int
) -> list[torch.Tensor]:
    """Call ``forward(i)`` for every row and concatenate each returned tensor over
    rows (dim 0). Nothing is synchronised inside the loop."""
    parts = None
    with torch.no_grad():
        for i in range(n):
            outputs = forward(i)
            if parts is None:
                parts = [[] for _ in outputs]
            for part, output in zip(parts, outputs):
                part.append(output)
    if parts is None:
        return []
    return [torch.cat(part) for part in parts]


def line_order(lengths: Sequence[int]) -> list[int]:
    """Dataset rows sorted by length, longest first, ties in dataset order: the order
    the released ``downsample_and_sort`` (a stable Arrow sort) produces."""
    return sorted(range(len(lengths)), key=lambda i: -lengths[i])


def lines_with_token(
    order: Sequence[int], token_sets: Sequence[set[int]], token: int
) -> list[int]:
    """The rows containing ``token``, in ``order``: filtering keeps dataset order and
    the stable sort then gives the same sequence as sorting the filtered rows."""
    return [i for i in order if token in token_sets[i]]


def delete_token(input_ids: Sequence[int], token: int) -> list[int]:
    """The released deletion: remove the first occurrence of ``token``."""
    ids = list(input_ids)
    del ids[ids.index(token)]
    return ids
