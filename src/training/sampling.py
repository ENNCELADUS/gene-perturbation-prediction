"""Training epochs: sharded GeneEffect batches and anchor-balanced response batches."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.data.batches import DependencyBatch, ResponseBatch
from src.data.datasets import DependencyDataset, ResponseDataset
from src.data.prepared import PreparedInputs


def balanced_responses(
    dataset: ResponseDataset, *, batch_size: int, epoch: int, rank: int
) -> Iterator[ResponseBatch]:
    """Endless batches with ``batch_size // anchors`` conditions from every anchor.

    Each anchor cycles through all of its conditions in a fresh seeded
    permutation per pass; there is no held-out condition.
    """
    anchors = dataset.inputs.response_anchors
    pools = [
        np.asarray([i for i, (m, _) in enumerate(dataset.keys) if m == anchor])
        for anchor in anchors
    ]
    for anchor, pool in zip(anchors, pools, strict=True):
        if not len(pool):
            raise ValueError(f"response anchor {anchor} has no prepared conditions")
    generators = [
        np.random.default_rng(np.random.SeedSequence([0, epoch, rank, position]))
        for position in range(len(anchors))
    ]
    orders = [rng.permutation(pool) for rng, pool in zip(generators, pools)]
    positions = [0] * len(anchors)
    per_anchor = batch_size // len(anchors)
    while True:
        indices: list[int] = []
        for anchor in range(len(anchors)):
            for _ in range(per_anchor):
                if positions[anchor] == len(orders[anchor]):
                    orders[anchor] = generators[anchor].permutation(pools[anchor])
                    positions[anchor] = 0
                indices.append(int(orders[anchor][positions[anchor]]))
                positions[anchor] += 1
        yield dataset.collate(indices)


def make_training_loaders(
    inputs: PreparedInputs,
    config: Mapping[str, Any],
    epoch: int,
    accelerator: Any,
) -> tuple[DataLoader[DependencyBatch], Iterator[ResponseBatch]]:
    """One shuffled GeneEffect epoch for this rank and its endless response batches.

    GeneEffect rows are split across ranks by a seeded ``DistributedSampler`` with
    the incomplete tail dropped, so every rank takes the same number of updates.
    Do not pass the loader through ``accelerator.prepare``. The datasets keep their
    tables on the rank's device, so the loader collates in-process, without workers.
    """
    train = config["train"]
    rank, world = accelerator.process_index, accelerator.num_processes
    dependency = DependencyDataset(inputs, "train", device=accelerator.device)
    sampler = DistributedSampler(
        dependency, num_replicas=world, rank=rank, shuffle=True, seed=0, drop_last=True
    )
    sampler.set_epoch(epoch)
    loader = DataLoader(
        dependency,
        batch_size=train["dependency_batch_size"],
        sampler=sampler,
        drop_last=True,
        collate_fn=dependency.collate,
        generator=torch.Generator().manual_seed(epoch * world + rank),
    )
    responses = balanced_responses(
        ResponseDataset(inputs, device=accelerator.device),
        batch_size=train["response_batch_size"],
        epoch=epoch,
        rank=rank,
    )
    return loader, responses
