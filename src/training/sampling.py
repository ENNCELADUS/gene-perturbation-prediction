"""Deterministic dependency epochs and independently cycling anchor replay."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.data.batches import DependencyBatch, ResponseBatch
from src.data.datasets import DependencyDataset, ResponseDataset, _train_config
from src.data.prepared import PreparedInputs


def _balanced_replay(
    dataset: ResponseDataset, *, batch_size: int, epoch: int, rank: int
) -> Iterator[ResponseBatch]:
    per_anchor = batch_size // 4
    pools = [
        np.asarray(
            [
                index
                for index in dataset.indices
                if dataset.cache.model_ids[index] == anchor
            ],
            dtype=np.int64,
        )
        for anchor in dataset.inputs.response_anchors
    ]
    generators = [
        np.random.default_rng(np.random.SeedSequence([0, epoch, rank, anchor]))
        for anchor in range(4)
    ]
    orders = [
        rng.permutation(pool) for rng, pool in zip(generators, pools, strict=True)
    ]
    positions = [0] * 4
    while True:
        batch_indices: list[int] = []
        for anchor, (pool, rng) in enumerate(zip(pools, generators, strict=True)):
            for _ in range(per_anchor):
                if positions[anchor] == len(orders[anchor]):
                    orders[anchor] = rng.permutation(pool)
                    positions[anchor] = 0
                batch_indices.append(int(orders[anchor][positions[anchor]]))
                positions[anchor] += 1
        yield dataset.collate(batch_indices)


def make_training_loaders(
    inputs: PreparedInputs,
    config: Mapping[str, Any],
    epoch: int,
    accelerator: Any,
) -> tuple[DataLoader[DependencyBatch], Iterator[ResponseBatch]]:
    """Build one full-batch dependency epoch and unlimited equal-anchor replay.

    Dependency rows use DistributedSampler exactly once; do not subsequently pass
    this loader through Accelerate.prepare/prepare_data_loader. Evaluation loaders
    instead use Accelerate sharding and its per-iteration gather_for_metrics tail
    trimming. Replay is a plain iterator and never changes Accelerate loader state.
    """
    train = _train_config(config)
    seeds = config.get("seeds", {})
    if not isinstance(seeds, Mapping) or any(
        seeds.get(name, 0) != 0 for name in ("train", "collator", "projection")
    ):
        raise ValueError("runtime train, collator, and projection base seeds must be 0")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    dependency_size = int(train.get("dependency_batch_size", 256))
    response_size = int(train.get("response_batch_size", 64))
    if dependency_size <= 0:
        raise ValueError("dependency_batch_size must be positive")
    if response_size <= 0 or response_size % 4:
        raise ValueError("response_batch_size must be positive and divisible by four")
    rank = int(accelerator.process_index) if accelerator is not None else 0
    world_size = int(accelerator.num_processes) if accelerator is not None else 1
    dependency_dataset = DependencyDataset(inputs, "train")
    response_dataset = ResponseDataset(inputs, holdout=False)
    anchors = inputs.response_anchors
    if len(anchors) != 4 or len(set(anchors)) != 4:
        raise ValueError("response replay requires four distinct anchors")
    for anchor in anchors:
        if not any(
            response_dataset.cache.model_ids[index] == anchor
            for index in response_dataset.indices
        ):
            raise ValueError(
                f"response replay anchor {anchor} has no training conditions"
            )
    sampler = DistributedSampler(
        dependency_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    sampler.set_epoch(epoch)
    loader = DataLoader(
        dependency_dataset,
        batch_size=dependency_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=dependency_dataset.collate,
        generator=torch.Generator().manual_seed(epoch * world_size + rank),
    )
    if len(loader) == 0:
        raise ValueError(
            "dependency training has no full batch per rank; "
            "reduce dependency_batch_size"
        )
    return loader, _balanced_replay(
        response_dataset, batch_size=response_size, epoch=epoch, rank=rank
    )


__all__ = ["make_training_loaders"]
