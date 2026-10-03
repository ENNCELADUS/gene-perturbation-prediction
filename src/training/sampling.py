"""Training epochs: sharded or gene-blocked GeneEffect rows, balanced responses."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler

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


class GeneBlockSampler(Sampler[list[int]]):
    """Batches of every training row of ``genes_per_block`` genes, for one rank.

    Each epoch draws one permutation of the genes that have rows, seeded by the
    epoch alone so every rank sees the same order, and cuts it into blocks. Blocks
    are dealt to ranks round-robin; an incomplete block and the blocks that do not
    fill a whole round are dropped, so every gene appears at most once per epoch
    and every rank takes the same number of updates.
    """

    def __init__(
        self,
        rows_by_gene: Sequence[np.ndarray],
        *,
        genes_per_block: int,
        epoch: int,
        rank: int,
        world: int,
    ) -> None:
        if genes_per_block < 1:
            raise ValueError("genes_per_block must be positive")
        self.rows_by_gene = rows_by_gene
        self.genes = np.flatnonzero([len(rows) for rows in rows_by_gene])
        self.genes_per_block = genes_per_block
        self.epoch, self.rank, self.world = epoch, rank, world

    def __len__(self) -> int:
        return len(self.genes) // self.genes_per_block // self.world

    def __iter__(self) -> Iterator[list[int]]:
        order = np.random.default_rng(
            np.random.SeedSequence([0, self.epoch])
        ).permutation(self.genes)
        blocks = order[: len(self) * self.world * self.genes_per_block].reshape(
            -1, self.genes_per_block
        )
        for block in blocks[self.rank :: self.world]:
            yield np.concatenate([self.rows_by_gene[gene] for gene in block]).tolist()


def dependency_loader(
    dataset: DependencyDataset,
    config: Mapping[str, Any],
    epoch: int,
    accelerator: Any,
) -> DataLoader[DependencyBatch]:
    """One shuffled GeneEffect epoch of training batches for this rank.

    ``pearson_blocks`` takes gene blocks (``GeneBlockSampler``); every other
    objective takes ``dependency_batch_size`` rows from a seeded
    ``DistributedSampler`` with the incomplete tail dropped. Either way every rank
    takes the same number of updates. Do not pass the loader through
    ``accelerator.prepare``. The dataset keeps its tables on the rank's device, so
    the loader collates in-process, without workers.
    """
    train = config["train"]
    rank, world = accelerator.process_index, accelerator.num_processes
    if train["objective"] == "pearson_blocks":
        sampler = GeneBlockSampler(
            dataset.rows_by_gene(),
            genes_per_block=train["genes_per_block"],
            epoch=epoch,
            rank=rank,
            world=world,
        )
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate)
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True, seed=0, drop_last=True
    )
    sampler.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_size=train["dependency_batch_size"],
        sampler=sampler,
        drop_last=True,
        collate_fn=dataset.collate,
        generator=torch.Generator().manual_seed(epoch * world + rank),
    )


def response_stream(
    inputs: PreparedInputs,
    config: Mapping[str, Any],
    epoch: int,
    accelerator: Any,
) -> Iterator[ResponseBatch] | None:
    """This rank's endless response batches, or ``None`` without response replay."""
    train = config["train"]
    if train["response_weight"] <= 0:
        return None
    return balanced_responses(
        ResponseDataset(inputs, device=accelerator.device),
        batch_size=train["response_batch_size"],
        epoch=epoch,
        rank=accelerator.process_index,
    )
