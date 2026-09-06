"""Deterministic gene-major batching for the Stage-2 Pearson objective."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math

import numpy as np


def _hash_rank(seed: int, epoch: int, gene: int, context: int) -> bytes:
    return hashlib.sha256(f"{seed}|{epoch}|{gene}|{context}".encode()).digest()


@dataclass(frozen=True)
class GeneContextRow:
    gene_index: int
    context_indices: tuple[int, ...]
    label_mask: tuple[bool, ...]


@dataclass(frozen=True)
class GeneContextBatchIndex:
    """One fixed-shape ``[genes, contexts]`` dependency minibatch."""

    rows: tuple[GeneContextRow, ...]
    is_rank_padding: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        if not self.rows:
            raise ValueError("gene-context batch cannot be empty")
        widths = {len(row.context_indices) for row in self.rows}
        if len(widths) != 1 or any(
            len(row.label_mask) != len(row.context_indices) for row in self.rows
        ):
            raise ValueError("gene-context rows must have one fixed aligned width")
        return len(self.rows), next(iter(widths))

    @property
    def objective_weight(self) -> float:
        return 0.0 if self.is_rank_padding else 1.0

    def flattened_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (row.gene_index, context)
            for row in self.rows
            for context in row.context_indices
        )

    def mask_array(self) -> np.ndarray:
        return np.asarray([row.label_mask for row in self.rows], dtype=bool)

    def as_rank_padding(self) -> GeneContextBatchIndex:
        return replace(self, is_rank_padding=True)


def build_epoch_batches(
    label_mask: np.ndarray,
    g_var_mask: np.ndarray,
    *,
    genes_per_batch: int = 8,
    contexts_per_gene: int = 32,
    seed: int = 20_260_828,
    epoch: int = 0,
) -> tuple[GeneContextBatchIndex, ...]:
    """Cover each labeled pair once while keeping Pearson genes on every batch."""
    labels = np.asarray(label_mask)
    g_var = np.asarray(g_var_mask)
    if labels.ndim != 2 or labels.dtype != bool:
        raise ValueError("label_mask must be a boolean [genes, contexts] array")
    if g_var.shape != (labels.shape[0],) or g_var.dtype != bool:
        raise ValueError("g_var_mask must be boolean [genes]")
    if genes_per_batch <= 0 or contexts_per_gene < 3:
        raise ValueError("batch widths must be positive and contexts_per_gene >= 3")
    if not labels.any() or not g_var.any():
        raise ValueError("sampler requires labeled pairs and G_var genes")

    chunks: list[GeneContextRow] = []
    g_var_chunks: list[GeneContextRow] = []
    for gene_index in range(labels.shape[0]):
        contexts = np.flatnonzero(labels[gene_index]).tolist()
        if not contexts:
            continue
        contexts.sort(key=lambda context: _hash_rank(seed, epoch, gene_index, context))
        for offset in range(0, len(contexts), contexts_per_gene):
            real = contexts[offset : offset + contexts_per_gene]
            padded = list(real)
            while len(padded) < contexts_per_gene:
                padded.append(real[(len(padded) - len(real)) % len(real)])
            row = GeneContextRow(
                gene_index=gene_index,
                context_indices=tuple(padded),
                label_mask=tuple(
                    [True] * len(real) + [False] * (len(padded) - len(real))
                ),
            )
            chunks.append(row)
            if g_var[gene_index] and len(real) >= 3:
                g_var_chunks.append(row)

    batch_count = math.ceil(len(chunks) / genes_per_batch)
    if len(g_var_chunks) < batch_count:
        raise ValueError(
            "not enough scorable G_var chunks to place one on every batch: "
            f"need {batch_count}, have {len(g_var_chunks)}"
        )
    g_var_chunks.sort(
        key=lambda row: _hash_rank(seed, epoch, row.gene_index, row.context_indices[0])
    )
    anchors = g_var_chunks[:batch_count]
    anchor_ids = {id(row) for row in anchors}
    remaining = [row for row in chunks if id(row) not in anchor_ids]
    remaining.sort(
        key=lambda row: _hash_rank(
            seed + 1, epoch, row.gene_index, row.context_indices[0]
        )
    )

    batches: list[GeneContextBatchIndex] = []
    cursor = 0
    for anchor in anchors:
        rows = [anchor]
        rows.extend(remaining[cursor : cursor + genes_per_batch - 1])
        cursor += genes_per_batch - 1
        while len(rows) < genes_per_batch:
            rows.append(
                GeneContextRow(
                    gene_index=anchor.gene_index,
                    context_indices=anchor.context_indices,
                    label_mask=tuple(False for _ in anchor.label_mask),
                )
            )
        batches.append(GeneContextBatchIndex(rows=tuple(rows)))
    if cursor < len(remaining):
        raise RuntimeError("internal sampler packing error: unassigned gene chunks")
    batches.sort(
        key=lambda batch: _hash_rank(
            seed + 2, epoch, batch.rows[0].gene_index, batch.rows[0].context_indices[0]
        )
    )
    return tuple(batches)


def shard_batches(
    batches: tuple[GeneContextBatchIndex, ...], *, rank: int, world_size: int
) -> tuple[GeneContextBatchIndex, ...]:
    """Even-step DDP sharding with explicit zero-weight padding batches."""
    if not batches:
        raise ValueError("cannot shard an empty batch sequence")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid DDP rank/world_size")
    target = math.ceil(len(batches) / world_size) * world_size
    padded = list(batches)
    for index in range(target - len(batches)):
        padded.append(batches[index % len(batches)].as_rank_padding())
    shard = tuple(padded[rank::world_size])
    if len(shard) != target // world_size:
        raise RuntimeError("DDP sampler produced uneven rank step counts")
    return shard
