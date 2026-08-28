from __future__ import annotations

import numpy as np
import pytest

from aivc_model.geneeffect_sampler import build_epoch_batches, shard_batches


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    labels = np.ones((16, 35), dtype=bool)
    labels[3, -2:] = False
    labels[9, -4:] = False
    g_var = np.zeros(16, dtype=bool)
    g_var[::2] = True
    return labels, g_var


def test_every_labeled_pair_is_covered_once_and_padding_is_masked() -> None:
    labels, g_var = _fixture()
    batches = build_epoch_batches(
        labels, g_var, genes_per_batch=8, contexts_per_gene=32
    )
    counts = np.zeros_like(labels, dtype=np.int64)
    for batch in batches:
        assert batch.shape == (8, 32)
        mask = batch.mask_array()
        assert any(
            g_var[row.gene_index] and mask[position].sum() >= 3
            for position, row in enumerate(batch.rows)
        )
        for position, row in enumerate(batch.rows):
            for context, valid in zip(row.context_indices, mask[position], strict=True):
                if valid:
                    counts[row.gene_index, context] += 1
    np.testing.assert_array_equal(counts, labels.astype(np.int64))


def test_epoch_is_deterministic_but_changes_order() -> None:
    labels, g_var = _fixture()
    first = build_epoch_batches(labels, g_var, epoch=0)
    again = build_epoch_batches(labels, g_var, epoch=0)
    next_epoch = build_epoch_batches(labels, g_var, epoch=1)
    assert first == again
    assert first != next_epoch


def test_ddp_shards_have_equal_steps_and_explicit_padding() -> None:
    labels, g_var = _fixture()
    batches = build_epoch_batches(labels, g_var)
    shards = [shard_batches(batches, rank=rank, world_size=3) for rank in range(3)]
    assert len({len(shard) for shard in shards}) == 1
    flattened = [batch for shard in shards for batch in shard]
    assert sum(not batch.is_rank_padding for batch in flattened) == len(batches)
    assert sum(batch.is_rank_padding for batch in flattened) < 3
    assert all(
        batch.objective_weight == 0.0 for batch in flattened if batch.is_rank_padding
    )


def test_sampler_rejects_too_few_gvar_chunks() -> None:
    labels = np.ones((16, 35), dtype=bool)
    g_var = np.zeros(16, dtype=bool)
    g_var[0] = True
    with pytest.raises(ValueError, match="not enough"):
        build_epoch_batches(labels, g_var)


def test_sampler_validates_masks_and_ddp_coordinates() -> None:
    labels, g_var = _fixture()
    with pytest.raises(ValueError, match="boolean"):
        build_epoch_batches(labels.astype(int), g_var)
    batches = build_epoch_batches(labels, g_var)
    with pytest.raises(ValueError, match="rank"):
        shard_batches(batches, rank=2, world_size=2)
