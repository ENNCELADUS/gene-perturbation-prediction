"""Joint dependency coverage, deterministic epochs and equal full DDP batches."""

from types import SimpleNamespace
import pytest
from src.data.datasets import DependencyDataset
from src.data.prepared import load_inputs
from src.training.sampling import make_training_loaders
from test_joint_data import make_prepared_fixture


def _keys(loader):
    return [
        (model_id, gene)
        for batch in loader
        for model_id, gene in zip(
            batch.conditions.model_ids, batch.conditions.genes, strict=True
        )
    ]


def test_single_rank_unit_batches_cover_every_labeled_pair_once(tmp_path):
    config = make_prepared_fixture(tmp_path)
    config["train"]["dependency_batch_size"] = 1
    inputs = load_inputs(config)
    loader, _ = make_training_loaders(inputs, config, 0, None)
    actual = _keys(loader)
    rows = DependencyDataset(inputs, "train").rows
    expected = set(zip(rows.model_id, rows.gene_symbol, strict=True))
    assert len(actual) == len(set(actual)) == len(expected)
    assert set(actual) == expected


def test_epoch_is_deterministic_but_changes_order(tmp_path):
    config = make_prepared_fixture(tmp_path)
    config["train"]["dependency_batch_size"] = 1
    inputs = load_inputs(config)

    def epoch(index):
        loader, _ = make_training_loaders(inputs, config, index, None)
        return _keys(loader)

    assert epoch(0) == epoch(0)
    assert epoch(0) != epoch(1)
    assert set(epoch(0)) == set(epoch(1))


def test_ddp_shards_are_disjoint_equal_full_batches_with_documented_tail_drop(tmp_path):
    config = make_prepared_fixture(tmp_path)
    config["train"]["dependency_batch_size"] = 2
    inputs = load_inputs(config)
    world = 4
    shards = []
    steps = []
    for rank in range(world):
        loader, _ = make_training_loaders(
            inputs, config, 0, SimpleNamespace(process_index=rank, num_processes=world)
        )
        steps.append(len(loader))
        assert all(len(batch.conditions.genes) == 2 for batch in loader)
        shards.append(_keys(loader))
    assert len(set(steps)) == 1
    all_keys = [key for shard in shards for key in shard]
    assert len(all_keys) == len(set(all_keys))
    n_rows = len(DependencyDataset(inputs, "train"))
    assert len(all_keys) == (n_rows // (world * 2)) * (world * 2)


def test_sampler_rejects_invalid_rank(tmp_path):
    config = make_prepared_fixture(tmp_path)
    inputs = load_inputs(config)
    with pytest.raises(ValueError, match="rank"):
        make_training_loaders(
            inputs, config, 0, SimpleNamespace(process_index=2, num_processes=2)
        )
