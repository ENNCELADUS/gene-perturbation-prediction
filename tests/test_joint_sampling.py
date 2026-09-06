"""Four-anchor replay, epoch restart, and tail-preserving evaluation."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from test_joint_data import make_prepared_fixture

from src.data.datasets import DependencyDataset, make_evaluation_loaders
from src.data.prepared import load_inputs
from src.training.sampling import make_training_loaders


def keys(batch):
    if hasattr(batch, "conditions"):
        batch = batch.conditions
    return tuple(zip(batch.model_ids, batch.genes, strict=True))


def test_balanced_replay_cycles_excludes_holdout_preserves_dependency(tmp_path):
    config = make_prepared_fixture(tmp_path)
    config["train"]["response_batch_size"] = 64
    inputs = load_inputs(config)
    dependency, response = make_training_loaders(inputs, config, 2, None)
    dependency_keys = {key for batch in dependency for key in keys(batch)}
    assert all(
        model_id in inputs.split.supervised_train for model_id, _ in dependency_keys
    )
    eligible = DependencyDataset(inputs, "train").rows
    assert inputs.response_holdout.issubset(
        set(zip(eligible.model_id, eligible.gene_symbol, strict=True))
    )
    seen = set()
    for _ in range(8):
        batch = next(response)
        assert Counter(batch.model_ids) == dict.fromkeys(inputs.response_anchors, 16)
        assert not set(keys(batch)) & inputs.response_holdout
        assert all("@" not in gene for gene in batch.genes)
        seen.update(keys(batch))
    assert seen == set(inputs.response_targets.keys) - inputs.response_holdout
    assert ("ACH-A0", "G1") in seen and ("ACH-A1", "G1") in seen


def test_epoch_rank_restart_reproduces_both_streams_without_global_rng_use(tmp_path):
    config = make_prepared_fixture(tmp_path)
    inputs = load_inputs(config)
    torch_state = torch.get_rng_state().clone()

    def sample(epoch, rank):
        accelerator = SimpleNamespace(process_index=rank, num_processes=2)
        dep, response = make_training_loaders(inputs, config, epoch, accelerator)
        return [keys(batch) for batch in dep], [keys(next(response)) for _ in range(5)]

    first = sample(2, 0)
    assert first == sample(2, 0)
    second_rank = sample(2, 1)
    assert second_rank == sample(2, 1)
    assert first != second_rank
    assert first != sample(3, 0)
    assert not set(sum(first[0], ())) & set(sum(second_rank[0], ()))
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


@pytest.mark.parametrize("size", [0, 3, 6, -4])
def test_invalid_response_batch_size_fails_at_construction(tmp_path, size):
    config = make_prepared_fixture(tmp_path)
    inputs = load_inputs(config)
    config["train"]["response_batch_size"] = size
    with pytest.raises(ValueError, match="divisible by four"):
        make_training_loaders(inputs, config, 0, None)


@pytest.mark.parametrize("name", ["train", "collator", "projection"])
def test_nonzero_runtime_seed_rejected(tmp_path, name):
    config = make_prepared_fixture(tmp_path)
    inputs = load_inputs(config)
    config["seeds"][name] = 13
    with pytest.raises(ValueError, match="base seeds must be 0"):
        make_training_loaders(inputs, config, 0, None)


def test_cpu_accelerate_evaluation_retains_tail_and_keys(tmp_path):
    from accelerate import Accelerator

    config = make_prepared_fixture(tmp_path)
    inputs = load_inputs(config)
    accelerator = Accelerator(cpu=True)
    dependency, response = make_evaluation_loaders(inputs, config, "val", accelerator)
    dep_rows = []
    sizes = []
    for batch in dependency:
        sizes.append(len(keys(batch)))
        dep_rows.extend(
            accelerator.gather_for_metrics(
                [{"key": key} for key in keys(batch)], use_gather_object=True
            )
        )
    assert sizes == [2, 1]
    assert {row["key"] for row in dep_rows} == {
        ("ACH-VAL", gene) for gene in inputs.genes
    }
    response_rows = []
    for batch in response:
        response_rows.extend(
            accelerator.gather_for_metrics(
                [{"key": key} for key in keys(batch)], use_gather_object=True
            )
        )
    assert {row["key"] for row in response_rows} == inputs.response_holdout
    assert len(response_rows) == len(inputs.response_holdout)
