"""Self-contained epoch resume reproduces next update, optimizer and RNG."""

import copy
from pathlib import Path

from accelerate import Accelerator
import pytest
import torch

from src.training import trainer
from src.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
    TrainState,
)
from test_joint_training import (
    assert_tree_equal,
    fresh_model,
    restore_model,
    tiny_training_config,
)


def test_epoch_resume_exact_next_update_and_remaining_replay(tmp_path, monkeypatch):
    config = tiny_training_config(tmp_path / "data")
    accelerator = Accelerator(cpu=True)
    snapshots = []
    original = trainer.train_update
    count = 0

    def update(*args):
        nonlocal count
        metrics = original(*args)
        count += 1
        if count in (8, 15):  # first update of uninterrupted/resumed epoch 2
            snapshots.append(
                (
                    copy.deepcopy(args[0].state_dict()),
                    copy.deepcopy(args[1].state_dict()),
                    metrics,
                )
            )
        return metrics

    monkeypatch.setattr(trainer, "train_update", update)
    model, inputs = fresh_model(config)
    trainer.fit(model, inputs, config, tmp_path / "full", accelerator)
    expected = load_checkpoint(tmp_path / "full" / "last.pt")
    model, inputs = fresh_model(config)
    # Do not count the independent first epoch in the next-update spy.
    monkeypatch.setattr(trainer, "train_update", original)
    original_save = trainer.save_checkpoint

    def interrupt_after_epoch(path, *args):
        original_save(path, *args)
        if path.name == "last.pt":
            raise RuntimeError("simulated interruption after epoch checkpoint")

    monkeypatch.setattr(trainer, "save_checkpoint", interrupt_after_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.fit(model, inputs, config, tmp_path / "part", accelerator)
    monkeypatch.setattr(trainer, "save_checkpoint", original_save)
    boundary = load_checkpoint(tmp_path / "part" / "last.pt")
    assert boundary["train_state"]["global_step"] == 7
    # Neither external initialization source is needed for restoration.
    Path(config["paths"]["esm2_embeddings"]).unlink()
    Path(config["paths"]["state_checkpoint"]).unlink()
    model, inputs = restore_model(boundary)
    torch.testing.assert_close(
        model.backbone.perturbations.esm_matrix,
        boundary["preprocessing"]["esm2_vectors"],
        rtol=0,
        atol=0,
    )

    def no_fit(*args, **kwargs):
        raise AssertionError("resume fitted feature scales")

    monkeypatch.setattr(trainer, "fit_startup_standardizer", no_fit)
    monkeypatch.setattr(trainer, "train_update", update)
    trainer.fit(
        model, inputs, config, tmp_path / "part", accelerator, restored=boundary
    )
    actual = load_checkpoint(tmp_path / "part" / "last.pt")
    assert len(snapshots) == 2
    assert_tree_equal(snapshots[0], snapshots[1])
    for name in (
        "model_state",
        "optimizer",
        "train_state",
        "rng_states",
        "preprocessing",
        "projection_state",
        "normalization_state",
        "architecture",
        "amp_state",
    ):
        assert_tree_equal(actual[name], expected[name])
    assert actual["train_state"]["global_step"] == 14


def test_resume_rejects_world_size_change_before_fitting(tmp_path):
    config = tiny_training_config(tmp_path / "data")
    model, inputs = fresh_model(config)
    with pytest.raises(ValueError, match="world size changed"):
        trainer.fit(
            model,
            inputs,
            config,
            tmp_path / "run",
            Accelerator(cpu=True),
            restored={
                "train_state": vars(TrainState()),
                "world_size": 2,
            },
        )


def test_atomic_checkpoint_failure_preserves_previous_file_and_rng(
    tmp_path, monkeypatch
):
    config = tiny_training_config(tmp_path / "data")
    model, inputs = fresh_model(config)
    accelerator = Accelerator(cpu=True)
    from src.model.normalization import fit_startup_standardizer

    fit_startup_standardizer(model, inputs)
    optimizer = trainer.make_optimizer(model, config)
    path = tmp_path / "last.pt"
    path.write_bytes(b"previous checkpoint")
    before = capture_rng_state(accelerator.device)

    def fail(payload, temporary):
        Path(temporary).write_bytes(b"partial checkpoint")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        save_checkpoint(
            path,
            model,
            optimizer,
            TrainState(),
            config,
            inputs.preprocessing_state(),
            accelerator,
        )
    assert path.read_bytes() == b"previous checkpoint"
    assert not path.with_name("last.pt.tmp").exists()
    assert_tree_equal(before, capture_rng_state(accelerator.device))
