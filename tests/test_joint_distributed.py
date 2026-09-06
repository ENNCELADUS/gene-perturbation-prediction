"""Real CPU/Gloo wrapper parity, gathered tails and every-rank resume."""

import copy
import json
import os
from pathlib import Path
import random
import socket
import time

import conftest  # noqa: F401 -- child imports preserve OpenMP/xgboost initialization
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from src.data.datasets import DependencyDataset, ResponseDataset
from src.eval.geneeffect import evaluate_model
from src.model.normalization import fit_startup_standardizer
from src.training import trainer
from src.training.checkpoint import load_checkpoint, save_checkpoint, TrainState
from src.training.distributed import require_distinct_devices
from test_joint_training import (
    assert_tree_equal,
    fresh_model,
    restore_model,
    tiny_training_config,
)


def _accelerator(rank, port):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
    )
    accelerator = Accelerator(
        cpu=True,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=True, static_graph=False
            )
        ],
    )
    assert accelerator.num_processes == 2
    require_distinct_devices(accelerator)
    return accelerator


def _spawn(worker, args):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = torch.multiprocessing.spawn(
        worker, args=(port, *args), nprocs=2, join=False
    )
    try:
        deadline = time.monotonic() + 90
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("two-rank training did not finish within 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def _parity_worker(rank, port, initial, output, replay, fail):
    accelerator = _accelerator(rank, port)
    try:
        saved = load_checkpoint(Path(initial))
        model, inputs = restore_model(saved)
        config = saved["config"]
        optimizer = trainer.make_optimizer(model, config)
        model, optimizer = accelerator.prepare(model, optimizer)
        assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
        dependency = DependencyDataset(inputs, "train").collate(
            [rank * 2, rank * 2 + 1]
        )
        response = ResponseDataset(inputs, holdout=False)
        response_batch = (
            response.collate(response.indices[rank * 4 : rank * 4 + 4])
            if replay
            else None
        )
        if fail and rank == 1:
            dependency.residual[0] = float("nan")
        try:
            trainer.train_update(
                model, optimizer, dependency, response_batch, config, accelerator
            )
        except RuntimeError as exc:
            if not fail:
                raise
            error = str(exc)
            obstructed = Path(output, "not-a-directory")
            if rank == 0:
                obstructed.write_text("obstruct checkpoint directory")
            try:
                save_checkpoint(
                    obstructed / "last.pt",
                    model,
                    optimizer,
                    TrainState(),
                    config,
                    inputs.preprocessing_state(),
                    accelerator,
                )
            except RuntimeError as checkpoint_error:
                assert "failed on rank zero" in str(checkpoint_error)
                assert "FileExistsError" in str(checkpoint_error)
            else:
                raise AssertionError("rank-zero checkpoint failure did not propagate")
            Path(output, f"rank-{rank}.json").write_text(json.dumps({"error": error}))
            return
        result = evaluate_model(
            model, inputs, config, split="val", accelerator=accelerator
        )
        torch.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": result.metrics,
                "pairs": len(result.predictions),
                "response": len(result.response),
            },
            Path(output, f"rank-{rank}.pt"),
        )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("replay,fail", [(False, False), (True, False), (False, True)])
def test_ddp_update_matches_effective_serial_batch_and_trims_eval_tail(
    tmp_path, replay, fail
):
    config = tiny_training_config(tmp_path / "data", transformer_dropout=0.0)
    model, inputs = fresh_model(config)
    assert all(
        module.p == 0
        for module in model.modules()
        if isinstance(module, torch.nn.Dropout)
    )
    accelerator = Accelerator(cpu=True)
    fit_startup_standardizer(model, inputs, batch_size=2)
    optimizer = trainer.make_optimizer(model, config)
    initial = tmp_path / "initial.pt"
    save_checkpoint(
        initial,
        model,
        optimizer,
        TrainState(),
        config,
        inputs.preprocessing_state(),
        accelerator,
    )
    _spawn(_parity_worker, (str(initial), str(tmp_path), replay, fail))
    if fail:
        for rank in range(2):
            error = json.loads((tmp_path / f"rank-{rank}.json").read_text())["error"]
            assert "training forward/loss failed on a rank" in error
            assert "non-finite GeneEffect targets" in error
        return
    dependency = DependencyDataset(inputs, "train").collate([0, 1, 2, 3])
    response = ResponseDataset(inputs, holdout=False)
    response_batch = response.collate(response.indices[:8]) if replay else None
    model, optimizer = accelerator.prepare(model, optimizer)
    trainer.train_update(
        model, optimizer, dependency, response_batch, config, accelerator
    )
    expected = evaluate_model(
        model, inputs, config, split="val", accelerator=accelerator
    )
    first = None
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        if first is not None:
            assert_tree_equal(first, actual)
        first = actual
        for key, value in model.state_dict().items():
            torch.testing.assert_close(
                actual["model"][key], value, atol=3e-6, rtol=1e-5
            )
        for key, value in optimizer.state_dict()["state"].items():
            for name, tensor in value.items():
                torch.testing.assert_close(
                    actual["optimizer"]["state"][key][name],
                    tensor,
                    atol=1e-7,
                    rtol=2e-4,
                )
        assert actual["pairs"] == 3
        assert actual["response"] == 4
        for key, value in expected.metrics.items():
            if value is None:
                assert actual["metrics"][key] is None
            else:
                assert actual["metrics"][key] == pytest.approx(
                    value, rel=1e-5, abs=1e-5
                )


def _resume_worker(rank, port, config, output):
    accelerator = _accelerator(rank, port)
    output = Path(output)
    original_update, original_save = trainer.train_update, trainer.save_checkpoint
    try:
        count, snapshots = 0, []

        def update(*args):
            nonlocal count
            metrics = original_update(*args)
            count += 1
            if count == 4:  # three local full batches in epoch 1
                snapshots.append(
                    (
                        copy.deepcopy(args[0].state_dict()),
                        copy.deepcopy(args[1].state_dict()),
                        metrics,
                    )
                )
            return metrics

        def new_model():
            model, inputs = fresh_model(config)
            random.seed(100 + rank)
            np.random.seed(200 + rank)
            torch.manual_seed(300 + rank)
            return model, inputs

        trainer.train_update = update
        model, inputs = new_model()
        trainer.fit(model, inputs, config, output / "full", accelerator)
        expected = load_checkpoint(output / "full" / "last.pt")
        trainer.train_update = original_update

        def interrupt(path, *args):
            original_save(path, *args)
            if path.name == "last.pt":
                raise RuntimeError("epoch interruption")

        trainer.save_checkpoint = interrupt
        model, inputs = new_model()
        try:
            trainer.fit(model, inputs, config, output / "part", accelerator)
        except RuntimeError as exc:
            assert "epoch interruption" in str(exc)
        else:
            raise AssertionError("epoch interruption missing")
        boundary = load_checkpoint(output / "part" / "last.pt")
        assert boundary["world_size"] == 2
        assert boundary["train_state"]["global_step"] == 3
        assert not torch.equal(
            boundary["rng_states"][0]["torch"], boundary["rng_states"][1]["torch"]
        )
        model, inputs = restore_model(boundary)
        count = 3
        trainer.train_update, trainer.save_checkpoint = update, original_save
        trainer.fit(
            model, inputs, config, output / "part", accelerator, restored=boundary
        )
        actual = load_checkpoint(output / "part" / "last.pt")
        assert len(snapshots) == 2
        assert_tree_equal(snapshots[0], snapshots[1])
        for name in ("model_state", "optimizer", "train_state", "rng_states"):
            assert_tree_equal(actual[name], expected[name])
        Path(output, f"resume-{rank}.json").write_text(
            json.dumps(
                {
                    "next_update_equal": True,
                    "global_step": actual["train_state"]["global_step"],
                    "next_update_replay": snapshots[1][2]["train_response_loss"]
                    is not None,
                }
            )
        )
    finally:
        trainer.train_update, trainer.save_checkpoint = original_update, original_save
        torch.distributed.destroy_process_group()


def test_two_rank_real_fit_epoch_resume_restores_each_rank_rng_and_next_update(
    tmp_path,
):
    config = tiny_training_config(tmp_path / "data")
    _spawn(_resume_worker, (config, str(tmp_path)))
    for rank in range(2):
        assert json.loads((tmp_path / f"resume-{rank}.json").read_text()) == {
            "next_update_equal": True,
            "global_step": 6,
            "next_update_replay": False,
        }
