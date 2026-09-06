"""Epoch-boundary training state and GeneEffect-loss checkpoint selection."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from src.training.distributed import run_rank_zero_or_raise


@dataclass
class TrainState:
    next_epoch: int = 0
    global_step: int = 0
    best_loss: float = math.inf
    best_epoch: int = -1
    bad_epochs: int = 0


def record_validation(
    state: TrainState, metrics: Mapping[str, Any], epoch: int
) -> bool:
    """Record a completed epoch; only a strict GeneEffect-loss decrease wins."""
    loss = float(metrics["val_geneeffect_loss"])
    if not math.isfinite(loss):
        raise ValueError("val_geneeffect_loss must be finite")
    improved = loss < state.best_loss
    if improved:
        state.best_loss, state.best_epoch, state.bad_epochs = loss, epoch, 0
    else:
        state.bad_epochs += 1
    state.next_epoch = epoch + 1
    return improved


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Use weights-only-loadable types, including the NumPy MT19937 words."""
    numpy = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy": (numpy[0], torch.tensor(numpy[1].astype(np.int64)), *numpy[2:]),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng_state(saved: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(saved["python"])
    numpy = saved["numpy"]
    np.random.set_state((numpy[0], numpy[1].numpy().astype(np.uint32), *numpy[2:]))
    torch.set_rng_state(saved["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(saved["mps"])


def save_checkpoint(
    path: Path, model, optimizer, state: TrainState, config, preprocessing, accelerator
) -> None:
    """Collect every rank's RNG, then atomically save ordinary checkpoint data."""
    rng_states = [capture_rng_state(accelerator.device)]
    if accelerator.num_processes > 1:
        local = rng_states[0]
        rng_states = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(rng_states, local)

    def write():
        unwrapped = accelerator.unwrap_model(model)
        payload = {
            "architecture": unwrapped.architecture,
            "model_state": unwrapped.state_dict(),
            "projection_state": unwrapped.projection.to_state(),
            "normalization_state": unwrapped.standardizer.to_state(),
            "preprocessing": dict(preprocessing),
            "optimizer": optimizer.state_dict(),
            "train_state": asdict(state),
            "config": dict(config),
            "world_size": accelerator.num_processes,
            "amp_state": {
                "mixed_precision": accelerator.mixed_precision,
                "scaler": None
                if accelerator.scaler is None
                else accelerator.scaler.state_dict(),
            },
            "rng_states": rng_states,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    run_rank_zero_or_raise(accelerator, f"checkpoint {path}", write)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load tensors and Python data on CPU, never serialized model classes."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "architecture",
        "model_state",
        "projection_state",
        "normalization_state",
        "preprocessing",
        "optimizer",
        "train_state",
        "config",
        "world_size",
        "amp_state",
        "rng_states",
    }
    if not isinstance(saved, dict) or required - saved.keys():
        raise ValueError(f"incomplete joint checkpoint: {path}")
    if saved["world_size"] < 1 or len(saved["rng_states"]) != saved["world_size"]:
        raise ValueError("checkpoint requires one RNG state per rank")
    TrainState(**saved["train_state"])
    return saved
