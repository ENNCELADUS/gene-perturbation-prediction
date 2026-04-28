"""Runtime helpers for Accelerate-backed training."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from accelerate import Accelerator


class AccelerateRuntime:
    """Small wrapper around :class:`accelerate.Accelerator`."""

    def __init__(self, config: Mapping[str, object]):
        device_config = config.get("device_config", {})
        if not isinstance(device_config, Mapping):
            device_config = {}
        mixed_precision = (
            "fp16" if bool(device_config.get("use_mixed_precision", False)) else "no"
        )
        self.accelerator = Accelerator(mixed_precision=mixed_precision)

    @property
    def device(self) -> torch.device:
        """Return the active Accelerate device."""
        return self.accelerator.device

    @property
    def is_main_process(self) -> bool:
        """Whether this process should write logs and checkpoints."""
        return self.accelerator.is_main_process

    def prepare(self, *objects):
        """Delegate object preparation to Accelerate."""
        return self.accelerator.prepare(*objects)

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate through Accelerate."""
        self.accelerator.backward(loss)

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors across processes for metric computation."""
        return self.accelerator.gather_for_metrics(tensor)

    def clip_grad_norm_(self, parameters, max_norm: float) -> None:
        """Clip gradients through Accelerate."""
        self.accelerator.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the unwrapped model for checkpointing."""
        return self.accelerator.unwrap_model(model)

    def wait_for_everyone(self) -> None:
        """Synchronize processes."""
        self.accelerator.wait_for_everyone()

    def save_state_dict(self, model: torch.nn.Module, path: str | Path) -> None:
        """Save an unwrapped model state dict from the main process."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.unwrap_model(model).state_dict()
        self.accelerator.save(state_dict, output_path)
