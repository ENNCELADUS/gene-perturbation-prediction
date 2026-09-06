"""Tests for src.training.distributed DDP invariants, using a fake accelerator
(no real process group -- torch.distributed calls are monkeypatched)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.training.distributed import (
    assert_all_ranks_stepped,
    require_distinct_devices,
    run_rank_zero_or_raise,
)


class FakeAccelerator:
    """Minimal stand-in for accelerate.Accelerator's fields used here."""

    def __init__(
        self,
        num_processes: int,
        device_type: str = "cpu",
        device_index: int | None = None,
        is_main_process: bool = True,
        gathered: torch.Tensor | None = None,
    ) -> None:
        self.num_processes = num_processes
        self.device = (
            torch.device(device_type, device_index)
            if device_index is not None
            else torch.device(device_type)
        )
        self.is_main_process = is_main_process
        self._gathered = gathered

    def gather(self, local: torch.Tensor) -> torch.Tensor:
        assert self._gathered is not None, "test must supply `gathered`"
        return self._gathered


def test_require_distinct_devices_single_process_is_noop(monkeypatch):
    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("no collective should run for num_processes == 1")

    monkeypatch.setattr(torch.distributed, "all_gather_object", _boom)
    accelerator = FakeAccelerator(num_processes=1, device_type="cpu")
    require_distinct_devices(accelerator)  # must not raise, must not gather


def test_require_distinct_devices_raises_on_duplicate_cuda_index(monkeypatch):
    def _fake_all_gather_object(object_list: list, obj: object) -> None:
        object_list[0] = ("cuda", 0)
        object_list[1] = ("cuda", 0)

    monkeypatch.setattr(torch.distributed, "all_gather_object", _fake_all_gather_object)
    accelerator = FakeAccelerator(num_processes=2, device_type="cuda", device_index=0)
    with pytest.raises(RuntimeError, match="distinct CUDA"):
        require_distinct_devices(accelerator)


def test_require_distinct_devices_raises_on_non_cuda_rank(monkeypatch):
    def _fake_all_gather_object(object_list: list, obj: object) -> None:
        object_list[0] = ("cuda", 0)
        object_list[1] = ("cpu", None)

    monkeypatch.setattr(torch.distributed, "all_gather_object", _fake_all_gather_object)
    accelerator = FakeAccelerator(num_processes=2, device_type="cuda", device_index=0)
    with pytest.raises(RuntimeError, match="CUDA on every rank"):
        require_distinct_devices(accelerator)


def test_require_distinct_devices_memoizes_the_collective(monkeypatch):
    calls = {"count": 0}

    def _fake_all_gather_object(object_list: list, obj: object) -> None:
        calls["count"] += 1
        object_list[0] = ("cuda", 0)
        object_list[1] = ("cuda", 1)

    monkeypatch.setattr(torch.distributed, "all_gather_object", _fake_all_gather_object)
    accelerator = FakeAccelerator(num_processes=2, device_type="cuda", device_index=0)
    require_distinct_devices(accelerator)
    require_distinct_devices(accelerator)
    assert calls["count"] == 1


def test_assert_all_ranks_stepped_raises_on_zero_steps():
    accelerator = FakeAccelerator(
        num_processes=2,
        gathered=torch.tensor([5, 0], dtype=torch.int64),
    )
    with pytest.raises(RuntimeError, match="must all be positive"):
        assert_all_ranks_stepped(accelerator, local_steps=5)


def test_assert_all_ranks_stepped_returns_counts():
    accelerator = FakeAccelerator(
        num_processes=2,
        gathered=torch.tensor([5, 3], dtype=torch.int64),
    )
    counts = assert_all_ranks_stepped(accelerator, local_steps=5)
    assert counts == (5, 3)


def test_run_rank_zero_or_raise_reraises_rank_zero_failure():
    accelerator = FakeAccelerator(num_processes=1, is_main_process=True)

    def _failing_action() -> None:
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match=r"label failed on rank zero.*boom"):
        run_rank_zero_or_raise(accelerator, "label", _failing_action)
