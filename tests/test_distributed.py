"""Tests for src.training.distributed DDP invariants, using a fake accelerator
(no real process group -- torch.distributed calls are monkeypatched)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.training.distributed import (
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
    ) -> None:
        self.num_processes = num_processes
        self.device = (
            torch.device(device_type, device_index)
            if device_index is not None
            else torch.device(device_type)
        )
        self.is_main_process = is_main_process


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
    with pytest.raises(RuntimeError, match="CPU on every rank or distinct CUDA"):
        require_distinct_devices(accelerator)


def test_require_distinct_devices_checks_current_assignments(monkeypatch):
    calls = {"count": 0}

    def _fake_all_gather_object(object_list: list, obj: object) -> None:
        calls["count"] += 1
        object_list[0] = ("cuda", 0)
        object_list[1] = ("cuda", 1)

    monkeypatch.setattr(torch.distributed, "all_gather_object", _fake_all_gather_object)
    accelerator = FakeAccelerator(num_processes=2, device_type="cuda", device_index=0)
    require_distinct_devices(accelerator)
    require_distinct_devices(accelerator)
    assert calls["count"] == 2


def test_require_distinct_devices_allows_cpu_gloo_ranks(monkeypatch):
    def gather(assignments, local):
        assignments[:] = [("cpu", None), ("cpu", None)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    require_distinct_devices(FakeAccelerator(num_processes=2))


def test_remote_rank_failure_is_reported_before_dependent_work(monkeypatch):
    from src.training.distributed import raise_rank_errors

    def gather(errors, local):
        assert local is None
        errors[:] = [None, "ValueError: invalid batch"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="invalid batch"):
        raise_rank_errors(FakeAccelerator(num_processes=2), "batch", None)


def test_run_rank_zero_or_raise_reraises_rank_zero_failure():
    accelerator = FakeAccelerator(num_processes=1, is_main_process=True)

    def _failing_action() -> None:
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match=r"label failed on rank zero.*boom"):
        run_rank_zero_or_raise(accelerator, "label", _failing_action)
