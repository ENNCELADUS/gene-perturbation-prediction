"""Process-wide test setup: import-order and threading guards that must run first."""

from __future__ import annotations

import os

import pytest

# Set before torch initialises: cdist's backward (the energy distance) has no MPS
# kernel.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# scipy and torch bundle their own libomp; one OpenMP thread keeps them from
# segfaulting together on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture
def cpu(monkeypatch):
    """Keep Accelerate off a visible MPS device, on one torch thread."""
    import torch

    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    torch.set_num_threads(1)
