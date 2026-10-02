"""Process-wide test setup: import-order and threading guards that must run first."""

from __future__ import annotations

import os

# MPS fallback guard for torch tests. Some losses use
# torch.cdist, whose backward is unimplemented on the MPS backend; the energy
# distance therefore needs the CPU fallback. This env var must be set before
# torch initializes the MPS backend, so it lives here in conftest (imported
# before any test module) rather than inside an individual test file.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# macOS OpenMP thread-pool guard. scipy and torch each bundle their own libomp;
# converting a scipy sparse matrix and then building a torch sparse tensor in the
# same process (exp10 ddgcn graph utils) segfaults under the default multi-thread
# OpenMP pool. Pinning a single OpenMP thread before torch initializes avoids the
# crash. setdefault so an explicit caller value still wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# macOS OpenMP load-order guard. xgboost and torch each bundle their own libomp;
# loading torch's runtime first and then fitting xgboost segfaults the process.
# conftest is imported before any test module, so importing xgboost here pins the
# safe (xgboost-then-torch) load order for the whole suite. Guarded so the suite
# still collects if xgboost is absent.
try:  # noqa: SIM105
    import xgboost  # noqa: F401
except ImportError:
    pass
