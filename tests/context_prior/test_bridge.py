"""Per-gene affine bridge and its quality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.bridge import bridge_quality, fit_bridge


def test_bridge_recovers_an_affine_map_and_handles_a_constant_gene():
    rng = np.random.default_rng(0)
    lines = [f"L{i}" for i in range(30)]
    pseudo = pd.DataFrame(rng.normal(size=(30, 2)), index=lines, columns=["A", "B"])
    pseudo["B"] = 1.0
    bulk = pd.DataFrame(
        {"A": 2 * pseudo["A"] + 1, "B": rng.normal(3, 1, 30)}, index=lines
    )
    bridge = fit_bridge(pseudo, bulk)
    assert np.allclose(bridge.slope, [2.0, 0.0])
    assert np.isclose(bridge.intercept[1], bulk["B"].mean())
    bridged = bridge.apply(pseudo)
    assert np.allclose(bridged["A"], bulk["A"])
    quality = bridge_quality(bridged, bulk)
    assert np.isclose(quality["A"], 1.0) and np.isnan(quality["B"])


def test_bridge_requires_aligned_frames():
    frame = pd.DataFrame({"A": [1.0, 2.0]}, index=["L1", "L2"])
    with pytest.raises(ValueError, match="aligned"):
        fit_bridge(frame, frame.rename(index={"L2": "L3"}))
