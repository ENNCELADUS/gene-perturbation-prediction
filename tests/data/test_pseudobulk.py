"""Pseudo-bulk: summed raw UMI, whole-library CPM, log1p; round trip."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.data.pseudobulk import PSEUDOBULK_DIR, pseudobulk, read_pseudobulk


def test_pseudobulk_sums_cells_and_duplicate_symbols():
    counts = sparse.csr_matrix(np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 1.0]]))
    values = pseudobulk(counts, ["A", "B", "A"], ["A", "B", "C"])
    assert np.isclose(values[0], np.log1p(5 * 1e6 / 7))
    assert np.isclose(values[1], np.log1p(2 * 1e6 / 7))
    assert np.isnan(values[2])


def test_normalised_input_is_refused():
    with pytest.raises(ValueError, match="raw integer"):
        pseudobulk(sparse.csr_matrix(np.array([[0.5]])), ["A"], ["A"])


def test_read_pseudobulk_round_trip(tmp_path):
    root = tmp_path / PSEUDOBULK_DIR
    root.mkdir()
    frame = pd.DataFrame({"A": [1.0]}, index=pd.Index(["L1"], name="model_id"))
    frame.to_parquet(root / "pseudobulk.parquet")
    (root / "manifest.json").write_text(json.dumps({"genes": ["A"], "lines": ["L1"]}))
    assert read_pseudobulk(tmp_path).equals(frame)
