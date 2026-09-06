"""Observed response cache: actual gene order, array round-trip and corruption."""

import json
import numpy as np
import pandas as pd
import pytest

from src.data.response_cache import (
    open_response_targets_cache,
    write_response_targets_cache,
)


def write_fixture(root):
    bags = [
        np.arange(6, dtype=np.float32).reshape(3, 2),
        np.ones((2, 2), dtype=np.float32),
    ]
    metadata = pd.DataFrame(
        {"model_id": ["A", "B"], "perturbation_gene": ["G", "G"], "n_cells": [3, 2]}
    )
    write_response_targets_cache(
        root,
        genes=["G@A", "G@B"],
        target_bags=bags,
        metadata=metadata,
        hvg_order=["Z", "X"],
    )
    return bags


def test_roundtrip_mmap_and_distinct_anchor_keys(tmp_path):
    expected = write_fixture(tmp_path)
    result = open_response_targets_cache(tmp_path, expected_hvg_order=["Z", "X"])
    assert result.keys == (("A", "G"), ("B", "G"))
    for index, bag in enumerate(expected):
        np.testing.assert_array_equal(result.target_bag(index), bag)
    with pytest.raises(ValueError, match="order"):
        open_response_targets_cache(tmp_path, expected_hvg_order=["X", "Z"])


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("header", "hvg_order"),
        ("offsets", "offset"),
        ("identity", "identity|genes|metadata"),
    ],
)
def test_rejects_malformed_with_preparation_guidance(tmp_path, mutation, match):
    write_fixture(tmp_path)
    root = tmp_path / "response_targets"
    if mutation == "header":
        header = json.loads((root / "manifest.json").read_text())
        header.pop("hvg_order")
        (root / "manifest.json").write_text(json.dumps(header))
    elif mutation == "offsets":
        np.save(root / "offsets.npy", np.asarray([0, 4, 3], dtype=np.int64))
    else:
        np.save(root / "genes.npy", np.asarray(["WRONG@A", "G@B"], dtype=object))
    with pytest.raises(ValueError, match=match) as caught:
        open_response_targets_cache(tmp_path, expected_hvg_order=["Z", "X"])
    assert str(root) in str(caught.value)
    assert "hpc/run.sh prepare" in str(caught.value)


def test_missing_cache_has_actionable_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="hpc/run.sh prepare") as caught:
        open_response_targets_cache(tmp_path, expected_hvg_order=["Z", "X"])
    assert str(tmp_path) in str(caught.value)


def test_writer_rejects_bad_alignment_before_replacing_cache(tmp_path):
    expected = write_fixture(tmp_path)
    metadata = pd.DataFrame(
        {"model_id": ["A"], "perturbation_gene": ["G"], "n_cells": [9]}
    )
    with pytest.raises(ValueError, match="differs from assembled"):
        write_response_targets_cache(
            tmp_path,
            genes=["G@A"],
            target_bags=[expected[0]],
            metadata=metadata,
            hvg_order=["Z", "X"],
        )
    actual = open_response_targets_cache(tmp_path, expected_hvg_order=["Z", "X"])
    np.testing.assert_array_equal(actual.target_bag(0), expected[0])
