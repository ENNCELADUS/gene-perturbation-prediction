"""Memory-bounded streaming of X-Atlas-Orion reservoirs."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import test_prepare as base
from src.data import response as response_streaming
from src.data.response import prefetched, read_xatlas_response_cells
from test_prepare_equivalence import CAPPED, _write_xatlas, naive_xatlas_cells


# --- prefetched -----------------------------------------------------------


def test_prefetched_keeps_order_and_bounds_reads_ahead() -> None:
    lock = threading.Lock()
    running, peak = 0, 0

    def read(item: int) -> int:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.02 if item % 2 else 0.0)  # odd items finish last
        with lock:
            running -= 1
        return item * 10

    assert list(prefetched(read, range(9), threads=3)) == [i * 10 for i in range(9)]
    assert 1 < peak <= 3


# --- read_xatlas_response_cells -------------------------------------------


def _read(source: dict, caps: dict, seed: int):
    return read_xatlas_response_cells(
        source["shard_dir"],
        source["gene_metadata_path"],
        model_id=base.HCT116,
        shard_glob=source["shard_glob"],
        control_label=source["control_label"],
        pass_guide_filter_value=source["pass_guide_filter_value"],
        symbol_col="gene_name",
        gene_order=base.HVG,
        max_cells_per_gene=caps["response_max_cells_per_gene"],
        seed=seed,
    )


@pytest.mark.parametrize(
    "caps",
    [
        CAPPED,
        {"response_max_cells_per_gene": 3},
    ],
)
def test_reservoir_matches_whole_cell_reference(tmp_path, caps) -> None:
    """Several shards, rejected rows, duplicate and unknown tokens, two caps."""
    source = _write_xatlas(tmp_path, seed=300)
    cells = _read(source, caps, seed=11)
    labels, sizes, aligned = naive_xatlas_cells(source, base.HVG, caps, seed=11)
    np.testing.assert_array_equal(cells.labels, labels)
    assert cells.library_sizes.tobytes() == sizes.tobytes()
    assert cells.hvg_counts.toarray().tobytes() == aligned.tobytes()
    np.testing.assert_array_equal(cells.present, (aligned > 0).any(axis=0))
    assert not cells.present[base.HVG.index("H1")]  # absent from the metadata


def test_only_rows_a_reservoir_keeps_are_converted(tmp_path, monkeypatch) -> None:
    source = _write_xatlas(tmp_path, seed=300)
    converted = []
    original = response_streaming.xatlas_token_rows

    def spy(table, rows):
        converted.append(len(rows))
        return original(table, rows)

    monkeypatch.setattr(response_streaming, "xatlas_token_rows", spy)
    caps = {"response_max_cells_per_gene": 1}
    cells = _read(source, caps, seed=0)
    n_genes = len(set(cells.labels.tolist()))
    assert len(converted) == 5  # one conversion per shard
    assert all(0 < count <= n_genes for count in converted)


def test_unknown_tokens_of_kept_cells_are_reported(tmp_path, caplog) -> None:
    source = _write_xatlas(tmp_path, seed=300)
    with caplog.at_level("WARNING", logger="src.data.response"):
        _read(source, CAPPED, seed=0)
    assert [r.message for r in caplog.records if "missing from" in r.message] == [
        f"{base.HCT116}: dropped 2 distinct gene tokens missing from the gene "
        "metadata index"
    ]
