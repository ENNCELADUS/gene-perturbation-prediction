from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.builders.materialize_exp13_original47_raw_umi import (
    map_symbols_to_ensembl,
    stable_rank,
    update_top_cells,
)


def test_top_cells_keeps_smallest_stable_hashes() -> None:
    heap = []
    for cell_id in ("c3", "c1", "c4", "c2"):
        update_top_cells(
            heap,
            rank=stable_rank(0, "ACH-TEST", cell_id),
            cell_id=cell_id,
            genes=[3],
            expressions=[1],
            max_cells=2,
        )
    expected = sorted(
        ("c3", "c1", "c4", "c2"),
        key=lambda cell_id: stable_rank(0, "ACH-TEST", cell_id),
    )[:2]
    assert sorted(item[1] for item in heap) == sorted(expected)


def test_symbol_mapping_drops_ambiguous_symbols_and_ensembl_ids() -> None:
    metadata = pd.DataFrame(
        {
            "ensembl_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG3", "ENSG4"],
            "gene_symbol": ["A", "B", "B", "C", "D"],
        }
    )
    indices, ensembl, symbols = map_symbols_to_ensembl(
        np.asarray(["A", "B", "C", "D", "X"]), metadata
    )
    assert indices.tolist() == [0, 3]
    assert ensembl.tolist() == ["ENSG1", "ENSG4"]
    assert symbols.tolist() == ["A", "D"]
