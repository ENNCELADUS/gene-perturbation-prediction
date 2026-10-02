"""Memory-bounded CSR assembly of per-gene X-Atlas-Orion reservoirs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.data.response_streaming import (
    drain_gene_reservoirs_to_matrix,
    resolve_total_budget_keep_mask,
)


# --- resolve_total_budget_keep_mask ----------------------------------------


def test_keep_mask_none_keeps_everything() -> None:
    mask = resolve_total_budget_keep_mask(100, None, seed=0)
    assert mask.dtype == bool
    assert mask.all()
    assert mask.sum() == 100


def test_keep_mask_budget_above_n_cells_keeps_everything() -> None:
    mask = resolve_total_budget_keep_mask(10, total_cells=50, seed=0)
    assert mask.all()


def test_keep_mask_trims_to_exact_budget() -> None:
    mask = resolve_total_budget_keep_mask(1000, total_cells=100, seed=0)
    assert mask.sum() == 100


def test_keep_mask_deterministic_for_same_seed() -> None:
    first = resolve_total_budget_keep_mask(500, total_cells=50, seed=3)
    second = resolve_total_budget_keep_mask(500, total_cells=50, seed=3)
    np.testing.assert_array_equal(first, second)


def test_keep_mask_different_seed_changes_selection() -> None:
    first = resolve_total_budget_keep_mask(500, total_cells=50, seed=3)
    second = resolve_total_budget_keep_mask(500, total_cells=50, seed=99)
    assert not np.array_equal(first, second)


# --- drain_gene_reservoirs_to_matrix: total-cell budget --------------------


def _reservoir_cell(value: float, barcode: str) -> tuple:
    return (
        np.array([0], dtype=np.int64),
        np.array([value], dtype=np.float32),
        barcode,
        "s",
    )


def test_drain_applies_total_cell_budget_after_per_gene_cap() -> None:
    reservoirs = {
        "GENE_A": [_reservoir_cell(1.0, f"a{i}") for i in range(6)],
        "GENE_B": [_reservoir_cell(2.0, f"b{i}") for i in range(6)],
    }
    metadata = pd.DataFrame({"ensembl_id": ["ENSG0"], "gene_name": ["G0"]}, index=[0])
    matrix, var, genes, barcodes, samples = drain_gene_reservoirs_to_matrix(
        reservoirs,
        metadata,
        metadata_var_columns=("ensembl_id", "gene_name"),
        total_cells=5,
        seed=0,
    )
    assert matrix.shape[0] == 5
    assert len(genes) == 5
    assert len(barcodes) == 5
    assert len(samples) == 5


def test_drain_total_cells_none_keeps_all() -> None:
    reservoirs = {
        "GENE_A": [_reservoir_cell(1.0, f"a{i}") for i in range(4)],
    }
    metadata = pd.DataFrame({"ensembl_id": ["ENSG0"], "gene_name": ["G0"]}, index=[0])
    matrix, _var, genes, _barcodes, _samples = drain_gene_reservoirs_to_matrix(
        reservoirs,
        metadata,
        metadata_var_columns=("ensembl_id", "gene_name"),
        total_cells=None,
    )
    assert matrix.shape[0] == 4
    assert len(genes) == 4


def test_drain_drains_reservoir_slots_to_none() -> None:
    """Every consumed slot is released."""
    reservoirs = {"GENE_A": [_reservoir_cell(1.0, "a")]}
    drain_gene_reservoirs_to_matrix(
        reservoirs,
        pd.DataFrame({"ensembl_id": ["ENSG0"], "gene_name": ["G0"]}, index=[0]),
        metadata_var_columns=("ensembl_id", "gene_name"),
    )
    assert reservoirs["GENE_A"][0] is None


def test_drain_drops_tokens_missing_from_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A token absent from ``metadata`` drops that entry, not the cell."""
    reservoirs = {
        "GENE_A": [
            (
                np.array([0, 1, 2], dtype=np.int64),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "a",
                "s",
            )
        ],
    }
    metadata = pd.DataFrame({"ensembl_id": ["ENSG0"], "gene_name": ["G0"]}, index=[0])
    with caplog.at_level("WARNING", logger="src.data.response_streaming"):
        matrix, var, _genes, _barcodes, _samples = drain_gene_reservoirs_to_matrix(
            reservoirs, metadata, metadata_var_columns=("ensembl_id", "gene_name")
        )
    assert matrix.shape == (1, 1)
    assert matrix.toarray().tolist() == [[1.0]]
    assert var.index.tolist() == ["ENSG0"]
    warnings = [
        record.message
        for record in caplog.records
        if "missing from gene metadata index" in record.message
    ]
    assert len(warnings) == 1
    assert "2/3" in warnings[0]


def test_drain_matches_coo_reference_with_duplicates_missing_and_budget() -> None:
    """Direct CSR construction equals a COO reference build."""
    reservoirs = {
        "GENE_B": [
            (
                np.array([1, 3], dtype=np.int64),
                np.array([7.0, 8.0], dtype=np.float32),
                "b0",
                "sb",
            ),
            (
                np.array([5], dtype=np.int64),
                np.array([6.0], dtype=np.float32),
                "b1",
                "sb",
            ),
        ],
        "GENE_A": [
            (
                np.array([5, 1, 5, 99], dtype=np.int64),
                np.array([1.0, 2.0, 3.0, 9.0], dtype=np.float32),
                "a0",
                "sa",
            ),
            (
                np.array([3, 88], dtype=np.int64),
                np.array([4.0, 5.0], dtype=np.float32),
                "a1",
                "sa",
            ),
        ],
    }
    original = {gene: list(cells) for gene, cells in reservoirs.items()}
    metadata = pd.DataFrame(
        {
            "ensembl_id": ["ENSG5", "ENSG1", "ENSG3"],
            "gene_name": ["G5", "G1", "G3"],
        },
        index=[5, 1, 3],
    )
    keep_mask = resolve_total_budget_keep_mask(4, total_cells=3, seed=1)

    selected = []
    global_index = 0
    for gene in sorted(original):
        for cell in original[gene]:
            if keep_mask[global_index]:
                selected.append((gene, cell))
            global_index += 1
    tokens = sorted(
        {
            int(token)
            for _gene, cell in selected
            for token in cell[0]
            if token in metadata.index
        }
    )
    token_to_col = {token: col for col, token in enumerate(tokens)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, (_gene, cell) in enumerate(selected):
        for token, value in zip(cell[0], cell[1], strict=True):
            if int(token) in token_to_col:
                rows.append(row)
                columns.append(token_to_col[int(token)])
                values.append(float(value))
    expected = csr_matrix(
        (values, (rows, columns)), shape=(len(selected), len(tokens)), dtype=np.float32
    )

    matrix, var, genes, barcodes, samples = drain_gene_reservoirs_to_matrix(
        reservoirs,
        metadata,
        metadata_var_columns=("ensembl_id", "gene_name"),
        total_cells=3,
        seed=1,
    )

    np.testing.assert_array_equal(matrix.indptr, expected.indptr)
    np.testing.assert_array_equal(matrix.indices, expected.indices)
    np.testing.assert_array_equal(matrix.data, expected.data)
    assert var.equals(metadata.loc[tokens].set_index("ensembl_id", drop=False))
    np.testing.assert_array_equal(genes, [gene for gene, _cell in selected])
    np.testing.assert_array_equal(barcodes, [cell[2] for _gene, cell in selected])
    np.testing.assert_array_equal(samples, [cell[3] for _gene, cell in selected])
    assert all(cell is None for bucket in reservoirs.values() for cell in bucket)
