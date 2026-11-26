"""The MuSL harness (hpc/musl/prepare.py): MAGIC imputation of a line's single cells,
the genes x cells matrix, the training rows of a line, and the official statistical
features in chunks."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.experiments.run_dir import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "musl_prepare", REPO_ROOT / "hpc" / "musl" / "prepare.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_a_line_with_18_cells_is_imputed_without_the_library_filter():
    pytest.importorskip("magic")
    pytest.importorskip("scprep")
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(
        rng.poisson(0.5, (18, 40)), columns=[f"G{i}" for i in range(40)]
    )  # library size < 1500
    out = prepare.impute(counts, np.random.default_rng(0))
    assert out.shape == (18, 40) and np.isfinite(out.to_numpy()).all()


def test_line_matrix_is_genes_by_cells_with_zero_rows_for_absent_genes():
    imputed = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=["G1", "G2"]
    )  # 3 cells
    out = prepare.line_matrix(imputed, pd.Index(["G2", "G9"]))
    assert (
        out.tolist() == [[2.0, 4.0, 6.0], [0.0, 0.0, 0.0]] and out.dtype == np.float32
    )


def test_fit_samples_are_all_sl_rows_and_as_many_non_sl_rows_of_each_line():
    rows = pd.DataFrame(
        {
            "model_id": ["L1"] * 5 + ["L2"] * 3,
            "label": [1, 0, 0, 0, 0, 1, 1, 0],
            "gene_1": list("AAAAABBB"),
            "gene_2": list("BCDEFCDE"),
        }
    )
    out = prepare.line_samples(rows, np.random.default_rng(0))
    assert out.groupby("model_id")["label"].agg(["sum", "size"]).values.tolist() == [
        [1, 2],
        [2, 3],
    ]


def test_pooled_cells_are_drawn_evenly_from_the_lines():
    sizes = {"L1": 3, "L2": 10, "L3": 10}
    picks = prepare.pooled_cells(sizes, 9, np.random.default_rng(0))
    counts = pd.Series([m for m, _ in picks]).value_counts()
    assert len(picks) == 9 and counts.to_dict() == {"L1": 3, "L2": 3, "L3": 3}
    assert len(set(picks)) == 9


def test_features_do_not_depend_on_the_chunk_size_and_have_35_columns():
    rng = np.random.default_rng(0)
    matrix = rng.gamma(2.0, 1.0, (6, 30)).astype(np.float32)
    pairs = np.array([[0, 1], [2, 3], [4, 5], [1, 4], [0, 5]])
    whole = prepare.pair_features(matrix, pairs, chunk_rows=100)
    chunked = prepare.pair_features(matrix, pairs, chunk_rows=2)
    assert whole.shape == (5, 35) and np.isfinite(whole).all()
    assert np.allclose(whole, chunked)


def test_a_gene_without_expression_gives_finite_features_and_histograms():
    rng = np.random.default_rng(0)
    matrix = rng.gamma(2.0, 1.0, (3, 30)).astype(np.float32)
    matrix[1] = 0.0  # a zero row: the gene is a node but has no expression in the line
    pairs = np.array([[0, 1], [1, 2]])
    features = prepare.pair_features(matrix, pairs)
    assert np.isfinite(
        features
    ).all()  # std ratio (column 13) is inf: torch's nan_to_num caps it
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")  # the official module imports it
    dataset = prepare.official("module").EdgeDataset(
        torch.from_numpy(pairs.T.copy()),
        torch.zeros(2).long(),
        torch.from_numpy(matrix),
        state=torch.from_numpy(features),
    )
    assert torch.isfinite(dataset[0]["image"]).all()


def test_gene_symbols_come_from_the_first_known_column_and_duplicates_are_summed():
    var = pd.DataFrame({"gene_name": ["A", "B", "A"]}, index=["e1", "e2", "e3"])
    out = prepare.counts_frame(np.array([[1, 2, 3], [4, 5, 6]]), var)
    assert out.columns.tolist() == ["A", "B"] and out.to_numpy().tolist() == [
        [4, 2],
        [10, 5],
    ]
