"""Per-line cell inputs of the stage-one ladder."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.data import ladder_cells as cells


def _poisson_line(rng, n_cells=4000, n_genes=6):
    depth = rng.integers(2_000, 20_000, size=n_cells).astype(float)
    rate = np.array([0.5, 2.0, 10.0, 50.0, 200.0, 5.0])[:n_genes]  # CP10K
    counts = rng.poisson(np.outer(depth / 1e4, rate))
    # A large filler gene carries the rest of each cell's library.
    filler = np.maximum(depth - counts.sum(axis=1), 0)
    return np.column_stack([counts, filler]), rate


def test_pie_input_and_pseudobulk_follow_their_definitions():
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(50, 5)).astype(float)
    counts[0] = 0  # an empty cell keeps zero, not NaN
    counts[0, 4] = 7
    symbols = ["g1", "G2", "C1orf1", "MT-CO1", "Z"]
    genes = ["G1", "G2", "C1ORF1", "ABSENT"]
    out = cells.gene_summaries(
        sparse.csr_matrix(counts), symbols, genes, rng=np.random.default_rng(1)
    )
    library = counts.sum(axis=1)
    cp10k = counts[:, :3] * 1e4 / library[:, None]
    assert np.allclose(out.pie_input[:3], np.log1p(cp10k).mean(axis=0), atol=1e-6)
    assert np.allclose(out.pseudobulk[:3], np.log1p(cp10k.mean(axis=0)), atol=1e-6)
    assert np.allclose(out.detected[:3], (counts[:, :3] > 0).mean(axis=0))
    assert out.listed.tolist() == [True, True, True, False]
    for value in (out.pie_input, out.pseudobulk, out.sd, out.sd_expected):
        assert np.isnan(value[3])


def test_spread_excess_is_zero_for_poisson_cells():
    counts, _ = _poisson_line(np.random.default_rng(2))
    symbols = [f"G{i}" for i in range(counts.shape[1])]
    out = cells.gene_summaries(
        sparse.csr_matrix(counts), symbols, symbols[:6], rng=np.random.default_rng(3)
    )
    assert np.allclose(out.detected, out.detected_expected, atol=0.02)
    assert np.allclose(out.sd, out.sd_expected, atol=0.03)


def test_overdispersed_cells_show_excess_spread():
    rng = np.random.default_rng(4)
    n = 4000
    depth = np.full(n, 10_000.0)
    # Half the cells express the gene at 40 CP10K, half not at all.
    rate = np.where(np.arange(n) < n // 2, 40.0, 0.0)
    gene = rng.poisson(depth / 1e4 * rate)
    counts = np.column_stack([gene, depth - gene])
    out = cells.gene_summaries(
        sparse.csr_matrix(counts), ["A", "F"], ["A"], rng=np.random.default_rng(5)
    )
    assert out.detected[0] < out.detected_expected[0] - 0.3
    assert out.sd[0] > out.sd_expected[0] + 0.5


def test_hvg_shuffle_keeps_marginals_and_removes_co_expression():
    rng = np.random.default_rng(6)
    n = 2000
    shared = rng.poisson(5.0, size=n)
    counts = np.column_stack([shared, shared, rng.poisson(3.0, size=(n, 2))])
    counts = counts + 1
    hvg = cells.hvg_cells(
        sparse.csr_matrix(counts.astype(float)),
        ["H1", "h2", "H3", "X"],
        ["H1", "H2", "H3", "H4"],
        1e4,
        rng=np.random.default_rng(7),
    )
    assert hvg.listed.tolist() == [True, True, True, False]
    for j in range(4):
        assert np.array_equal(np.sort(hvg.cells[:, j]), np.sort(hvg.shuffled[:, j]))
    assert np.corrcoef(hvg.cells[:, 0], hvg.cells[:, 1])[0, 1] > 0.5
    assert abs(np.corrcoef(hvg.shuffled[:, 0], hvg.shuffled[:, 1])[0, 1]) < 0.1
    library = counts.sum(axis=1)
    expected = np.log1p((counts[:, :3] * 1e4 / library[:, None]).mean(axis=0))
    assert np.allclose(hvg.mean_cell[:3], expected, atol=1e-5)


def test_count_shuffle_keeps_each_genes_counts():
    rng = np.random.default_rng(8)
    counts = sparse.random(300, 40, density=0.2, random_state=1, format="csr")
    counts.data = np.ceil(counts.data * 9)
    shuffled = cells.shuffle_counts(counts, rng)
    assert shuffled.shape == counts.shape
    dense, moved = counts.toarray(), shuffled.toarray()
    for j in range(dense.shape[1]):
        assert np.array_equal(np.sort(dense[:, j]), np.sort(moved[:, j]))
    assert not np.array_equal(dense, moved)


def test_cell_qc_columns():
    counts = np.array([[5, 3, 2, 0], [0, 0, 0, 10], [1, 1, 1, 1]], dtype=float)
    qc = cells.cell_qc(counts, ["MT-CO1", "rpl3", "RPS27A", "ACTB"])
    assert qc.shape == (3, len(cells.QC_COLUMNS))
    assert np.allclose(qc[:, 0], np.log([10, 10, 4]))
    assert np.allclose(qc[:, 1], [3, 1, 4])
    assert np.allclose(qc[:, 2], [0.5, 0.0, 0.25])
    assert np.allclose(qc[:, 3], [0.5, 0.0, 0.5])


def test_bulk_rescaling_to_cp10k():
    tpm = np.array([0.0, 99.0, 9999.0])
    assert np.allclose(cells.bulk_to_pie_input(np.log2(tpm + 1)), np.log1p(tpm / 100))


def test_symbol_keys_match_case_insensitively_and_drop_ambiguous():
    keys = ["C1orf112", "TP53", "Abc", "ABC"]
    assert cells.symbol_key(["C1ORF112", "TP53", "ABC", "NONE"], keys) == {
        "C1ORF112": "C1orf112",
        "TP53": "TP53",
    }


def test_line_rng_depends_on_line_and_purpose():
    draw = cells.line_rng("ACH-1", 0, "a").integers(1 << 30)
    assert draw == cells.line_rng("ACH-1", 0, "a").integers(1 << 30)
    assert draw != cells.line_rng("ACH-2", 0, "a").integers(1 << 30)
    assert draw != cells.line_rng("ACH-1", 0, "b").integers(1 << 30)


@pytest.mark.parametrize("n_cells", [50, 300])
def test_doublet_scores(n_cells):
    from src.experiments.ladder_inputs import SCRUBLET_MIN_CELLS, doublet_scores

    rng = np.random.default_rng(9)
    counts = rng.poisson(rng.gamma(0.5, 4.0, size=200), size=(n_cells, 200))
    scores = doublet_scores(counts)
    assert scores.shape == (n_cells,)
    if n_cells < SCRUBLET_MIN_CELLS:
        assert np.isnan(scores).all()
    else:
        assert np.isfinite(scores).all()
