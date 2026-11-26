"""SLMGAE's GO views (hpc/slmgae/go_similarity.py): Wang's term similarity and the
best-match gene similarity (the measure of the released GOSemSim matrices), the weights
of the two relations, and the k-nearest-neighbour graph of the official code."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import scipy.sparse

from src.experiments.run_dir import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "go_similarity", REPO_ROOT / "hpc" / "slmgae" / "go_similarity.py"
)
go = importlib.util.module_from_spec(spec)
spec.loader.exec_module(go)

PARENTS = {"R": [], "A": [("R", 0.8)], "B": [("R", 0.8)], "C": [("A", 0.8)]}  # all is_a


def test_wang_term_similarity_on_a_small_graph():
    sim = go.term_similarity(PARENTS, ["A", "B", "C"])
    assert sim[0, 1] == pytest.approx(
        1.6 / 3.6
    )  # A, B share R: (0.8 + 0.8) / (1.8 + 1.8)
    assert sim[0, 2] == pytest.approx(3.24 / 4.24)  # A, C share A and R
    assert np.allclose(np.diag(sim), 1.0) and np.allclose(sim, sim.T)


def test_gene_similarity_is_the_best_match_average():
    sim = go.term_similarity(PARENTS, ["A", "B", "C"])
    gene_terms = scipy.sparse.csr_matrix(
        np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
    )  # g0 {A}, g1 {B, C}, g2 {}
    out = go.gene_similarity(sim, gene_terms)
    assert out[0, 1] == pytest.approx((3.24 / 4.24 + 1.6 / 3.6 + 3.24 / 4.24) / 3)
    assert out[2].tolist() == [0.0, 0.0, 0.0]


def test_part_of_weighs_0_6(tmp_path):
    obo = tmp_path / "go.obo"
    obo.write_text(
        "[Term]\nid: GO:1\n\n[Term]\nid: GO:2\nis_a: GO:1 ! x\n\n"
        "[Term]\nid: GO:3\nrelationship: part_of GO:1 ! x\n\n"
        "[Term]\nid: GO:4\nis_obsolete: true\n"
    )
    assert go.read_obo(obo) == {
        "GO:1": [],
        "GO:2": [("GO:1", 0.8)],
        "GO:3": [("GO:1", 0.6)],
    }


def test_knn_graph_equals_the_official_construction():
    rng = np.random.default_rng(0)
    sim = rng.random((30, 30))
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 0)
    top = np.zeros_like(sim)
    for i in range(30):  # official build_KNN_mateix
        keep = np.argsort(sim[i])[::-1][:5]
        top[i, keep] = sim[i, keep]
    expected = ((top + top.T) > 0).astype(float)  # official array2coo, then symmetric
    assert np.array_equal(go.knn_graph(sim, 5).toarray() > 0, expected > 0)


def test_loss_index_equals_the_official_set_construction():
    spec = importlib.util.spec_from_file_location(
        "slmgae_train", REPO_ROOT / "hpc" / "slmgae" / "train.py"
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    n, held = 7, [(0, 3), (2, 6), (5, 6)]
    x, y = np.triu_indices(n, k=1)  # official: set of all pairs minus the masked pairs
    official = set(zip(x, y)) - set(held)
    index = train.loss_index(
        n, np.array([a for a, _ in held]), np.array([b for _, b in held])
    )
    assert index.dtype == np.int32 and set(map(tuple, index.tolist())) == official


def test_released_matrix_is_a_ragged_upper_triangle(tmp_path):
    path = (
        tmp_path / "gosim.txt"
    )  # 4 genes: line i holds the values of genes i+1 onward
    path.write_text("0.1\t0.2\t0.3\t\n0.4\t0.5\t\n0.6\t\n")
    out = go.read_released(path, 4)
    assert out[0].tolist() == [0.0, 0.1, 0.2, 0.3]
    assert out[1, 2] == 0.4 and out[1, 3] == 0.5 and out[2, 3] == 0.6
    assert np.array_equal(out, out.T)
    path.write_text("0.1\t0.2\t0.3\t\n0.4\t0.5\t\n")
    with pytest.raises(ValueError):
        go.read_released(path, 5)
