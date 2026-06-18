# tests/test_ddgcn_graph.py
from __future__ import annotations

import numpy as np
import torch


def test_build_fold_adjacency_is_symmetric_binary() -> None:
    from ddgcn.graph import build_fold_adjacency

    pair_index = np.array([[0, 1], [1, 2], [0, 1]])  # duplicate (0,1)
    adj = build_fold_adjacency(pair_index, n_gene=3).toarray()
    assert adj.shape == (3, 3)
    assert np.array_equal(adj, adj.T)  # symmetric
    assert set(np.unique(adj)).issubset({0.0, 1.0})  # binary
    assert adj[0, 1] == 1.0 and adj[1, 0] == 1.0
    assert adj[1, 2] == 1.0 and adj[2, 1] == 1.0
    assert adj[0, 2] == 0.0


def test_build_fold_adjacency_empty() -> None:
    from ddgcn.graph import build_fold_adjacency

    adj = build_fold_adjacency(np.zeros((0, 2), dtype=int), n_gene=4).toarray()
    assert adj.shape == (4, 4)
    assert adj.sum() == 0.0


def test_normalize_adj_row_and_column_symmetric() -> None:
    import scipy.sparse as sp

    from ddgcn.graph import normalize_adj

    # adj + I on a 2-node graph with one edge -> each node degree 2
    base = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    norm = normalize_adj(base, "Row&Column").toarray()
    # D^-0.5 (A) D^-0.5 with D=diag(2,2) -> all entries 0.5
    assert np.allclose(norm, np.full((2, 2), 0.5))


def test_to_torch_sparse_roundtrip() -> None:
    import scipy.sparse as sp

    from ddgcn.graph import to_torch_sparse

    mat = sp.coo_matrix(np.array([[0.0, 2.0], [2.0, 0.0]]))
    t = to_torch_sparse(mat)
    assert t.is_sparse
    assert torch.allclose(t.to_dense(), torch.tensor([[0.0, 2.0], [2.0, 0.0]]))


def test_identity_features_shape() -> None:
    from ddgcn.graph import identity_features

    feat = identity_features(5)
    assert feat.shape == (5, 5)
    assert torch.allclose(feat, torch.eye(5))
