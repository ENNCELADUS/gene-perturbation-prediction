"""Adjacency construction and normalization for the DDGCN port.

Mirrors the math in ``data/SL_benchmark/src/utils/ddgcn_utils.py`` but builds
the graph from our CV fold index arrays (9471-gene universe) and targets
torch 2.12 (``torch.sparse_coo_tensor``).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def build_fold_adjacency(pair_index: np.ndarray, n_gene: int) -> sp.csr_matrix:
    """Build a symmetric binary adjacency from a pair-index array.

    Args:
        pair_index: Integer array of shape ``(m, 2)`` mapping each pair to its
            two gene indices into the universe.
        n_gene: Total number of genes (matrix dimension).

    Returns:
        A symmetric binary ``csr_matrix`` of shape ``(n_gene, n_gene)``.
    """
    if len(pair_index) == 0:
        return sp.csr_matrix((n_gene, n_gene), dtype=np.float32)
    rows = pair_index[:, 0]
    cols = pair_index[:, 1]
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_gene, n_gene))
    adj = adj + adj.T
    adj = adj.tocsr()
    adj.data[:] = 1.0  # collapse duplicates / symmetric overlaps to binary
    return adj


def normalize_adj(adj: sp.spmatrix, normal_dim: str) -> sp.coo_matrix:
    """Symmetric (or row) normalization of an adjacency matrix.

    Args:
        adj: Sparse adjacency (caller adds self-loops before calling).
        normal_dim: ``"Row&Column"`` for ``D^-0.5 A D^-0.5`` or ``"Row"`` for
            ``D^-1 A``.

    Returns:
        The normalized matrix as a ``coo_matrix``.

    Raises:
        ValueError: If ``normal_dim`` is unsupported.
    """
    if normal_dim == "Row&Column":
        rowsum = np.array(adj.sum(1))
        inv = np.power(rowsum, -0.5).flatten()
        inv[np.isinf(inv)] = 0.0
        d_inv_sqrt = sp.diags(inv)
        return adj.dot(d_inv_sqrt).transpose().dot(d_inv_sqrt).tocoo()
    if normal_dim == "Row":
        rowsum = np.array(adj.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        return sp.diags(r_inv).dot(adj).tocoo()
    raise ValueError(f"unsupported normal_dim: {normal_dim!r}")


def to_torch_sparse(mat: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse COO float tensor.

    Args:
        mat: Any scipy sparse matrix.

    Returns:
        A coalesced ``torch.sparse_coo_tensor`` (float32).
    """
    coo = mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape)).coalesce()


def identity_features(n_gene: int) -> torch.Tensor:
    """Return the dense identity feature matrix used as encoder stream x1.

    Args:
        n_gene: Number of genes (feature dimension).

    Returns:
        An ``(n_gene, n_gene)`` float tensor equal to ``torch.eye(n_gene)``.
    """
    return torch.eye(n_gene)
