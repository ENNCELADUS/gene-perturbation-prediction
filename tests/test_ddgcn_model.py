# tests/test_ddgcn_model.py
from __future__ import annotations

import numpy as np
import torch

from ddgcn.graph import (
    build_fold_adjacency,
    identity_features,
    normalize_adj,
    to_torch_sparse,
)


def _tiny_adj(n: int = 4) -> torch.Tensor:
    import scipy.sparse as sp

    pair_index = np.array([[0, 1], [2, 3]])
    adj = build_fold_adjacency(pair_index, n)
    adj = adj + sp.eye(n)
    return to_torch_sparse(normalize_adj(adj, "Row&Column"))


def test_autoencoder_forward_shapes() -> None:
    from ddgcn.model import GraphAutoEncoder

    n = 4
    adj = _tiny_adj(n)
    x1 = identity_features(n)
    x2 = torch.from_numpy(
        build_fold_adjacency(np.array([[0, 1], [2, 3]]), n).toarray()
    ).float()
    model = GraphAutoEncoder(
        nfeat=n,
        nhid1=8,
        nhid2=4,
        dropout=0.5,
        init="Kaiming",
        use_bias=False,
        is_sparse_feat1=True,
        is_sparse_feat2=True,
    )
    model.eval()
    logit1, logit2 = model(x1, x2, adj)
    assert logit1.shape == (n, n)
    assert logit2.shape == (n, n)


def test_inner_product_decoder_output_is_symmetric() -> None:
    from ddgcn.model import InnerProductDecoder

    dec = InnerProductDecoder(dropout=0.5)
    dec.eval()  # no dropout mask in eval
    z = torch.randn(5, 3)
    out1, out2 = dec(z, z)
    assert torch.allclose(out1, out1.t(), atol=1e-5)
    assert torch.allclose(out1, out2)  # same input -> same output


def test_dual_dropout_active_only_in_training() -> None:
    from ddgcn.model import GCNEncoder

    torch.manual_seed(0)
    n = 6
    adj = _tiny_adj(n)
    x1 = identity_features(n)
    x2 = identity_features(n)
    enc = GCNEncoder(
        nfeat=n,
        nhid1=8,
        nhid2=4,
        dropout=0.5,
        init="Kaiming",
        use_bias=False,
        is_sparse_feat1=True,
        is_sparse_feat2=True,
    )
    enc.eval()
    a1, _ = enc(x1, x2, adj)
    a2, _ = enc(x1, x2, adj)
    assert torch.allclose(a1, a2)  # deterministic in eval
    enc.train()
    torch.manual_seed(1)
    b1, _ = enc(x1, x2, adj)
    torch.manual_seed(2)
    b2, _ = enc(x1, x2, adj)
    assert not torch.allclose(b1, b2)  # dropout randomizes in train


def test_objective_weights_formula() -> None:
    from ddgcn.model import objective_weights

    n = 4
    target = torch.from_numpy(
        build_fold_adjacency(np.array([[0, 1], [2, 3]]), n).toarray()
    ).float()
    target = target + torch.eye(n)
    pos_weight, norm = objective_weights(target)
    e = float(target.sum())
    assert abs(pos_weight - (n**2 - e) / e) < 1e-6
    assert abs(norm - n**2 / ((n**2 - e) * 2)) < 1e-6
