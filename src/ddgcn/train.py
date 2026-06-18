# src/ddgcn/train.py
"""Single-fold DDGCN training loop producing a fused score matrix.

Faithful to the vendored ``train_ddgcn.py`` loss-plateau schedule: max-epoch
cap, a minimum tolerance epoch before early-stop is considered, and a relative
loss-change stop threshold. No validation split, no best-epoch checkpointing —
the final-epoch fused score matrix is returned (zero leakage from test pairs).
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from ddgcn.config import DdgcnConfig
from ddgcn.graph import (
    build_fold_adjacency,
    identity_features,
    normalize_adj,
    to_torch_sparse,
)
from ddgcn.model import GraphAutoEncoder, objective_weights

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed torch (CPU+CUDA) and numpy for per-fold determinism."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def resolve_device() -> torch.device:
    """Return the CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fuse(logit1: torch.Tensor, logit2: torch.Tensor, rho: float) -> np.ndarray:
    """Weighted geometric mean of two logit matrices' sigmoids, diag zeroed."""
    p1 = torch.sigmoid(logit1).cpu().numpy()
    p2 = torch.sigmoid(logit2).cpu().numpy()
    fused = np.power(p1 * np.power(p2, rho), 1.0 / (1.0 + rho))
    n = fused.shape[0]
    fused[np.arange(n), np.arange(n)] = 0.0
    return fused


def train_fold(
    pos_index: np.ndarray,
    neg_index: np.ndarray,
    n_gene: int,
    config: DdgcnConfig,
    device: torch.device | None = None,
) -> np.ndarray:
    """Train one DDGCN fold and return its fused score matrix.

    Args:
        pos_index: ``(p, 2)`` train-positive gene-index pairs.
        neg_index: ``(q, 2)`` train-negative gene-index pairs.
        n_gene: Universe size (matrix dimension).
        config: Hyperparameters (dropout, lr, dims, schedule, seed, rho).
        device: Torch device; defaults to :func:`resolve_device`.

    Returns:
        Fused ``(n_gene, n_gene)`` score matrix in ``[0, 1]`` with zero diagonal.
    """
    device = device or resolve_device()
    set_seed(config.seed)

    graph_pos = build_fold_adjacency(pos_index, n_gene)
    graph_neg = build_fold_adjacency(neg_index, n_gene)

    adj_norm = to_torch_sparse(
        normalize_adj(graph_pos + sp.eye(n_gene), config.normal_dim)
    ).to(device)
    feature1 = identity_features(n_gene).to(device)
    feature2 = torch.from_numpy(graph_pos.toarray().astype(np.float32)).to(device)

    target = torch.from_numpy(graph_pos.toarray().astype(np.float32)) + torch.eye(
        n_gene
    )
    pair_mask = torch.from_numpy((graph_pos + graph_neg).toarray().astype(np.float32))
    pos_weight, norm = objective_weights(target)
    pos_weight_t = torch.tensor(pos_weight)

    model = GraphAutoEncoder(
        nfeat=n_gene,
        nhid1=config.hidden1,
        nhid2=config.hidden2,
        dropout=config.dropout,
        init=config.init_type,
        use_bias=config.use_bias,
        is_sparse_feat1=True,
        is_sparse_feat2=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, amsgrad=True)
    last_loss = 1e-5
    for epoch in range(config.max_epochs):
        model.train()
        logit1, logit2 = model(feature1, feature2, adj_norm)
        loss1 = norm * binary_cross_entropy_with_logits(
            logit1.cpu(),
            target,
            weight=pair_mask,
            pos_weight=pos_weight_t,
            reduction="mean",
        )
        loss2 = norm * binary_cross_entropy_with_logits(
            logit2.cpu(),
            target,
            weight=pair_mask,
            pos_weight=pos_weight_t,
            reduction="mean",
        )
        loss = loss1 + config.rho * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stop = (
            epoch > config.tolerance_epoch
            and abs((loss.item() - last_loss) / last_loss) < config.stop_threshold
        )
        if (
            (epoch + 1) % config.eval_interval == 0
            or stop
            or epoch + 1 >= config.max_epochs
        ):
            logger.info(
                "fold epoch %d/%d loss=%.6f", epoch + 1, config.max_epochs, loss.item()
            )
            if stop or epoch + 1 >= config.max_epochs:
                break
        last_loss = loss.item()

    model.eval()
    with torch.no_grad():
        logit1, logit2 = model(feature1, feature2, adj_norm)
    return _fuse(logit1, logit2, config.rho)
