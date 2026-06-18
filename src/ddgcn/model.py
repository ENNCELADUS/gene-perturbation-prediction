# src/ddgcn/model.py
"""DDGCN model: dual-stream GCN auto-encoder with dual dropout.

Ported from ``data/SL_benchmark/src/models/ddgcn.py`` (which matches the
official Cai et al. 2020 repo line-for-line). Changes: torch 2.12 sparse API,
``logging`` instead of ``print``, Python 3.11 type hints, no wandb/eval helpers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits  # noqa: F401
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """Kipf-style GCN layer: ``D^-0.5 A D^-0.5 X W`` (sparse-aware)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: str,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.use_bias = use_bias
        self.reset_parameters(init)

    def reset_parameters(self, init: str) -> None:
        """Initialize weights/bias per the configured scheme."""
        if init == "Xavier":
            fan_in, fan_out = self.weight.shape
            init_range = math.sqrt(6.0 / (fan_in + fan_out))
            self.weight.data.uniform_(-init_range, init_range)
            if self.use_bias:
                nn.init.constant_(self.bias, 0.0)
        elif init == "Kaiming":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.use_bias:
                fan_in, _ = self.weight.shape
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.use_bias:
                self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Propagate ``inputs`` through the normalized adjacency ``adj``."""
        if inputs.is_sparse:
            support = torch.sparse.mm(inputs, self.weight)
        else:
            support = torch.mm(inputs, self.weight)
        outputs = torch.sparse.mm(adj, support)
        if self.use_bias:
            return outputs + self.bias
        return outputs


class GCNEncoder(nn.Module):
    """Two-layer GCN encoder shared across both feature streams."""

    def __init__(
        self,
        nfeat: int,
        nhid1: int,
        nhid2: int,
        dropout: float,
        init: str,
        use_bias: bool,
        is_sparse_feat1: bool,
        is_sparse_feat2: bool,
    ) -> None:
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1, init, use_bias)
        self.gc2 = GraphConvolution(nhid1, nhid2, init, use_bias)
        self.dropout = dropout
        self.is_sparse_feat1 = is_sparse_feat1
        self.is_sparse_feat2 = is_sparse_feat2

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both streams with a shared hidden dual-dropout mask."""
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        if self.is_sparse_feat1:
            x1 = x1.to_sparse()
        if self.is_sparse_feat2:
            x2 = x2.to_sparse()
        x1 = F.relu(self.gc1(x1, adj))
        x2 = F.relu(self.gc1(x2, adj))
        if self.training:
            mask = torch.bernoulli(
                x1.data.new(x1.data.size()).fill_(1 - self.dropout)
            ) / (1 - self.dropout)
            x1 = x1 * mask
            x2 = x2 * mask
        x1 = self.gc2(x1, adj)
        x2 = self.gc2(x2, adj)
        return x1, x2


class InnerProductDecoder(nn.Module):
    """Link-prediction decoder: ``Z Z^T`` per stream with shared dropout mask."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(
        self, inputs1: torch.Tensor, inputs2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode both embeddings into reconstructed adjacency logits."""
        if self.training:
            mask = torch.bernoulli(
                inputs1.data.new(inputs1.data.size()).fill_(1 - self.dropout)
            ) / (1 - self.dropout)
            inputs1 = inputs1 * mask
            inputs2 = inputs2 * mask
        outputs1 = torch.mm(inputs1, inputs1.t())
        outputs2 = torch.mm(inputs2, inputs2.t())
        return outputs1, outputs2


class GraphAutoEncoder(nn.Module):
    """Full DDGCN: dual-stream GCN encoder + inner-product decoder."""

    def __init__(
        self,
        nfeat: int,
        nhid1: int,
        nhid2: int,
        dropout: float,
        init: str,
        use_bias: bool,
        is_sparse_feat1: bool,
        is_sparse_feat2: bool,
    ) -> None:
        super().__init__()
        self.encoder = GCNEncoder(
            nfeat,
            nhid1,
            nhid2,
            dropout,
            init,
            use_bias,
            is_sparse_feat1,
            is_sparse_feat2,
        )
        self.decoder = InnerProductDecoder(dropout)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two reconstructed adjacency logit matrices."""
        z1, z2 = self.encoder(x1, x2, adj)
        return self.decoder(z1, z2)


def objective_weights(target_adj: torch.Tensor) -> tuple[float, float]:
    """Compute class-balancing weights for the reconstruction BCE loss.

    Args:
        target_adj: Target adjacency (train-positive graph + identity).

    Returns:
        ``(pos_weight, norm)`` where ``pos_weight = (N^2 - E) / E`` and
        ``norm = N^2 / (2 (N^2 - E))``, ``E = target_adj.sum()``.
    """
    num_edges = float(target_adj.sum())
    num_nodes = target_adj.shape[0]
    pos_weight = (num_nodes**2 - num_edges) / num_edges
    norm = num_nodes**2 / ((num_nodes**2 - num_edges) * 2)
    return pos_weight, norm
