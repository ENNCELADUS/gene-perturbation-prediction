# DDGCN K562 SL-Pair Reproduction (exp10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the DDGCN model (Dual-Dropout GCN, Cai et al. 2020) into `src/ddgcn/` and evaluate it on the K562 SL-pair benchmark under the exp06/07/08/09 CV1/CV2/CV3 protocol and official per-anchor metrics.

**Architecture:** A clean PyTorch port of the model/loss/training loop (reference: `data/SL_benchmark/src/models/ddgcn.py`), wired to the existing `sl_benchmark_baseline` protocol harness via a `score_matrix(symbols, gene_effects)` producer (the same seam exp08 uses). One DDGCN trained per `(CV-split, fold)` on that fold's train-positive SL adjacency; produces a 9471x9471 fused score matrix scored by `official_classification_metrics` + `official_ranking_metrics`.

**Tech Stack:** Python 3.11, PyTorch 2.12, NumPy, SciPy (sparse), pandas, scikit-learn, PyYAML. `uv` for all invocations. pytest for tests.

## Global Constraints

- Pinned hyperparameters (official defaults): `dropout=0.5`, `lr=0.01`.
- Candidate universe is exactly **9471** genes (K562-filtered). Assert it. Never use DDGCN's native `num_node=9845`.
- Gene key columns: `gene_a_unified_id` / `gene_b_unified_id` (integer ids; what `_build_gene_universe` keys on).
- Metrics come ONLY from `sl_benchmark_baseline.metrics.official_classification_metrics` and `official_ranking_metrics`. Never reimplement; never use DDGCN-native `cal_metrics` / `Evaluator` / flat F1@0.987.
- Reuse verbatim (import, never copy): `load_benchmark`, `fold_split` (`sl_benchmark_baseline.data`); `GeneUniverse`, `_build_gene_universe`, `_pair_indices`, `_metric_rows`, `_summarize` (`sl_benchmark_baseline.evaluate`).
- Stopping: faithful loss-plateau, NO validation split, final-epoch score matrix. Test pairs never touch training.
- Negatives: Rand 1:1, CV1/CV2/CV3 only. Train/test partition comes from `fold_split()` on our CSVs.
- Adjacency is symmetric (`A = A | A.T`, binary), diagonal handled by the `+I` in normalization; final score-matrix diagonal zeroed.
- torch 2.12: use `torch.sparse_coo_tensor`, NOT deprecated `torch.sparse.FloatTensor`.
- Code style: absolute imports, Google-style docstrings, no `print` in library code (use `logging`), functions < 50 lines, files < 600 lines, no hardcoded paths/thresholds (use config), no bare `except`.
- All commands prefixed with `uv run`. Default input CSV: `data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv`. Default output dir: `results/experiments/10_k562_sl_pair_ddgcn/run`.
- Add `src/ddgcn` to `[tool.hatch.build.targets.wheel] packages` in `pyproject.toml`.
- Commit after each task with Conventional Commits (`feat`/`test`/`docs`/`chore`).

---

### Task 1: Config dataclass + YAML loader

**Files:**
- Create: `src/ddgcn/__init__.py`
- Create: `src/ddgcn/config.py`
- Test: `tests/test_ddgcn_config.py`
- Modify: `pyproject.toml` (add `src/ddgcn` to wheel packages)

**Interfaces:**
- Produces: `DdgcnConfig` frozen dataclass; `load_config(path: Path) -> DdgcnConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ddgcn_config.py
from __future__ import annotations

from pathlib import Path


def test_defaults_match_official_ddgcn() -> None:
    from ddgcn.config import DdgcnConfig

    c = DdgcnConfig()
    assert c.dropout == 0.5
    assert c.lr == 0.01
    assert c.hidden1 == 512
    assert c.hidden2 == 256
    assert c.init_type == "Kaiming"
    assert c.use_bias is False
    assert c.rho == 1.0
    assert c.max_epochs == 2000
    assert c.tolerance_epoch == 1000
    assert c.stop_threshold == 1e-5
    assert c.eval_interval == 50
    assert c.seed == 456
    assert c.ranking_k == (10, 20, 50)
    assert c.folds == (0, 1, 2, 3, 4)
    assert c.split_types is None


def test_load_config_coerces_paths_and_tuples(tmp_path: Path) -> None:
    from ddgcn.config import load_config

    yaml_text = (
        "input_csv: data/x.csv\n"
        "output_dir: results/y\n"
        "split_types: [CV1, CV2]\n"
        "folds: [0, 1]\n"
        "ranking_k: [10, 20]\n"
        "dropout: 0.5\n"
        "lr: 0.01\n"
    )
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    c = load_config(p)
    assert c.input_csv == Path("data/x.csv")
    assert c.output_dir == Path("results/y")
    assert c.split_types == ("CV1", "CV2")
    assert c.folds == (0, 1)
    assert c.ranking_k == (10, 20)


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    from ddgcn.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("bogus_key: 1\n")
    try:
        load_config(p)
    except ValueError as exc:
        assert "bogus_key" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown key")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_ddgcn_config.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ddgcn'`)

- [ ] **Step 3: Create the package marker and config**

```python
# src/ddgcn/__init__.py
"""DDGCN reproduction on the K562 SL-pair benchmark (exp10)."""
```

```python
# src/ddgcn/config.py
"""Configuration for the exp10 DDGCN reproduction run."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DdgcnConfig:
    """Defaults and hyperparameters for the DDGCN reproduction.

    Defaults match the official Dual-Dropout GCN repo (Cai et al. 2020) and the
    vendored port in ``data/SL_benchmark/src/``.

    Attributes:
        input_csv: Canonical all-CV balanced benchmark CSV.
        output_dir: Run directory for metrics and manifest.
        split_types: CV splits to evaluate; ``None`` auto-discovers the input.
        folds: CV fold ids to evaluate.
        ranking_k: Cutoffs for NDCG/Recall/Precision@k.
        seed: Per-fold seed for torch/numpy.
        dropout: Dropout probability (input, hidden, decoder).
        lr: Adam learning rate.
        hidden1: First GCN layer output dim.
        hidden2: Second GCN layer (node embedding) dim.
        init_type: Weight init scheme (``"Kaiming"``/``"Xavier"``/other).
        use_bias: Whether GCN layers use a bias term.
        rho: Geometric-mean / second-stream loss weight.
        normal_dim: Adjacency normalization mode (``"Row&Column"``/``"Row"``).
        max_epochs: Maximum training epochs per fold.
        tolerance_epoch: Minimum epochs before early-stop is considered.
        stop_threshold: Relative loss-change early-stop threshold.
        eval_interval: Epoch cadence for the early-stop check.
    """

    input_csv: Path = Path(
        "data/SL_benchmark/derived/k562_depmap_rand_1to1/"
        "all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv"
    )
    output_dir: Path = Path("results/experiments/10_k562_sl_pair_ddgcn/run")
    split_types: tuple[str, ...] | None = None
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    ranking_k: tuple[int, ...] = (10, 20, 50)
    seed: int = 456

    dropout: float = 0.5
    lr: float = 0.01
    hidden1: int = 512
    hidden2: int = 256
    init_type: str = "Kaiming"
    use_bias: bool = False
    rho: float = 1.0
    normal_dim: str = "Row&Column"
    max_epochs: int = 2000
    tolerance_epoch: int = 1000
    stop_threshold: float = 1e-5
    eval_interval: int = 50


_PATH_FIELDS = {"input_csv", "output_dir"}
_TUPLE_FIELDS = {"split_types", "folds", "ranking_k"}


def load_config(path: Path) -> DdgcnConfig:
    """Load a :class:`DdgcnConfig` from YAML, coercing paths and tuples.

    Args:
        path: Path to a YAML file with a subset of ``DdgcnConfig`` fields.

    Returns:
        The constructed :class:`DdgcnConfig`.

    Raises:
        ValueError: If the YAML contains keys not present on ``DdgcnConfig``.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    valid = {f.name for f in fields(DdgcnConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    kwargs: dict[str, object] = {}
    for key, value in raw.items():
        if key in _PATH_FIELDS and value is not None:
            kwargs[key] = Path(value)
        elif key in _TUPLE_FIELDS and value is not None:
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return DdgcnConfig(**kwargs)
```

- [ ] **Step 4: Add the package to the wheel build**

In `pyproject.toml`, change the wheel packages line to include `src/ddgcn`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/dependency_baseline", "src/aivc_model", "src/sl_benchmark_baseline", "src/sl_dl_model", "src/ddgcn"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ddgcn_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/ddgcn/__init__.py src/ddgcn/config.py tests/test_ddgcn_config.py pyproject.toml
git commit -m "feat: add ddgcn config dataclass and yaml loader (exp10)"
```

---

### Task 2: Graph utilities (adjacency build, normalize, sparse, features)

**Files:**
- Create: `src/ddgcn/graph.py`
- Test: `tests/test_ddgcn_graph.py`

**Interfaces:**
- Consumes: `GeneUniverse` from `sl_benchmark_baseline.evaluate` (uses `.symbols` length + `_pair_indices` output).
- Produces:
  - `build_fold_adjacency(pair_index: np.ndarray, n_gene: int) -> scipy.sparse.csr_matrix` — symmetric binary adjacency from a `(m, 2)` index array.
  - `normalize_adj(adj: csr_matrix, normal_dim: str) -> scipy.sparse.coo_matrix` — `D^-0.5 (adj) D^-0.5` for `"Row&Column"`.
  - `to_torch_sparse(mat) -> torch.Tensor` — scipy sparse -> torch sparse COO float tensor.
  - `identity_features(n_gene: int) -> torch.Tensor` — dense `eye(n)` float tensor.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_ddgcn_graph.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ddgcn.graph'`)

- [ ] **Step 3: Write the implementation**

```python
# src/ddgcn/graph.py
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
    return torch.sparse_coo_tensor(
        indices, values, torch.Size(coo.shape)
    ).coalesce()


def identity_features(n_gene: int) -> torch.Tensor:
    """Return the dense identity feature matrix used as encoder stream x1.

    Args:
        n_gene: Number of genes (feature dimension).

    Returns:
        An ``(n_gene, n_gene)`` float tensor equal to ``torch.eye(n_gene)``.
    """
    return torch.eye(n_gene)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ddgcn_graph.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ddgcn/graph.py tests/test_ddgcn_graph.py
git commit -m "feat: add ddgcn graph adjacency utilities (exp10)"
```

---

### Task 3: Model (encoder, decoder, autoencoder) + objective weights

**Files:**
- Create: `src/ddgcn/model.py`
- Test: `tests/test_ddgcn_model.py`

**Interfaces:**
- Produces:
  - `GraphConvolution(in_features, out_features, init, use_bias=False)` (`nn.Module`, `forward(inputs, adj)`).
  - `GCNEncoder(nfeat, nhid1, nhid2, dropout, init, use_bias, is_sparse_feat1, is_sparse_feat2)` (`forward(x1, x2, adj) -> (z1, z2)`).
  - `InnerProductDecoder(dropout)` (`forward(z1, z2) -> (logit1, logit2)`).
  - `GraphAutoEncoder(nfeat, nhid1, nhid2, dropout, init, use_bias, is_sparse_feat1, is_sparse_feat2)` (`forward(x1, x2, adj) -> (logit1, logit2)`).
  - `objective_weights(target_adj: torch.Tensor) -> tuple[float, float]` returning `(pos_weight, norm)`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_ddgcn_model.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ddgcn.model'`)

- [ ] **Step 3: Write the implementation (layers + encoder)**

```python
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
```

- [ ] **Step 4: Append decoder, autoencoder, and objective weights**

Append to `src/ddgcn/model.py`:

```python
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
            nfeat, nhid1, nhid2, dropout, init, use_bias,
            is_sparse_feat1, is_sparse_feat2,
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ddgcn_model.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/ddgcn/model.py tests/test_ddgcn_model.py
git commit -m "feat: port ddgcn dual-dropout gcn auto-encoder (exp10)"
```

---

### Task 4: Per-fold training loop -> score matrix

**Files:**
- Create: `src/ddgcn/train.py`
- Test: `tests/test_ddgcn_train.py`

**Interfaces:**
- Consumes: `DdgcnConfig`; `build_fold_adjacency`, `normalize_adj`, `to_torch_sparse`, `identity_features` (Task 2); `GraphAutoEncoder`, `objective_weights` (Task 3).
- Produces:
  - `set_seed(seed: int) -> None` — seeds torch + numpy + cuda.
  - `resolve_device() -> torch.device` — cuda if available else cpu.
  - `train_fold(pos_index: np.ndarray, neg_index: np.ndarray, n_gene: int, config: DdgcnConfig, device: torch.device | None = None) -> np.ndarray` — trains one fold, returns the fused `(n_gene, n_gene)` score matrix with diagonal zeroed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ddgcn_train.py
from __future__ import annotations

import dataclasses

import numpy as np

from ddgcn.config import DdgcnConfig


def _fast_config() -> DdgcnConfig:
    # Tiny + fast: few epochs, small hidden dims, low tolerance for early stop.
    return dataclasses.replace(
        DdgcnConfig(),
        hidden1=8,
        hidden2=4,
        max_epochs=20,
        tolerance_epoch=2,
        eval_interval=5,
    )


def test_train_fold_returns_zero_diag_score_matrix() -> None:
    from ddgcn.train import train_fold

    n = 8
    pos_index = np.array([[0, 1], [2, 3], [4, 5]])
    neg_index = np.array([[0, 4], [1, 5], [2, 6]])
    sm = train_fold(pos_index, neg_index, n_gene=n, config=_fast_config())
    assert sm.shape == (n, n)
    assert np.isfinite(sm).all()
    assert np.allclose(np.diag(sm), 0.0)


def test_train_fold_is_deterministic_for_fixed_seed() -> None:
    from ddgcn.train import train_fold

    n = 8
    pos_index = np.array([[0, 1], [2, 3], [4, 5]])
    neg_index = np.array([[0, 4], [1, 5], [2, 6]])
    cfg = _fast_config()
    sm1 = train_fold(pos_index, neg_index, n_gene=n, config=cfg)
    sm2 = train_fold(pos_index, neg_index, n_gene=n, config=cfg)
    assert np.allclose(sm1, sm2)


def test_train_fold_scores_in_unit_interval() -> None:
    from ddgcn.train import train_fold

    n = 8
    pos_index = np.array([[0, 1], [2, 3], [4, 5]])
    neg_index = np.array([[0, 4], [1, 5], [2, 6]])
    sm = train_fold(pos_index, neg_index, n_gene=n, config=_fast_config())
    # Fused geometric mean of two sigmoids -> within [0, 1].
    assert sm.min() >= 0.0
    assert sm.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_ddgcn_train.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ddgcn.train'`)

- [ ] **Step 3: Write the implementation**

```python
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

    target = torch.from_numpy(
        graph_pos.toarray().astype(np.float32)
    ) + torch.eye(n_gene)
    pair_mask = torch.from_numpy(
        (graph_pos + graph_neg).toarray().astype(np.float32)
    )
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
            logit1.cpu(), target, weight=pair_mask,
            pos_weight=pos_weight_t, reduction="mean",
        )
        loss2 = norm * binary_cross_entropy_with_logits(
            logit2.cpu(), target, weight=pair_mask,
            pos_weight=pos_weight_t, reduction="mean",
        )
        loss = loss1 + config.rho * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stop = (
            epoch > config.tolerance_epoch
            and abs((loss.item() - last_loss) / last_loss) < config.stop_threshold
        )
        if (epoch + 1) % config.eval_interval == 0 or stop or epoch + 1 >= config.max_epochs:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ddgcn_train.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ddgcn/train.py tests/test_ddgcn_train.py
git commit -m "feat: add ddgcn per-fold training loop (exp10)"
```

---

### Task 5: Producer + per-fold protocol wrapper

**Files:**
- Create: `src/ddgcn/scoring.py`
- Test: `tests/test_ddgcn_scoring.py`

**Interfaces:**
- Consumes: `fold_split` (`sl_benchmark_baseline.data`); `_build_gene_universe`, `_pair_indices`, `_metric_rows` (`sl_benchmark_baseline.evaluate`); `train_fold`, `resolve_device` (Task 4); `DdgcnConfig` (Task 1).
- Produces:
  - `DdgcnProducer(config, device=None)` with `score_matrix_for_fold(pos_index, neg_index, n_gene) -> np.ndarray`.
  - `run_fold_ddgcn(frame: pd.DataFrame, split_type: str, fold_id: int, config: DdgcnConfig, universe, device=None) -> list[dict]` — returns long-form metric rows (model name `"ddgcn"`, slice `"full_universe"`).

This mirrors the exp08 `run_fold_with_producer` data flow: build universe once, `fold_split`, `_pair_indices` for pos/neg/seen, train, score, `_metric_rows`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ddgcn_scoring.py
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from ddgcn.config import DdgcnConfig


def _fast_config() -> DdgcnConfig:
    return dataclasses.replace(
        DdgcnConfig(), hidden1=8, hidden2=4, max_epochs=15,
        tolerance_epoch=2, eval_interval=5,
    )


def _toy_frame() -> pd.DataFrame:
    # 6 genes, one CV1 fold, train+test pos/neg rows.
    rows = [
        # split_type, fold_id, split_role, sl_label, a_id, b_id, a_sym, b_sym, a_eff, b_eff
        ("CV1", 0, "train", 1, 0, 1, "G0", "G1", -0.5, -0.4),
        ("CV1", 0, "train", 1, 2, 3, "G2", "G3", -0.6, -0.3),
        ("CV1", 0, "train", 0, 0, 4, "G0", "G4", -0.5, 0.1),
        ("CV1", 0, "train", 0, 1, 5, "G1", "G5", -0.4, 0.2),
        ("CV1", 0, "test", 1, 0, 2, "G0", "G2", -0.5, -0.6),
        ("CV1", 0, "test", 0, 3, 5, "G3", "G5", -0.3, 0.2),
    ]
    cols = [
        "split_type", "fold_id", "split_role", "sl_label",
        "gene_a_unified_id", "gene_b_unified_id",
        "gene_a_symbol", "gene_b_symbol",
        "gene_a_k562_gene_effect", "gene_b_k562_gene_effect",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_producer_score_matrix_shape_and_diag() -> None:
    from ddgcn.scoring import DdgcnProducer

    prod = DdgcnProducer(_fast_config())
    pos_index = np.array([[0, 1], [2, 3]])
    neg_index = np.array([[0, 4], [1, 5]])
    sm = prod.score_matrix_for_fold(pos_index, neg_index, n_gene=6)
    assert sm.shape == (6, 6)
    assert np.allclose(np.diag(sm), 0.0)


def test_run_fold_ddgcn_emits_official_metric_rows() -> None:
    from sl_benchmark_baseline.evaluate import _build_gene_universe

    from ddgcn.scoring import run_fold_ddgcn

    frame = _toy_frame()
    universe = _build_gene_universe(frame)
    rows = run_fold_ddgcn(frame, "CV1", 0, _fast_config(), universe)
    assert len(rows) > 0
    assert {r["model"] for r in rows} == {"ddgcn"}
    assert {r["split_type"] for r in rows} == {"CV1"}
    assert {r["slice"] for r in rows} == {"full_universe"}
    metrics = {r["metric"] for r in rows}
    assert "auroc" in metrics
    assert "ndcg@10" in metrics
    for r in rows:
        assert np.isfinite(r["value"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_ddgcn_scoring.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ddgcn.scoring'`)

- [ ] **Step 3: Write the implementation**

```python
# src/ddgcn/scoring.py
"""Bridge the DDGCN model to the sl_benchmark_baseline protocol harness.

Mirrors ``sl_dl_model/scoring.py``: build the gene universe once, slice the
fold, map pairs to universe indices, train DDGCN on train-positive/negative
adjacency, and feed the fused score matrix to the official metric functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from sl_benchmark_baseline.data import fold_split
from sl_benchmark_baseline.evaluate import (
    GeneUniverse,
    _metric_rows,
    _pair_indices,
)

from ddgcn.config import DdgcnConfig
from ddgcn.train import resolve_device, train_fold


class DdgcnProducer:
    """Trains a DDGCN per fold and returns its fused score matrix."""

    def __init__(self, config: DdgcnConfig, device: torch.device | None = None) -> None:
        """Store config and resolve the compute device once.

        Args:
            config: DDGCN hyperparameters.
            device: Torch device; defaults to :func:`resolve_device`.
        """
        self.config = config
        self.device = device or resolve_device()

    def score_matrix_for_fold(
        self, pos_index: np.ndarray, neg_index: np.ndarray, n_gene: int
    ) -> np.ndarray:
        """Train on this fold's train edges and return the fused score matrix.

        Args:
            pos_index: ``(p, 2)`` train-positive gene-index pairs.
            neg_index: ``(q, 2)`` train-negative gene-index pairs.
            n_gene: Universe size.

        Returns:
            Fused ``(n_gene, n_gene)`` score matrix with zero diagonal.
        """
        return train_fold(pos_index, neg_index, n_gene, self.config, self.device)


def run_fold_ddgcn(
    frame: pd.DataFrame,
    split_type: str,
    fold_id: int,
    config: DdgcnConfig,
    universe: GeneUniverse,
    device: torch.device | None = None,
) -> list[dict[str, object]]:
    """Train DDGCN on one fold and return official long-form metric rows.

    Args:
        frame: Full benchmark DataFrame (all splits/folds/roles).
        split_type: CV split type (``"CV1"``/``"CV2"``/``"CV3"``).
        fold_id: Fold id to evaluate.
        config: DDGCN hyperparameters.
        universe: Prebuilt gene universe from ``_build_gene_universe(frame)``.
        device: Torch device; defaults to :func:`resolve_device`.

    Returns:
        Long-form metric row dicts (model ``"ddgcn"``, slice
        ``"full_universe"``).
    """
    train_df, test_df = fold_split(frame, split_type, fold_id)
    n_gene = len(universe.symbols)

    train_pos = train_df[train_df["sl_label"] == 1]
    train_neg = train_df[train_df["sl_label"] == 0]
    test_pos = test_df[test_df["sl_label"] == 1]
    test_neg = test_df[test_df["sl_label"] == 0]

    pos_train_index = _pair_indices(train_pos, universe)
    neg_train_index = _pair_indices(train_neg, universe)
    pos_index = _pair_indices(test_pos, universe)
    neg_index = _pair_indices(test_neg, universe)
    seen_index = _pair_indices(train_pos, universe)

    producer = DdgcnProducer(config, device)
    score_matrix = producer.score_matrix_for_fold(
        pos_train_index, neg_train_index, n_gene
    )
    return _metric_rows(
        split_type,
        "ddgcn",
        fold_id,
        "full_universe",
        score_matrix,
        pos_index,
        neg_index,
        seen_index,
        config.ranking_k,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ddgcn_scoring.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ddgcn/scoring.py tests/test_ddgcn_scoring.py
git commit -m "feat: add ddgcn producer and protocol fold wrapper (exp10)"
```

---

### Task 6: CV runner + artifact writer

**Files:**
- Create: `src/ddgcn/evaluate.py`
- Test: `tests/test_ddgcn_evaluate.py`

**Interfaces:**
- Consumes: `load_benchmark` (`sl_benchmark_baseline.data`); `_build_gene_universe`, `_summarize`, `VALID_SPLIT_TYPES` (`sl_benchmark_baseline.evaluate` / `.data`); `run_fold_ddgcn` (Task 5); `DdgcnConfig` (Task 1).
- Produces:
  - `run_cv(config: DdgcnConfig) -> pd.DataFrame` — runs all `split_types x folds`, writes artifacts (flat + per-split subdirs + combined summary + manifest), returns the summary DataFrame.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ddgcn_evaluate.py
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd

from ddgcn.config import DdgcnConfig


def _toy_csv(path: Path) -> None:
    rows = [
        ("CV1", 0, "train", 1, 0, 1, "G0", "G1", -0.5, -0.4),
        ("CV1", 0, "train", 1, 2, 3, "G2", "G3", -0.6, -0.3),
        ("CV1", 0, "train", 0, 0, 4, "G0", "G4", -0.5, 0.1),
        ("CV1", 0, "train", 0, 1, 5, "G1", "G5", -0.4, 0.2),
        ("CV1", 0, "test", 1, 0, 2, "G0", "G2", -0.5, -0.6),
        ("CV1", 0, "test", 0, 3, 5, "G3", "G5", -0.3, 0.2),
    ]
    cols = [
        "split_type", "fold_id", "split_role", "sl_label",
        "gene_a_unified_id", "gene_b_unified_id",
        "gene_a_symbol", "gene_b_symbol",
        "gene_a_k562_gene_effect", "gene_b_k562_gene_effect",
    ]
    frame = pd.DataFrame(rows, columns=cols)
    # load_benchmark requires a pair_id column; synthesize a unique one.
    frame["pair_id"] = [f"p{i}" for i in range(len(frame))]
    frame.to_csv(path, index=False)


def test_run_cv_writes_artifacts(tmp_path: Path) -> None:
    from ddgcn.evaluate import run_cv

    csv = tmp_path / "bench.csv"
    _toy_csv(csv)
    out = tmp_path / "run"
    cfg = dataclasses.replace(
        DdgcnConfig(),
        input_csv=csv,
        output_dir=out,
        split_types=("CV1",),
        folds=(0,),
        hidden1=8,
        hidden2=4,
        max_epochs=15,
        tolerance_epoch=2,
        eval_interval=5,
    )
    summary = run_cv(cfg)

    assert (out / "fold_metrics.csv").exists()
    assert (out / "summary.csv").exists()
    assert (out / "manifest.json").exists()
    assert (out / "official_metrics_summary.csv").exists()
    assert (out / "CV1" / "fold_metrics.csv").exists()
    assert (out / "CV1" / "summary.csv").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["candidate_gene_count"] == 6
    assert manifest["split_types"] == ["CV1"]
    assert manifest["dropout"] == 0.5
    assert manifest["lr"] == 0.01
    assert "input_csv_sha256" in manifest
    assert "train_edge_counts" in manifest

    assert set(summary.columns) == {
        "split_type", "model", "slice", "metric", "mean", "std"
    }
    assert (summary["model"] == "ddgcn").all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_ddgcn_evaluate.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ddgcn.evaluate'`)

- [ ] **Step 3: Write the implementation**

```python
# src/ddgcn/evaluate.py
"""CV runner and artifact writer for the DDGCN reproduction (exp10).

Reuses the sl_benchmark_baseline protocol (load, universe, summarize) and the
exp08 artifact layout (flat files + per-split subdirs + combined summary +
manifest). The metric path is never reimplemented here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from sl_benchmark_baseline.data import VALID_SPLIT_TYPES, fold_split, load_benchmark
from sl_benchmark_baseline.evaluate import _build_gene_universe, _summarize

from ddgcn.config import DdgcnConfig
from ddgcn.scoring import run_fold_ddgcn

logger = logging.getLogger(__name__)

OFFICIAL_METRIC_SOURCE = "data/SL_benchmark/src/preprocess.py:cal_metrics"
RANKING_SEMANTICS = (
    "Per-anchor candidate-partner ranking over the K562-filtered gene universe; "
    "train-positive pairs masked from candidate rankings. Identical to exp06-09."
)
MODEL_NOTES = (
    "DDGCN is transductive and featureless (learns from the SL adjacency "
    "itself). Strong CV1 reflects topology/degree memorization; CV3 cold-start "
    "is expected near-floor. CV2/CV3 are the meaningful comparison surfaces."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _resolve_split_types(
    frame: pd.DataFrame, requested: tuple[str, ...] | None
) -> tuple[str, ...]:
    available = set(frame["split_type"].unique())
    if requested is None:
        return tuple(s for s in VALID_SPLIT_TYPES if s in available)
    invalid = [s for s in requested if s not in VALID_SPLIT_TYPES]
    if invalid:
        raise ValueError(f"split_types must be in {VALID_SPLIT_TYPES}, got {invalid}")
    missing = [s for s in requested if s not in available]
    if missing:
        raise ValueError(f"requested split_types not present in input: {missing}")
    return requested


def _train_edge_counts(
    frame: pd.DataFrame, split_types: tuple[str, ...], folds: tuple[int, ...]
) -> dict[str, int]:
    """Count train-positive rows per (split, fold) for the manifest."""
    counts: dict[str, int] = {}
    for split_type in split_types:
        for fold_id in folds:
            train_df, _ = fold_split(frame, split_type, fold_id)
            n_pos = int((train_df["sl_label"] == 1).sum())
            counts[f"{split_type}_fold{fold_id}"] = n_pos
    return counts


def _build_manifest(
    config: DdgcnConfig,
    split_types: tuple[str, ...],
    candidate_gene_count: int,
    train_edge_counts: dict[str, int],
) -> dict[str, object]:
    """Assemble the run manifest dict."""
    import torch

    return {
        "input_csv": str(config.input_csv),
        "input_csv_sha256": _file_sha256(config.input_csv),
        "split_types": list(split_types),
        "folds": list(config.folds),
        "ranking_k": list(config.ranking_k),
        "candidate_gene_count": candidate_gene_count,
        "seed": config.seed,
        "dropout": config.dropout,
        "lr": config.lr,
        "rho": config.rho,
        "hidden1": config.hidden1,
        "hidden2": config.hidden2,
        "init_type": config.init_type,
        "use_bias": config.use_bias,
        "max_epochs": config.max_epochs,
        "tolerance_epoch": config.tolerance_epoch,
        "stop_threshold": config.stop_threshold,
        "train_edge_counts": train_edge_counts,
        "torch_version": torch.__version__,
        "model": "ddgcn",
        "official_metric_source": OFFICIAL_METRIC_SOURCE,
        "ranking_semantics": RANKING_SEMANTICS,
        "model_notes": MODEL_NOTES,
    }


def _write_split_dirs(
    output_dir: Path,
    fold_metrics: pd.DataFrame,
    split_types: tuple[str, ...],
    config: DdgcnConfig,
    candidate_gene_count: int,
    train_edge_counts: dict[str, int],
) -> None:
    """Write per-split subdirectories with their own metrics + manifest."""
    for split_type in split_types:
        split_rows = fold_metrics[fold_metrics["split_type"] == split_type]
        if split_rows.empty:
            continue
        split_dir = output_dir / split_type
        split_dir.mkdir(parents=True, exist_ok=True)
        split_rows.to_csv(split_dir / "fold_metrics.csv", index=False)
        _summarize(split_rows).to_csv(split_dir / "summary.csv", index=False)
        split_manifest = _build_manifest(
            config, (split_type,), candidate_gene_count, train_edge_counts
        )
        (split_dir / "manifest.json").write_text(json.dumps(split_manifest, indent=2))


def run_cv(config: DdgcnConfig) -> pd.DataFrame:
    """Run DDGCN across split_types x folds, write artifacts, return summary.

    Args:
        config: Run configuration.

    Returns:
        Summary DataFrame with columns
        ``split_type, model, slice, metric, mean, std``.

    Raises:
        RuntimeError: If no metric rows were produced.
    """
    frame = load_benchmark(config.input_csv)
    universe = _build_gene_universe(frame)
    candidate_gene_count = len(universe.symbols)
    split_types = _resolve_split_types(frame, config.split_types)

    all_rows: list[dict[str, object]] = []
    for split_type in split_types:
        for fold_id in config.folds:
            logger.info("training DDGCN: split=%s fold=%d", split_type, fold_id)
            all_rows.extend(
                run_fold_ddgcn(frame, split_type, fold_id, config, universe)
            )
    if not all_rows:
        raise RuntimeError("no metric rows produced; check split_types and data")

    fold_metrics = pd.DataFrame(all_rows).sort_values(
        ["split_type", "fold_id", "model", "slice", "metric"]
    ).reset_index(drop=True)
    summary = _summarize(fold_metrics)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_edge_counts = _train_edge_counts(frame, split_types, config.folds)
    manifest = _build_manifest(
        config, split_types, candidate_gene_count, train_edge_counts
    )

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    summary.to_csv(output_dir / "official_metrics_summary.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_split_dirs(
        output_dir, fold_metrics, split_types, config,
        candidate_gene_count, train_edge_counts,
    )
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ddgcn_evaluate.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/ddgcn/evaluate.py tests/test_ddgcn_evaluate.py
git commit -m "feat: add ddgcn cv runner and artifact writer (exp10)"
```

---

### Task 7: CLI entrypoint (`python -m ddgcn`)

**Files:**
- Create: `src/ddgcn/__main__.py`
- Test: `tests/test_ddgcn_cli.py`

**Interfaces:**
- Consumes: `load_config` (Task 1), `run_cv` (Task 6).
- Produces: `main(argv: list[str] | None = None) -> None`; CLI `run-cv --config <yaml> [--split-type CV1] [--log-file <path>]`.

The `--split-type` flag overrides `config.split_types` for partial reruns. `--help` must work without importing torch (lazy imports in `main`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ddgcn_cli.py
from __future__ import annotations

import subprocess
import sys


def test_help_works_without_torch() -> None:
    # --help must not require torch/cuda; exits 0.
    result = subprocess.run(
        [sys.executable, "-m", "ddgcn", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "run-cv" in result.stdout


def test_parser_run_cv_requires_config() -> None:
    from ddgcn.__main__ import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["run-cv", "--config", "x.yaml"])
    assert args.command == "run-cv"
    assert str(args.config) == "x.yaml"
    assert args.split_type is None


def test_parser_accepts_split_type_override() -> None:
    from ddgcn.__main__ import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["run-cv", "--config", "x.yaml", "--split-type", "CV2"]
    )
    assert args.split_type == "CV2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_ddgcn_cli.py -v`
Expected: FAIL (`ModuleNotFoundError` / no `__main__`)

- [ ] **Step 3: Write the implementation**

```python
# src/ddgcn/__main__.py
"""CLI for the exp10 DDGCN reproduction.

Usage::

    uv run python -m ddgcn run-cv \\
        --config configs/experiments/10_k562_sl_pair_ddgcn/ddgcn_cv.yaml

Add ``--split-type CV2`` to run a single split (partial reruns). ``--help``
works without importing torch (imports are deferred into ``main``).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="ddgcn",
        description="exp10 DDGCN reproduction on the K562 SL-pair benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run-cv",
        help="Run CV and write official metrics to the configured output dir.",
    )
    run.add_argument(
        "--config", type=Path, required=True, help="Path to a DdgcnConfig YAML file."
    )
    run.add_argument(
        "--split-type",
        choices=["CV1", "CV2", "CV3"],
        default=None,
        help="Run only this CV split (overrides config.split_types).",
    )
    run.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write log output (appended).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m ddgcn``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    args = _build_parser().parse_args(argv)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    from ddgcn.config import load_config
    from ddgcn.evaluate import run_cv

    config = load_config(args.config)
    if args.split_type is not None:
        config = dataclasses.replace(config, split_types=(args.split_type,))
    summary = run_cv(config)
    logging.getLogger(__name__).info(
        "wrote %d summary rows to %s", len(summary), config.output_dir
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ddgcn_cli.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ddgcn/__main__.py tests/test_ddgcn_cli.py
git commit -m "feat: add ddgcn cli entrypoint (exp10)"
```

---

### Task 8: Config YAML, lint clean, full suite

**Files:**
- Create: `configs/experiments/10_k562_sl_pair_ddgcn/ddgcn_cv.yaml`
- Create: `configs/experiments/10_k562_sl_pair_ddgcn/README.md`

**Interfaces:**
- Consumes: everything above. No new code interfaces.

- [ ] **Step 1: Write the run config**

```yaml
# configs/experiments/10_k562_sl_pair_ddgcn/ddgcn_cv.yaml
# DDGCN reproduction (Cai et al. 2020) on the K562 SL-pair benchmark.
# Run: uv run python -m ddgcn run-cv --config <this file>
# Defaults are the official DDGCN settings (dropout=0.5, lr=0.01).
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
output_dir: results/experiments/10_k562_sl_pair_ddgcn/run
split_types: [CV1, CV2, CV3]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 456
dropout: 0.5
lr: 0.01
hidden1: 512
hidden2: 256
init_type: Kaiming
use_bias: false
rho: 1.0
max_epochs: 2000
tolerance_epoch: 1000
stop_threshold: 1.0e-5
eval_interval: 50
```

- [ ] **Step 2: Write the config README**

```markdown
# exp10: DDGCN reproduction on the K562 SL-pair benchmark

Reproduces DDGCN (Dual-Dropout GCN, Cai et al. 2020,
https://github.com/CXX1113/Dual-DropoutGCN) under the exp06/07/08/09 CV1/CV2/CV3
protocol and official per-anchor metrics.

Run all splits:

    uv run python -m ddgcn run-cv --config configs/experiments/10_k562_sl_pair_ddgcn/ddgcn_cv.yaml

Single split (partial rerun):

    uv run python -m ddgcn run-cv --config configs/experiments/10_k562_sl_pair_ddgcn/ddgcn_cv.yaml --split-type CV2

Hyperparameters are the official DDGCN defaults (dropout=0.5, lr=0.01, hidden
512->256, Kaiming, no bias, rho=1.0, Adam amsgrad, max 2000 epochs with
loss-plateau early stop). The model is transductive and featureless; CV1 is
topology-gameable (cf. exp06 degree probe) and CV3 cold-start is expected
near-floor. Compare on CV2/CV3.
```

- [ ] **Step 3: Lint, format, and run the full DDGCN test set**

Run:
```bash
uv run ruff check src/ddgcn tests/test_ddgcn_*.py
uv run ruff format src/ddgcn tests/test_ddgcn_*.py
uv run python -m pytest tests/test_ddgcn_config.py tests/test_ddgcn_graph.py tests/test_ddgcn_model.py tests/test_ddgcn_train.py tests/test_ddgcn_scoring.py tests/test_ddgcn_evaluate.py tests/test_ddgcn_cli.py -v
```
Expected: ruff clean; all DDGCN tests PASS.

- [ ] **Step 4: Commit**

```bash
git add configs/experiments/10_k562_sl_pair_ddgcn/
git commit -m "feat: add ddgcn exp10 run config and readme"
```

---

### Task 9: Run the reproduction + write the experiment doc

**Files:**
- Create: `docs/experiment/10_k562_sl_pair_ddgcn.md`
- (Produces, not committed — gitignored artifacts) `results/experiments/10_k562_sl_pair_ddgcn/run/`

**Interfaces:**
- Consumes: the full pipeline. No new code.

- [ ] **Step 1: Run a fast smoke (CV1 only, capped epochs) to confirm end-to-end on real data**

Run:
```bash
uv run python -c "
import dataclasses
from ddgcn.config import DdgcnConfig
from ddgcn.evaluate import run_cv
cfg = dataclasses.replace(
    DdgcnConfig(),
    output_dir='results/experiments/10_k562_sl_pair_ddgcn/_smoke',
    split_types=('CV1',), folds=(0,), max_epochs=60, tolerance_epoch=10, eval_interval=20,
)
s = run_cv(cfg)
print(s[s['metric'].isin(['auroc','ndcg@10'])].to_string())
"
```
Expected: completes; prints finite AUROC + NDCG@10 for `ddgcn` on CV1 fold 0; manifest shows `candidate_gene_count: 9471`. (Smoke output dir can be deleted afterward.)

- [ ] **Step 2: Launch the full run**

Run:
```bash
uv run python -m ddgcn run-cv \
  --config configs/experiments/10_k562_sl_pair_ddgcn/ddgcn_cv.yaml \
  --log-file results/experiments/10_k562_sl_pair_ddgcn/run/train.log
```
Expected: writes `fold_metrics.csv`, `summary.csv`, `official_metrics_summary.csv`, `manifest.json`, and `CV1/`, `CV2/`, `CV3/` subdirs. If on CPU this is long; consider running per-split with `--split-type` in separate invocations.

- [ ] **Step 3: Verify acceptance assertions on the real output**

Run:
```bash
uv run python -c "
import json, pandas as pd
m = json.load(open('results/experiments/10_k562_sl_pair_ddgcn/run/manifest.json'))
assert m['candidate_gene_count'] == 9471, m['candidate_gene_count']
assert m['dropout'] == 0.5 and m['lr'] == 0.01
s = pd.read_csv('results/experiments/10_k562_sl_pair_ddgcn/run/summary.csv')
piv = s[s['metric'].isin(['auroc','aupr','ndcg@10','map@10'])]
print(piv.pivot_table(index=['split_type'], columns='metric', values='mean').to_string())
"
```
Expected: `candidate_gene_count == 9471`; a per-split table of mean AUROC/AUPR/NDCG@10/MAP@10.

- [ ] **Step 4: Write the experiment doc with the comparison table**

Create `docs/experiment/10_k562_sl_pair_ddgcn.md` following the exp06/08 doc
structure. Fill the DDGCN rows from the real `summary.csv` (replace the
bracketed values with the produced numbers). Include this comparison table:

```markdown
# exp10: DDGCN reproduction on the K562 SL-pair benchmark

DDGCN (Dual-Dropout GCN, Cai et al. 2020) reproduced under the exp06-09
CV1/CV2/CV3 protocol and official per-anchor metrics. Featureless, transductive
graph auto-encoder; defaults dropout=0.5, lr=0.01.

## Results vs the exp06 floor

| model | split | AUROC | AUPR | NDCG@10 | MAP@10 |
|---|---|---|---|---|---|
| exp06 XGB (B) | CV2 | 0.704 | 0.732 | 0.042 | 0.034 |
| exp06 XGB (B) | CV3 | 0.596 | — | 0.002 | — |
| exp06 degree probe (C) | CV1 | — | — | 0.197 | — |
| ddgcn | CV1 | [fill] | [fill] | [fill] | [fill] |
| ddgcn | CV2 | [fill] | [fill] | [fill] | [fill] |
| ddgcn | CV3 | [fill] | [fill] | [fill] | [fill] |

## Interpretation

DDGCN is transductive and featureless — it learns from the SL adjacency itself,
so it behaves like a learned nonlinear degree probe. Read CV2/CV3 as the
generalization surfaces; a CV1-only win mirrors the exp06 degree probe
(NDCG@10 0.197) and is not evidence of learned biology. CV3 cold-start has no
training edges touching test genes and is expected near-floor.

## Reproduction notes

- Model ported from `data/SL_benchmark/src/models/ddgcn.py` (matches the
  official PyTorch repo line-for-line); torch 2.12 sparse API.
- 9,471-gene K562 universe; official metrics reused verbatim from
  `sl_benchmark_baseline.metrics`.
- Loss-plateau stopping, no validation split (zero test leakage).
- Artifacts: `results/experiments/10_k562_sl_pair_ddgcn/run/`.
```

- [ ] **Step 5: Commit the doc**

```bash
git add docs/experiment/10_k562_sl_pair_ddgcn.md
git commit -m "docs: add exp10 ddgcn reproduction results and comparison"
```

---






