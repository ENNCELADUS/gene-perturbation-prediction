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
