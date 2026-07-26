# src/ddgcn/scoring.py
"""Bridge the DDGCN model to the sl_benchmark_baseline protocol harness.

Build the gene universe once, slice the
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
