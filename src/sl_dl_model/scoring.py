# src/sl_dl_model/scoring.py
"""Per-fold scoring: producer -> universe embeddings -> official metric rows.

Reuses the exp06/07 baseline scoring harness verbatim. Phase 0 fits the exp07
``_transcript`` sklearn models on producer-supplied embeddings.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sl_benchmark_baseline.data import fold_split
from sl_benchmark_baseline.evaluate import (
    GeneUniverse,
    _build_augmented_score_matrix,
    _build_gene_universe,
    _covered_pair_mask,
    _metric_rows,
    _pair_indices,
)
from sl_benchmark_baseline.features import Standardizer, build_augmented_pair_features
from sl_dl_model.config import SLDLConfig

logger = logging.getLogger(__name__)


def train_symbols_for_fold(train_df: pd.DataFrame) -> set[str]:
    """Return upper-cased gene symbols appearing in this fold's train pairs.

    Args:
        train_df: Train-split rows for one fold; must have ``gene_a_symbol``
            and ``gene_b_symbol`` columns.

    Returns:
        Set of upper-cased symbol strings from train pairs only.
    """
    a = train_df["gene_a_symbol"].astype(str).str.upper()
    b = train_df["gene_b_symbol"].astype(str).str.upper()
    return set(a) | set(b)


def _augmented_raw(
    frame: pd.DataFrame,
    universe: GeneUniverse,
    include_coverage_flag: bool,
) -> np.ndarray:
    """Build unstandardized augmented features for a frame of pairs."""
    idx = _pair_indices(frame, universe)
    a, b = idx[:, 0], idx[:, 1]
    return build_augmented_pair_features(
        frame["gene_a_k562_gene_effect"].to_numpy(),
        frame["gene_b_k562_gene_effect"].to_numpy(),
        universe.embeddings[a],
        universe.embeddings[b],
        universe.coverage_mask[a],
        universe.coverage_mask[b],
        include_coverage_flag=include_coverage_flag,
    )


def _fold_data(
    frame: pd.DataFrame,
    universe: GeneUniverse,
    standardizer: Standardizer,
    config: SLDLConfig,
) -> object:
    """Build a :class:`~sl_benchmark_baseline.models.FoldData` for a frame."""
    from sl_benchmark_baseline.models import FoldData

    raw = _augmented_raw(frame, universe, config.include_coverage_flag)
    return FoldData(
        df=frame,
        features=standardizer.transform(raw),
        labels=frame["sl_label"].to_numpy(dtype=int),
    )


def _as_baseline_config(config: SLDLConfig) -> object:
    """Convert an :class:`SLDLConfig` to a minimal :class:`SLBaselineConfig`.

    Only fields consumed by ``build_augmented_models`` need to be populated.
    """
    from sl_benchmark_baseline.config import SLBaselineConfig

    return SLBaselineConfig(
        input_csv=config.input_csv,
        output_dir=config.output_dir,
        split_types=config.split_types,
        folds=config.folds,
        ranking_k=config.ranking_k,
        seed=config.seed,
        fallback_strategy=config.fallback_strategy,
        include_coverage_flag=config.include_coverage_flag,
        bags_npz=config.bags_npz,
    )


def run_fold_with_producer(
    frame: pd.DataFrame,
    split_type: str,
    fold_id: int,
    config: SLDLConfig,
    producer: object,
) -> list[dict[str, object]]:
    """Produce fold embeddings via ``producer``, score universe, return metric rows.

    The producer trains (or constructs) embeddings for universe genes using only
    train-fold symbols, then this function reuses the exp07 augmented score matrix
    and official metric harness verbatim.

    Args:
        frame: Full benchmark DataFrame (all splits/folds/roles).
        split_type: CV split type (``"CV1"``, ``"CV2"``, or ``"CV3"``).
        fold_id: Fold id to evaluate.
        config: :class:`SLDLConfig` for this run.
        producer: An :class:`~sl_dl_model.evaluate.EmbeddingProducer` instance;
            ``produce(symbols, train_symbols)`` returns
            ``(embeddings (n, dim), coverage_mask (n,))``.

    Returns:
        Long-form metric row dicts with keys ``split_type``, ``model``,
        ``fold_id``, ``slice``, ``metric``, ``value``.
    """
    from sl_benchmark_baseline.models import build_augmented_models

    train_df, test_df = fold_split(frame, split_type, fold_id)
    base_universe = _build_gene_universe(frame)
    train_symbols = train_symbols_for_fold(train_df)

    embeddings, coverage = producer.produce(base_universe.symbols, train_symbols)
    universe = GeneUniverse(
        keys=base_universe.keys,
        symbols=base_universe.symbols,
        gene_effects=base_universe.gene_effects,
        index_by_key=base_universe.index_by_key,
        embeddings=embeddings,
        coverage_mask=coverage,
    )

    test_pos = test_df[test_df["sl_label"] == 1]
    test_neg = test_df[test_df["sl_label"] == 0]
    train_pos = train_df[train_df["sl_label"] == 1]
    pos_index = _pair_indices(test_pos, universe)
    neg_index = _pair_indices(test_neg, universe)
    seen_index = _pair_indices(train_pos, universe)

    aug_std = Standardizer.fit(
        _augmented_raw(train_df, universe, config.include_coverage_flag)
    )
    train_aug = _fold_data(train_df, universe, aug_std, config)

    proxy_config = _as_baseline_config(config)
    pos_cov = pos_index[_covered_pair_mask(pos_index, universe)]
    neg_cov = neg_index[_covered_pair_mask(neg_index, universe)]

    rows: list[dict[str, object]] = []

    # DL path: use the trained pair head's score matrix directly.
    if hasattr(producer, "score_matrix"):
        sm = producer.score_matrix(universe.symbols, universe.gene_effects)
        rows.extend(
            _metric_rows(
                split_type,
                "state_dl",
                fold_id,
                "full_universe",
                sm,
                pos_index,
                neg_index,
                seen_index,
                config.ranking_k,
            )
        )
        if len(pos_cov) > 0 and len(neg_cov) > 0:
            rows.extend(
                _metric_rows(
                    split_type,
                    "state_dl",
                    fold_id,
                    "covered_pairs",
                    sm,
                    pos_cov,
                    neg_cov,
                    seen_index,
                    config.ranking_k,
                )
            )
        return rows

    for model in build_augmented_models(proxy_config):
        if not model.name.endswith("_transcript"):
            continue
        model.fit(train_aug)
        sm = _build_augmented_score_matrix(
            model, universe, aug_std, config.include_coverage_flag
        )
        rows.extend(
            _metric_rows(
                split_type,
                model.name,
                fold_id,
                "full_universe",
                sm,
                pos_index,
                neg_index,
                seen_index,
                config.ranking_k,
            )
        )
        if len(pos_cov) > 0 and len(neg_cov) > 0:
            rows.extend(
                _metric_rows(
                    split_type,
                    model.name,
                    fold_id,
                    "covered_pairs",
                    sm,
                    pos_cov,
                    neg_cov,
                    seen_index,
                    config.ranking_k,
                )
            )
        else:
            logger.debug(
                "split %s fold %s: covered_pairs slice skipped for %s "
                "(covered positives=%d, negatives=%d)",
                split_type,
                fold_id,
                model.name,
                len(pos_cov),
                len(neg_cov),
            )
    return rows


def make_fold_producer(
    config: SLDLConfig,
    caches: object,
    frame: pd.DataFrame,
    split_type: str,
    fold_id: int,
) -> object:
    """Build a fold-specific StateDlProducer from shared caches + fold train pairs.

    This factory is used by the ``state_dl`` path in
    :func:`~sl_dl_model.evaluate.run_cv`. It is a Phase-0 stub — the full
    implementation is added in Task 2.4.

    Args:
        config: :class:`SLDLConfig` for this run.
        caches: :class:`~sl_dl_model.evaluate.StateDlCaches` loaded once.
        frame: Full benchmark DataFrame.
        split_type: CV split type.
        fold_id: Fold id.

    Returns:
        A :class:`~sl_dl_model.train.StateDlProducer` for this fold.
    """
    from sl_dl_model.train import StateDlProducer

    train_df, _ = fold_split(frame, split_type, fold_id)
    train_pairs = [
        (
            str(r["gene_a_symbol"]).upper(),
            str(r["gene_b_symbol"]).upper(),
            int(r["sl_label"]),
            float(r["gene_a_k562_gene_effect"]),
            float(r["gene_b_k562_gene_effect"]),
        )
        for _, r in train_df.iterrows()
    ]
    return StateDlProducer(
        config,
        esm=caches.esm,
        bags=caches.bags,
        train_pairs=train_pairs,
        input_dim=caches.input_dim,
        output_dim=caches.output_dim,
    )
