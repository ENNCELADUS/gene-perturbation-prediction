# src/sl_dl_model/evaluate.py
"""Per-fold CV runner: turns a per-gene embedding table into official metrics.

Reuses the exp06/07 baseline scoring harness verbatim. exp08's DL model plugs
in as an :class:`EmbeddingProducer`; the metric/scoring path is never
reimplemented here.

Phase-0 version: ``run_cv`` accepts any :class:`EmbeddingProducer` instance
(e.g. :class:`ZeroEmbeddingProducer`) or the string ``"state_dl"`` (Task 2.4).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from sl_benchmark_baseline.data import load_benchmark
from sl_benchmark_baseline.evaluate import _summarize
from sl_dl_model.config import SLDLConfig

logger = logging.getLogger(__name__)


class EmbeddingProducer(Protocol):
    """Protocol: produce per-gene embeddings + coverage mask for a fold.

    Implementations may be stateless (e.g. :class:`ZeroEmbeddingProducer`) or
    stateful per-fold trainers (e.g. ``StateDlProducer`` in Task 2.3+).
    """

    def produce(
        self,
        symbols: np.ndarray,
        train_symbols: set[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(embeddings (n_gene, dim), coverage_mask (n_gene,))``.

        Args:
            symbols: Universe gene symbols in canonical order, shape
                ``(n_gene,)``.
            train_symbols: Upper-cased gene symbols from this fold's train
                pairs only (for leakage-safe per-fold training).

        Returns:
            A pair ``(embeddings, coverage_mask)`` where ``embeddings`` has
            shape ``(n_gene, dim)`` and ``coverage_mask`` has shape
            ``(n_gene,)`` with values 0 or 1.
        """
        ...


class ZeroEmbeddingProducer:
    """All-zero embeddings + all-zero mask: GeneEffect-only exp06-equivalent.

    This producer makes ``run_cv`` reproduce the exp06 dependency-only baseline
    in-harness. The transcript models trained on zero embeddings are equivalent
    to GeneEffect-only logistic regression / XGBoost.
    """

    dim: int = 1

    def produce(
        self,
        symbols: np.ndarray,
        train_symbols: set[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return zero embeddings and zero coverage mask for all universe genes.

        Args:
            symbols: Universe gene symbols, shape ``(n_gene,)``.
            train_symbols: Ignored (no training in this producer).

        Returns:
            ``(zeros (n_gene, 1), zeros (n_gene,))``.
        """
        n = len(symbols)
        return (
            np.zeros((n, self.dim), dtype=float),
            np.zeros(n, dtype=int),
        )


def run_cv(
    config: SLDLConfig,
    producer: "EmbeddingProducer | str",
) -> pd.DataFrame:
    """Run CV across split_types x folds, write metrics, return summary.

    Phase-0 version: loops folds and calls
    :func:`~sl_dl_model.scoring.run_fold_with_producer` for each.

    Args:
        config: :class:`SLDLConfig` controlling splits, folds, output dir, etc.
        producer: Either a reusable :class:`EmbeddingProducer` instance (e.g.
            :class:`ZeroEmbeddingProducer`) or the string ``"state_dl"``. The
            string path loads shared caches once and builds a per-fold
            ``StateDlProducer`` (Task 2.4).

    Returns:
        Summary :class:`pandas.DataFrame` with columns
        ``split_type, model, slice, metric, mean, std``.

    Raises:
        RuntimeError: If no metric rows were produced (e.g. split_types filter
            yields no matching folds in the loaded benchmark frame).
    """
    from sl_dl_model.scoring import make_fold_producer, run_fold_with_producer

    frame = load_benchmark(config.input_csv)
    split_types = config.split_types or ("CV1", "CV2", "CV3")
    available = set(frame["split_type"].unique())
    split_types = tuple(s for s in split_types if s in available)

    shared: StateDlCaches | None = None
    if producer == "state_dl":
        shared = _load_state_dl_caches(config)

    all_rows: list[dict[str, object]] = []
    for split_type in split_types:
        for fold_id in config.folds:
            fold_producer = (
                make_fold_producer(config, shared, frame, split_type, fold_id)
                if producer == "state_dl"
                else producer
            )
            all_rows.extend(
                run_fold_with_producer(
                    frame, split_type, fold_id, config, fold_producer
                )
            )

    # FIX 2: guard empty metric rows — indicates a config/data mismatch
    if not all_rows:
        logger.error(
            "no metric rows produced — split_types=%s not found in frame "
            "(available: %s); check split_types and training data",
            list(config.split_types or ("CV1", "CV2", "CV3")),
            sorted(available),
        )
        raise RuntimeError(
            "no metric rows produced; check split_types and training data"
        )

    fold_metrics = pd.DataFrame(all_rows)
    summary = _summarize(fold_metrics)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    # FIX 1: compute candidate_gene_count from the loaded frame
    candidate_gene_count = len(
        set(frame["gene_a_symbol"]) | set(frame["gene_b_symbol"])
    )

    # FIX 1: gwps_coverage_gene_count — available only on state_dl path
    gwps_coverage_gene_count: int | None = None
    if shared is not None:
        bags_obj = getattr(shared, "bags", None)
        if bags_obj is not None:
            bags_by_symbol = getattr(bags_obj, "bags_by_symbol", None)
            if bags_by_symbol is not None:
                gwps_coverage_gene_count = len(bags_by_symbol)

    manifest: dict[str, object] = {
        "input_csv": str(config.input_csv),
        "split_types": list(split_types),
        "folds": list(config.folds),
        "ranking_k": list(config.ranking_k),
        "seed": config.seed,
        "embedding_method": config.embedding_method,
        "fallback_strategy": config.fallback_strategy,
        "include_coverage_flag": config.include_coverage_flag,
        "esm2_model": config.esm2_model,
        "state_checkpoint": str(config.state_checkpoint),
        # FIX 1: new required manifest fields
        "candidate_gene_count": candidate_gene_count,
        "pooling": config.pooling,
        "loss_weights": {
            "lambda_sl": config.lambda_sl,
            "lambda_distill": config.lambda_distill,
            "lambda_distill_after_warmup": config.lambda_distill_after_warmup,
            "lambda_bag": config.lambda_bag,
            "lambda_rank": config.lambda_rank,
        },
        "coverage_flag_included": config.include_coverage_flag,
        "gwps_coverage_gene_count": gwps_coverage_gene_count,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary


@dataclass(frozen=True)
class StateDlCaches:
    """Shared, fold-independent caches for the state_dl producer.

    Loaded once by :func:`_load_state_dl_caches` and passed to every
    per-fold :class:`~sl_dl_model.train.StateDlProducer` via
    :func:`~sl_dl_model.scoring.make_fold_producer`.
    """

    esm: object  # Esm2EmbeddingTable
    bags: object  # GwpsBags
    input_dim: int
    output_dim: int


def _load_state_dl_caches(config: SLDLConfig) -> StateDlCaches:
    """Load ESM2 + gwps-bags caches once; shared across all folds (Task 2.4).

    Args:
        config: :class:`SLDLConfig` with ``esm2_npz`` and ``bags_npz`` paths.

    Returns:
        A :class:`StateDlCaches` dataclass.

    Raises:
        ValueError: If ``config.esm2_npz`` is not set.
    """
    from sl_dl_model.bags import build_gwps_bags, load_bags_npz
    from sl_dl_model.gene_embeddings import load_esm2_embeddings

    if config.esm2_npz is None:
        raise ValueError("state_dl producer requires config.esm2_npz")

    esm = load_esm2_embeddings(config.esm2_npz)
    if config.bags_npz is not None and Path(config.bags_npz).exists():
        bags = load_bags_npz(config.bags_npz)
    else:
        # FIX 3: warn when bags_npz is not set — full h5ad will be loaded
        logger.warning(
            "bags_npz is not set; the full gwps h5ad will be loaded into memory "
            "(%s). Pre-build the bags NPZ with `save_bags_npz` to avoid this.",
            config.gwps_h5ad,
        )
        bags = build_gwps_bags(config, rng_seed=config.seed)

    return StateDlCaches(
        esm=esm,
        bags=bags,
        input_dim=bags.input_dim,
        output_dim=bags.input_dim,
    )
