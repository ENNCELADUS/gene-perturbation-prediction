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
from accelerate import PartialState
from accelerate.utils import gather_object

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


def _run_local_jobs(
    config: SLDLConfig,
    shared: "StateDlCaches | None",
    frame: pd.DataFrame,
    jobs: list[tuple[str, int]],
    producer: "EmbeddingProducer | str",
) -> list[dict[str, object]]:
    """Run this rank's assigned ``(split_type, fold_id)`` jobs; return metric rows.

    Each job is run through the unchanged per-fold path
    (:func:`~sl_dl_model.scoring.run_fold_with_producer`), so the rows are
    identical to a serial run of the same jobs. A failure in any job propagates
    (fail-fast); callers must not swallow it.

    Args:
        config: Run configuration.
        shared: Shared caches for the ``state_dl`` path, else ``None``.
        frame: Full benchmark DataFrame.
        jobs: This rank's ``(split_type, fold_id)`` slice from :func:`_shard_jobs`.
        producer: ``"state_dl"`` or a reusable :class:`EmbeddingProducer`.

    Returns:
        Metric row dicts for this rank's jobs (possibly empty if no jobs).
    """
    from sl_dl_model.scoring import make_fold_producer, run_fold_with_producer

    rows: list[dict[str, object]] = []
    for split_type, fold_id in jobs:
        fold_producer = (
            make_fold_producer(config, shared, frame, split_type, fold_id)
            if producer == "state_dl"
            else producer
        )
        rows.extend(
            run_fold_with_producer(frame, split_type, fold_id, config, fold_producer)
        )
    return rows


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
    frame = load_benchmark(config.input_csv)
    split_types = config.split_types or ("CV1", "CV2", "CV3")
    available = set(frame["split_type"].unique())
    split_types = tuple(s for s in split_types if s in available)

    shared: StateDlCaches | None = None
    if producer == "state_dl":
        shared = _load_state_dl_caches(config)

    # Build the full ordered job list, shard it across ranks, run only ours.
    state = PartialState()
    jobs = [(s, f) for s in split_types for f in config.folds]
    local_jobs = _shard_jobs(jobs, state.process_index, state.num_processes)
    local_rows = _run_local_jobs(config, shared, frame, local_jobs, producer)

    # Collective: every rank contributes its rows; all ranks receive the union.
    # gather_object preserves rank order, so rank r's rows land contiguously.
    gathered: list[list[dict[str, object]]] = gather_object([local_rows])
    all_rows = [row for rank_rows in gathered for row in rank_rows]

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
    # Canonical ordering so 1-process and N-process runs are byte-identical.
    sort_cols = ["split_type", "fold_id", "model", "slice", "metric"]
    fold_metrics = fold_metrics.sort_values(sort_cols).reset_index(drop=True)
    summary = _summarize(fold_metrics)

    # FIX 1: only the main process writes artifacts (multi-process safe).
    if not PartialState().is_main_process:
        return summary

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # FIX 1: compute candidate_gene_count from the loaded frame
    candidate_gene_count = len(
        set(frame["gene_a_symbol"]) | set(frame["gene_b_symbol"])
    )

    # FIX 1: gwps_coverage_gene_count — available only on state_dl path
    gwps_coverage_gene_count = _gwps_coverage_count(shared)

    manifest = _build_manifest(
        config,
        split_types=split_types,
        candidate_gene_count=candidate_gene_count,
        gwps_coverage_gene_count=gwps_coverage_gene_count,
    )

    # Top-level (flat) artifacts — retained for backward compatibility.
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # FIX 7: per-split subdirs (spec §7 "mirror exp06 layout") + combined summary.
    for split_type in split_types:
        split_rows = fold_metrics[fold_metrics["split_type"] == split_type]
        if split_rows.empty:
            continue
        split_dir = output_dir / split_type
        split_dir.mkdir(parents=True, exist_ok=True)
        split_rows.to_csv(split_dir / "fold_metrics.csv", index=False)
        _summarize(split_rows).to_csv(split_dir / "summary.csv", index=False)
        split_manifest = _build_manifest(
            config,
            split_types=(split_type,),
            candidate_gene_count=candidate_gene_count,
            gwps_coverage_gene_count=gwps_coverage_gene_count,
        )
        (split_dir / "manifest.json").write_text(json.dumps(split_manifest, indent=2))

    # Combined official summary across all splits.
    summary.to_csv(output_dir / "official_metrics_summary.csv", index=False)
    return summary


def _gwps_coverage_count(shared: "StateDlCaches | None") -> int | None:
    """Return the number of gwps-covered genes, or None off the state_dl path."""
    if shared is None:
        return None
    bags_obj = getattr(shared, "bags", None)
    if bags_obj is None:
        return None
    bags_by_symbol = getattr(bags_obj, "bags_by_symbol", None)
    if bags_by_symbol is None:
        return None
    return len(bags_by_symbol)


def _build_manifest(
    config: SLDLConfig,
    *,
    split_types: tuple[str, ...],
    candidate_gene_count: int,
    gwps_coverage_gene_count: int | None,
) -> dict[str, object]:
    """Assemble the run manifest dict (spec §7 fields).

    Args:
        config: Run configuration.
        split_types: CV split types covered by this manifest.
        candidate_gene_count: Unique-gene count of the benchmark universe.
        gwps_coverage_gene_count: Number of gwps-covered genes, or ``None`` when
            not running the state_dl path.

    Returns:
        JSON-serializable manifest dict.
    """
    return {
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


def _shard_jobs(
    jobs: list[tuple[str, int]],
    rank: int,
    num_processes: int,
) -> list[tuple[str, int]]:
    """Return the round-robin slice of CV jobs owned by ``rank``.

    Round-robin (``jobs[rank::num_processes]``) keeps load balanced across
    ranks when per-fold cost varies. Every job is owned by exactly one rank and
    the union across all ranks reconstructs ``jobs`` in order.

    Args:
        jobs: Ordered ``(split_type, fold_id)`` pairs to distribute.
        rank: Zero-based process index of the calling rank.
        num_processes: Total number of ranks.

    Returns:
        The sublist of ``jobs`` this rank should run (possibly empty).
    """
    return jobs[rank::num_processes]
