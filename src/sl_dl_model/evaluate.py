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
import time
import traceback as _tb
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from accelerate import PartialState

from sl_benchmark_baseline.data import load_benchmark
from sl_benchmark_baseline.evaluate import _summarize
from sl_dl_model import fold_queue as fq
from sl_dl_model.config import SLDLConfig
from sl_dl_model.scoring import make_fold_producer, run_fold_with_producer

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


def _run_worker_queue(
    config: SLDLConfig,
    shared: "StateDlCaches | None",
    frame: pd.DataFrame,
    jobs: list[tuple[str, int]],
    producer: "EmbeddingProducer | str",
    state: PartialState,
) -> None:
    """Walk ``jobs`` in order, atomically claiming and running each unfinished one.

    Every rank runs the identical loop. Atomic ``mkdir`` claims guarantee each
    job runs on exactly one rank; a lost claim or an already-done job is skipped
    immediately, so a rank never idles while work remains (decision C). A fold
    that raises is quarantined: a ``.failed`` marker with the traceback is
    written and the loop continues (decision B). No collective is used.

    Args:
        config: Run configuration.
        shared: Shared caches for the ``state_dl`` path, else ``None``.
        frame: Full benchmark DataFrame.
        jobs: Ordered ``(split_type, fold_id)`` pairs (the full job list).
        producer: ``"state_dl"`` or a reusable :class:`EmbeddingProducer`.
        state: Active :class:`PartialState` (for ``process_index`` in logs).
    """
    results_dir = fq.fold_results_dir(config)
    token = fq.run_token()
    (results_dir / ".claims" / token).mkdir(parents=True, exist_ok=True)

    for split_type, fold_id in jobs:
        if fq.is_done(results_dir, split_type, fold_id):
            continue
        if fq.failed_path(results_dir, split_type, fold_id).exists():
            continue
        if not fq.try_claim(results_dir, split_type, fold_id, run_token=token):
            continue
        # Re-check after winning the claim: a prior run may have produced a
        # result between our is_done check and the claim.
        if fq.is_done(results_dir, split_type, fold_id):
            continue
        try:
            fold_producer = (
                make_fold_producer(config, shared, frame, split_type, fold_id)
                if producer == "state_dl"
                else producer
            )
            rows = run_fold_with_producer(
                frame, split_type, fold_id, config, fold_producer
            )
            fq.atomic_write_json(fq.result_path(results_dir, split_type, fold_id), rows)
            logger.info(
                "[rank %d] fold %s/%d done (%d rows)",
                state.process_index,
                split_type,
                fold_id,
                len(rows),
            )
        except Exception:  # noqa: BLE001 — deliberate quarantine (decision B)
            fq.atomic_write_json(
                fq.failed_path(results_dir, split_type, fold_id),
                {
                    "split_type": split_type,
                    "fold_id": fold_id,
                    "rank": state.process_index,
                    "timestamp": time.time(),
                    "traceback": _tb.format_exc(),
                },
            )
            logger.error(
                "[rank %d] fold %s/%d FAILED; quarantined and continuing",
                state.process_index,
                split_type,
                fold_id,
            )


def run_cv(
    config: SLDLConfig,
    producer: "EmbeddingProducer | str",
) -> pd.DataFrame:
    """Run CV across split_types x folds, write metrics, return summary.

    Every rank walks the full ``(split_type, fold_id)`` job list and claims
    each unfinished job atomically via the filesystem work-queue
    (:func:`_run_worker_queue`); rank 0 then assembles the per-fold result
    files into the combined artifacts (:func:`_assemble`). No
    ``torch.distributed`` collective is used (Guard G1), so uneven fold
    runtimes can no longer cause a gather/NCCL timeout.

    Args:
        config: :class:`SLDLConfig` controlling splits, folds, output dir, etc.
        producer: Either a reusable :class:`EmbeddingProducer` instance (e.g.
            :class:`ZeroEmbeddingProducer`) or the string ``"state_dl"``. The
            string path loads shared caches once and builds a per-fold
            ``StateDlProducer``.

    Returns:
        Summary :class:`pandas.DataFrame` with columns
        ``split_type, model, slice, metric, mean, std`` on rank 0; an empty
        frame on non-main ranks.

    Raises:
        RuntimeError: If no metric rows were produced, or if any fold did not
            produce a result (quarantined or deadline-missed); see
            :func:`_assemble`.
    """
    frame = load_benchmark(config.input_csv)
    split_types = config.split_types or ("CV1", "CV2", "CV3")
    available = set(frame["split_type"].unique())
    split_types = tuple(s for s in split_types if s in available)

    shared: StateDlCaches | None = None
    if producer == "state_dl":
        shared = _load_state_dl_caches(config)

    # Build the full ordered job list; every rank walks the same list and
    # claims jobs atomically through the filesystem (no collective).
    state = PartialState()
    jobs = [(s, f) for s in split_types for f in config.folds]
    _run_worker_queue(config, shared, frame, jobs, producer, state)

    # Rank-0 assembles from the per-fold result files. Non-main ranks are done.
    if not state.is_main_process:
        return pd.DataFrame()
    return _assemble(config, jobs, split_types, frame, shared)


def _assemble(
    config: SLDLConfig,
    jobs: list[tuple[str, int]],
    split_types: tuple[str, ...],
    frame: pd.DataFrame,
    shared: "StateDlCaches | None",
) -> pd.DataFrame:
    """Poll for terminal markers, collect results, write combined artifacts.

    Rank-0 only. Replaces the old end-of-run NCCL all-gather barrier with a
    bounded
    filesystem poll (no collective). Stops when every job is terminal (result
    or failed) or ``assembly_timeout_seconds`` elapses, then assembles whatever
    results exist. Writes the succeeded folds' artifacts first, then — per
    decision B — raises if any fold lacks a result so the run exits non-zero.

    Args:
        config: Run configuration.
        jobs: Full ordered ``(split_type, fold_id)`` list.
        split_types: Split types covered by this run.
        frame: Full benchmark DataFrame (for candidate-gene count).
        shared: Shared caches (for gwps coverage count) or ``None``.

    Returns:
        Summary :class:`pandas.DataFrame` (only when every fold succeeded).

    Raises:
        RuntimeError: If no result rows were produced by any fold, or if any
            job lacks a result (quarantined or deadline-missed) — after the
            succeeded folds' artifacts have been written (decision B).
    """
    results_dir = fq.fold_results_dir(config)
    deadline = time.monotonic() + float(config.assembly_timeout_seconds)
    while time.monotonic() < deadline:
        terminal = all(
            fq.result_path(results_dir, s, f).exists()
            or fq.failed_path(results_dir, s, f).exists()
            for s, f in jobs
        )
        if terminal:
            break
        time.sleep(float(config.assembly_poll_seconds))

    all_rows: list[dict[str, object]] = []
    produced: set[tuple[str, int]] = set()
    for split_type, fold_id in jobs:
        rpath = fq.result_path(results_dir, split_type, fold_id)
        if rpath.exists():
            all_rows.extend(fq.read_json(rpath))
            produced.add((split_type, fold_id))

    if not all_rows:
        logger.error(
            "no metric rows produced — split_types=%s; check splits and data",
            list(split_types),
        )
        raise RuntimeError(
            "no metric rows produced; check split_types and training data"
        )

    fold_metrics = pd.DataFrame(all_rows)
    sort_cols = ["split_type", "fold_id", "model", "slice", "metric"]
    fold_metrics = fold_metrics.sort_values(sort_cols).reset_index(drop=True)
    summary = _summarize(fold_metrics)

    _write_assembly_artifacts(config, fold_metrics, summary, split_types, frame, shared)

    # Decision B: artifacts for succeeded folds are now safely on disk; fail
    # the run (non-zero exit) if any fold did not produce a result.
    failed = [job for job in jobs if job not in produced]
    if failed:
        logger.error("assembly: %d fold(s) missing results: %s", len(failed), failed)
        raise RuntimeError(
            f"{len(failed)} fold(s) did not produce results: {failed}; "
            "succeeded folds' artifacts were written — resubmit to re-run the rest"
        )
    return summary


def _write_assembly_artifacts(
    config: SLDLConfig,
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    split_types: tuple[str, ...],
    frame: pd.DataFrame,
    shared: "StateDlCaches | None",
) -> None:
    """Write top-level + per-split metric/summary/manifest artifacts.

    Identical layout to the pre-work-queue writer: a flat
    ``fold_metrics.csv`` / ``summary.csv`` / ``manifest.json``, per-split
    subdirectories, and a combined ``official_metrics_summary.csv``.

    Args:
        config: Run configuration.
        fold_metrics: Canonically-sorted long-form metric rows.
        summary: Summary frame from :func:`_summarize`.
        split_types: Split types covered by this run.
        frame: Full benchmark DataFrame (for candidate-gene count).
        shared: Shared caches (for gwps coverage count) or ``None``.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_gene_count = len(
        set(frame["gene_a_symbol"]) | set(frame["gene_b_symbol"])
    )
    gwps_coverage_gene_count = _gwps_coverage_count(shared)
    manifest = _build_manifest(
        config,
        split_types=split_types,
        candidate_gene_count=candidate_gene_count,
        gwps_coverage_gene_count=gwps_coverage_gene_count,
    )
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

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

    summary.to_csv(output_dir / "official_metrics_summary.csv", index=False)


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
        "batch_pairs": config.batch_pairs,
        "early_stop_patience": config.early_stop_patience,
        "early_stop_metric": "val_pair_auroc",
        "val_source": "test_fold",
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
    from sl_dl_model.bags import (
        build_gwps_bags,
        load_bags_npz,
        state_checkpoint_input_dim,
    )
    from sl_dl_model.gene_embeddings import load_esm2_embeddings

    if config.esm2_npz is None:
        raise ValueError("state_dl producer requires config.esm2_npz")

    esm = load_esm2_embeddings(config.esm2_npz)
    expected_input_dim = state_checkpoint_input_dim(config)
    if config.bags_npz is not None and Path(config.bags_npz).exists():
        bags = load_bags_npz(config.bags_npz)
        if expected_input_dim is not None and bags.input_dim != expected_input_dim:
            msg = (
                f"bags_npz {config.bags_npz} has input_dim={bags.input_dim}, "
                f"but STATE checkpoint expects input_dim={expected_input_dim}. "
                "Rebuild it once with "
                "`uv run python scripts/setup_exp08_assets.py bags` before "
                "launching the multi-rank training job."
            )
            raise ValueError(msg)
    else:
        # FIX 3: warn when bags_npz is not set — full h5ad will be loaded
        logger.warning(
            "bags_npz is not set; the full gwps h5ad will be loaded into memory "
            "(%s). Pre-build the bags NPZ with `save_bags_npz` to avoid this.",
            config.gwps_h5ad,
        )
        bags = build_gwps_bags(config, rng_seed=config.seed)

    if expected_input_dim is not None and bags.input_dim != expected_input_dim:
        msg = (
            f"GWPS bags input_dim={bags.input_dim} does not match STATE "
            f"checkpoint input_dim={expected_input_dim}; rebuild bags_npz with "
            "checkpoint gene alignment"
        )
        raise ValueError(msg)

    return StateDlCaches(
        esm=esm,
        bags=bags,
        input_dim=bags.input_dim,
        output_dim=bags.input_dim,
    )
