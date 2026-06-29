"""Step 2 pass: train SL heads from cached fold embeddings."""

from __future__ import annotations

import logging
import time
import traceback as _tb

import pandas as pd
from accelerate import PartialState

from sl_benchmark_baseline.data import fold_split, load_benchmark
from sl_dl_model import fold_queue as fq
from sl_dl_model.evaluate import _assemble
from sl_dl_model.exp08b_artifacts import embedding_cache_path
from sl_dl_model.exp08b_config import Exp08bConfig, SlHeadConfig
from sl_dl_model.exp08b_queue import (
    read_step2_failed_cache_fp,
    read_step2_result_cache_fp,
    step2_fold_fingerprint,
    step2_metric_config,
    step2_metric_model_name,
)
from sl_dl_model.exp08b_runner import jobs, raise_if_step_incomplete
from sl_dl_model.exp08b_sl_head import CachedEmbeddingPairHeadProducer
from sl_dl_model.scoring import run_fold_with_producer

logger = logging.getLogger(__name__)


def _train_pairs(train_df: pd.DataFrame) -> list[tuple[str, str, int, float, float]]:
    """Return train pair tuples consumed by the cached-embedding pair trainer."""
    pairs: list[tuple[str, str, int, float, float]] = []
    for row in train_df.itertuples(index=False):
        pairs.append(
            (
                str(row.gene_a_symbol),
                str(row.gene_b_symbol),
                int(row.sl_label),
                float(row.gene_a_k562_gene_effect),
                float(row.gene_b_k562_gene_effect),
            )
        )
    return pairs


def _same_cache_result(
    results_dir, split_type: str, fold_id: int, fp: str, cache_fp: str
) -> bool:
    return (
        fq.is_done(results_dir, split_type, fold_id, fingerprint=fp)
        and read_step2_result_cache_fp(results_dir, split_type, fold_id) == cache_fp
    )


def _same_cache_failure(
    results_dir, split_type: str, fold_id: int, fp: str, cache_fp: str
) -> bool:
    return (
        fq.is_failed(results_dir, split_type, fold_id, fingerprint=fp)
        and read_step2_failed_cache_fp(results_dir, split_type, fold_id) == cache_fp
    )


def run_train_sl_head(config: Exp08bConfig):
    """Run the cached-embedding SL-head pass over the filesystem queue."""
    frame = load_benchmark(config.input_csv)
    runtime = PartialState()
    token = fq.run_token()
    metric_config = step2_metric_config(config)
    fp = fq.fingerprint(metric_config)
    results_dir = fq.fold_results_dir(metric_config)
    (results_dir / ".claims" / token).mkdir(parents=True, exist_ok=True)
    job_list = jobs(frame, metric_config)
    split_types = tuple(dict.fromkeys(split for split, _fold in job_list))

    for split_type, fold_id in job_list:
        cache_fp = step2_fold_fingerprint(config, split_type, fold_id)
        if _same_cache_result(results_dir, split_type, fold_id, fp, cache_fp):
            continue
        if _same_cache_failure(results_dir, split_type, fold_id, fp, cache_fp):
            continue
        if not fq.try_claim(results_dir, split_type, fold_id, run_token=token):
            continue
        if _same_cache_result(results_dir, split_type, fold_id, fp, cache_fp):
            continue
        try:
            cache_path = embedding_cache_path(config, split_type, fold_id)
            if not cache_path.exists():
                raise RuntimeError(
                    f"missing Step 1 cache for {split_type}/fold{fold_id}: "
                    f"{cache_path}"
                )
            train_df, _test_df = fold_split(frame, split_type, fold_id)
            producer = CachedEmbeddingPairHeadProducer(
                SlHeadConfig.from_exp08b(config),
                cache_path=cache_path,
                train_pairs=_train_pairs(train_df),
                metric_model_name=step2_metric_model_name(config, split_type, fold_id),
                device=runtime.device,
            )
            rows = run_fold_with_producer(
                frame, split_type, fold_id, metric_config, producer
            )
            fq.write_result(
                results_dir,
                split_type,
                fold_id,
                rows,
                fingerprint=fp,
                extra={"cache_fp": cache_fp},
            )
            logger.info(
                "[rank %d] SL head %s/%d done",
                runtime.process_index,
                split_type,
                fold_id,
            )
        except Exception:  # noqa: BLE001 - quarantine and continue
            fq.write_failed(
                results_dir,
                split_type,
                fold_id,
                {
                    "split_type": split_type,
                    "fold_id": int(fold_id),
                    "rank": runtime.process_index,
                    "timestamp": time.time(),
                    "traceback": _tb.format_exc(),
                    "cache_fp": cache_fp,
                },
                fingerprint=fp,
            )
            logger.error(
                "[rank %d] SL head %s/%d failed",
                runtime.process_index,
                split_type,
                fold_id,
            )

    if not runtime.is_main_process:
        return pd.DataFrame()
    try:
        return _assemble(metric_config, job_list, split_types, frame, shared=None)
    except RuntimeError:
        raise_if_step_incomplete(results_dir, job_list, fp, "sl_head")
        raise
