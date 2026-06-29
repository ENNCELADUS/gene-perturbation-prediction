"""Step 1 pass: train fold-local generators and cache embeddings."""

from __future__ import annotations

import logging
import time
import traceback as _tb

import numpy as np
import pandas as pd
from accelerate import PartialState

from sl_benchmark_baseline.data import fold_split, load_benchmark
from sl_dl_model import fold_queue as fq
from sl_dl_model.exp08b_config import Exp08bConfig
from sl_dl_model.exp08b_generator import Step1GeneratorTrainer
from sl_dl_model.exp08b_queue import step_results_dir
from sl_dl_model.exp08b_runner import jobs, raise_if_step_incomplete
from sl_dl_model.state_dl_caches import load_state_dl_caches

logger = logging.getLogger(__name__)


def _train_symbols(train_df: pd.DataFrame) -> set[str]:
    """Return upper-cased train-pair genes for the generator pass."""
    a = train_df["gene_a_symbol"].astype(str).str.upper()
    b = train_df["gene_b_symbol"].astype(str).str.upper()
    return set(a) | set(b)


def _universe_symbols(frame: pd.DataFrame) -> np.ndarray:
    a = frame["gene_a_symbol"].astype(str).str.upper()
    b = frame["gene_b_symbol"].astype(str).str.upper()
    return np.asarray(sorted(set(a) | set(b)), dtype=object)


def run_train_generator(config: Exp08bConfig) -> None:
    """Run the generator pass over the filesystem queue."""
    frame = load_benchmark(config.input_csv)
    shared = load_state_dl_caches(config)
    runtime = PartialState()
    token = fq.run_token()
    fp = fq.fingerprint(config)
    results_dir = step_results_dir(config, "generator")
    (results_dir / ".claims" / token).mkdir(parents=True, exist_ok=True)
    job_list = jobs(frame, config)
    symbols = _universe_symbols(frame)

    trainer = Step1GeneratorTrainer(
        config,
        esm=shared.esm,
        bags=shared.bags,
        input_dim=shared.input_dim,
        output_dim=shared.output_dim,
        device=runtime.device,
    )

    for split_type, fold_id in job_list:
        if fq.is_done(results_dir, split_type, fold_id, fingerprint=fp):
            continue
        if fq.is_failed(results_dir, split_type, fold_id, fingerprint=fp):
            continue
        if not fq.try_claim(results_dir, split_type, fold_id, run_token=token):
            continue
        if fq.is_done(results_dir, split_type, fold_id, fingerprint=fp):
            continue
        try:
            train_df, _test_df = fold_split(frame, split_type, fold_id)
            result = trainer.train_fold(
                split_type=split_type,
                fold_id=fold_id,
                symbols=symbols,
                train_symbols=_train_symbols(train_df),
            )
            rows = [
                {
                    "split_type": split_type,
                    "fold_id": int(fold_id),
                    "embedding_path": str(result.embedding_path),
                    "manifest_path": str(result.manifest_path),
                    "bag_scale": float(result.bag_scale),
                    "train_bag_gene_count": int(result.train_bag_gene_count),
                    "val_bag_gene_count": int(result.val_bag_gene_count),
                }
            ]
            fq.write_result(results_dir, split_type, fold_id, rows, fingerprint=fp)
            logger.info(
                "[rank %d] generator %s/%d done",
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
                },
                fingerprint=fp,
            )
            logger.error(
                "[rank %d] generator %s/%d failed",
                runtime.process_index,
                split_type,
                fold_id,
            )

    if runtime.is_main_process:
        raise_if_step_incomplete(results_dir, job_list, fp, "generator")
