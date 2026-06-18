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

    fold_metrics = (
        pd.DataFrame(all_rows)
        .sort_values(["split_type", "fold_id", "model", "slice", "metric"])
        .reset_index(drop=True)
    )
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
        output_dir,
        fold_metrics,
        split_types,
        config,
        candidate_gene_count,
        train_edge_counts,
    )
    return summary
