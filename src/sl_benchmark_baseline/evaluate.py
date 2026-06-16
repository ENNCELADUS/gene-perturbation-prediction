"""Per-fold CV loop, aggregation, and output writing for the SL baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.data import VALID_SPLIT_TYPES, fold_split, load_benchmark
from sl_benchmark_baseline.features import Standardizer, build_pair_features
from sl_benchmark_baseline.metrics import (
    classification_metrics,
    ranking_metrics,
)
from sl_benchmark_baseline.models import FoldData, build_models

LEAKAGE_NOTES = (
    "GeneEffect(K562, g) as a feature against Rand negatives is low leakage "
    "risk. This becomes high risk under Exp/Dep negative sampling. CV1 is a "
    "pair-level split: results are not held-out-gene generalization."
)
RANKING_SEMANTICS = (
    "Ranking metrics are pair-level over the flat test list; this differs from "
    "the official per-gene-anchor candidate ranking and is not claimed "
    "equivalent. Ties are broken by pair_id."
)
MODEL_C_F1_NOTE = (
    "Model C min-max normalizes train-positive degree-product scores within each "
    "test fold. Its f1@0.5 is fold-relative and should not be interpreted as a "
    "calibrated probability threshold."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _build_fold_data(frame: pd.DataFrame, standardizer: Standardizer) -> FoldData:
    raw = build_pair_features(
        frame["gene_a_k562_gene_effect"].to_numpy(),
        frame["gene_b_k562_gene_effect"].to_numpy(),
    )
    return FoldData(
        df=frame,
        features=standardizer.transform(raw),
        labels=frame["sl_label"].to_numpy(dtype=int),
    )


def run_fold(
    frame: pd.DataFrame, split_type: str, fold_id: int, config: SLBaselineConfig
) -> list[dict[str, object]]:
    """Fit all models on one fold and return long-form metric rows."""
    train_df, test_df = fold_split(frame, split_type, fold_id)
    train_raw = build_pair_features(
        train_df["gene_a_k562_gene_effect"].to_numpy(),
        train_df["gene_b_k562_gene_effect"].to_numpy(),
    )
    standardizer = Standardizer.fit(train_raw)
    train = _build_fold_data(train_df, standardizer)
    test = _build_fold_data(test_df, standardizer)
    pair_ids = test_df["pair_id"].astype(str).tolist()

    rows: list[dict[str, object]] = []
    for model in build_models(config):
        model.fit(train)
        scores = model.predict_proba(test)
        metrics = classification_metrics(test.labels, scores)
        metrics.update(ranking_metrics(test.labels, scores, pair_ids, config.ranking_k))
        for metric, value in metrics.items():
            rows.append(
                {
                    "split_type": split_type,
                    "model": model.name,
                    "fold_id": fold_id,
                    "metric": metric,
                    "value": float(value),
                }
            )
    return rows


def _summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold_metrics.groupby(["split_type", "model", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return summary.sort_values(["split_type", "model", "metric"]).reset_index(drop=True)


def _resolve_split_types(
    frame: pd.DataFrame, requested: tuple[str, ...] | None
) -> tuple[str, ...]:
    available = set(frame["split_type"].unique())
    if requested is None:
        return tuple(split for split in VALID_SPLIT_TYPES if split in available)

    invalid = [split for split in requested if split not in VALID_SPLIT_TYPES]
    if invalid:
        raise ValueError(f"split_types must be in {VALID_SPLIT_TYPES}, got {invalid}")
    missing = [split for split in requested if split not in available]
    if missing:
        raise ValueError(
            f"requested split_types not present in input: {missing}; "
            f"available split_types: {sorted(available)}"
        )
    return requested


def run_cv(config: SLBaselineConfig) -> pd.DataFrame:
    """Run the full CV1 loop, write outputs, and return the summary table."""
    frame = load_benchmark(config.input_csv)
    split_types = _resolve_split_types(frame, config.split_types)
    all_rows: list[dict[str, object]] = []
    for split_type in split_types:
        for fold_id in config.folds:
            all_rows.extend(run_fold(frame, split_type, fold_id, config))
    fold_metrics = pd.DataFrame(all_rows)
    summary = _summarize(fold_metrics)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    manifest = {
        "input_csv": str(config.input_csv),
        "input_csv_sha256": _file_sha256(config.input_csv),
        "split_types": list(split_types),
        "folds": list(config.folds),
        "ranking_k": list(config.ranking_k),
        "seed": config.seed,
        "models": ["A", "B", "C"],
        "leakage_notes": LEAKAGE_NOTES,
        "ranking_semantics": RANKING_SEMANTICS,
        "model_c_f1_note": MODEL_C_F1_NOTE,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary
