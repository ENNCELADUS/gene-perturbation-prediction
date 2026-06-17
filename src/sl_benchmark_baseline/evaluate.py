"""Per-fold CV loop, aggregation, and output writing for the SL baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.data import VALID_SPLIT_TYPES, fold_split, load_benchmark
from sl_benchmark_baseline.features import Standardizer, build_pair_features
from sl_benchmark_baseline.metrics import (
    official_classification_metrics,
    official_ranking_metrics,
)
from sl_benchmark_baseline.models import FoldData, FrequencyProbeModel, build_models

LEAKAGE_NOTES = (
    "GeneEffect(K562, g) as a feature against Rand negatives is low leakage "
    "risk. This becomes high risk under Exp/Dep negative sampling. CV1 is a "
    "pair-level split: results are not held-out-gene generalization."
)
RANKING_SEMANTICS = (
    "Ranking metrics follow SL_benchmark cal_metrics: per-anchor candidate-partner "
    "ranking over the K562-filtered candidate gene universe, with train-positive "
    "pairs masked from candidate rankings."
)
OFFICIAL_METRIC_SOURCE = "data/SL_benchmark/src/preprocess.py:cal_metrics"
SCORE_MATRIX_CHUNK_ROWS = 64


@dataclass(frozen=True)
class GeneUniverse:
    """Compressed candidate-gene universe used for official score matrices."""

    keys: tuple[object, ...]
    symbols: np.ndarray
    gene_effects: np.ndarray
    index_by_key: dict[object, int]


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


def _gene_key_columns(frame: pd.DataFrame) -> tuple[str, str]:
    if {"gene_a_unified_id", "gene_b_unified_id"}.issubset(frame.columns):
        return "gene_a_unified_id", "gene_b_unified_id"
    return "gene_a_symbol", "gene_b_symbol"


def _build_gene_universe(frame: pd.DataFrame) -> GeneUniverse:
    gene_a_key, gene_b_key = _gene_key_columns(frame)
    gene_a = pd.DataFrame(
        {
            "key": frame[gene_a_key],
            "symbol": frame["gene_a_symbol"],
            "gene_effect": frame["gene_a_k562_gene_effect"],
        }
    )
    gene_b = pd.DataFrame(
        {
            "key": frame[gene_b_key],
            "symbol": frame["gene_b_symbol"],
            "gene_effect": frame["gene_b_k562_gene_effect"],
        }
    )
    genes = (
        pd.concat([gene_a, gene_b], ignore_index=True)
        .drop_duplicates("key")
        .sort_values("key")
        .reset_index(drop=True)
    )
    keys = tuple(genes["key"].tolist())
    return GeneUniverse(
        keys=keys,
        symbols=genes["symbol"].astype(str).to_numpy(),
        gene_effects=genes["gene_effect"].to_numpy(dtype=float),
        index_by_key={key: index for index, key in enumerate(keys)},
    )


def _pair_indices(frame: pd.DataFrame, universe: GeneUniverse) -> np.ndarray:
    gene_a_key, gene_b_key = _gene_key_columns(frame)
    return np.column_stack(
        [
            frame[gene_a_key].map(universe.index_by_key).to_numpy(dtype=int),
            frame[gene_b_key].map(universe.index_by_key).to_numpy(dtype=int),
        ]
    )


def _build_score_matrix(
    model: object,
    universe: GeneUniverse,
    standardizer: Standardizer,
) -> np.ndarray:
    if isinstance(model, FrequencyProbeModel):
        score_matrix = model.predict_score_matrix(universe.symbols)
        np.fill_diagonal(score_matrix, 0.0)
        return score_matrix

    n_gene = len(universe.symbols)
    score_matrix = np.zeros((n_gene, n_gene), dtype=float)
    all_gene_indices = np.arange(n_gene)
    for start in range(0, n_gene, SCORE_MATRIX_CHUNK_ROWS):
        stop = min(start + SCORE_MATRIX_CHUNK_ROWS, n_gene)
        row_indices = np.arange(start, stop)
        gene_a_indices = np.repeat(row_indices, n_gene)
        gene_b_indices = np.tile(all_gene_indices, len(row_indices))
        raw = build_pair_features(
            universe.gene_effects[gene_a_indices],
            universe.gene_effects[gene_b_indices],
        )
        features = standardizer.transform(raw)
        pair_df = pd.DataFrame(
            {
                "gene_a_symbol": universe.symbols[gene_a_indices],
                "gene_b_symbol": universe.symbols[gene_b_indices],
            }
        )
        fold_data = FoldData(
            df=pair_df,
            features=features,
            labels=np.zeros(len(features), dtype=int),
        )
        score_matrix[start:stop, :] = model.predict_proba(fold_data).reshape(
            len(row_indices), n_gene
        )
    np.fill_diagonal(score_matrix, 0.0)
    return score_matrix


def run_fold(
    frame: pd.DataFrame,
    split_type: str,
    fold_id: int,
    config: SLBaselineConfig,
    universe: GeneUniverse,
) -> list[dict[str, object]]:
    """Fit all models on one fold and return long-form metric rows."""
    train_df, test_df = fold_split(frame, split_type, fold_id)
    train_raw = build_pair_features(
        train_df["gene_a_k562_gene_effect"].to_numpy(),
        train_df["gene_b_k562_gene_effect"].to_numpy(),
    )
    standardizer = Standardizer.fit(train_raw)
    train = _build_fold_data(train_df, standardizer)
    test_pos = test_df[test_df["sl_label"] == 1]
    test_neg = test_df[test_df["sl_label"] == 0]
    train_pos = train_df[train_df["sl_label"] == 1]
    pos_index = _pair_indices(test_pos, universe)
    neg_index = _pair_indices(test_neg, universe)
    seen_index = _pair_indices(train_pos, universe)

    rows: list[dict[str, object]] = []
    for model in build_models(config):
        model.fit(train)
        score_matrix = _build_score_matrix(model, universe, standardizer)
        metrics = official_classification_metrics(score_matrix, pos_index, neg_index)
        metrics.update(
            official_ranking_metrics(
                score_matrix,
                pos_index,
                seen_index=seen_index,
                ks=config.ranking_k,
            )
        )
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
    universe = _build_gene_universe(frame)
    split_types = _resolve_split_types(frame, config.split_types)
    all_rows: list[dict[str, object]] = []
    for split_type in split_types:
        for fold_id in config.folds:
            all_rows.extend(run_fold(frame, split_type, fold_id, config, universe))
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
        "candidate_gene_count": len(universe.symbols),
        "seed": config.seed,
        "models": ["A", "B", "C"],
        "leakage_notes": LEAKAGE_NOTES,
        "ranking_semantics": RANKING_SEMANTICS,
        "official_metric_source": OFFICIAL_METRIC_SOURCE,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary
