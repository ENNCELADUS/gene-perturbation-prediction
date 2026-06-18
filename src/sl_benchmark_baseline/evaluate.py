"""Per-fold CV loop, aggregation, and output writing for the SL baseline."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.data import VALID_SPLIT_TYPES, fold_split, load_benchmark
from sl_benchmark_baseline.embeddings import GeneEmbeddingTable, align_to_universe
from sl_benchmark_baseline.features import (
    Standardizer,
    build_augmented_pair_features,
    build_pair_features,
    build_selectivity_pair_features,
)
from sl_benchmark_baseline.metrics import (
    official_classification_metrics,
    official_ranking_metrics,
)
from sl_benchmark_baseline.models import (
    FoldData,
    FrequencyProbeModel,
    build_augmented_models,
    build_models,
    build_selectivity_models,
)
from sl_benchmark_baseline.selectivity import (
    UniverseSelectivity,
    align_selectivity_to_universe,
    build_selectivity_table_from_depmap,
)

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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneUniverse:
    """Compressed candidate-gene universe used for official score matrices."""

    keys: tuple[object, ...]
    symbols: np.ndarray
    gene_effects: np.ndarray
    index_by_key: dict[object, int]
    embeddings: np.ndarray | None = None
    coverage_mask: np.ndarray | None = None
    entrez: np.ndarray | None = None
    selectivity: UniverseSelectivity | None = None


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


def _build_gene_universe(
    frame: pd.DataFrame,
    embedding_table: GeneEmbeddingTable | None = None,
    fallback_strategy: str = "zero",
) -> GeneUniverse:
    gene_a_key, gene_b_key = _gene_key_columns(frame)
    has_entrez = {"gene_a_entrez_id", "gene_b_entrez_id"}.issubset(frame.columns)
    gene_a_cols = {
        "key": frame[gene_a_key],
        "symbol": frame["gene_a_symbol"],
        "gene_effect": frame["gene_a_k562_gene_effect"],
    }
    gene_b_cols = {
        "key": frame[gene_b_key],
        "symbol": frame["gene_b_symbol"],
        "gene_effect": frame["gene_b_k562_gene_effect"],
    }
    if has_entrez:
        gene_a_cols["entrez"] = frame["gene_a_entrez_id"]
        gene_b_cols["entrez"] = frame["gene_b_entrez_id"]
    gene_a = pd.DataFrame(gene_a_cols)
    gene_b = pd.DataFrame(gene_b_cols)
    genes = (
        pd.concat([gene_a, gene_b], ignore_index=True)
        .drop_duplicates("key")
        .sort_values("key")
        .reset_index(drop=True)
    )
    keys = tuple(genes["key"].tolist())
    symbols = genes["symbol"].astype(str).to_numpy()
    entrez = genes["entrez"].to_numpy() if has_entrez else None
    embeddings = None
    coverage_mask = None
    if embedding_table is not None:
        embeddings, coverage_mask = align_to_universe(
            embedding_table, symbols, fallback_strategy
        )
    return GeneUniverse(
        keys=keys,
        symbols=symbols,
        gene_effects=genes["gene_effect"].to_numpy(dtype=float),
        index_by_key={key: index for index, key in enumerate(keys)},
        embeddings=embeddings,
        coverage_mask=coverage_mask,
        entrez=entrez,
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


def _build_augmented_score_matrix(
    model: object,
    universe: GeneUniverse,
    standardizer: Standardizer,
    include_coverage_flag: bool,
) -> np.ndarray:
    """Score all candidate pairs using augmented GeneEffect+transcript features."""
    if universe.embeddings is None or universe.coverage_mask is None:
        raise ValueError("augmented score matrix requires universe embeddings")
    n_gene = len(universe.symbols)
    score_matrix = np.zeros((n_gene, n_gene), dtype=float)
    all_idx = np.arange(n_gene)
    for start in range(0, n_gene, SCORE_MATRIX_CHUNK_ROWS):
        stop = min(start + SCORE_MATRIX_CHUNK_ROWS, n_gene)
        rows = np.arange(start, stop)
        a_idx = np.repeat(rows, n_gene)
        b_idx = np.tile(all_idx, len(rows))
        raw = build_augmented_pair_features(
            universe.gene_effects[a_idx],
            universe.gene_effects[b_idx],
            universe.embeddings[a_idx],
            universe.embeddings[b_idx],
            universe.coverage_mask[a_idx],
            universe.coverage_mask[b_idx],
            include_coverage_flag=include_coverage_flag,
        )
        features = standardizer.transform(raw)
        pair_df = pd.DataFrame(
            {
                "gene_a_symbol": universe.symbols[a_idx],
                "gene_b_symbol": universe.symbols[b_idx],
            }
        )
        fold_data = FoldData(
            df=pair_df, features=features, labels=np.zeros(len(features), dtype=int)
        )
        score_matrix[start:stop, :] = model.predict_proba(fold_data).reshape(
            len(rows), n_gene
        )
    np.fill_diagonal(score_matrix, 0.0)
    return score_matrix


def _build_selectivity_score_matrix(
    model: object,
    universe: GeneUniverse,
    standardizer: Standardizer,
) -> np.ndarray:
    """Score all candidate pairs using GeneEffect+selectivity features."""
    if universe.selectivity is None:
        raise ValueError("selectivity score matrix requires universe.selectivity")
    sel = universe.selectivity
    n_gene = len(universe.symbols)
    score_matrix = np.zeros((n_gene, n_gene), dtype=float)
    all_idx = np.arange(n_gene)
    for start in range(0, n_gene, SCORE_MATRIX_CHUNK_ROWS):
        stop = min(start + SCORE_MATRIX_CHUNK_ROWS, n_gene)
        rows = np.arange(start, stop)
        a_idx = np.repeat(rows, n_gene)
        b_idx = np.tile(all_idx, len(rows))
        raw = np.column_stack(
            [
                build_pair_features(
                    universe.gene_effects[a_idx], universe.gene_effects[b_idx]
                ),
                build_selectivity_pair_features(
                    sel.sel_matrix[a_idx, b_idx],
                    sel.sel_matrix[b_idx, a_idx],
                    sel.pan_essential[a_idx],
                    sel.pan_essential[b_idx],
                ),
            ]
        )
        features = standardizer.transform(raw)
        pair_df = pd.DataFrame(
            {
                "gene_a_symbol": universe.symbols[a_idx],
                "gene_b_symbol": universe.symbols[b_idx],
            }
        )
        fold_data = FoldData(
            df=pair_df, features=features, labels=np.zeros(len(features), dtype=int)
        )
        score_matrix[start:stop, :] = model.predict_proba(fold_data).reshape(
            len(rows), n_gene
        )
    np.fill_diagonal(score_matrix, 0.0)
    return score_matrix


def _metric_rows(
    split_type: str,
    model_name: str,
    fold_id: int,
    slice_name: str,
    score_matrix: np.ndarray,
    pos_index: np.ndarray,
    neg_index: np.ndarray,
    seen_index: np.ndarray,
    ks: tuple[int, ...],
) -> list[dict[str, object]]:
    """Assemble long-form metric rows for one model/slice on one fold."""
    metrics = official_classification_metrics(score_matrix, pos_index, neg_index)
    metrics.update(
        official_ranking_metrics(score_matrix, pos_index, seen_index=seen_index, ks=ks)
    )
    return [
        {
            "split_type": split_type,
            "model": model_name,
            "fold_id": fold_id,
            "slice": slice_name,
            "metric": metric,
            "value": float(value),
        }
        for metric, value in metrics.items()
    ]


def _selectivity_raw(
    frame: pd.DataFrame, universe: GeneUniverse, config: SLBaselineConfig
) -> np.ndarray:
    """Unstandardized GeneEffect(5) + selectivity(3) block for a frame."""
    if universe.selectivity is None:
        raise ValueError("selectivity score requires universe.selectivity")
    pair_idx = _pair_indices(frame, universe)
    a_idx, b_idx = pair_idx[:, 0], pair_idx[:, 1]
    sel = universe.selectivity
    sel_ab = sel.sel_matrix[a_idx, b_idx]
    sel_ba = sel.sel_matrix[b_idx, a_idx]
    pan_a = sel.pan_essential[a_idx]
    pan_b = sel.pan_essential[b_idx]
    return np.column_stack(
        [
            build_pair_features(
                frame["gene_a_k562_gene_effect"].to_numpy(),
                frame["gene_b_k562_gene_effect"].to_numpy(),
            ),
            build_selectivity_pair_features(sel_ab, sel_ba, pan_a, pan_b),
        ]
    )


def _augmented_raw(
    frame: pd.DataFrame, universe: GeneUniverse, config: SLBaselineConfig
) -> np.ndarray:
    """Unstandardized augmented features for a frame (used to fit a standardizer)."""
    pair_idx = _pair_indices(frame, universe)
    a_idx, b_idx = pair_idx[:, 0], pair_idx[:, 1]
    return build_augmented_pair_features(
        frame["gene_a_k562_gene_effect"].to_numpy(),
        frame["gene_b_k562_gene_effect"].to_numpy(),
        universe.embeddings[a_idx],
        universe.embeddings[b_idx],
        universe.coverage_mask[a_idx],
        universe.coverage_mask[b_idx],
        include_coverage_flag=config.include_coverage_flag,
    )


def _augmented_fold_data(
    frame: pd.DataFrame,
    universe: GeneUniverse,
    standardizer: Standardizer,
    config: SLBaselineConfig,
) -> FoldData:
    """Build standardized augmented FoldData for a frame."""
    raw = _augmented_raw(frame, universe, config)
    return FoldData(
        df=frame,
        features=standardizer.transform(raw),
        labels=frame["sl_label"].to_numpy(dtype=int),
    )


def _covered_pair_mask(index: np.ndarray, universe: GeneUniverse) -> np.ndarray:
    """Boolean mask of pairs whose two genes are both gwps-covered."""
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    return (universe.coverage_mask[index[:, 0]] == 1) & (
        universe.coverage_mask[index[:, 1]] == 1
    )


def _non_pan_essential_mask(
    index: np.ndarray, universe: GeneUniverse, max_essential_fraction: float = 0.5
) -> np.ndarray:
    """Pairs where neither gene is broadly essential (essential_fraction <= thr)."""
    if len(index) == 0 or universe.selectivity is None:
        return np.zeros(len(index), dtype=bool)
    ess = universe.selectivity.essential_fraction
    return (ess[index[:, 0]] <= max_essential_fraction) & (
        ess[index[:, 1]] <= max_essential_fraction
    )


def _selectivity_covered_mask(index: np.ndarray, universe: GeneUniverse) -> np.ndarray:
    """Pairs whose two genes both cleared the selectivity n_min coverage bar."""
    if len(index) == 0 or universe.selectivity is None:
        return np.zeros(len(index), dtype=bool)
    cov = universe.selectivity.coverage_flag
    return (cov[index[:, 0]] == 1) & (cov[index[:, 1]] == 1)


def _run_fold_augmented(
    train_df: pd.DataFrame,
    train_base: FoldData,
    base_std: Standardizer,
    pos_index: np.ndarray,
    neg_index: np.ndarray,
    seen_index: np.ndarray,
    split_type: str,
    fold_id: int,
    config: SLBaselineConfig,
    universe: GeneUniverse,
) -> list[dict[str, object]]:
    """Fit baseline + transcript models, emitting full and covered slices."""
    aug_std = Standardizer.fit(_augmented_raw(train_df, universe, config))
    train_aug = _augmented_fold_data(train_df, universe, aug_std, config)
    pos_cov = pos_index[_covered_pair_mask(pos_index, universe)]
    neg_cov = neg_index[_covered_pair_mask(neg_index, universe)]
    rows: list[dict[str, object]] = []
    for model in build_augmented_models(config):
        if model.name.endswith("_transcript"):
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
                logger.warning(
                    "split %s fold %s: covered_pairs slice skipped for %s "
                    "(covered test positives=%d, negatives=%d); "
                    "check embedding coverage of this fold's test genes.",
                    split_type,
                    fold_id,
                    model.name,
                    len(pos_cov),
                    len(neg_cov),
                )
        else:
            model.fit(train_base)
            sm = _build_score_matrix(model, universe, base_std)
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
    return rows


def _run_fold_selectivity(
    train_df: pd.DataFrame,
    train_base: FoldData,
    base_std: Standardizer,
    pos_index: np.ndarray,
    neg_index: np.ndarray,
    seen_index: np.ndarray,
    split_type: str,
    fold_id: int,
    config: SLBaselineConfig,
    universe: GeneUniverse,
) -> list[dict[str, object]]:
    """Fit baseline A/B/C + A_xcl/B_xcl, emitting full + diagnostic slices."""
    sel_std = Standardizer.fit(_selectivity_raw(train_df, universe, config))
    train_sel = FoldData(
        df=train_df,
        features=sel_std.transform(_selectivity_raw(train_df, universe, config)),
        labels=train_df["sl_label"].to_numpy(dtype=int),
    )
    rows: list[dict[str, object]] = []
    for model in build_selectivity_models(config):
        if model.name.endswith("_xcl"):
            model.fit(train_sel)
            sm = _build_selectivity_score_matrix(model, universe, sel_std)
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
            for slice_name, mask_fn in (
                ("non_pan_essential", _non_pan_essential_mask),
                ("covered_pairs", _selectivity_covered_mask),
            ):
                pos_s = pos_index[mask_fn(pos_index, universe)]
                neg_s = neg_index[mask_fn(neg_index, universe)]
                if len(pos_s) > 0 and len(neg_s) > 0:
                    rows.extend(
                        _metric_rows(
                            split_type,
                            model.name,
                            fold_id,
                            slice_name,
                            sm,
                            pos_s,
                            neg_s,
                            seen_index,
                            config.ranking_k,
                        )
                    )
                else:
                    logger.warning(
                        "split %s fold %s: %s slice skipped for %s (pos=%d, neg=%d)",
                        split_type,
                        fold_id,
                        slice_name,
                        model.name,
                        len(pos_s),
                        len(neg_s),
                    )
        else:
            model.fit(train_base)
            sm = _build_score_matrix(model, universe, base_std)
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
    return rows


def run_fold(
    frame: pd.DataFrame,
    split_type: str,
    fold_id: int,
    config: SLBaselineConfig,
    universe: GeneUniverse,
) -> list[dict[str, object]]:
    """Fit all models on one fold and return long-form metric rows."""
    train_df, test_df = fold_split(frame, split_type, fold_id)
    base_std = Standardizer.fit(
        build_pair_features(
            train_df["gene_a_k562_gene_effect"].to_numpy(),
            train_df["gene_b_k562_gene_effect"].to_numpy(),
        )
    )
    train_base = _build_fold_data(train_df, base_std)
    test_pos = test_df[test_df["sl_label"] == 1]
    test_neg = test_df[test_df["sl_label"] == 0]
    train_pos = train_df[train_df["sl_label"] == 1]
    pos_index = _pair_indices(test_pos, universe)
    neg_index = _pair_indices(test_neg, universe)
    seen_index = _pair_indices(train_pos, universe)

    if config.selectivity:
        return _run_fold_selectivity(
            train_df,
            train_base,
            base_std,
            pos_index,
            neg_index,
            seen_index,
            split_type,
            fold_id,
            config,
            universe,
        )
    if not config.augmented:
        rows: list[dict[str, object]] = []
        for model in build_models(config):
            model.fit(train_base)
            score_matrix = _build_score_matrix(model, universe, base_std)
            rows.extend(
                _metric_rows(
                    split_type,
                    model.name,
                    fold_id,
                    "full_universe",
                    score_matrix,
                    pos_index,
                    neg_index,
                    seen_index,
                    config.ranking_k,
                )
            )
        return rows
    return _run_fold_augmented(
        train_df,
        train_base,
        base_std,
        pos_index,
        neg_index,
        seen_index,
        split_type,
        fold_id,
        config,
        universe,
    )


def _summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold_metrics.groupby(["split_type", "model", "slice", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return summary.sort_values(["split_type", "model", "slice", "metric"]).reset_index(
        drop=True
    )


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
    if config.selectivity:
        from sl_benchmark_baseline.data import assert_rand_only

        assert_rand_only(frame)
    embedding_table = None
    if config.augmented:
        from sl_benchmark_baseline.embeddings import load_gene_embeddings

        embedding_table = load_gene_embeddings(config.bags_npz)
    universe = _build_gene_universe(
        frame,
        embedding_table=embedding_table,
        fallback_strategy=config.fallback_strategy,
    )
    selectivity = None
    if config.selectivity:
        if universe.entrez is None:
            raise ValueError(
                "selectivity mode requires gene_a_entrez_id/gene_b_entrez_id "
                "columns in the benchmark CSV"
            )
        table = build_selectivity_table_from_depmap(
            config.depmap_dir, tuple(int(e) for e in universe.entrez), config
        )
        selectivity = align_selectivity_to_universe(
            table, [int(e) for e in universe.entrez], config.sel_lambda
        )
        universe = replace(universe, selectivity=selectivity)
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
        "leakage_notes": LEAKAGE_NOTES,
        "ranking_semantics": RANKING_SEMANTICS,
        "official_metric_source": OFFICIAL_METRIC_SOURCE,
    }
    coverage_count = (
        int(universe.coverage_mask.sum()) if universe.coverage_mask is not None else 0
    )
    pair_coverage_fraction = None
    if config.augmented and universe.coverage_mask is not None:
        pair_idx = _pair_indices(frame, universe)
        both_covered = (universe.coverage_mask[pair_idx[:, 0]] == 1) & (
            universe.coverage_mask[pair_idx[:, 1]] == 1
        )
        pair_coverage_fraction = float(both_covered.mean()) if len(pair_idx) else 0.0
        gene_fraction = (
            coverage_count / len(universe.symbols) if len(universe.symbols) else 0.0
        )
        if gene_fraction < 0.5:
            logger.warning(
                "Low gwps gene coverage: %.1f%% (%d/%d). Check that bags_npz "
                "matches the benchmark universe (symbol casing/aliases).",
                gene_fraction * 100,
                coverage_count,
                len(universe.symbols),
            )
    if config.selectivity:
        model_names = ["A", "B", "C", "A_xcl", "B_xcl"]
    elif config.augmented:
        model_names = ["A", "B", "A_transcript", "B_transcript"]
    else:
        model_names = ["A", "B", "C"]
    manifest["models"] = model_names
    manifest["augmented"] = config.augmented
    manifest["bags_npz"] = None if config.bags_npz is None else str(config.bags_npz)
    manifest["embedding_method"] = config.embedding_method
    observed_feature_set = (
        embedding_table.feature_set if embedding_table is not None else None
    )
    observed_embedding_dim = (
        embedding_table.dim if embedding_table is not None else None
    )
    manifest["embedding_method_is_user_label"] = True
    manifest["observed_npz_feature_set"] = observed_feature_set
    manifest["observed_embedding_dim"] = observed_embedding_dim
    if (
        observed_feature_set is not None
        and observed_feature_set not in config.embedding_method
    ):
        logger.warning(
            "embedding_method=%r does not mention the NPZ's actual feature_set "
            "%r; the --embedding-method flag is a label only and is not "
            "validated against the bags NPZ contents.",
            config.embedding_method,
            observed_feature_set,
        )
    manifest["fallback_strategy"] = config.fallback_strategy
    manifest["include_coverage_flag"] = config.include_coverage_flag
    manifest["gwps_coverage_gene_count"] = coverage_count
    manifest["gwps_coverage_fraction"] = (
        coverage_count / len(universe.symbols) if len(universe.symbols) else 0.0
    )
    manifest["gwps_coverage_pair_fraction"] = pair_coverage_fraction
    manifest["selectivity"] = config.selectivity
    if config.selectivity:
        manifest["depmap_dir"] = str(config.depmap_dir)
        manifest["cn_loss_thr"] = config.cn_loss_thr
        manifest["expr_low_quantile"] = config.expr_low_quantile
        manifest["sel_n_min"] = config.sel_n_min
        manifest["sel_lambda"] = config.sel_lambda
        manifest["selectivity_coverage_gene_count"] = (
            int(selectivity.coverage_flag.sum()) if selectivity is not None else 0
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary
