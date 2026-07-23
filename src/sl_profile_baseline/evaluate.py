"""Official Feng CV evaluation for strict two-profile GeneEffect baselines."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    ndcg_score,
    precision_recall_curve,
    roc_auc_score,
)

from sl_profile_baseline.config import ProfileBaselineConfig
from sl_profile_baseline.data import (
    FengFold,
    ProfileUniverse,
    candidate_gene_count,
    load_feng_fold,
    load_profile_universe,
)
from sl_profile_baseline.features import (
    PROFILE_FEATURE_NAMES,
    ProfileFeatureCache,
    build_cached_pair_features,
    build_raw_pair_features,
    prepare_feature_cache,
)
from sl_profile_baseline.models import (
    ProbabilityModel,
    fit_raw_sgd,
    fit_summary_logreg,
    fit_summary_xgb,
    positive_probability,
)

logger = logging.getLogger(__name__)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(labels, scores)),
        "f1_oracle": float(f1.max()),
    }


def _average_precision_at_k(relevance: np.ndarray, k: int) -> float:
    values = np.asarray(relevance[:k], dtype=float)
    if values.sum() == 0:
        return 0.0
    precision = np.cumsum(values) / np.arange(1, len(values) + 1)
    return float((precision * values).sum() / min(values.sum(), k))


def _ranking_metrics_low_memory(
    matrix: np.ndarray,
    positives: np.ndarray,
    seen: np.ndarray,
    ks: tuple[int, ...],
) -> dict[str, float]:
    """Match official ranking semantics without two float64 matrix copies."""
    np.fill_diagonal(matrix, 0.0)
    matrix[seen[:, 0], seen[:, 1]] = -999999.0
    matrix[seen[:, 1], seen[:, 0]] = -999999.0
    by_anchor: dict[int, set[int]] = {}
    for gene_a, gene_b in positives:
        by_anchor.setdefault(int(gene_a), set()).add(int(gene_b))
        by_anchor.setdefault(int(gene_b), set()).add(int(gene_a))
    max_k = max(ks)
    relevance_rows: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    positive_counts: list[int] = []
    for anchor in sorted(by_anchor):
        order = np.argsort(matrix[anchor])[::-1][:max_k]
        relevant = np.fromiter(
            (int(candidate) in by_anchor[anchor] for candidate in order),
            dtype=int,
            count=len(order),
        )
        relevance_rows.append(relevant)
        score_rows.append(matrix[anchor, order])
        positive_counts.append(len(by_anchor[anchor]))
    if not relevance_rows:
        return {
            name: 0.0
            for k in ks
            for name in (f"ndcg@{k}", f"recall@{k}", f"precision@{k}", f"map@{k}")
        }
    return _ranking_at_ks(relevance_rows, score_rows, positive_counts, ks)


def _ranking_at_ks(
    relevance_rows: list[np.ndarray],
    score_rows: list[np.ndarray],
    positive_counts: list[int],
    ks: tuple[int, ...],
) -> dict[str, float]:
    relevance = np.asarray(relevance_rows)
    scores = np.asarray(score_rows)
    counts = np.asarray(positive_counts)
    output: dict[str, float] = {}
    for k in ks:
        hits = relevance[:, :k].sum(axis=1)
        output[f"ndcg@{k}"] = float(ndcg_score(relevance, scores, k=k))
        output[f"recall@{k}"] = float((hits / counts).mean())
        output[f"precision@{k}"] = float((hits / np.minimum(counts, k)).mean())
        output[f"map@{k}"] = float(
            np.mean([_average_precision_at_k(row, k) for row in relevance])
        )
    return output


def _slice_masks(
    pairs: np.ndarray, universe: ProfileUniverse, config: ProfileBaselineConfig
) -> dict[str, np.ndarray]:
    a_index = pairs[:, 0]
    b_index = pairs[:, 1]
    covered = universe.coverage[a_index] & universe.coverage[b_index]
    non_pan = (
        universe.essential_fraction[a_index] <= config.pan_essential_fraction
    ) & (universe.essential_fraction[b_index] <= config.pan_essential_fraction)
    return {
        "full_universe": np.ones(len(pairs), dtype=bool),
        "covered_pairs": covered,
        "non_pan_essential": covered & non_pan,
    }


def _classification_rows(
    split_type: str,
    fold_id: int,
    model_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    masks: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slice_name, mask in masks.items():
        sliced_labels = labels[mask]
        if len(np.unique(sliced_labels)) != 2:
            continue
        for metric, value in _classification_metrics(
            sliced_labels, scores[mask]
        ).items():
            rows.append(
                {
                    "split_type": split_type,
                    "fold_id": fold_id,
                    "model": model_name,
                    "slice": slice_name,
                    "metric": metric,
                    "value": value,
                    "n_pairs": int(mask.sum()),
                }
            )
    return rows


def _fixed_profile_scores(
    model_name: str, features: np.ndarray
) -> np.ndarray:
    feature_index = {
        "pearson_abs": PROFILE_FEATURE_NAMES.index("pearson"),
        "spearman_abs": PROFILE_FEATURE_NAMES.index("spearman"),
    }
    return np.abs(features[:, feature_index[model_name]]).astype(float)


def _residual_pair_scores(
    cache: ProfileFeatureCache, pairs: np.ndarray
) -> np.ndarray:
    a_index = pairs[:, 0]
    b_index = pairs[:, 1]
    product = (
        cache.line_residual_centered[a_index]
        * cache.line_residual_centered[b_index]
    ).sum(axis=1)
    denominator = (
        cache.line_residual_norm[a_index] * cache.line_residual_norm[b_index]
    )
    return np.abs(product / denominator).astype(float)


def _fit_supervised_models(
    fold: FengFold,
    compact_train: np.ndarray,
    universe: ProfileUniverse,
    config: ProfileBaselineConfig,
) -> dict[str, ProbabilityModel]:
    models: dict[str, ProbabilityModel] = {}
    if "summary_logreg" in config.models:
        models["summary_logreg"] = fit_summary_logreg(
            compact_train, fold.train_labels, config
        )
    if "summary_xgb" in config.models:
        models["summary_xgb"] = fit_summary_xgb(
            compact_train, fold.train_labels, config
        )
    if "raw_sgd" in config.models:
        logger.info("building raw profile train matrix")
        raw_train = build_raw_pair_features(universe.profiles, fold.train_pairs)
        models["raw_sgd"] = fit_raw_sgd(raw_train, fold.train_labels, config)
        del raw_train
    return models


def _test_scores(
    fold: FengFold,
    compact_test: np.ndarray,
    universe: ProfileUniverse,
    cache: ProfileFeatureCache,
    models: dict[str, ProbabilityModel],
    config: ProfileBaselineConfig,
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    fixed_models = (
        "pearson_abs",
        "spearman_abs",
    )
    for model_name in fixed_models:
        if model_name in config.models:
            scores[model_name] = _fixed_profile_scores(model_name, compact_test)
    if "cell_line_residual_pearson_abs" in config.models:
        scores["cell_line_residual_pearson_abs"] = _residual_pair_scores(
            cache, fold.test_pairs
        )
    if "missingness_only" in config.models:
        a_count = universe.finite_counts[fold.test_pairs[:, 0]]
        b_count = universe.finite_counts[fold.test_pairs[:, 1]]
        scores["missingness_only"] = np.minimum(a_count, b_count) / len(
            universe.cell_lines
        )
    for model_name in ("summary_logreg", "summary_xgb"):
        if model_name in models:
            scores[model_name] = positive_probability(models[model_name], compact_test)
    if "raw_sgd" in models:
        raw_test = build_raw_pair_features(universe.profiles, fold.test_pairs)
        scores["raw_sgd"] = positive_probability(models["raw_sgd"], raw_test)
        del raw_test
    return scores


def _coessentiality_matrix(
    cache: ProfileFeatureCache, model_name: str, path: Path
) -> np.memmap:
    if model_name == "pearson_abs":
        normalized = cache.centered / cache.centered_norm[:, None]
    elif model_name == "spearman_abs":
        normalized = cache.rank_centered / cache.rank_norm[:, None]
    elif model_name == "cell_line_residual_pearson_abs":
        normalized = (
            cache.line_residual_centered / cache.line_residual_norm[:, None]
        )
    else:
        raise ValueError(f"unsupported coessentiality model: {model_name}")
    n_gene = len(normalized)
    scores = np.memmap(path, mode="w+", dtype=np.float32, shape=(n_gene, n_gene))
    for start in range(0, n_gene, 64):
        stop = min(start + 64, n_gene)
        scores[start:stop] = np.abs(normalized[start:stop] @ normalized.T)
    np.fill_diagonal(scores, 0.0)
    scores.flush()
    return scores


def _summary_score_matrix(
    model: ProbabilityModel,
    cache: ProfileFeatureCache,
    config: ProfileBaselineConfig,
    path: Path,
) -> np.memmap:
    n_gene = len(cache.profiles)
    matrix = np.memmap(path, mode="w+", dtype=np.float32, shape=(n_gene, n_gene))
    all_indices = np.arange(n_gene)
    for start in range(0, n_gene, config.score_matrix_chunk_rows):
        stop = min(start + config.score_matrix_chunk_rows, n_gene)
        rows = np.arange(start, stop)
        pairs = np.column_stack(
            [
                np.repeat(rows, n_gene),
                np.tile(all_indices, len(rows)),
            ]
        )
        features = build_cached_pair_features(
            cache, pairs, chunk_size=config.pair_feature_chunk_size
        )
        matrix[start:stop] = positive_probability(model, features).reshape(
            len(rows), n_gene
        )
    np.fill_diagonal(matrix, 0.0)
    matrix.flush()
    return matrix


def _ranking_rows(
    split_type: str,
    fold_id: int,
    model_name: str,
    matrix: np.ndarray,
    fold: FengFold,
    ks: tuple[int, ...],
) -> list[dict[str, object]]:
    test_pos = fold.test_pairs[fold.test_labels == 1]
    train_pos = fold.train_pairs[fold.train_labels == 1]
    metrics = _ranking_metrics_low_memory(matrix, test_pos, train_pos, ks)
    return [
        {
            "split_type": split_type,
            "fold_id": fold_id,
            "model": model_name,
            "slice": "full_universe",
            "metric": metric,
            "value": value,
            "n_pairs": len(test_pos),
        }
        for metric, value in metrics.items()
    ]


def _slice_count_rows(
    split_type: str,
    fold_id: int,
    fold: FengFold,
    universe: ProfileUniverse,
    config: ProfileBaselineConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slice_name, mask in _slice_masks(fold.test_pairs, universe, config).items():
        labels = fold.test_labels[mask]
        rows.append(
            {
                "split_type": split_type,
                "fold_id": fold_id,
                "slice": slice_name,
                "n_pairs": int(mask.sum()),
                "n_positive": int(labels.sum()),
                "n_negative": int(len(labels) - labels.sum()),
                "positive_prevalence": float(labels.mean()),
            }
        )
    return rows


def _run_fold(
    split_type: str,
    fold_id: int,
    fold: FengFold,
    universe: ProfileUniverse,
    cache: ProfileFeatureCache,
    config: ProfileBaselineConfig,
) -> list[dict[str, object]]:
    compact_train = build_cached_pair_features(
        cache, fold.train_pairs, config.pair_feature_chunk_size
    )
    compact_test = build_cached_pair_features(
        cache, fold.test_pairs, config.pair_feature_chunk_size
    )
    models = _fit_supervised_models(fold, compact_train, universe, config)
    scores = _test_scores(fold, compact_test, universe, cache, models, config)
    masks = _slice_masks(fold.test_pairs, universe, config)
    rows: list[dict[str, object]] = []
    for model_name, model_scores in scores.items():
        rows.extend(
            _classification_rows(
                split_type,
                fold_id,
                model_name,
                fold.test_labels,
                model_scores,
                masks,
            )
        )
    del compact_train, compact_test

    with tempfile.TemporaryDirectory(prefix="sl-profile-scores-") as temp_dir:
        for model_name in config.ranking_models:
            if model_name not in config.models:
                continue
            logger.info("scoring candidate matrix: model=%s", model_name)
            matrix_path = Path(temp_dir) / f"{model_name}.float32"
            if model_name in {
                "pearson_abs",
                "spearman_abs",
                "cell_line_residual_pearson_abs",
            }:
                matrix = _coessentiality_matrix(cache, model_name, matrix_path)
            elif model_name in models:
                matrix = _summary_score_matrix(
                    models[model_name], cache, config, matrix_path
                )
            else:
                raise ValueError(f"ranking model is unavailable: {model_name}")
            rows.extend(
                _ranking_rows(
                    split_type,
                    fold_id,
                    model_name,
                    matrix,
                    fold,
                    config.ranking_k,
                )
            )
            del matrix
    return rows


def _summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_metrics.groupby(["split_type", "model", "slice", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["split_type", "model", "slice", "metric"])
        .reset_index(drop=True)
    )


def _manifest(
    config: ProfileBaselineConfig,
    universe: ProfileUniverse,
    split_hashes: dict[str, str],
) -> dict[str, object]:
    config_values = asdict(config)
    for key, value in config_values.items():
        if isinstance(value, Path):
            config_values[key] = str(value)
        elif isinstance(value, tuple):
            config_values[key] = list(value)
    source_dir = Path(__file__).parent
    source_hashes = {
        path.name: _file_sha256(path) for path in sorted(source_dir.glob("*.py"))
    }
    return {
        "task": "Feng Rand 1:1 benchmark-label candidate prioritization",
        "probability_semantics": "P(sampled Feng label=1 | balanced benchmark)",
        "biological_sl_probability": False,
        "feature_boundary": "supervised models use only the two queried profiles",
        "transductive_control": (
            "cell_line_residual_pearson_abs subtracts a label-independent "
            "per-cell-line median estimated across covered candidate genes"
        ),
        "depmap_release": "Public 26Q1",
        "gene_effect_csv": str(config.gene_effect_csv),
        "gene_effect_sha256": _file_sha256(config.gene_effect_csv),
        "entities_csv": str(config.entities_csv),
        "entities_sha256": _file_sha256(config.entities_csv),
        "split_sha256": split_hashes,
        "config_sha256": (
            _file_sha256(config.config_source)
            if config.config_source is not None
            else None
        ),
        "source_sha256": source_hashes,
        "resolved_config": config_values,
        "candidate_gene_count": len(universe.symbols),
        "covered_gene_count": int(universe.coverage.sum()),
        "cell_line_count": len(universe.cell_lines),
        "min_finite_lines": config.min_finite_lines,
        "missing_profile_fallback": "all-zero neutral profile plus coverage feature",
        "within_profile_missing_fallback": "per-gene median",
        "residual_control": "subtract per-cell-line median across covered genes",
        "models": list(config.models),
        "ranking_models": list(config.ranking_models),
        "supervised_model_ranking": False,
        "ranking_note": (
            "summary_logreg, summary_xgb, and raw_sgd are classification-only; "
            "full 9845x9845 ranking is reported for fixed coessentiality scores"
        ),
        "classification_metric_note": (
            "aupr matches Feng trapezoidal PR-AUC; average_precision has the "
            "slice prevalence as its random baseline; f1_oracle selects the "
            "threshold on each test fold"
        ),
        "score_matrix_storage": "temporary float32 memmap",
        "split_types": list(config.split_types),
        "cv2_cv3_auxiliary_exposure": (
            "held-out genes are unseen to Feng SL-pair/graph training, but their "
            "ground-truth DepMap GeneEffect profiles remain available as inputs"
        ),
        "folds": list(config.folds),
        "ranking_k": list(config.ranking_k),
        "feature_names": list(PROFILE_FEATURE_NAMES),
        "seed": config.seed,
    }


def run_cv(config: ProfileBaselineConfig) -> pd.DataFrame:
    """Run the fixed model ladder over official Feng CV1/CV2/CV3 folds."""
    first_split = config.split_path(config.split_types[0])
    n_gene = candidate_gene_count(first_split)
    universe = load_profile_universe(
        config.entities_csv,
        config.gene_effect_csv,
        n_gene,
        config.min_finite_lines,
        config.dependency_threshold,
    )
    cache = prepare_feature_cache(
        universe.profiles, universe.coverage, config.dependency_threshold
    )
    rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    split_hashes: dict[str, str] = {}
    for split_type in config.split_types:
        split_path = config.split_path(split_type)
        split_hashes[split_type] = _file_sha256(split_path)
        if candidate_gene_count(split_path) != n_gene:
            raise ValueError(f"candidate universe differs for {split_type}")
        for fold_id in config.folds:
            logger.info(
                "running profile baseline: split=%s fold=%d", split_type, fold_id
            )
            fold = load_feng_fold(split_path, fold_id)
            slice_rows.extend(
                _slice_count_rows(split_type, fold_id, fold, universe, config)
            )
            rows.extend(
                _run_fold(split_type, fold_id, fold, universe, cache, config)
            )
    fold_metrics = pd.DataFrame(rows).sort_values(
        ["split_type", "fold_id", "model", "slice", "metric"]
    )
    summary = _summarize(fold_metrics)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(output_dir / "slice_counts.csv", index=False)
    pd.DataFrame(
        {
            "feng_unified_id": np.arange(len(universe.symbols)),
            "feng_symbol": universe.symbols,
            "depmap_entrez_id": universe.entrez_ids,
            "symbol_match": [value is not None for value in universe.entrez_ids],
            "finite_line_count": universe.finite_counts,
            "profile_covered": universe.coverage,
        }
    ).to_csv(output_dir / "gene_mapping.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(_manifest(config, universe, split_hashes), indent=2)
    )
    return summary
