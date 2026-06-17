"""Configuration for the K562 SL-pair dependency-only baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SLBaselineConfig:
    """Defaults and hyperparameters for the SL-pair baseline run.

    Attributes:
        input_csv: Canonical all-CV (CV1/CV2/CV3) balanced benchmark CSV.
        output_dir: Run directory for metrics and manifest.
        split_types: CV split types to evaluate; ``None`` auto-discovers input
            (all of CV1/CV2/CV3 present in the canonical CSV).
        folds: CV fold ids to evaluate.
        ranking_k: Cutoffs for NDCG/Recall/Precision@k.
        seed: Global seed for deterministic model fits.
        logreg_c: Inverse regularization strength for model A.
        logreg_max_iter: Max solver iterations for model A.
        xgb_n_estimators: Number of trees for model B.
        xgb_max_depth: Max tree depth for model B.
        xgb_learning_rate: Learning rate for model B.
    """

    input_csv: Path = Path(
        "data/SL_benchmark/derived/k562_depmap_rand_1to1/"
        "all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv"
    )
    output_dir: Path = Path(
        "results/experiments/06_k562_sl_pair_dependency_only_mvp/run"
    )
    split_types: tuple[str, ...] | None = None
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    ranking_k: tuple[int, ...] = (10, 20, 50)
    seed: int = 17
    logreg_c: float = 1.0
    logreg_max_iter: int = 1000
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.1
