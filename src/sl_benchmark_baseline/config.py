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
        bags_npz: Optional exp03 cell-bags NPZ enabling exp07 augmented mode.
        embedding_method: Pooling/representation tag recorded in the manifest.
        fallback_strategy: Uncovered-gene fallback (``"zero"`` or
            ``"global_mean"``).
        include_coverage_flag: Whether to append swap-invariant coverage columns.
        depmap_dir: DepMap CSV directory enabling exp09 selectivity mode.
        cn_loss_thr: Copy-number threshold below which a gene is loss-flagged.
        expr_low_quantile: Quantile defining low-expression flagging.
        sel_n_min: Minimum cell-line count for a usable selectivity statistic.
        sel_lambda: Selectivity-feature regularization/blend weight.
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
    bags_npz: Path | None = None
    embedding_method: str = "pca_delta_meanpool"
    fallback_strategy: str = "zero"
    include_coverage_flag: bool = True
    depmap_dir: Path | None = None
    cn_loss_thr: float = 0.8
    expr_low_quantile: float = 0.10
    sel_n_min: int = 20
    sel_lambda: float = 0.0

    @property
    def augmented(self) -> bool:
        """True when a cell-bags NPZ is supplied (exp07 augmented mode)."""
        return self.bags_npz is not None

    @property
    def selectivity(self) -> bool:
        """True when a DepMap dir is supplied (exp09 selectivity mode)."""
        return self.depmap_dir is not None
