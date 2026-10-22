"""baselines / residual."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.data.labels import ResidualTargets, fit_gene_means
from src.data.splits import FixedSplit, assert_fit_eligible

_LOGGER = logging.getLogger(__name__)


GENE_MEAN: str = "gene_mean"


COPY_PRIOR: str = "copy_prior"


NEAREST_LINE: str = "nearest_line"


CONTEXT_PCA_RIDGE: str = "context_pca_ridge"


VAL_SLICE: str = "val"


TEST_SLICE: str = "test"


_MIN_TRAIN_LINES_FOR_CONTEXT: int = 2


_FEATURE_CONSTANT_EPS: float = 1e-12


@dataclass(frozen=True)
class _LadderConfig:
    """Ladder-wide tuning knobs, bundled to shorten call sites below."""

    pca_components: int
    ridge_alpha: float
    min_lines: int


def _view_method_name(base: str, view_name: str) -> str:
    """Per-context-view method name, e.g. ``nearest_line[tx1]``."""
    return f"{base}[{view_name}]"


def _row(
    slice_name: str, model_id: str, method: str, held: pd.DataFrame, gene: str, pred
) -> dict:
    """One predictions.csv row. ``held`` is this line's rt.long, indexed by gene."""
    return {
        "slice": slice_name,
        "model_id": model_id,
        "gene_symbol": gene,
        "method": method,
        "gene_effect": float(held.loc[gene, "gene_effect"]),
        "residual": float(held.loc[gene, "residual"]),
        "residual_prediction": float(pred),
    }


def _build_fold_targets(
    labels: pd.DataFrame, train_lines: Sequence[str], min_lines: int
) -> ResidualTargets:
    """Fit mu_hat_g on ``train_lines`` only; attach it to every label row.

    This is the TARGET-centering mean -- safe to fit per-fold/per-slice
    (see module docstring, fact 1). Mirrors
    ``residual_target.build_residual_targets``, but forwards ``min_lines``
    to :func:`fit_gene_means` -- ``build_residual_targets`` always uses
    that function's own default (3) with no override.
    """
    gene_mean = fit_gene_means(labels, train_lines, min_lines=min_lines)
    n_before = len(labels)
    long = labels.merge(
        gene_mean.rename("gene_mean"),
        left_on="gene_symbol",
        right_index=True,
        how="inner",
    )
    n_dropped = n_before - len(long)
    if n_dropped:
        _LOGGER.warning(
            "dropped %d label row(s) whose gene did not survive min_lines=%d",
            n_dropped,
            min_lines,
        )
    missing_labels = int(long["gene_effect"].isna().sum())
    if missing_labels:
        _LOGGER.warning("dropped %d row(s) without GeneEffect labels", missing_labels)
        long = long.loc[long["gene_effect"].notna()]
    long = long.copy()
    long["residual"] = long["gene_effect"] - long["gene_mean"]
    long = long[["model_id", "gene_symbol", "gene_effect", "gene_mean", "residual"]]
    long = long.reset_index(drop=True)
    return ResidualTargets(
        long=long,
        gene_mean=gene_mean,
        train_lines=tuple(train_lines),
        n_genes=long["gene_symbol"].nunique(),
        n_lines=long["model_id"].nunique(),
    )


def _gene_mean_rows(
    slice_name: str, rt: ResidualTargets, held_out_id: str
) -> list[dict]:
    """gene_mean rows: delta_hat identically 0.0 for every gene."""
    held = rt.long.loc[rt.long["model_id"] == held_out_id].set_index("gene_symbol")
    return [_row(slice_name, held_out_id, GENE_MEAN, held, g, 0.0) for g in held.index]


def _copy_prior_rows(
    slice_name: str,
    rt: ResidualTargets,
    held_out_id: str,
    prior: pd.Series,
    mu_bar: pd.Series,
) -> list[dict]:
    """copy_prior rows: delta_hat = prior_g - mu_bar_g (fold-independent)."""
    held = rt.long.loc[rt.long["model_id"] == held_out_id].set_index("gene_symbol")
    genes = held.index.intersection(prior.index).intersection(mu_bar.index)
    if len(genes) == 0:
        _LOGGER.warning("copy_prior: no overlapping genes for line %s", held_out_id)
        return []
    delta = prior.loc[genes] - mu_bar.loc[genes]
    return [
        _row(slice_name, held_out_id, COPY_PRIOR, held, g, delta.loc[g]) for g in genes
    ]


def _prepare_fixed_context(
    view: pd.DataFrame,
    train_ids: list[str],
    eval_ids: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit one train-only scaler and transform every fixed-split eval line."""
    feature_columns = list(view.columns)
    train_raw = view.loc[train_ids, feature_columns].to_numpy(dtype=float)
    spread = train_raw.max(axis=0) - train_raw.min(axis=0)
    keep = spread > _FEATURE_CONSTANT_EPS
    if not keep.any():
        _LOGGER.warning("fixed split: no non-constant context features")
        return None
    train_raw = train_raw[:, keep]
    eval_raw = view.loc[eval_ids, feature_columns].to_numpy(dtype=float)[:, keep]
    scaler = StandardScaler().fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    eval_scaled = scaler.transform(eval_raw)
    if not np.isfinite(train_scaled).all() or not np.isfinite(eval_scaled).all():
        raise ValueError("non-finite standardized context in fixed split")
    return train_scaled, eval_scaled


def _run_fixed_evals(
    labels: pd.DataFrame,
    train_lines: Sequence[str],
    eval_specs: Sequence[tuple[str, str]],
    mu_bar: pd.Series,
    context_views: Mapping[str, pd.DataFrame],
    copy_prior: pd.Series | None,
    config: _LadderConfig,
) -> list[dict]:
    """Fit each fixed-split baseline once and batch-predict val plus test."""
    rt = _build_fold_targets(labels, train_lines, config.min_lines)
    train_ids = list(train_lines)
    eval_ids = [held_out_id for _slice_name, held_out_id in eval_specs]
    held_by_id = {
        model_id: frame.set_index("gene_symbol")
        for model_id, frame in rt.long.loc[rt.long["model_id"].isin(eval_ids)].groupby(
            "model_id", sort=False
        )
    }
    missing_labels = sorted(set(eval_ids) - set(held_by_id))
    if missing_labels:
        raise ValueError(f"fixed split has no surviving labels for {missing_labels}")

    rows: list[dict] = []
    for slice_name, held_out_id in eval_specs:
        rows.extend(_gene_mean_rows(slice_name, rt, held_out_id))
        if copy_prior is not None:
            rows.extend(
                _copy_prior_rows(slice_name, rt, held_out_id, copy_prior, mu_bar)
            )

    train_label_rows = rt.long.loc[rt.long["model_id"].isin(train_ids)]
    train_by_gene = {
        gene: frame.set_index("model_id")[["gene_effect", "residual"]]
        for gene, frame in train_label_rows.groupby("gene_symbol", sort=False)
    }
    eval_row_by_id = {model_id: index for index, model_id in enumerate(eval_ids)}

    for view_name, view in context_views.items():
        view_ids = set(view.index)
        missing_eval = sorted(set(eval_ids) - view_ids)
        if missing_eval:
            raise ValueError(
                f"context view {view_name!r} lacks fixed-split eval rows: "
                f"{missing_eval}"
            )
        context_train_ids = [model_id for model_id in train_ids if model_id in view_ids]
        if len(context_train_ids) < _MIN_TRAIN_LINES_FOR_CONTEXT:
            raise ValueError(
                f"context view {view_name!r} has too few fixed-split train rows"
            )
        prepared = _prepare_fixed_context(view, context_train_ids, eval_ids)
        if prepared is None:
            continue
        train_scaled, eval_scaled = prepared
        context_row_by_id = {
            model_id: index for index, model_id in enumerate(context_train_ids)
        }

        distances = np.linalg.norm(
            eval_scaled[:, np.newaxis, :] - train_scaled[np.newaxis, :, :], axis=2
        )
        donor_orders = np.argsort(distances, axis=1, kind="stable")
        nearest_method = _view_method_name(NEAREST_LINE, view_name)
        for slice_name, held_out_id in eval_specs:
            held = held_by_id[held_out_id]
            donor_order = donor_orders[eval_row_by_id[held_out_id]]
            for gene in held.index:
                gene_train = train_by_gene[gene]
                donor_id = next(
                    (
                        context_train_ids[index]
                        for index in donor_order
                        if context_train_ids[index] in gene_train.index
                    ),
                    None,
                )
                if donor_id is None:
                    raise ValueError(
                        f"nearest_line[{view_name}]: no labeled train donor for {gene}"
                    )
                prediction = float(gene_train.loc[donor_id, "gene_effect"]) - float(
                    mu_bar.loc[gene]
                )
                rows.append(
                    _row(
                        slice_name,
                        held_out_id,
                        nearest_method,
                        held,
                        gene,
                        prediction,
                    )
                )

        n_train, n_features = train_scaled.shape
        n_components = min(config.pca_components, n_train - 1, n_features)
        if n_components < 1:
            continue
        pca = PCA(n_components=n_components, svd_solver="full").fit(train_scaled)
        train_pcs = pca.transform(train_scaled)
        eval_pcs = pca.transform(eval_scaled)
        ridge_method = _view_method_name(CONTEXT_PCA_RIDGE, view_name)
        eval_genes = set().union(*(set(held.index) for held in held_by_id.values()))
        for gene in rt.gene_mean.index:
            if gene not in eval_genes:
                continue
            gene_rows = train_by_gene[gene]
            gene_rows = gene_rows.loc[gene_rows.index.isin(context_row_by_id)]
            if len(gene_rows) < _MIN_TRAIN_LINES_FOR_CONTEXT:
                raise ValueError(
                    f"context_pca_ridge[{view_name}]: gene {gene} has fewer than "
                    f"{_MIN_TRAIN_LINES_FOR_CONTEXT} labeled context-train lines"
                )
            indices = [context_row_by_id[model_id] for model_id in gene_rows.index]
            targets = gene_rows["residual"].to_numpy(dtype=float)
            ridge = Ridge(alpha=config.ridge_alpha).fit(train_pcs[indices], targets)
            gene_predictions = np.asarray(ridge.predict(eval_pcs), dtype=float)
            for slice_name, held_out_id in eval_specs:
                held = held_by_id[held_out_id]
                if gene in held.index:
                    rows.append(
                        _row(
                            slice_name,
                            held_out_id,
                            ridge_method,
                            held,
                            gene,
                            gene_predictions[eval_row_by_id[held_out_id]],
                        )
                    )
    return rows


def _validate_split(labels: pd.DataFrame, split: FixedSplit) -> tuple[str, ...]:
    """Return the supervised train lines; every fit line passes the eligibility guard.

    Train membership may include contexts without GeneEffect labels. They stay
    registered train contexts but enter no fit and no donor set. Validation and
    test lines must be labeled. ``split`` membership itself was validated when
    it was loaded.
    """
    known = set(labels["model_id"])
    for name, ids in (("val", split.val), ("test", split.test)):
        unknown = sorted(set(ids) - known)
        if unknown:
            raise ValueError(f"split {name!r} has unknown model_id(s): {unknown}")
    unknown_train = set(split.train) - known
    if unknown_train != set(split.unlabeled_train):
        raise ValueError(
            "split train label coverage does not match declared unlabeled_train: "
            f"unknown={sorted(unknown_train)}, "
            f"declared={sorted(split.unlabeled_train)}"
        )
    supervised_train = tuple(model_id for model_id in split.train if model_id in known)
    for model_id in supervised_train:
        assert_fit_eligible(model_id, split)
    return supervised_train


def run_r1_ladder(
    labels: pd.DataFrame,
    context_views: Mapping[str, pd.DataFrame],
    copy_prior: pd.Series | None,
    *,
    split: FixedSplit,
    pca_components: int = 8,
    ridge_alpha: float = 1.0,
    min_lines: int = 3,
) -> pd.DataFrame:
    """The control ladder's residual predictions on the split's validation and test
    lines, every baseline fitted once on its supervised training lines: one row per
    (slice, line, gene, method)."""
    config = _LadderConfig(pca_components, ridge_alpha, min_lines)
    train = _validate_split(labels, split)
    mu_bar = fit_gene_means(labels, train, min_lines=min_lines)
    eval_specs = [(VAL_SLICE, held) for held in split.val] + [
        (TEST_SLICE, held) for held in split.test
    ]
    rows = _run_fixed_evals(
        labels, train, eval_specs, mu_bar, context_views, copy_prior, config
    )
    if not rows:
        raise ValueError("R1 ladder produced no predictions; check inputs")
    return (
        pd.DataFrame(rows)
        .sort_values(["slice", "method", "model_id", "gene_symbol"], kind="stable")
        .reset_index(drop=True)
    )


