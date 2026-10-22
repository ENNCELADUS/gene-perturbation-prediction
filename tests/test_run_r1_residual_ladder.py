"""The residual baseline ladder: residual predictions on a fixed split's validation
and test lines, every baseline fitted once on the supervised training lines.

All fixtures are synthetic. Predictions are centred on the fold-independent
``mu_bar_g``: a prediction centred on a fold-varying mean is an affine function of
the label it predicts and scores a mechanical +/-1.0 on the per-gene axis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.baselines.residual as residual_ladder
from src.baselines.residual import COPY_PRIOR, GENE_MEAN
from src.data.labels import fit_gene_means
from src.data.splits import FixedSplit

_BASE_GENES = [f"G{i}" for i in range(12)]


def _labels_no_signal(
    lines: list[str], genes: list[str], seed: int = 0
) -> pd.DataFrame:
    """Synthetic labels with a per-gene mean plus small iid noise only."""
    rng = np.random.default_rng(seed)
    mu = {gene: -1.0 - 0.05 * i for i, gene in enumerate(genes)}
    rows = [
        {
            "model_id": line,
            "gene_symbol": gene,
            "gene_effect": mu[gene] + float(rng.normal(0.0, 0.05)),
        }
        for line in lines
        for gene in genes
    ]
    return pd.DataFrame(rows)


def _labels_with_signal(
    lines: list[str], genes: list[str], signal: dict[str, float], seed: int = 0
) -> pd.DataFrame:
    """gene_effect(gene, line) = mu_g + slope_g * signal[line] + small noise."""
    rng = np.random.default_rng(seed)
    mu = {gene: -1.0 - 0.05 * i for i, gene in enumerate(genes)}
    slope = {gene: 0.8 + 0.1 * (i % 4) for i, gene in enumerate(genes)}
    rows = [
        {
            "model_id": line,
            "gene_symbol": gene,
            "gene_effect": (
                mu[gene] + slope[gene] * signal[line] + float(rng.normal(0.0, 0.02))
            ),
        }
        for line in lines
        for gene in genes
    ]
    return pd.DataFrame(rows)


def _context(lines: list[str], features: dict[str, dict[str, float]]) -> pd.DataFrame:
    """A context view: one row per line, one column per feature."""
    return pd.DataFrame(
        {name: [values[line] for line in lines] for name, values in features.items()},
        index=pd.Index(lines, name="model_id"),
    )


def _prior(genes: list[str]) -> pd.Series:
    return pd.Series([-1.0 - 0.05 * i for i in range(len(genes))], index=genes)


def _split(train, val, test, unlabeled_train=()) -> FixedSplit:
    return FixedSplit(
        train=tuple(train),
        val=tuple(val),
        test=tuple(test),
        unlabeled_train=tuple(unlabeled_train),
    )


def _ladder(labels, split, contexts=None, prior=None, pca_components=3):
    return residual_ladder.run_r1_ladder(
        labels, contexts or {}, prior, split=split, pca_components=pca_components
    )


def test_predictions_cover_val_and_test_lines_only() -> None:
    lines = [f"L{i}" for i in range(9)]
    labels = _labels_no_signal(lines, _BASE_GENES, seed=3)
    train, val, test = lines[:6], lines[6:8], lines[8:]
    view = _context(
        lines, {"signal": {line: float(i) - 4.0 for i, line in enumerate(lines)}}
    )

    predictions = _ladder(labels, _split(train, val, test), {"ctx": view})

    assert list(predictions.columns) == [
        "slice",
        "model_id",
        "gene_symbol",
        "method",
        "gene_effect",
        "residual",
        "residual_prediction",
    ]
    assert set(predictions.loc[predictions["slice"] == "val", "model_id"]) == set(val)
    assert set(predictions.loc[predictions["slice"] == "test", "model_id"]) == set(test)
    assert set(predictions["model_id"]).isdisjoint(train)
    assert set(predictions["method"]) == {
        GENE_MEAN,
        "nearest_line[ctx]",
        "context_pca_ridge[ctx]",
    }


def test_context_pca_ridge_recovers_a_planted_signal() -> None:
    lines = [f"L{i}" for i in range(12)]
    signal = {line: float(i) - 6.0 for i, line in enumerate(lines)}
    labels = _labels_with_signal(lines, _BASE_GENES, signal)
    predictions = _ladder(
        labels,
        _split(lines[:8], lines[8:], ()),
        {"signal": _context(lines, {"s": signal})},
    )
    ridge = predictions.loc[predictions["method"] == "context_pca_ridge[signal]"]
    for _, gene in ridge.groupby("gene_symbol"):
        assert np.corrcoef(gene["residual_prediction"], gene["residual"])[0, 1] > 0.9


def test_gene_mean_predicts_zero_and_copy_prior_a_per_gene_constant() -> None:
    lines = [f"L{i}" for i in range(8)]
    labels = _labels_no_signal(lines, _BASE_GENES)
    predictions = _ladder(
        labels, _split(lines[:5], lines[5:7], lines[7:]), prior=_prior(_BASE_GENES)
    )
    gene_mean = predictions.loc[predictions["method"] == GENE_MEAN]
    assert len(gene_mean) == 3 * len(_BASE_GENES)
    assert (gene_mean["residual_prediction"] == 0.0).all()
    copy_prior = predictions.loc[predictions["method"] == COPY_PRIOR]
    spread = copy_prior.groupby("gene_symbol")["residual_prediction"].nunique()
    assert (spread == 1).all()
    # centred on the training-line mean, never a fold-varying one
    means = fit_gene_means(labels, lines[:5], min_lines=3)
    expected = _prior(_BASE_GENES) - means
    first = copy_prior.drop_duplicates("gene_symbol").set_index("gene_symbol")
    assert np.allclose(first["residual_prediction"], expected.loc[first.index])


def test_constant_context_feature_is_dropped_not_nan() -> None:
    lines = [f"L{i}" for i in range(8)]
    signal = {line: float(i) for i, line in enumerate(lines)}
    labels = _labels_with_signal(lines, _BASE_GENES, signal)
    view = _context(
        lines, {"constant": {line: 7.0 for line in lines}, "signal": signal}
    )
    predictions = _ladder(labels, _split(lines[:6], lines[6:], ()), {"ctx": view})
    context_rows = predictions.loc[
        predictions["method"].isin(["nearest_line[ctx]", "context_pca_ridge[ctx]"])
    ]
    assert len(context_rows) > 0
    assert np.isfinite(context_rows["residual_prediction"].to_numpy()).all()


def test_fixed_split_train_never_enters_val_fit() -> None:
    """mu_hat/mu_bar come from train only: an outlier validation label stays out."""
    lines = [f"L{i}" for i in range(7)]
    train, val, test = lines[:5], [lines[5]], [lines[6]]
    rows = []
    for gene in ("G0", "G1", "G2"):
        for line in train + test:
            rows.append({"model_id": line, "gene_symbol": gene, "gene_effect": -1.0})
        rows.append({"model_id": val[0], "gene_symbol": gene, "gene_effect": -50.0})
    predictions = _ladder(pd.DataFrame(rows), _split(train, val, test))
    val_gene_mean = predictions.loc[
        (predictions["slice"] == "val") & (predictions["method"] == GENE_MEAN)
    ]
    assert (val_gene_mean["residual"] <= -48.0).all()


def test_fixed_split_keeps_unlabeled_train_members_out_of_supervised_fit() -> None:
    lines = [f"L{i}" for i in range(7)]
    labels = _labels_no_signal(lines, _BASE_GENES, seed=4)
    unlabeled = "PC9_NO_GENEEFFECT"
    signal = {line: float(i) for i, line in enumerate([*lines, unlabeled])}
    predictions = _ladder(
        labels,
        _split([*lines[:5], unlabeled], [lines[5]], [lines[6]], [unlabeled]),
        {"ctx": _context([*lines, unlabeled], {"signal": signal})},
    )
    assert unlabeled not in set(predictions["model_id"])
    assert predictions["residual_prediction"].notna().all()


def test_fixed_split_sparse_train_labels_keep_exact_baseline_coverage() -> None:
    """Sparse train labels fit per gene without changing the evaluation mask."""
    train = [f"L{i}" for i in range(6)]
    val, test = ["V0"], ["T0"]
    genes = ["G0", "G1", "G2"]
    missing_train_pair = {"G0": "L0", "G1": "L1", "G2": "L2"}
    context_value = {
        "L0": 0.0,
        "L1": 1.0,
        "L2": 2.0,
        "L3": 3.0,
        "L4": 4.0,
        "L5": 5.0,
        "V0": 0.1,
        "T0": 4.9,
    }
    rows = []
    for line in [*train, *val, *test]:
        for gene_index, gene in enumerate(genes):
            if line == missing_train_pair.get(gene):
                continue
            rows.append(
                {
                    "model_id": line,
                    "gene_symbol": gene,
                    "gene_effect": -1.0 + gene_index + 0.25 * context_value[line],
                }
            )
    labels = pd.DataFrame(rows)
    predictions = _ladder(
        labels,
        _split(train, val, test),
        {"ctx": _context([*train, *val, *test], {"signal": context_value})},
        _prior(genes),
    )
    expected_methods = {
        GENE_MEAN,
        COPY_PRIOR,
        "nearest_line[ctx]",
        "context_pca_ridge[ctx]",
    }
    keys = ["slice", "model_id", "gene_symbol"]
    truth_keys = set(
        predictions.loc[predictions["method"] == GENE_MEAN, keys].itertuples(
            index=False, name=None
        )
    )
    assert set(predictions["method"]) == expected_methods
    for method in expected_methods:
        method_keys = set(
            predictions.loc[predictions["method"] == method, keys].itertuples(
                index=False, name=None
            )
        )
        assert method_keys == truth_keys

    train_g0 = labels.loc[
        labels["model_id"].isin(train) & (labels["gene_symbol"] == "G0")
    ]
    expected = float(
        train_g0.loc[train_g0["model_id"] == "L1", "gene_effect"].iloc[0]
        - train_g0["gene_effect"].mean()
    )
    actual = predictions.loc[
        (predictions["slice"] == "val")
        & (predictions["model_id"] == "V0")
        & (predictions["gene_symbol"] == "G0")
        & (predictions["method"] == "nearest_line[ctx]"),
        "residual_prediction",
    ].item()
    assert actual == pytest.approx(expected)


def test_fixed_split_fits_one_pca_per_view_and_one_ridge_per_gene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit count scales with genes, not evaluated-line by gene products."""
    train = [f"L{i}" for i in range(8)]
    val = ["V0", "V1", "V2"]
    test = ["T0", "T1"]
    lines = [*train, *val, *test]
    genes = [f"G{i}" for i in range(20)]
    labels = _labels_no_signal(lines, genes)
    signal = {model_id: float(index) for index, model_id in enumerate(lines)}
    fit_counts = {"pca": 0, "ridge": 0}
    original_pca_fit = residual_ladder.PCA.fit
    original_ridge_fit = residual_ladder.Ridge.fit

    def counted_pca_fit(self, *args, **kwargs):
        fit_counts["pca"] += 1
        return original_pca_fit(self, *args, **kwargs)

    def counted_ridge_fit(self, *args, **kwargs):
        fit_counts["ridge"] += 1
        return original_ridge_fit(self, *args, **kwargs)

    monkeypatch.setattr(residual_ladder.PCA, "fit", counted_pca_fit)
    monkeypatch.setattr(residual_ladder.Ridge, "fit", counted_ridge_fit)
    predictions = _ladder(
        labels, _split(train, val, test), {"ctx": _context(lines, {"signal": signal})}
    )
    assert fit_counts == {"pca": 1, "ridge": len(genes)}
    assert len(predictions) == 3 * (len(val) + len(test)) * len(genes)


def test_fixed_split_rejects_unlabeled_validation() -> None:
    lines = [f"L{i}" for i in range(6)]
    labels = _labels_no_signal(lines, _BASE_GENES)
    with pytest.raises(ValueError, match="split 'val' has unknown model_id"):
        _ladder(labels, _split(lines[:5], ["UNKNOWN_VAL"], [lines[5]]))


def test_fixed_split_rejects_undeclared_unknown_train() -> None:
    lines = [f"L{i}" for i in range(6)]
    labels = _labels_no_signal(lines, _BASE_GENES)
    with pytest.raises(ValueError, match="does not match declared unlabeled_train"):
        _ladder(labels, _split([*lines[:5], "TYPO"], [lines[5]], []))


# --- The command on synthetic prepared inputs ---------------------------


def _synthetic_inputs():
    """Eleven lines on 12 genes; ACH-000551 (the K562 copy-prior donor) trains."""
    from src.data.context_pca import fit_context_pca, pooled_context
    from src.data.prepared import PreparedInputs, PreparedLine
    from src.data.q_sc import QScFeatures

    rng = np.random.default_rng(0)
    train = ("ACH-000551", *(f"ACH-T{i}" for i in range(5)))
    val, test = tuple(f"ACH-V{i}" for i in range(4)), ("ACH-X0",)
    lines = (*train, *val, *test)
    genes = tuple(_BASE_GENES)
    signal = {line: float(i) - 4.0 for i, line in enumerate(lines)}
    labels = _labels_with_signal(list(lines), list(genes), signal)
    means = fit_gene_means(labels, train).reindex(list(genes))
    labels["residual"] = labels.gene_effect - labels.gene_symbol.map(means)
    q_sc = QScFeatures(
        symbols=genes, values=np.zeros((len(genes), 3)), available=np.ones(12, bool)
    )
    prepared = {
        line: PreparedLine(
            controls_tx1=(rng.normal(size=(16, 5)) + signal[line]).astype(np.float32),
            basal_hvg=(rng.normal(size=(16, 7)) + 0.5 * signal[line]).astype(
                np.float32
            ),
            q_sc=q_sc,
        )
        for line in lines
    }
    return PreparedInputs(
        split=FixedSplit(train=train, val=val, test=test),
        labels=labels,
        genes=genes,
        train_gene_means=means,
        variable_genes=frozenset(genes),
        selective_genes=frozenset(genes),
        residual_scale=pd.Series(1.0, index=list(genes), name="residual_scale"),
        context_pca=fit_context_pca(
            np.stack([pooled_context(prepared[line].controls_tx1) for line in train]),
            2,
        ),
        hvg_order=tuple(f"H{i}" for i in range(7)),
        esm2_symbols=genes,
        esm2_vectors=np.zeros((len(genes), 2), dtype=np.float32),
        lines=prepared,
        response_targets=None,
        response_anchors=(),
        target_sum=1000.0,
    )


def test_baselines_run_on_synthetic_inputs(tmp_path: Path) -> None:
    from src.experiments.baselines import run_baselines

    inputs = _synthetic_inputs()
    result = run_baselines({}, split="val", out_dir=tmp_path / "val", inputs=inputs)

    assert set(result.summary) == {
        GENE_MEAN,
        COPY_PRIOR,
        "nearest_line[tx1]",
        "nearest_line[hvg]",
        "context_pca_ridge[tx1]",
        "context_pca_ridge[hvg]",
    }
    gene_mean = result.summary[GENE_MEAN]
    assert gene_mean["val_residual_pearson_macro_per_gene"] is None  # undefined
    assert gene_mean["val_residual_pearson_per_gene_scored"] == 0
    assert gene_mean["val_residual_pearson_per_gene_undefined"] == len(_BASE_GENES)
    ridge = result.summary["context_pca_ridge[hvg]"]
    assert ridge["val_residual_pearson_macro_per_gene"] > 0.5  # the planted signal
    evaluated = set(result.predictions.model_id)
    assert evaluated == set(inputs.split.val)  # no train or test line is scored
    for name in ("predictions.parquet", "per_line.csv", "per_gene.csv", "metrics.json"):
        assert (tmp_path / "val" / name).is_file()
    assert not (tmp_path / "test").exists()


def test_baselines_add_the_prior_controls_with_a_prior(tmp_path: Path) -> None:
    import dataclasses

    from src.baselines.prior_controls import PRIOR, PRIOR_PLUS_TX1_RIDGE
    from src.data.prior_offsets import PriorOffsets
    from src.experiments.baselines import run_baselines

    inputs = _synthetic_inputs()
    lines = (*inputs.split.supervised_train, *inputs.split.val)
    values = pd.DataFrame(
        np.outer(0.1 * np.arange(len(lines)), np.ones(len(inputs.genes))),
        index=list(lines),
        columns=list(inputs.genes),
    )
    prior = dataclasses.replace(
        inputs,
        prior=PriorOffsets(
            {"run_id": "r", "reference": {}}, values, inputs.residual_scale
        ),
    )
    result = run_baselines({}, split="val", out_dir=tmp_path / "val", inputs=prior)
    assert {PRIOR, PRIOR_PLUS_TX1_RIDGE} <= set(result.summary)
    assert result.summary[PRIOR]["val_selective_spearman"] is not None
    controls = result.predictions.loc[result.predictions.method == PRIOR]
    assert set(controls.model_id) == set(inputs.split.val)
