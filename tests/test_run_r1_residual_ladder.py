"""Tests for the R1 DepMap GeneEffect residual baseline ladder CLI.

All fixtures are synthetic CSVs/JSON written to ``tmp_path`` -- no
gitignored data dependency. See ``aivc_model.residual_ladder`` for the
delta-only prediction convention and the leave-one-out centering artifact
these tests specifically guard against: a prediction centered by a
fold-*varying* mean (mu_hat_g^(-c)) is an exact affine function of the very
label being predicted, so it can score a mechanical +/-1.0 on the per-gene
axis. Centering predictions with a fold-*independent* mean (mu_bar_g)
removes that artifact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aivc_model.residual_ladder as residual_ladder
from aivc_model.residual_ladder import (
    COPY_PRIOR,
    GENE_MEAN,
    _build_fold_targets,
)
from aivc_model.residual_metrics import score_predictions
from aivc_model.residual_target import fit_gene_means
from scripts.run_r1_residual_ladder import main

_BASE_LINES = [f"L{i}" for i in range(6)]
_BASE_GENES = [f"G{i}" for i in range(12)]
_SIGNAL_BY_LINE = {
    "L0": -2.5,
    "L1": -1.5,
    "L2": -0.5,
    "L3": 0.5,
    "L4": 1.5,
    "L5": 2.5,
}


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
    """Synthetic labels where gene_effect linearly depends on ``signal``.

    gene_effect(gene, line) = mu_g + slope_g * signal[line] + small noise,
    with slope_g nonzero and gene-varying, so the per-gene axis genuinely
    depends on the line-level context via a real (not spurious) mechanism.
    """
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


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def _write_context_csv(
    path: Path, lines: list[str], features: dict[str, dict[str, float]]
) -> Path:
    """Write a wide context CSV: model_id + one column per feature name."""
    frame = pd.DataFrame({"model_id": lines})
    for name, values in features.items():
        frame[name] = [values[line] for line in lines]
    frame.to_csv(path, index=False)
    return path


def _write_split_json(
    path: Path,
    train: list[str],
    val: list[str],
    test: list[str],
    unlabeled_train: list[str] | None = None,
) -> Path:
    payload = {
        "train": train,
        "val": val,
        "test": test,
        "unlabeled_train": unlabeled_train or [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_cli(
    tmp_path: Path,
    labels: pd.DataFrame,
    *,
    context_paths: dict[str, Path] | None = None,
    prior_path: Path | None = None,
    pca_components: int = 3,
    out_name: str = "out",
    outer: str = "lolo",
    split_json: Path | None = None,
) -> Path:
    """Write ``labels`` to CSV, invoke the CLI, and return its out-dir."""
    labels_csv = _write_csv(tmp_path / "labels.csv", labels)
    out_dir = tmp_path / out_name
    argv = [
        "--labels",
        str(labels_csv),
        "--out-dir",
        str(out_dir),
        "--pca-components",
        str(pca_components),
        "--outer",
        outer,
    ]
    if outer == "fixed":
        assert split_json is not None, "fixed outer requires split_json in this helper"
        argv += ["--split-json", str(split_json)]
    for name, path in (context_paths or {}).items():
        argv += ["--context", f"{name}={path}"]
    if prior_path is not None:
        argv += ["--copy-prior", str(prior_path)]
    exit_code = main(argv)
    assert exit_code == 0
    return out_dir


def _write_prior_csv(path: Path, genes: list[str]) -> Path:
    effects = [-1.0 - 0.05 * i for i in range(len(genes))]
    frame = pd.DataFrame({"gene_symbol": genes, "gene_effect": effects})
    frame.to_csv(path, index=False)
    return path


def _read_slice_methods(out_dir: Path, slice_name: str = "lolo") -> dict:
    """Load summary.json and return one slice's ``methods`` dict."""
    summary = json.loads((out_dir / "summary.json").read_text())
    return summary["slices"][slice_name]["methods"]


# --- End-to-end (LOLO) --------------------------------------------------


def test_end_to_end_writes_expected_outputs(tmp_path: Path) -> None:
    """A small synthetic LOLO run completes and writes all four output files."""
    labels = _labels_no_signal(_BASE_LINES, _BASE_GENES)
    feat1 = {line: float(i) for i, line in enumerate(_BASE_LINES)}
    context_csv = _write_context_csv(
        tmp_path / "ctx.csv", _BASE_LINES, {"feat0": _SIGNAL_BY_LINE, "feat1": feat1}
    )
    out_dir = _run_cli(tmp_path, labels, context_paths={"ctx": context_csv})

    for name in ("summary.json", "per_line.csv", "per_gene.csv", "predictions.csv"):
        assert (out_dir / name).exists()

    predictions = pd.read_csv(out_dir / "predictions.csv")
    assert list(predictions.columns) == [
        "slice",
        "model_id",
        "gene_symbol",
        "method",
        "gene_effect",
        "residual",
        "residual_prediction",
    ]
    assert set(predictions["slice"]) == {"lolo"}
    per_line = pd.read_csv(out_dir / "per_line.csv")
    assert list(per_line.columns) == ["slice", "method", "model_id", "spearman"]
    per_gene = pd.read_csv(out_dir / "per_gene.csv")
    assert list(per_gene.columns) == ["slice", "method", "gene_symbol", "spearman"]

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["outer"] == "lolo"
    lolo = summary["slices"]["lolo"]
    assert lolo["n_lines_evaluated"] == len(_BASE_LINES)
    assert lolo["n_genes_evaluated"] == len(_BASE_GENES)
    expected_methods = {GENE_MEAN, "nearest_line[ctx]", "context_pca_ridge[ctx]"}
    assert expected_methods.issubset(lolo["methods"].keys())
    for entry in lolo["methods"].values():
        assert "ci_per_line" in entry
        assert "ci_per_gene" in entry


# --- Planted signal: proves the harness can detect real context signal -


def test_context_pca_ridge_detects_planted_signal(tmp_path: Path) -> None:
    """context_pca_ridge must clearly beat the NaN gene_mean baseline.

    One context dimension linearly drives the residual by construction
    (see :func:`_labels_with_signal`). If context_pca_ridge cannot recover
    a strong, well-defined per-gene score here, the harness cannot detect
    real context signal at all, and every negative result it produces
    elsewhere is uninterpretable.
    """
    labels = _labels_with_signal(_BASE_LINES, _BASE_GENES, _SIGNAL_BY_LINE)
    context_csv = _write_context_csv(
        tmp_path / "ctx.csv", _BASE_LINES, {"signal": _SIGNAL_BY_LINE}
    )
    out_dir = _run_cli(tmp_path, labels, context_paths={"signal": context_csv})
    methods = _read_slice_methods(out_dir)

    gene_mean_entry = methods[GENE_MEAN]
    assert gene_mean_entry["macro_per_gene"] is None or math.isnan(
        gene_mean_entry["macro_per_gene"]
    )

    ridge_entry = methods["context_pca_ridge[signal]"]
    macro_per_gene = ridge_entry["macro_per_gene"]
    assert macro_per_gene is not None and not math.isnan(macro_per_gene)
    assert macro_per_gene > 0.8, (
        f"context_pca_ridge should strongly recover the planted per-gene "
        f"signal; got macro_per_gene={macro_per_gene}"
    )


def test_per_gene_axis_invariant_to_truth_column(tmp_path: Path) -> None:
    """Per-gene Spearman of delta_hat vs. gene_effect == vs. residual.

    Per-gene Spearman is invariant to subtracting any per-gene constant
    from the truth column, so scoring the same predictions against raw
    gene_effect and against the fold-fit residual must agree exactly.
    This is the free correctness check the R1 protocol relies on to prove
    the residual construction did not leak.
    """
    labels = _labels_with_signal(_BASE_LINES, _BASE_GENES, _SIGNAL_BY_LINE)
    context_csv = _write_context_csv(
        tmp_path / "ctx.csv", _BASE_LINES, {"signal": _SIGNAL_BY_LINE}
    )
    out_dir = _run_cli(tmp_path, labels, context_paths={"signal": context_csv})
    predictions = pd.read_csv(out_dir / "predictions.csv")

    subframe = predictions.loc[predictions["method"] == "context_pca_ridge[signal]"]
    assert len(subframe) > 0
    via_residual = score_predictions(
        subframe, truth_col="residual", pred_col="residual_prediction"
    )
    via_raw = score_predictions(
        subframe, truth_col="gene_effect", pred_col="residual_prediction"
    )
    assert not math.isnan(via_residual.macro_per_gene)
    assert via_raw.macro_per_gene == pytest.approx(
        via_residual.macro_per_gene, abs=1e-6
    )


def test_per_line_axis_is_scored_against_residual(tmp_path: Path) -> None:
    """The reported per-line axis must pair delta_hat against delta (residual).

    Unlike the per-gene axis, per-line Spearman is *not* invariant to
    truth-column choice: mu_g varies with gene, which is exactly the axis
    correlated over within one line, so subtracting it changes ranks.
    Scoring delta_hat against raw gene_effect would put mu_g back into
    only one side of the comparison (the prediction has no mu component at
    all) and measure accidental alignment with the gene main effect
    instead of genuine deviation-prediction skill. Pin the correct
    (delta_hat vs. delta) convention here.
    """
    labels = _labels_with_signal(_BASE_LINES, _BASE_GENES, _SIGNAL_BY_LINE)
    context_csv = _write_context_csv(
        tmp_path / "ctx.csv", _BASE_LINES, {"signal": _SIGNAL_BY_LINE}
    )
    out_dir = _run_cli(tmp_path, labels, context_paths={"signal": context_csv})
    predictions = pd.read_csv(out_dir / "predictions.csv")
    methods = _read_slice_methods(out_dir)

    subframe = predictions.loc[predictions["method"] == "context_pca_ridge[signal]"]
    via_residual = score_predictions(
        subframe, truth_col="residual", pred_col="residual_prediction"
    )
    via_raw = score_predictions(
        subframe, truth_col="gene_effect", pred_col="residual_prediction"
    )
    # The two conventions must actually differ for this fixture (otherwise
    # the test can't discriminate between them), and the emitted summary
    # must match the residual-truth convention specifically.
    assert via_raw.macro_per_line != pytest.approx(
        via_residual.macro_per_line, abs=1e-6
    )
    reported = methods["context_pca_ridge[signal]"]["macro_per_line"]
    assert reported == pytest.approx(via_residual.macro_per_line, abs=1e-9)


# --- Null control: no spurious signal from pure noise context ----------


def test_context_pca_ridge_null_control_near_zero(tmp_path: Path) -> None:
    """Pure-noise context must not yield a spuriously high per-gene score."""
    lines = [f"L{i}" for i in range(8)]
    genes = [f"G{i}" for i in range(20)]
    labels = _labels_no_signal(lines, genes, seed=1)
    rng = np.random.default_rng(99)
    noise_features = {
        "noise0": dict(zip(lines, rng.normal(size=len(lines)), strict=True)),
        "noise1": dict(zip(lines, rng.normal(size=len(lines)), strict=True)),
    }
    context_csv = _write_context_csv(tmp_path / "ctx.csv", lines, noise_features)
    out_dir = _run_cli(
        tmp_path,
        labels,
        context_paths={"noise": context_csv},
        pca_components=2,
    )
    methods = _read_slice_methods(out_dir)

    ridge_entry = methods["context_pca_ridge[noise]"]
    macro_per_gene = ridge_entry["macro_per_gene"]
    assert macro_per_gene is not None and not math.isnan(macro_per_gene)
    assert abs(macro_per_gene) < 0.4, (
        f"pure-noise context should not yield a strong per-gene score; "
        f"got macro_per_gene={macro_per_gene}"
    )

    gene_mean_entry = methods[GENE_MEAN]
    assert gene_mean_entry["macro_per_gene"] is None or math.isnan(
        gene_mean_entry["macro_per_gene"]
    )


# --- gene_mean is NaN, never coerced to 0 -------------------------------


def test_gene_mean_is_nan_on_both_axes(tmp_path: Path) -> None:
    """gene_mean's delta_hat is identically 0.0, so both axes are NaN."""
    labels = _labels_no_signal(_BASE_LINES, _BASE_GENES)
    out_dir = _run_cli(tmp_path, labels)
    methods = _read_slice_methods(out_dir)

    entry = methods[GENE_MEAN]
    assert entry["macro_per_line"] is None or math.isnan(entry["macro_per_line"])
    assert entry["macro_per_gene"] is None or math.isnan(entry["macro_per_gene"])
    assert entry["n_lines"] == 0
    assert entry["n_genes"] == 0
    assert entry["n_line_undefined"] == len(_BASE_LINES)
    assert entry["n_gene_undefined"] == len(_BASE_GENES)

    predictions = pd.read_csv(out_dir / "predictions.csv")
    gene_mean_rows = predictions.loc[predictions["method"] == GENE_MEAN]
    assert len(gene_mean_rows) == len(_BASE_LINES) * len(_BASE_GENES)
    assert (gene_mean_rows["residual_prediction"] == 0.0).all()


# --- Fix 1 regression: fold-independent centering, not fold-fit --------


def test_copy_prior_per_gene_is_nan_not_leaked(tmp_path: Path) -> None:
    """copy_prior's delta_hat = prior_g - mu_bar_g is a true per-gene constant.

    Before the fix, copy_prior centered its prediction with the fold-*fit*
    mean (mu_hat_g^(-c)), which is an exact affine function of the very
    label being predicted -- scoring +1.0 on the per-gene axis by
    construction, regardless of the prior's actual content. With a
    fold-*independent* mu_bar_g, the prediction is a true per-gene
    constant (same value for every evaluated line), so it is undefined
    (NaN), not a leaked correlation.
    """
    labels = _labels_no_signal(_BASE_LINES, _BASE_GENES)
    prior_csv = _write_prior_csv(tmp_path / "prior.csv", _BASE_GENES)
    out_dir = _run_cli(tmp_path, labels, prior_path=prior_csv)
    methods = _read_slice_methods(out_dir)

    entry = methods[COPY_PRIOR]
    macro = entry["macro_per_gene"]
    assert macro is None or math.isnan(macro)
    assert entry["n_gene_undefined"] == len(_BASE_GENES)

    predictions = pd.read_csv(out_dir / "predictions.csv")
    copy_prior_rows = predictions.loc[predictions["method"] == COPY_PRIOR]
    # Same gene must predict the identical value for every evaluated line.
    spread = copy_prior_rows.groupby("gene_symbol")["residual_prediction"].nunique()
    assert (spread == 1).all()


def test_no_method_hits_perfect_per_gene_correlation(tmp_path: Path) -> None:
    """No method's per-gene macro is a leakage-signature +/-1.0.

    A perfect correlation across a dozen genes with no planted signal is
    the signature of the leave-one-out centering artifact (Fix 1), never
    a genuine result -- pin it as a general guard across every method in
    the ladder, not just copy_prior specifically.
    """
    labels = _labels_no_signal(_BASE_LINES, _BASE_GENES, seed=7)
    feat1 = {line: float(i) for i, line in enumerate(_BASE_LINES)}
    context_csv = _write_context_csv(
        tmp_path / "ctx.csv", _BASE_LINES, {"feat0": _SIGNAL_BY_LINE, "feat1": feat1}
    )
    prior_csv = _write_prior_csv(tmp_path / "prior.csv", _BASE_GENES)
    out_dir = _run_cli(
        tmp_path, labels, context_paths={"ctx": context_csv}, prior_path=prior_csv
    )
    methods = _read_slice_methods(out_dir)

    for method, entry in methods.items():
        macro = entry["macro_per_gene"]
        if macro is None or (isinstance(macro, float) and math.isnan(macro)):
            continue
        assert abs(abs(macro) - 1.0) > 1e-6, (
            f"{method} scored a leakage-signature +/-1.0 per-gene macro: {macro}"
        )


# --- LOO-artifact regression: pins the bug Fix 1 corrected --------------


def test_loo_artifact_regression(tmp_path: Path) -> None:
    """Emitting mu_hat_g^(-c) itself (instead of delta_hat=0) is a -1.0 artifact.

    Leave-one-out means are an exact affine (decreasing) function of the
    held-out value: mu_hat_g^(-c) = (S_g - y_gc) / (n - 1) for a fixed
    per-gene total S_g. Scoring that quantity directly against the true
    gene_effect on the per-gene axis therefore gives Spearman == -1.0 *by
    construction*, regardless of the data -- a mechanical leakage
    artifact, not a genuine result. This test pins both halves: the wrong
    quantity really does score -1.0, and the runner's actual gene_mean
    output (delta_hat identically 0) does not.
    """
    labels = _labels_no_signal(_BASE_LINES, _BASE_GENES)

    wrong_rows = []
    for held_out in _BASE_LINES:
        train_lines = [line for line in _BASE_LINES if line != held_out]
        mu_hat = fit_gene_means(labels, train_lines, min_lines=3)
        held = labels.loc[labels["model_id"] == held_out].set_index("gene_symbol")
        for gene in mu_hat.index:
            wrong_rows.append(
                {
                    "model_id": held_out,
                    "gene_symbol": gene,
                    "gene_effect": float(held.loc[gene, "gene_effect"]),
                    "residual_prediction": float(mu_hat.loc[gene]),
                }
            )
    wrong_frame = pd.DataFrame(wrong_rows)
    wrong_score = score_predictions(
        wrong_frame, truth_col="gene_effect", pred_col="residual_prediction"
    )
    assert wrong_score.macro_per_gene == pytest.approx(-1.0, abs=1e-9)
    assert wrong_score.n_gene_undefined == 0

    out_dir = _run_cli(tmp_path, labels)
    methods = _read_slice_methods(out_dir)
    entry = methods[GENE_MEAN]
    assert entry["macro_per_gene"] is None or math.isnan(entry["macro_per_gene"])


# --- Leakage: fold-fit mu_hat_g must never see the held-out line's own label --


def test_fold_fit_gene_mean_differs_from_all_lines_mean(tmp_path: Path) -> None:
    """The held-out line's own label must not enter its own centering."""
    lines = _BASE_LINES
    # G0's value on L0 is a strong outlier, so a leaky (all-lines) mean and
    # a correct (train-only) mean must differ measurably for that fold.
    rows = [{"model_id": "L0", "gene_symbol": "G0", "gene_effect": -10.0}]
    rows += [
        {"model_id": line, "gene_symbol": "G0", "gene_effect": -1.0}
        for line in lines
        if line != "L0"
    ]
    labels = pd.DataFrame(rows)

    all_lines_mean = fit_gene_means(labels, lines, min_lines=3)
    train_lines = [line for line in lines if line != "L0"]
    fold_fit_mean = fit_gene_means(labels, train_lines, min_lines=3)

    assert fold_fit_mean.loc["G0"] != pytest.approx(all_lines_mean.loc["G0"])
    assert fold_fit_mean.loc["G0"] == pytest.approx(-1.0)

    # Also check it via the ladder's own fold-construction helper.
    rt = _build_fold_targets(labels, train_lines, min_lines=3)
    assert "L0" not in rt.train_lines
    assert rt.gene_mean.loc["G0"] == pytest.approx(-1.0)
    assert rt.gene_mean.loc["G0"] != pytest.approx(all_lines_mean.loc["G0"])


# --- Constant-within-fold features are dropped, not silently NaN-producing --


def test_constant_context_feature_is_dropped_not_nan(tmp_path: Path) -> None:
    """A globally-constant context column must be dropped, not divide by zero."""
    labels = _labels_with_signal(_BASE_LINES, _BASE_GENES, _SIGNAL_BY_LINE)
    context_csv = _write_context_csv(
        tmp_path / "ctx.csv",
        _BASE_LINES,
        {
            "constant": {line: 7.0 for line in _BASE_LINES},
            "signal": _SIGNAL_BY_LINE,
        },
    )
    out_dir = _run_cli(tmp_path, labels, context_paths={"ctx": context_csv})
    predictions = pd.read_csv(out_dir / "predictions.csv")

    context_rows = predictions.loc[
        predictions["method"].isin(["nearest_line[ctx]", "context_pca_ridge[ctx]"])
    ]
    assert len(context_rows) > 0
    assert context_rows["residual_prediction"].notna().all()
    assert np.isfinite(context_rows["residual_prediction"].to_numpy()).all()


# --- Fixed train/val/test split: the primary (default) protocol --------


def test_fixed_split_evaluates_val_and_test_slices(tmp_path: Path) -> None:
    """--outer fixed (default) fits once on train and scores val/test slices."""
    lines = [f"L{i}" for i in range(9)]
    genes = [f"G{i}" for i in range(12)]
    labels = _labels_no_signal(lines, genes, seed=3)
    train, val, test = lines[:6], lines[6:8], lines[8:]
    signal = {line: float(i) - 4.0 for i, line in enumerate(lines)}
    context_csv = _write_context_csv(tmp_path / "ctx.csv", lines, {"signal": signal})
    split_json = _write_split_json(tmp_path / "split.json", train, val, test)

    out_dir = _run_cli(
        tmp_path,
        labels,
        context_paths={"ctx": context_csv},
        outer="fixed",
        split_json=split_json,
    )
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["outer"] == "fixed"
    assert set(summary["slices"]) == {"val", "test"}
    assert summary["split"]["train"] == train
    assert summary["split"]["supervised_train"] == train
    assert summary["split"]["unlabeled_train"] == []
    assert summary["split"]["val"] == val
    assert summary["split"]["test"] == test
    assert summary["slices"]["val"]["n_lines_evaluated"] == len(val)
    assert summary["slices"]["test"]["n_lines_evaluated"] == len(test)

    predictions = pd.read_csv(out_dir / "predictions.csv")
    assert set(predictions.loc[predictions["slice"] == "val", "model_id"]) == set(val)
    assert set(predictions.loc[predictions["slice"] == "test", "model_id"]) == set(test)
    # Train lines must never appear as an evaluated (predicted-for) line.
    assert set(predictions["model_id"]).isdisjoint(train)


def test_fixed_split_requires_split_json(tmp_path: Path) -> None:
    """The default --outer=fixed must fail clearly without --split-json."""
    labels = _labels_no_signal(_BASE_LINES, _BASE_GENES)
    labels_csv = _write_csv(tmp_path / "labels.csv", labels)
    with pytest.raises(ValueError, match="split-json"):
        main(
            [
                "--labels",
                str(labels_csv),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )


def test_fixed_split_train_never_enters_val_fit(tmp_path: Path) -> None:
    """mu_hat/mu_bar for a fixed split must come from train only, once."""
    lines = [f"L{i}" for i in range(7)]
    genes = ["G0", "G1", "G2"]
    # G0's value on the sole val line is a strong outlier: if it leaked
    # into mu_hat/mu_bar, gene_mean's target residual for that line would
    # be artificially small (leaky centering pulls the mean toward it).
    train, val, test = lines[:5], [lines[5]], [lines[6]]
    rows = []
    for gene in genes:
        for line in train + test:
            rows.append({"model_id": line, "gene_symbol": gene, "gene_effect": -1.0})
        rows.append({"model_id": val[0], "gene_symbol": gene, "gene_effect": -50.0})
    labels = pd.DataFrame(rows)
    split_json = _write_split_json(tmp_path / "split.json", train, val, test)

    out_dir = _run_cli(tmp_path, labels, outer="fixed", split_json=split_json)
    predictions = pd.read_csv(out_dir / "predictions.csv")
    val_gene_mean = predictions.loc[
        (predictions["slice"] == "val") & (predictions["method"] == GENE_MEAN)
    ]
    # mu_hat fit on train only (all -1.0) means residual = -50.0 - (-1.0);
    # any leakage from the outlier val label would pull mu_hat toward it
    # and shrink this residual well below 49 in magnitude.
    assert (val_gene_mean["residual"] <= -48.0).all()


def test_fixed_split_keeps_unlabeled_train_members_out_of_supervised_fit(
    tmp_path: Path,
) -> None:
    """Unlabeled train membership is allowed but never becomes a label donor."""
    lines = [f"L{i}" for i in range(7)]
    genes = [f"G{i}" for i in range(12)]
    labels = _labels_no_signal(lines, genes, seed=4)
    unlabeled = "PC9_NO_GENEEFFECT"
    train, val, test = [*lines[:5], unlabeled], [lines[5]], [lines[6]]
    signal = {line: float(i) for i, line in enumerate([*lines, unlabeled])}
    context_csv = _write_context_csv(
        tmp_path / "ctx.csv", [*lines, unlabeled], {"signal": signal}
    )
    split_json = _write_split_json(
        tmp_path / "split.json", train, val, test, [unlabeled]
    )

    out_dir = _run_cli(
        tmp_path,
        labels,
        context_paths={"ctx": context_csv},
        outer="fixed",
        split_json=split_json,
    )
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["split"]["train"] == train
    assert summary["split"]["supervised_train"] == lines[:5]
    assert summary["split"]["unlabeled_train"] == [unlabeled]
    predictions = pd.read_csv(out_dir / "predictions.csv")
    assert unlabeled not in set(predictions["model_id"])
    assert predictions["residual_prediction"].notna().all()


def test_fixed_split_sparse_train_labels_keep_exact_baseline_coverage(
    tmp_path: Path,
) -> None:
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
                    "gene_effect": -1.0
                    + gene_index
                    + 0.25 * context_value[line],
                }
            )
    labels = pd.DataFrame(rows)
    context_csv = _write_context_csv(
        tmp_path / "ctx.csv",
        [*train, *val, *test],
        {"signal": context_value},
    )
    prior_csv = _write_prior_csv(tmp_path / "prior.csv", genes)
    split_json = _write_split_json(tmp_path / "split.json", train, val, test)

    out_dir = _run_cli(
        tmp_path,
        labels,
        context_paths={"ctx": context_csv},
        prior_path=prior_csv,
        outer="fixed",
        split_json=split_json,
    )
    predictions = pd.read_csv(out_dir / "predictions.csv")
    expected_methods = {
        GENE_MEAN,
        COPY_PRIOR,
        "nearest_line[ctx]",
        "context_pca_ridge[ctx]",
    }
    truth_keys = set(
        predictions.loc[
            predictions["method"] == GENE_MEAN,
            ["slice", "model_id", "gene_symbol"],
        ].itertuples(index=False, name=None)
    )
    assert set(predictions["method"]) == expected_methods
    for method in expected_methods:
        method_keys = set(
            predictions.loc[
                predictions["method"] == method,
                ["slice", "model_id", "gene_symbol"],
            ].itertuples(index=False, name=None)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fit count scales with genes, not evaluated-line by gene products."""
    train = [f"L{i}" for i in range(8)]
    val = ["V0", "V1", "V2"]
    test = ["T0", "T1"]
    lines = [*train, *val, *test]
    genes = [f"G{i}" for i in range(20)]
    labels = _labels_no_signal(lines, genes)
    signal = {model_id: float(index) for index, model_id in enumerate(lines)}
    context_csv = _write_context_csv(
        tmp_path / "ctx.csv", lines, {"signal": signal}
    )
    split_json = _write_split_json(tmp_path / "split.json", train, val, test)
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

    out_dir = _run_cli(
        tmp_path,
        labels,
        context_paths={"ctx": context_csv},
        outer="fixed",
        split_json=split_json,
    )

    assert fit_counts == {"pca": 1, "ridge": len(genes)}
    predictions = pd.read_csv(out_dir / "predictions.csv")
    assert len(predictions) == 3 * (len(val) + len(test)) * len(genes)


def test_fixed_split_fails_when_copy_prior_cannot_cover_truth_mask(
    tmp_path: Path,
) -> None:
    """A configured baseline may not silently evaluate fewer gene-line keys."""
    lines = [f"L{i}" for i in range(7)]
    genes = ["G0", "G1"]
    labels = _labels_no_signal(lines, genes)
    split_json = _write_split_json(
        tmp_path / "split.json", lines[:5], [lines[5]], [lines[6]]
    )
    incomplete_prior = _write_prior_csv(tmp_path / "prior.csv", ["G0"])

    with pytest.raises(ValueError, match="evaluated-key coverage differs"):
        _run_cli(
            tmp_path,
            labels,
            prior_path=incomplete_prior,
            outer="fixed",
            split_json=split_json,
        )


def test_fixed_split_rejects_unlabeled_validation(tmp_path: Path) -> None:
    """Only train may contain a context without GeneEffect labels."""
    labels = _labels_no_signal(_BASE_LINES, _BASE_GENES)
    split_json = _write_split_json(
        tmp_path / "split.json", _BASE_LINES[:5], ["UNKNOWN_VAL"], [_BASE_LINES[5]]
    )
    with pytest.raises(ValueError, match="split 'val' has unknown model_id"):
        _run_cli(tmp_path, labels, outer="fixed", split_json=split_json)


def test_fixed_split_rejects_undeclared_unknown_train(tmp_path: Path) -> None:
    labels = _labels_no_signal(_BASE_LINES, _BASE_GENES)
    split_json = _write_split_json(
        tmp_path / "split.json",
        [*_BASE_LINES[:5], "TYPO"],
        [_BASE_LINES[5]],
        [],
    )
    with pytest.raises(ValueError, match="does not match declared unlabeled_train"):
        _run_cli(tmp_path, labels, outer="fixed", split_json=split_json)


# --- No forbidden provenance/exposure apparatus in the plain JSON output --


def _flatten_keys(payload: object) -> list[str]:
    """Recursively collect every dict key in ``payload``."""
    keys: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.append(str(key))
            keys.extend(_flatten_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            keys.extend(_flatten_keys(item))
    return keys


def test_summary_json_has_no_forbidden_keys(tmp_path: Path) -> None:
    """summary.json must carry none of the removed provenance apparatus."""
    labels = _labels_no_signal(_BASE_LINES, _BASE_GENES)
    context_csv = _write_context_csv(
        tmp_path / "ctx.csv", _BASE_LINES, {"signal": _SIGNAL_BY_LINE}
    )
    out_dir = _run_cli(tmp_path, labels, context_paths={"ctx": context_csv})
    raw_text = (out_dir / "summary.json").read_text()
    summary = json.loads(raw_text)

    forbidden = ("sha256", "formal", "protocol_id", "exposure", "development_only")
    all_keys = " ".join(_flatten_keys(summary)).lower()
    lower_text = raw_text.lower()
    for token in forbidden:
        assert token not in all_keys, f"forbidden key fragment {token!r} in keys"
        assert token not in lower_text, f"forbidden token {token!r} in summary.json"
