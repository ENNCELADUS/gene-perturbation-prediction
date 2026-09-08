"""P1-C fold comparison against the pre-registered keep predicate.

Reads Task 4's run layout (``<root>/runs/<label>/<fold>/``) and applies the
predicate fixed by the design. Legs (a) internal anchors and (c) identity
advantage are per-fold requirements; leg (b), the held-out-context loss ratio,
is pooled equally across the four folds, so the variant-level verdict is
``kept_all`` in ``variants.csv``. The per-fold ``kept``/``kept_b`` in
``summary.csv`` are reported diagnostics, not the verdict. No model inference
or refitting happens here -- only re-derivation from existing exports.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.p1c import FOLDS, fold_membership
from src.eval.p1b_comparison import paired_difference
from src.eval.p1c_tier0 import decompose_effects


def _label_variant(label):
    return label.split("-lr")[0]


def equal_fold_difference(frames, metric, *, repeats=1000):
    """Pool per-fold paired differences with equal fold weight.

    ``frames`` is a list of ``(left, right)`` DataFrame pairs, one per fold,
    each carrying ``gene`` and ``metric``. Genes are resampled from the union
    across folds (synchronously -- one draw of the union feeds every fold's
    own per-gene sums/counts for that replicate) so a fold's own condition
    count never dominates the pooled interval.
    """
    grouped_by_fold = []
    for left, right in frames:
        aligned = left[["gene", metric]].merge(
            right[["gene", metric]],
            on="gene",
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        valid = aligned.dropna(subset=[metric + "_a", metric + "_b"]).copy()
        if valid.empty:
            continue
        valid["difference"] = valid[metric + "_a"] - valid[metric + "_b"]
        grouped_by_fold.append(
            valid.groupby("gene")["difference"].agg(["sum", "count"])
        )

    if not grouped_by_fold:
        return {"delta": None, "ci_low": None, "ci_high": None, "folds": 0, "pairs": 0}

    fold_means = [float(g["sum"].sum() / g["count"].sum()) for g in grouped_by_fold]
    delta = float(np.mean(fold_means))
    pairs = int(sum(int(g["count"].sum()) for g in grouped_by_fold))

    union = sorted(set().union(*(g.index for g in grouped_by_fold)))
    union_index = {gene: i for i, gene in enumerate(union)}
    n = len(union)
    fold_sums, fold_counts = [], []
    for grouped in grouped_by_fold:
        sums = np.zeros(n)
        counts = np.zeros(n)
        for gene, row in grouped.iterrows():
            idx = union_index[gene]
            sums[idx] = row["sum"]
            counts[idx] = row["count"]
        fold_sums.append(sums)
        fold_counts.append(counts)

    rng = np.random.default_rng(0)
    indices = rng.integers(0, n, size=(repeats, n))
    fold_values = []
    for sums, counts in zip(fold_sums, fold_counts):
        s = sums[indices].sum(axis=1)
        c = counts[indices].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(c > 0, s / c, np.nan)
        fold_values.append(values)
    boot = np.nanmean(np.stack(fold_values, axis=0), axis=0)
    boot = boot[np.isfinite(boot)]
    low, high = (
        (float("nan"), float("nan"))
        if boot.size == 0
        else np.quantile(boot, [0.025, 0.975])
    )
    return {
        "delta": delta,
        "ci_low": float(low),
        "ci_high": float(high),
        "folds": len(grouped_by_fold),
        "pairs": pairs,
    }


def equal_fold_ratio(frames, *, repeats=1000):
    """Pool per-fold loss *ratios* with equal fold weight.

    ``frames`` is a list of ``(model_rows, no_change_rows)`` DataFrame pairs,
    one per fold, each carrying ``gene`` and ``response_loss`` for the fold's
    held-out anchor. A replicate draws one sample of the union of genes and
    feeds it to every fold, so no fold's own condition count dominates; each
    fold contributes ``mean(model) / mean(no_change)`` on the resampled
    conditions, and the replicate's value is the equal-weight mean of those.

    This is the variant-level predicate quantity: the design's keep rule is
    stated as a ratio below 1, which a pooled *difference* below zero does not
    imply once anchors differ in scale.
    """
    metric = "response_loss"
    grouped_by_fold = []
    for model_rows, no_change_rows in frames:
        aligned = model_rows[["gene", metric]].merge(
            no_change_rows[["gene", metric]],
            on="gene",
            suffixes=("_model", "_no_change"),
            validate="one_to_one",
        )
        columns = [metric + "_model", metric + "_no_change"]
        valid = aligned.dropna(subset=columns)
        if valid.empty:
            continue
        grouped = valid.groupby("gene")[columns].sum()
        grouped["count"] = valid.groupby("gene")[columns[0]].count()
        grouped_by_fold.append(grouped)

    if not grouped_by_fold:
        return {"ratio": None, "ci_low": None, "ci_high": None, "folds": 0, "pairs": 0}

    model_column, no_change_column = metric + "_model", metric + "_no_change"
    fold_ratios = [
        float(g[model_column].sum() / g[no_change_column].sum())
        for g in grouped_by_fold
    ]
    ratio = float(np.mean(fold_ratios))
    pairs = int(sum(int(g["count"].sum()) for g in grouped_by_fold))

    union = sorted(set().union(*(g.index for g in grouped_by_fold)))
    union_index = {gene: i for i, gene in enumerate(union)}
    n = len(union)
    model_sums, no_change_sums = [], []
    for grouped in grouped_by_fold:
        model = np.zeros(n)
        no_change = np.zeros(n)
        for gene, row in grouped.iterrows():
            idx = union_index[gene]
            model[idx] = row[model_column]
            no_change[idx] = row[no_change_column]
        model_sums.append(model)
        no_change_sums.append(no_change)

    rng = np.random.default_rng(0)
    indices = rng.integers(0, n, size=(repeats, n))
    fold_values = []
    for model, no_change in zip(model_sums, no_change_sums):
        numerator = model[indices].sum(axis=1)
        denominator = no_change[indices].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            fold_values.append(
                np.where(denominator > 0, numerator / denominator, np.nan)
            )
    boot = np.nanmean(np.stack(fold_values, axis=0), axis=0)
    boot = boot[np.isfinite(boot)]
    low, high = (
        (float("nan"), float("nan"))
        if boot.size == 0
        else np.quantile(boot, [0.025, 0.975])
    )
    return {
        "ratio": ratio,
        "ci_low": float(low),
        "ci_high": float(high),
        "folds": len(grouped_by_fold),
        "pairs": pairs,
    }


def _read_export(directory):
    """Return ``(data, status)``; ``data`` is ``None`` for an absent/incomplete
    optional arm. A present but malformed export (completed status, missing
    conditions/cross-context/effects) raises."""
    directory = Path(directory)
    status_path = directory / "evaluation.json"
    if not status_path.exists():
        return None, "missing"
    status = json.loads(status_path.read_text())
    if status["status"] != "completed":
        return None, status["status"]
    conditions = pd.read_parquet(directory / "conditions.parquet")
    cross = pd.read_parquet(directory / "cross_context.parquet")
    effects = directory / "effects.npz"
    if not effects.exists():
        raise ValueError(f"malformed export: missing effects.npz in {directory}")
    return {"conditions": conditions, "cross": cross, "effects": effects}, "completed"


def _require_export(run_dir, side):
    """Like ``_read_export`` but raises: once training is completed, a
    missing, absent-``evaluation.json``, or non-completed export is a data
    problem, not an optional arm."""
    directory = run_dir / "evaluation" / side
    data, status = _read_export(directory)
    if data is None:
        raise ValueError(
            f"{run_dir}: {side} export missing or incomplete (status={status!r})"
        )
    return data


def _paired(left, right, metric, *, anchor, contrast):
    """``paired_difference`` guarded against an empty pairing silently
    evaluating a predicate leg as false."""
    result = paired_difference(left, right, metric)
    if result["pairs"] == 0:
        raise ValueError(f"no paired conditions for anchor {anchor!r} ({contrast})")
    return result


def fold_summary(run_dir):
    """One row describing a single ``<root>/runs/<label>/<fold>/`` run.

    Assumes ``training.json`` exists; the caller (``summarize``) is
    responsible for skipping labels/folds that never trained at all. Once
    training is ``completed``, both exports are required to exist and be
    ``completed`` -- their absence raises rather than degrading to
    ``kept=None``, which is reserved for training that never finished.

    The row's per-fold ``kept``/``kept_b`` are reported diagnostics; the
    variant-level verdict is ``summarize``'s pooled ratio (see
    :func:`equal_fold_ratio`).
    """
    run_dir = Path(run_dir)
    label = run_dir.parent.name
    fold = run_dir.name
    training = json.loads((run_dir / "training.json").read_text())
    sources, external_anchor = fold_membership(fold)

    row = {
        "label": label,
        "variant": training["variant"]
        if "variant" in training
        else _label_variant(label),
        "lr": training["lr"] if "lr" in training else None,
        "fold": fold,
        "external": external_anchor,
        "status": training["status"],
    }
    if training["status"] != "completed":
        row["kept"] = None
        return row
    missing_fields = [
        name for name in ("variant", "fold", "lr") if name not in training
    ]
    if missing_fields:
        # fit() writes status=completed before train_variant adds these; a run
        # killed in that window is an incomplete record, not a KeyError.
        row["status"] = "incomplete-record"
        row["missing_fields"] = ",".join(missing_fields)
        row["kept"] = None
        return row
    if training["fold"] != fold:
        raise ValueError(
            f"training.json fold {training['fold']!r} disagrees with directory {fold!r}"
        )

    internal = _require_export(run_dir, "internal")
    external = _require_export(run_dir, "external")

    row.update(
        best_epoch=training["best_epoch"],
        stopped_epoch=training["epoch"],
        best_loss=training["best_loss"],
    )

    internal_val = internal["conditions"]
    internal_val = internal_val[
        (internal_val.role == "val") & (internal_val.panel == "all")
    ]
    internal_ratio = {}
    no_change_means = []
    anchors_ci_below = 0
    for anchor in sources:
        model_rows = internal_val[
            (internal_val.model_id == anchor) & (internal_val.method == "model")
        ]
        no_change_rows = internal_val[
            (internal_val.model_id == anchor) & (internal_val.method == "no_change")
        ]
        no_change_means.append(float(no_change_rows.response_loss.mean()))
        internal_ratio[anchor] = float(
            model_rows.response_loss.mean() / no_change_rows.response_loss.mean()
        )
        interval = _paired(
            model_rows,
            no_change_rows,
            "response_loss",
            anchor=anchor,
            contrast="internal_vs_no_change",
        )
        if interval["ci_high"] is not None and interval["ci_high"] < 0:
            anchors_ci_below += 1
    internal_ratio_equal = float(np.mean(list(internal_ratio.values())))

    # Epoch 0 is the untrained checkpoint's own validation loss, recorded by
    # fit() before any update: what adaptation started from, on the same
    # conditions the internal ratio uses.
    history_path = run_dir / "history.json"
    epoch0_val_loss = None
    if history_path.exists():
        history = json.loads(history_path.read_text())
        if history:
            epoch0_val_loss = float(history[0]["val"]["response_loss"])
    no_change_equal = float(np.mean(no_change_means))
    epoch0_internal_ratio_equal = (
        None if epoch0_val_loss is None else epoch0_val_loss / no_change_equal
    )

    external_conditions = external["conditions"]
    external_ext = external_conditions[
        (external_conditions.role == "external")
        & (external_conditions.panel == "all")
        & (external_conditions.model_id == external_anchor)
    ]
    external_model = external_ext[external_ext.method == "model"]
    external_no_change = external_ext[external_ext.method == "no_change"]
    external_ratio = float(
        external_model.response_loss.mean() / external_no_change.response_loss.mean()
    )
    external_interval = _paired(
        external_model,
        external_no_change,
        "response_loss",
        anchor=external_anchor,
        contrast="external_vs_no_change",
    )

    identity_rows = external_model
    wrong = identity_rows.copy()
    wrong["response_loss"] = wrong["wrong_response_loss"]
    identity_interval = _paired(
        wrong,
        identity_rows,
        "response_loss",
        anchor=external_anchor,
        contrast="identity",
    )

    cross = external["cross"]
    cross_ext = cross[
        (cross.anchor_a == external_anchor) | (cross.anchor_b == external_anchor)
    ]
    key_columns = ["anchor_a", "anchor_b", "gene"]
    model_cross = cross_ext[cross_ext.method == "model"][
        key_columns + ["effect_difference_mse"]
    ]
    no_change_cross = cross_ext[cross_ext.method == "no_change"][
        key_columns + ["effect_difference_mse"]
    ]
    merged_cross = model_cross.merge(
        no_change_cross,
        on=key_columns,
        suffixes=("_model", "_no_change"),
        validate="one_to_one",
    )
    cross_mse_ratio = float(
        merged_cross.effect_difference_mse_model.mean()
        / merged_cross.effect_difference_mse_no_change.mean()
    )

    decomposed = decompose_effects(external["effects"], method="model")
    matching = decomposed[decomposed.model_id == external_anchor]
    if matching.empty:
        raise ValueError(
            f"no shared-bias decomposition for external anchor {external_anchor!r}"
        )
    shared_bias_fraction_external = float(matching.shared_bias_fraction.iloc[0])

    kept_a = anchors_ci_below >= 2
    kept_b = (
        external_interval["ci_high"] is not None and external_interval["ci_high"] < 0
    )
    kept_c = identity_interval["ci_low"] is not None and identity_interval["ci_low"] > 0
    kept = bool(kept_a and kept_b and kept_c)

    row.update(
        internal_ratio=internal_ratio,
        internal_ratio_equal=internal_ratio_equal,
        epoch0_val_loss=epoch0_val_loss,
        epoch0_internal_ratio_equal=epoch0_internal_ratio_equal,
        anchors_ci_below=anchors_ci_below,
        external_ratio=external_ratio,
        external_delta=external_interval["delta"],
        external_ci_low=external_interval["ci_low"],
        external_ci_high=external_interval["ci_high"],
        identity_advantage=identity_interval["delta"],
        identity_ci_low=identity_interval["ci_low"],
        identity_ci_high=identity_interval["ci_high"],
        cross_mse_ratio=cross_mse_ratio,
        shared_bias_fraction_external=shared_bias_fraction_external,
        kept_a=kept_a,
        kept_b=kept_b,
        kept_c=kept_c,
        kept=kept,
    )
    row["_external_frames"] = (
        external_model[["gene", "response_loss"]],
        external_no_change[["gene", "response_loss"]],
    )
    return row


def _flatten_summary_row(row):
    clean = {k: v for k, v in row.items() if not k.startswith("_")}
    internal_ratio = clean.pop("internal_ratio", None)
    if internal_ratio:
        for anchor, ratio in internal_ratio.items():
            clean[f"internal_ratio_{anchor}"] = ratio
    return clean


def _native_batch_index(export_name):
    if export_name == "N-native":
        return 0
    prefix = "N-native-b"
    if not export_name.startswith(prefix):
        raise ValueError(f"unrecognized native export name {export_name!r}")
    return int(export_name[len(prefix) :])


def _read_native_light_export(directory):
    """Batch-index != 0 native exports write only a raw ``summary.csv``
    (``evaluate_rows`` output plus ``role``) -- no ``conditions.parquet``, no
    ``method``/``panel`` columns, so no baseline is available there."""
    directory = Path(directory)
    status_path = directory / "evaluation.json"
    if not status_path.exists():
        return None, "missing"
    status = json.loads(status_path.read_text())
    if status["status"] != "completed":
        return None, status["status"]
    summary_path = directory / "summary.csv"
    if not summary_path.exists():
        raise ValueError(f"malformed export: missing summary.csv in {directory}")
    return pd.read_csv(summary_path), "completed"


NATIVE_PANEL = "native_all"


def _native_export_row(fold, export_name, export_dir, sources, external_anchor):
    """One descriptive row per ``N-native*`` export.

    ``restrict_to_native`` keeps only the ``native_common``/``native_all``
    panels, so there is no ``all`` panel to read here: every ratio is taken on
    ``native_all``, the same condition set the light (batch index != 0)
    summaries score.
    """
    row = {
        "label": "N-native",
        "variant": "N-native",
        "fold": fold,
        "export": export_name,
        "batch_index": _native_batch_index(export_name),
        "panel": NATIVE_PANEL,
        "kept": None,
    }
    if row["batch_index"] == 0:
        internal, internal_status = _read_export(export_dir / "internal")
        external, external_status = _read_export(export_dir / "external")
        row["internal_status"], row["external_status"] = (
            internal_status,
            external_status,
        )
        if internal is not None:
            internal_val = internal["conditions"]
            internal_val = internal_val[
                (internal_val.role == "val") & (internal_val.panel == NATIVE_PANEL)
            ]
            if internal_val.empty:
                raise ValueError(
                    f"{export_dir / 'internal'}: no val rows on panel {NATIVE_PANEL!r}"
                )
            ratios = []
            for anchor in sources:
                model_mean = internal_val[
                    (internal_val.model_id == anchor) & (internal_val.method == "model")
                ].response_loss.mean()
                no_change_mean = internal_val[
                    (internal_val.model_id == anchor)
                    & (internal_val.method == "no_change")
                ].response_loss.mean()
                ratios.append(model_mean / no_change_mean)
            row["internal_ratio_equal"] = float(np.mean(ratios)) if ratios else None
        else:
            row["internal_ratio_equal"] = None
        if external is not None:
            external_ext = external["conditions"]
            external_ext = external_ext[
                (external_ext.role == "external")
                & (external_ext.panel == NATIVE_PANEL)
                & (external_ext.model_id == external_anchor)
            ]
            if external_ext.empty:
                raise ValueError(
                    f"{export_dir / 'external'}: no external rows for anchor "
                    f"{external_anchor!r} on panel {NATIVE_PANEL!r}"
                )
            model_mean = external_ext[
                external_ext.method == "model"
            ].response_loss.mean()
            no_change_mean = external_ext[
                external_ext.method == "no_change"
            ].response_loss.mean()
            row["external_ratio"] = (
                float(model_mean / no_change_mean) if no_change_mean else None
            )
        else:
            row["external_ratio"] = None
        return row

    internal_frame, internal_status = _read_native_light_export(export_dir / "internal")
    external_frame, external_status = _read_native_light_export(export_dir / "external")
    row["internal_status"], row["external_status"] = internal_status, external_status
    if internal_frame is not None:
        internal_val = internal_frame[internal_frame.role == "val"]
        means = [
            internal_val[internal_val.model_id == anchor].response_loss.mean()
            for anchor in sources
        ]
        row["native_val_loss_equal"] = float(np.mean(means)) if means else None
    else:
        row["native_val_loss_equal"] = None
    if external_frame is not None:
        external_ext = external_frame[
            (external_frame.role == "external")
            & (external_frame.model_id == external_anchor)
        ]
        row["native_external_loss"] = (
            float(external_ext.response_loss.mean()) if not external_ext.empty else None
        )
    else:
        row["native_external_loss"] = None
    return row


def _native_export_rows(root):
    """Descriptive-only rows for ``N-native*`` exports (not part of the predicate)."""
    native_root = Path(root) / "runs" / "N-native"
    if not native_root.is_dir():
        return []
    rows = []
    for fold_dir in sorted(native_root.iterdir()):
        if not fold_dir.is_dir() or fold_dir.name not in FOLDS:
            continue
        fold = fold_dir.name
        sources, external_anchor = fold_membership(fold)
        evaluation_dir = fold_dir / "evaluation"
        if not evaluation_dir.is_dir():
            continue
        for export_dir in sorted(evaluation_dir.iterdir()):
            if not export_dir.is_dir():
                continue
            rows.append(
                _native_export_row(
                    fold, export_dir.name, export_dir, sources, external_anchor
                )
            )
    return rows


def _plot_learning_curves(root, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    runs_dir = root / "runs"
    fold_names = list(FOLDS)
    fig, axes = plt.subplots(
        1, len(fold_names), figsize=(4 * len(fold_names), 4), squeeze=False
    )
    axes = axes[0]
    for ax, fold in zip(axes, fold_names):
        if runs_dir.is_dir():
            for label_dir in sorted(runs_dir.iterdir()):
                if not label_dir.is_dir() or label_dir.name == "N-native":
                    continue
                history_path = label_dir / fold / "history.json"
                if not history_path.exists():
                    continue
                history = json.loads(history_path.read_text())
                steps = [entry["step"] for entry in history]
                losses = [entry["val"]["response_loss"] for entry in history]
                ax.plot(steps, losses, label=label_dir.name)
        ax.set(title=fold, xlabel="step", ylabel="val response_loss")
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


def _format_cell(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_analysis(variants_frame, out_dir):
    columns = list(variants_frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in variants_frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[c]) for c in columns) + " |")
    caveats = (
        "\n\nThe verdict is `kept_all`: legs (a) and (c) on every fold plus a "
        "pooled held-out ratio whose interval lies below 1. The per-fold `kept` "
        "and `kept_b` columns in `summary.csv` are reported diagnostics.\n"
        "\nSingle training seed 0. Jurkat was observed before this design was "
        "registered, so its fold is a diagnostic re-evaluation, not a held-out "
        "test. The four leave-one-anchor-out folds are related diagnostics on "
        "the same four lines, not independent contexts. ST/Tx1 pretraining "
        "exposure remains unresolved. Response results are not GeneEffect "
        "evidence.\n"
    )
    (out_dir / "analysis.md").write_text(
        "# P1-C fold comparison\n\n" + "\n".join(lines) + caveats
    )


def summarize(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = root / "runs"

    rows = []
    if runs_dir.is_dir():
        for label_dir in sorted(runs_dir.iterdir()):
            if not label_dir.is_dir() or label_dir.name == "N-native":
                continue
            for fold_dir in sorted(label_dir.iterdir()):
                if not fold_dir.is_dir() or fold_dir.name not in FOLDS:
                    continue
                if not (fold_dir / "training.json").exists():
                    continue
                rows.append(fold_summary(fold_dir))

    summary_rows = [_flatten_summary_row(row) for row in rows]
    summary_rows.extend(_native_export_rows(root))
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(out_dir / "summary.csv", index=False)

    variant_rows = []
    kept_by_label = {}
    canonical_labels = sorted(
        {row["label"] for row in rows if "-lr" not in row["label"]}
    )
    empty_pooled = {"ci_low": None, "ci_high": None, "folds": 0, "pairs": 0}
    for label in canonical_labels:
        label_rows = [row for row in rows if row["label"] == label]
        complete = [row for row in label_rows if row["kept"] is not None]
        folds = len(complete)
        kept_folds = sum(1 for row in complete if row["kept"])
        frames = [
            row["_external_frames"]
            for row in label_rows
            if "_external_frames" in row and row["_external_frames"] is not None
        ]
        pooled = (
            equal_fold_difference(frames, "response_loss")
            if frames
            else {**empty_pooled, "delta": None}
        )
        pooled_ratio = (
            equal_fold_ratio(frames) if frames else {**empty_pooled, "ratio": None}
        )
        # The verdict is the pooled held-out ratio, not a count of per-fold
        # verdicts: (a) and (c) must hold on every fold, but (b) is pooled
        # across folds so one fold's interval width cannot decide the variant.
        kept_all = bool(
            folds == len(FOLDS)
            and all(row["kept_a"] and row["kept_c"] for row in complete)
            and pooled_ratio["ci_high"] is not None
            and pooled_ratio["ci_high"] < 1
        )
        kept_by_label[label] = kept_all
        variant_rows.append(
            {
                "label": label,
                "folds": folds,
                "kept_folds": kept_folds,
                "pooled_external_delta": pooled["delta"],
                "pooled_external_ci_low": pooled["ci_low"],
                "pooled_external_ci_high": pooled["ci_high"],
                "pooled_ratio": pooled_ratio["ratio"],
                "pooled_ratio_ci_low": pooled_ratio["ci_low"],
                "pooled_ratio_ci_high": pooled_ratio["ci_high"],
                "kept_all": kept_all,
            }
        )
    variants_frame = pd.DataFrame(variant_rows)
    variants_frame.to_csv(out_dir / "variants.csv", index=False)

    kept_json = dict(kept_by_label)
    kept_json["V3_eligible"] = kept_by_label.get("V1", False) or kept_by_label.get(
        "V2", False
    )
    (out_dir / "kept.json").write_text(json.dumps(kept_json, indent=2) + "\n")

    _plot_learning_curves(root, out_dir)
    _write_analysis(variants_frame, out_dir)
    return out_dir
