"""P1-C fold comparison against the pre-registered keep predicate.

Reads Task 4's run layout (``<root>/runs/<label>/<fold>/``), applies the
predicate fixed by the design (``kept_a``/``kept_b``/``kept_c``) per fold, and
pools folds equally into a variant-level verdict. No model inference or
refitting happens here -- only re-derivation from existing exports.
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
    across folds so a fold's own condition count never dominates the pooled
    interval.
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


def _read_export(directory):
    """Return ``(data, status)``; ``data`` is ``None`` for an absent/incomplete
    optional arm. A present but malformed export (completed status, missing
    conditions/cross-context) raises."""
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


def fold_summary(run_dir):
    """One row describing a single ``<root>/runs/<label>/<fold>/`` run.

    Assumes ``training.json`` exists; the caller (``summarize``) is
    responsible for skipping labels/folds that never trained at all.
    """
    run_dir = Path(run_dir)
    label = run_dir.parent.name
    fold = run_dir.name
    training = json.loads((run_dir / "training.json").read_text())
    external_anchor = FOLDS[fold]
    sources, expected_external = fold_membership(fold)
    if expected_external != external_anchor:
        raise ValueError(f"fold membership mismatch for {fold!r}")

    row = {
        "label": label,
        "variant": training.get("variant", _label_variant(label)),
        "lr": training.get("lr"),
        "fold": fold,
        "external": external_anchor,
        "status": training["status"],
    }
    if training["status"] != "completed":
        row["kept"] = None
        return row
    if training["fold"] != fold:
        raise ValueError(
            f"training.json fold {training['fold']!r} disagrees with directory {fold!r}"
        )

    internal, internal_status = _read_export(run_dir / "evaluation" / "internal")
    external, external_status = _read_export(run_dir / "evaluation" / "external")
    if internal is None or external is None:
        reasons = []
        if internal is None:
            reasons.append(f"internal export {internal_status}")
        if external is None:
            reasons.append(f"external export {external_status}")
        row["status"] = "; ".join(reasons)
        row["kept"] = None
        return row

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
    anchors_ci_below = 0
    for anchor in sources:
        model_rows = internal_val[
            (internal_val.model_id == anchor) & (internal_val.method == "model")
        ]
        no_change_rows = internal_val[
            (internal_val.model_id == anchor) & (internal_val.method == "no_change")
        ]
        internal_ratio[anchor] = float(
            model_rows.response_loss.mean() / no_change_rows.response_loss.mean()
        )
        interval = paired_difference(model_rows, no_change_rows, "response_loss")
        if interval["ci_high"] is not None and interval["ci_high"] < 0:
            anchors_ci_below += 1
    internal_ratio_equal = float(np.mean(list(internal_ratio.values())))

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
    external_interval = paired_difference(
        external_model, external_no_change, "response_loss"
    )

    identity_rows = external_model
    wrong = identity_rows.copy()
    wrong["response_loss"] = wrong["wrong_response_loss"]
    identity_interval = paired_difference(wrong, identity_rows, "response_loss")

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


def _native_export_rows(root, out_dir):
    """Descriptive-only rows for ``N-native*`` exports (not part of the predicate)."""
    del out_dir
    native_root = Path(root) / "runs" / "N-native"
    if not native_root.is_dir():
        return []
    rows = []
    for fold_dir in sorted(native_root.iterdir()):
        if not fold_dir.is_dir() or fold_dir.name not in FOLDS:
            continue
        evaluation_dir = fold_dir / "evaluation"
        if not evaluation_dir.is_dir():
            continue
        for export_dir in sorted(evaluation_dir.iterdir()):
            if not export_dir.is_dir():
                continue
            row = {
                "label": "N-native",
                "variant": "N-native",
                "fold": fold_dir.name,
                "export": export_dir.name,
                "kept": None,
            }
            for side, column in (
                ("internal", "internal_ratio_equal"),
                ("external", "external_ratio"),
            ):
                data, status = _read_export(export_dir / side)
                row[f"{side}_status"] = status
                if data is None:
                    row[column] = None
                    continue
                conditions = data["conditions"]
                if "method" not in conditions or "panel" not in conditions:
                    row[column] = None
                    continue
                subset = conditions[conditions.panel == "all"]
                model_mean = subset[subset.method == "model"].response_loss.mean()
                no_change_mean = subset[
                    subset.method == "no_change"
                ].response_loss.mean()
                row[column] = (
                    float(model_mean / no_change_mean) if no_change_mean else None
                )
            rows.append(row)
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
        "\n\nSingle training seed 0. Jurkat was observed before this design was "
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
    summary_rows.extend(_native_export_rows(root, out_dir))
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(out_dir / "summary.csv", index=False)

    variant_rows = []
    kept_by_label = {}
    canonical_labels = sorted(
        {row["label"] for row in rows if "-lr" not in row["label"]}
    )
    for label in canonical_labels:
        label_rows = [row for row in rows if row["label"] == label]
        folds = sum(1 for row in label_rows if row["kept"] is not None)
        kept_folds = sum(1 for row in label_rows if row.get("kept"))
        frames = [
            row["_external_frames"]
            for row in label_rows
            if row.get("_external_frames") is not None
        ]
        pooled = (
            equal_fold_difference(frames, "response_loss")
            if frames
            else {"delta": None, "ci_low": None, "ci_high": None}
        )
        kept_all = bool(
            kept_folds == folds == len(FOLDS)
            and pooled["ci_high"] is not None
            and pooled["ci_high"] < 0
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
