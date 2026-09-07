"""Paired, single-seed P1-B summaries; genes are the resampling units."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.gene_splits import sha256_file


def paired_difference(left, right, metric, *, repeats=1000):
    keys = [k for k in ("model_id", "gene") if k in left and k in right]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("paired conditions must be unique")
    aligned = left[keys + [metric]].merge(
        right[keys + [metric]], on=keys, suffixes=("_a", "_b"), validate="one_to_one"
    )
    valid = aligned.dropna(subset=[metric + "_a", metric + "_b"])
    if valid.empty:
        return {
            "pairs": 0,
            "undefined": len(aligned),
            "delta": None,
            "relative_improvement_pct": None,
            "ci_low": None,
            "ci_high": None,
        }
    valid = valid.copy()
    valid["difference"] = valid[metric + "_a"] - valid[metric + "_b"]
    # When an effect gene spans multiple contexts, resample it synchronously.
    grouped = valid.groupby("gene")["difference"].agg(["sum", "count"]).sort_index()
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(0)
    boot = []
    for start in range(0, repeats, 32):
        indices = rng.integers(
            0, len(grouped), size=(min(32, repeats - start), len(grouped))
        )
        boot.extend((sums[indices].sum(1) / counts[indices].sum(1)).tolist())
    delta = float(valid.difference.mean())
    reference = float(valid[metric + "_b"].mean())
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "pairs": len(valid),
        "undefined": len(aligned) - len(valid),
        "unmatched_left": len(left) - len(aligned),
        "unmatched_right": len(right) - len(aligned),
        "delta": delta,
        "relative_improvement_pct": -100 * delta / reference
        if reference != 0
        and metric != "output_change_mse"
        and (
            metric.endswith("loss")
            or metric.endswith("mse")
            or metric == "energy_distance"
        )
        else None,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def matched_updates(frame):
    columns = [
        m for m in ("mean_delta_mse", "energy_distance", "response_loss") if m in frame
    ]
    left = frame[frame.arm == "B-unfreeze"][["step", *columns]]
    right = frame[frame.arm == "B-continue"][["step", *columns]]
    joined = left.merge(
        right, on="step", suffixes=("_unfreeze", "_continue"), validate="one_to_one"
    )
    for metric in columns:
        joined[metric + "_delta"] = (
            joined[metric + "_unfreeze"] - joined[metric + "_continue"]
        )
    return joined


def compare(prepared, runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, prepared = Path(runs), Path(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    frames, cross_frames = [], []
    current_checkpoints = {}
    for path in sorted((runs / "evaluation").glob("*/*/evaluation.json")):
        status = json.loads(path.read_text())
        if status["status"] == "unavailable":
            continue
        if status["status"] != "completed":
            raise ValueError(f"incomplete export: {path}")
        if status["bundle"] != manifest["bundle_sha256"]:
            raise ValueError("cannot compare exports from different bundles")
        state = status["state"]
        if state not in current_checkpoints:
            checkpoint = (
                prepared / f"{state}.pt"
                if state in ("B-init", "B-joint")
                else runs / state / "best.pt"
            )
            current_checkpoints[state] = sha256_file(checkpoint)
        if status["checkpoint_sha256"] != current_checkpoints[state]:
            raise ValueError(f"stale export checkpoint: {path}; rerun evaluate")
        frame = pd.read_parquet(path.parent / "conditions.parquet")
        frame["state"] = status["state"]
        frame["export"] = path.parent.name
        frames.append(frame)
        cross = pd.read_parquet(path.parent / "cross_context.parquet")
        cross["state"] = status["state"]
        cross["export"] = path.parent.name
        cross_frames.append(cross)
    if not frames:
        raise ValueError("no completed P1-B exports")
    frame = pd.concat(frames, ignore_index=True)
    directory = runs / "comparison"
    directory.mkdir(exist_ok=True)
    metrics = [
        "mean_delta_mse",
        "energy_distance",
        "response_loss",
        "without_own_response_loss",
        "identity_advantage",
        "output_change_mse",
    ]
    metrics = [m for m in metrics if m in frame]
    group = ["state", "export", "role", "model_id", "panel", "method"]
    frame.groupby(group)[metrics].agg(["mean", "count"]).to_csv(
        directory / "raw_summary.csv"
    )
    # Equal-anchor means remain separate from per-condition pooled means.
    frame.groupby(group)[metrics].mean().groupby(
        ["state", "export", "role", "panel", "method"]
    ).mean().to_csv(directory / "equal_anchor_summary.csv")
    comparisons = []
    identity_intervals = []
    arm_pairs = [
        ("B-joint", "B-init"),
        ("B-interface", "B-init"),
        ("B-unfreeze", "B-interface"),
        ("B-continue", "B-interface"),
        ("B-unfreeze", "B-continue"),
    ]
    for key, subset in frame.groupby(["export", "role", "model_id", "panel"]):
        metadata = dict(zip(["export", "role", "model_id", "panel"], key))
        for a, b in arm_pairs:
            left = subset[(subset.state == a) & (subset.method == "model")]
            right = subset[(subset.state == b) & (subset.method == "model")]
            if left.empty or right.empty:
                continue
            for metric in metrics:
                comparisons.append(
                    {
                        **metadata,
                        "arm": a,
                        "reference": b,
                        "metric": metric,
                        **paired_difference(left, right, metric),
                    }
                )
        for state, data in subset.groupby("state"):
            left = data[data.method == "model"]
            if key[1] != "train":
                for metric in (
                    "response_loss",
                    "mean_delta_mse",
                    "energy_distance",
                    "without_own_response_loss",
                ):
                    wrong_column = "wrong_" + metric
                    if (
                        wrong_column not in left
                        or left[wrong_column].notna().sum() == 0
                    ):
                        continue
                    wrong = left.copy()
                    wrong[metric] = wrong[wrong_column]
                    interval = paired_difference(wrong, left, metric)
                    interval["relative_improvement_pct"] = None
                    identity_intervals.append(
                        {
                            **metadata,
                            "state": state,
                            "metric": metric,
                            "contrast": "wrong_minus_correct",
                            **interval,
                        }
                    )
            for baseline in ("no_change", "global_mean", "perturbation_mean"):
                right = data[data.method == baseline]
                if right.empty:
                    continue
                for metric in (
                    "response_loss",
                    "mean_delta_mse",
                    "energy_distance",
                    "without_own_response_loss",
                ):
                    comparisons.append(
                        {
                            **metadata,
                            "arm": state,
                            "reference": baseline,
                            "metric": metric,
                            **paired_difference(left, right, metric),
                        }
                    )
    pd.DataFrame(comparisons).to_csv(directory / "paired_differences.csv", index=False)
    pd.DataFrame(identity_intervals).to_csv(
        directory / "identity_intervals.csv", index=False
    )
    cross = pd.concat(cross_frames, ignore_index=True)
    if not cross.empty:
        cross.groupby(["state", "export", "method", "anchor_a", "anchor_b"])[
            ["effect_difference_mse", "effect_difference_pearson"]
        ].agg(["mean", "count", lambda s: s.isna().sum()]).to_csv(
            directory / "cross_context_summary.csv"
        )
        cross_differences = []
        for key, subset in cross.groupby(["export", "anchor_a", "anchor_b"]):
            metadata = dict(zip(["export", "anchor_a", "anchor_b"], key))
            candidates = []
            for a, b in arm_pairs:
                candidates.append(
                    (
                        a,
                        b,
                        subset[(subset.state == a) & (subset.method == "model")],
                        subset[(subset.state == b) & (subset.method == "model")],
                    )
                )
            for state, data in subset.groupby("state"):
                for baseline in ("no_change", "global_mean", "perturbation_mean"):
                    candidates.append(
                        (
                            state,
                            baseline,
                            data[data.method == "model"],
                            data[data.method == baseline],
                        )
                    )
            for a, b, left, right in candidates:
                if left.empty or right.empty:
                    continue
                for metric in ("effect_difference_mse", "effect_difference_pearson"):
                    cross_differences.append(
                        {
                            **metadata,
                            "arm": a,
                            "reference": b,
                            "metric": metric,
                            **paired_difference(left, right, metric),
                        }
                    )
        pd.DataFrame(cross_differences).to_csv(
            directory / "cross_context_differences.csv", index=False
        )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    common_rows = []
    for arm in ("B-interface", "B-continue", "B-unfreeze"):
        path = runs / arm / "history.json"
        if path.exists():
            history = json.loads(path.read_text())
            ax = axes[0 if arm == "B-interface" else 1]
            ax.plot(
                [r["step"] for r in history],
                [r["val"]["response_loss"] for r in history],
                label=arm,
            )
            common_rows.extend(
                {"arm": arm, "step": r["step"], "epoch": r["epoch"], **r["val"]}
                for r in history
            )
    for ax, title in zip(
        axes, ("Stage 1: interface", "Stage 2: continuation vs unfreeze")
    ):
        ax.set(
            title=title,
            xlabel="Stage updates",
            ylabel="Source validation response loss",
        )
        if ax.lines:
            ax.legend()
    if common_rows:
        histories = pd.DataFrame(common_rows)
        histories.to_csv(directory / "epoch_metrics.csv", index=False)
        matched_updates(histories).to_csv(directory / "common_updates.csv", index=False)
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(fig)
    headline = frame[
        (frame.panel == "all")
        & (frame.method == "model")
        & ((frame.export == "internal") | (frame.role == "external"))
    ]
    headline = (
        headline.groupby(["state", "role", "model_id"])[
            ["mean_delta_mse", "energy_distance", "response_loss"]
        ]
        .mean()
        .groupby(["state", "role"])
        .mean()
    )
    table = ["| State | Role | MSE | Energy | Total |", "|---|---|---:|---:|---:|"]
    for (state, role), values in headline.iterrows():
        table.append(
            f"| {state} | {role} | {values['mean_delta_mse']:.7g} | "
            f"{values['energy_distance']:.7g} | {values['response_loss']:.7g} |"
        )
    (directory / "analysis.md").write_text(
        "# P1-B response diagnostics\n\n"
        + "\n".join(table)
        + "\n\n"
        + "Raw metrics: raw_summary.csv; equal-anchor means: equal_anchor_summary.csv. "
        "Paired differences: paired_differences.csv; positive identity advantage "
        "means correct identity predicts better. Its per-state intervals against "
        "zero are in identity_intervals.csv. Cross-context summaries and "
        "paired differences are exported separately. Curves and common updates "
        "describe actual early-stopping budgets.\n\n"
        "Single training seed 0; intervals use 1,000 paired perturbation resamples "
        "conditional on selected checkpoints. They do not estimate new-anchor or "
        "initialization variability. Jurkat is adaptation-held-out only; ST/Tx1 "
        "pretraining exposure remains unresolved. Native preprocessing is "
        "unavailable, so no native predictions are reported. A response "
        "improvement is not GeneEffect evidence.\n"
    )
    return directory
