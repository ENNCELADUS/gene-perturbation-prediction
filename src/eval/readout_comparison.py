"""Pair-aligned diagnostic differences and context-cluster uncertainty."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.eval.metrics import MIN_OBSERVATIONS

KEYS = ["model_id", "gene_symbol"]
CONTRASTS = (("A2", "A0"), ("A3", "A1"), ("A1", "A0"), ("A3", "A2"))


def align_predictions(left, right):
    """Require the same observed universe and targets; never join by array order."""
    aligned = []
    for frame in (left, right):
        if frame.duplicated(KEYS).any():
            raise ValueError("duplicate prediction keys; provide one method at a time")
        aligned.append(frame.set_index(KEYS).sort_index())
    left, right = aligned
    if not left.index.equals(right.index):
        raise ValueError("prediction row keys differ")
    for key in ("residual", "gene_effect"):
        if not np.allclose(left[key], right[key], rtol=0, atol=2e-7, equal_nan=False):
            raise ValueError("prediction targets differ")
    if not all(np.isfinite(frame["residual_prediction"]).all() for frame in aligned):
        raise ValueError("non-finite prediction on an observed pair")
    return left.reset_index(), right.reset_index()


def _gene_statistics(truth, predicted):
    valid = np.isfinite(truth)
    count = valid.sum(0)
    safe = np.maximum(count, 1)

    def centered(value):
        return np.where(valid, value - np.nansum(value, axis=0) / safe, 0.0)

    x, y = centered(truth), centered(predicted)
    xx, yy = (x * x).sum(0), (y * y).sum(0)
    # Match the production exact-constant policy before floating-point centering.
    for value, squared in ((truth, xx), (predicted, yy)):
        constant = np.where(valid, value, -np.inf).max(0) <= np.where(
            valid, value, np.inf
        ).min(0)
        squared[constant] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = (x * y).sum(0) / np.sqrt(xx * yy)
        ratio = np.sqrt(yy / xx)
    pearson[(count < MIN_OBSERVATIONS) | (xx == 0) | (yy == 0)] = np.nan
    ratio[(count < 2) | (xx == 0)] = np.nan
    rx = centered(rankdata(truth, axis=0, nan_policy="omit"))
    ry = centered(rankdata(predicted, axis=0, nan_policy="omit"))
    with np.errstate(divide="ignore", invalid="ignore"):
        spearman = (rx * ry).sum(0) / np.sqrt((rx * rx).sum(0) * (ry * ry).sum(0))
    spearman[(count < MIN_OBSERVATIONS) | (xx == 0) | (yy == 0)] = np.nan
    return {"pearson": pearson, "spearman": spearman, "sd_ratio": ratio}


def paired_diagnostic(left, right, variable_genes, *, groups, repeats=1000):
    """Right minus left; resample whole clusters, then recompute gene metrics."""
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    left, right = align_predictions(left, right)
    lines = sorted(left.model_id.unique())
    if set(lines) - groups.keys():
        raise ValueError("missing context cluster identities")
    clusters = sorted({groups[line] for line in lines})
    members = [
        np.array([i for i, line in enumerate(lines) if groups[line] == cluster])
        for cluster in clusters
    ]
    panel = list(variable_genes)

    def matrix(frame, column):
        return (
            frame.pivot(index="model_id", columns="gene_symbol", values=column)
            .reindex(index=lines, columns=panel)
            .to_numpy(float)
        )

    truth = matrix(left, "residual")
    predictions = [matrix(frame, "residual_prediction") for frame in (left, right)]
    losses = []
    counts = left.groupby("model_id").size().reindex(lines).to_numpy()
    for frame in (left, right):
        error = np.abs(frame.residual_prediction.to_numpy() - frame.residual.to_numpy())
        rows = frame.assign(loss=np.where(error < 1, 0.5 * error**2, error - 0.5))
        losses.append(rows.groupby("model_id").loss.sum().reindex(lines).to_numpy())

    def measure(indices):
        result = {
            "huber_delta": float(
                (losses[1][indices].sum() - losses[0][indices].sum())
                / counts[indices].sum()
            )
        }
        statistics = [
            _gene_statistics(truth[indices], prediction[indices])
            for prediction in predictions
        ]
        for name in ("pearson", "spearman", "sd_ratio"):
            a, b = (value[name] for value in statistics)
            common = np.isfinite(a) & np.isfinite(b)
            result[f"{name}_common_genes"] = int(common.sum())
            result[f"{name}_delta"] = (
                float((b[common] - a[common]).mean()) if common.any() else None
            )
        return result

    rng = np.random.default_rng(0)
    draws = [
        measure(
            np.concatenate(
                [members[i] for i in rng.integers(0, len(members), len(members))]
            )
        )
        for _ in range(repeats)
    ]
    intervals = {}
    for name in ("huber", "pearson", "spearman", "sd_ratio"):
        values = [
            row[f"{name}_delta"] for row in draws if row[f"{name}_delta"] is not None
        ]
        intervals[name] = {
            "interval_95": np.quantile(values, [0.025, 0.975]).tolist()
            if values
            else None,
            "defined_replicates": len(values),
        }
        if name != "huber":
            supports = [row[f"{name}_common_genes"] for row in draws]
            intervals[name]["common_gene_count_range"] = [min(supports), max(supports)]
    return {
        "point": measure(np.arange(len(lines))),
        "intervals": intervals,
        "clusters": len(clusters),
        "contexts": len(lines),
        "repeats": repeats,
        "seed": 0,
        "scope": "conditional on fixed checkpoint selection and head seed 0",
    }


def compare_runs(cache, root, destination, *, groups, repeats=1000, references=()):
    """Export four selected arms, paired gene changes and common-update curves."""
    root, destination = Path(root), Path(destination)
    predictions, details, summaries, curves = {}, {}, [], {}
    for arm in ("A0", "A1", "A2", "A3"):
        run = json.loads((root / arm / "run.json").read_text())
        if run["training"] != "completed" or run["evaluation"] != "completed":
            raise ValueError(f"{arm} is not complete")
        if run["cache_metadata"] != cache.metadata:
            raise ValueError("arms must share the same cache")
        for split in ("train", "val"):
            folder = root / arm / "evaluation" / "best" / split
            frame = pd.read_parquet(folder / "predictions.parquet")
            align_predictions(
                cache.labels(split).assign(residual_prediction=0.0), frame
            )
            summaries.append(
                {
                    "arm": arm,
                    "split": split,
                    "best_epoch": run["best_epoch"],
                    "best_step": run["best_step"],
                    "stopped_epoch": run["stopped_epoch"],
                    **json.loads((folder / "metrics.json").read_text()),
                }
            )
            if split == "val":
                predictions[arm] = frame
                details[arm] = pd.read_csv(folder / "per_gene.csv")
        curves[arm] = pd.read_json(root / arm / "metrics.jsonl", lines=True)
    for reference in references:
        align_predictions(predictions["A0"], pd.read_parquet(reference))
    destination.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summaries).to_csv(destination / "selected.csv", index=False)
    paired = {}
    for right, left in CONTRASTS:
        name = f"{right}-minus-{left}"
        paired[name] = paired_diagnostic(
            predictions[left],
            predictions[right],
            cache.variable_genes,
            groups=groups,
            repeats=repeats,
        )
        gene = details[right].merge(
            details[left],
            on="gene_symbol",
            suffixes=("_right", "_left"),
            validate="one_to_one",
        )
        for metric in ("pearson", "spearman", "sd_ratio", "rmse", "mae"):
            gene[f"{metric}_delta"] = gene[f"{metric}_right"] - gene[f"{metric}_left"]
        gene.to_csv(destination / f"{name}-per-gene.csv", index=False)
        common = curves[right].merge(
            curves[left],
            on=["epoch", "global_step"],
            suffixes=("_right", "_left"),
            validate="one_to_one",
        )
        common.to_csv(destination / f"{name}-common-updates.csv", index=False)
    (destination / "paired.json").write_text(
        json.dumps(paired, indent=2, allow_nan=False) + "\n"
    )
    return paired
