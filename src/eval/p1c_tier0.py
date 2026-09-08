"""Tier 0 analysis of existing P1-B exports: no inference, no fitting."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data.gene_splits import sha256_file
from src.eval.p1b import cross_context


def bias_decomposition(predicted, observed):
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(
        observed, dtype=np.float64
    )
    if error.ndim != 2 or error.shape[0] == 0:
        raise ValueError("effects must be a nonempty [conditions, coordinates] array")
    shared = error.mean(axis=0)
    total = float(np.mean(error**2))
    shared_bias = float(np.mean(shared**2))
    specific = float(np.mean((error - shared) ** 2))
    return {
        "total_mse": total,
        "shared_bias": shared_bias,
        "condition_specific": specific,
        "shared_bias_fraction": shared_bias / total if total > 0 else float("nan"),
        "n": int(error.shape[0]),
    }


def decompose_effects(effects_npz_path, method="model"):
    with np.load(effects_npz_path) as data:
        keys, predicted, observed = data["keys"], data["predicted"], data["observed"]
    mask = keys[:, 0] == method
    rows = []
    for anchor in sorted(set(keys[mask, 1])):
        anchor_mask = mask & (keys[:, 1] == anchor)
        decomposition = bias_decomposition(
            predicted[anchor_mask], observed[anchor_mask]
        )
        rows.append({"method": method, "model_id": anchor, **decomposition})
    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "model_id",
            "n",
            "total_mse",
            "shared_bias",
            "condition_specific",
            "shared_bias_fraction",
        ],
    )


def matched_coverage(conditions):
    rows = []
    for (role, model_id, panel), group in conditions.groupby(
        ["role", "model_id", "panel"]
    ):
        genes = set(group.gene)
        covered = set(group.loc[group.method == "perturbation_mean", "gene"])
        perturbation_mean_loss = group.loc[
            (group.method == "perturbation_mean") & group.gene.isin(covered),
            "response_loss",
        ].mean()
        for reference in ("no_change", "global_mean"):
            reference_rows = group[group.method == reference]
            on_covered = reference_rows[reference_rows.gene.isin(covered)]
            rows.append(
                {
                    "role": role,
                    "model_id": model_id,
                    "panel": panel,
                    "reference": reference,
                    "covered": len(covered),
                    "total": len(genes),
                    "reference_loss_on_covered": on_covered.response_loss.mean(),
                    "reference_loss_all": reference_rows.response_loss.mean(),
                    "perturbation_mean_loss": perturbation_mean_loss,
                }
            )
    return pd.DataFrame(rows)


def anchor_audit(cache_metadata, controls, missing_genes):
    rows = []
    for model_id, group in cache_metadata.groupby("model_id"):
        cells = group["n_cells"]
        rows.append(
            {
                "model_id": model_id,
                "conditions": len(group),
                "cells_median": float(cells.median()),
                "cells_p10": float(cells.quantile(0.1)),
                "cells_p90": float(cells.quantile(0.9)),
                "control_cells": int(controls[model_id]["hvg"].shape[0]),
                "zero_filled_coordinates": len(missing_genes[model_id]),
            }
        )
    return pd.DataFrame(rows)


def _effects_from_npz(effects_npz_path):
    with np.load(effects_npz_path) as data:
        keys, predicted, observed = data["keys"], data["predicted"], data["observed"]
    return {tuple(k): (p, o) for k, p, o in zip(keys, predicted, observed)}


def run_tier0(p1b_runs, p1b_prepared, out_dir):
    p1b_runs, p1b_prepared, out_dir = Path(p1b_runs), Path(p1b_prepared), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((p1b_prepared / "manifest.json").read_text())
    if sha256_file(p1b_prepared / "bundle.pt") != manifest["bundle_sha256"]:
        raise ValueError("prepared bundle identity changed")
    bundle = torch.load(
        p1b_prepared / "bundle.pt", map_location="cpu", weights_only=False
    )
    cache_metadata = pd.read_parquet(
        Path(bundle["response_cache"]) / "response_targets" / "metadata.parquet"
    )
    methods = ("model", "no_change", "global_mean", "perturbation_mean")
    bias_rows, coverage_rows, cross_rows = [], [], []
    for path in sorted((p1b_runs / "evaluation").glob("*/*/evaluation.json")):
        status = json.loads(path.read_text())
        if status["status"] == "unavailable":
            continue
        if status["status"] != "completed":
            raise ValueError(f"incomplete export: {path}")
        if status["bundle"] != manifest["bundle_sha256"]:
            raise ValueError("cannot compare exports from different bundles")
        state, export, directory = status["state"], path.parent.name, path.parent
        conditions = pd.read_parquet(directory / "conditions.parquet")
        for method in methods:
            decomposed = decompose_effects(directory / "effects.npz", method=method)
            decomposed["state"], decomposed["export"] = state, export
            bias_rows.append(decomposed)
        coverage = matched_coverage(conditions)
        coverage["state"], coverage["export"] = state, export
        coverage_rows.append(coverage)
        effects = _effects_from_npz(directory / "effects.npz")
        cross = cross_context(conditions, effects, bundle)
        cross["state"], cross["export"] = state, export
        cross_rows.append(cross)

    bias = (
        pd.concat(bias_rows, ignore_index=True)
        if bias_rows
        else pd.DataFrame(
            columns=[
                "method",
                "model_id",
                "n",
                "total_mse",
                "shared_bias",
                "condition_specific",
                "shared_bias_fraction",
                "state",
                "export",
            ]
        )
    )
    bias.to_csv(out_dir / "bias_decomposition.csv", index=False)

    coverage_all = (
        pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame()
    )
    coverage_all.to_csv(out_dir / "matched_coverage.csv", index=False)

    audit = anchor_audit(
        cache_metadata, bundle["controls"], manifest["source_missing_genes"]
    )
    audit.to_csv(out_dir / "anchor_audit.csv", index=False)

    cross_all = (
        pd.concat(cross_rows, ignore_index=True)
        if cross_rows
        else pd.DataFrame(
            columns=[
                "method",
                "anchor_a",
                "anchor_b",
                "gene",
                "effect_difference_mse",
                "effect_difference_pearson",
                "state",
                "export",
            ]
        )
    )
    if not cross_all.empty:
        cross_all.groupby(["state", "export", "method", "anchor_a", "anchor_b"])[
            ["effect_difference_mse", "effect_difference_pearson"]
        ].agg(["mean", "count", lambda s: s.isna().sum()]).to_csv(
            out_dir / "cross_context_guarded.csv"
        )
    else:
        pd.DataFrame(
            columns=["state", "export", "method", "anchor_a", "anchor_b"]
        ).to_csv(out_dir / "cross_context_guarded.csv", index=False)

    external = bundle["external"]
    jurkat_bias = bias[
        (bias.export == "external")
        & (bias.model_id == external)
        & (bias.method == "model")
    ]
    adapted_states = [
        s for s in jurkat_bias.state.unique() if s not in ("B-init", "B-joint")
    ]
    shared_bias_lines = [
        f"- {row.state}: shared_bias_fraction={row.shared_bias_fraction:.4g}"
        for row in jurkat_bias[jurkat_bias.state.isin(adapted_states)].itertuples()
    ]
    negative_transfer_rows = coverage_all[
        (coverage_all.covered > 0)
        & (coverage_all.reference == "no_change")
        & (coverage_all.perturbation_mean_loss > coverage_all.reference_loss_on_covered)
    ]
    negative_transfer_lines = [
        f"- {row.state}/{row.export} {row.model_id} ({row.panel}): "
        f"perturbation_mean_loss={row.perturbation_mean_loss:.4g} > "
        f"no_change_loss_on_covered={row.reference_loss_on_covered:.4g}"
        for row in negative_transfer_rows.itertuples()
    ]
    undefined_lines = []
    if not cross_all.empty:
        undefined_counts = (
            cross_all[cross_all.effect_difference_pearson.isna()]
            .groupby("method")
            .size()
        )
        undefined_lines = [
            f"- {method}: {count} undefined pairs"
            for method, count in undefined_counts.items()
        ]
    (out_dir / "tier0.md").write_text(
        "# P1-C Tier 0 analysis\n\n"
        "## Shared-bias fraction, adapted states, Jurkat (external)\n\n"
        + ("\n".join(shared_bias_lines) if shared_bias_lines else "None available.")
        + "\n\n"
        "## Matched-coverage negative transfer (perturbation_mean worse than "
        "no_change on covered genes)\n\n"
        + (
            "\n".join(negative_transfer_lines)
            if negative_transfer_lines
            else "None found."
        )
        + "\n\n"
        "## References with undefined cross-context Pearson (guarded)\n\n"
        + ("\n".join(undefined_lines) if undefined_lines else "None found.")
        + "\n\n"
        "Tier 0 re-derives statistics from existing P1-B exports only; no model "
        "inference or refitting was run to produce this analysis.\n"
    )
    return out_dir
