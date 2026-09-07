"""P1-B condition metrics, training-only bag baselines and paired diagnostics."""

import numpy as np
import pandas as pd
import torch
from src.model.response import energy_distance


def bag_terms(predicted, observed, coordinates):
    p, y = predicted.float()[:, coordinates], observed.float()[:, coordinates]
    mse = (p.mean(0) - y.mean(0)).square().mean()
    energy = energy_distance(p, y)
    if not torch.isfinite(mse + energy):
        raise ValueError("nonfinite response loss")
    return mse, energy


def score_bag(predicted, observed, basal, coordinates, *, own=None):
    mse, energy = bag_terms(predicted, observed, coordinates)
    without = [i for i in coordinates if i != own]
    om, oe = (
        bag_terms(predicted, observed, without)
        if without
        else (torch.tensor(float("nan")),) * 2
    )
    return {
        "mean_delta_mse": float(mse),
        "energy_distance": float(energy),
        "response_loss": float(mse + energy),
        "without_own_mean_delta_mse": float(om),
        "without_own_energy_distance": float(oe),
        "without_own_response_loss": float(om + oe),
        "own_removed": own in coordinates,
    }


def aggregate(frame):
    if frame.empty or frame.duplicated(["model_id", "gene"]).any():
        raise ValueError("nonempty unique response conditions required")
    columns = ["mean_delta_mse", "energy_distance", "response_loss"]
    if not np.isfinite(frame[columns]).all().all():
        raise ValueError("nonfinite response scores")
    return {
        k: float(v) for k, v in frame.groupby("model_id")[columns].mean().mean().items()
    }


def predictions(model, batch, device, genes=None):
    from contextlib import nullcontext
    from src.model.response import predict_bags

    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.device(device).type == "cuda"
        else nullcontext()
    )
    with context:
        return predict_bags(
            model, batch.controls_tx1, batch.genes if genes is None else genes, seed=0
        )


@torch.no_grad()
def evaluate_rows(model, view, indices, *, device, batch_size=32):
    rows = []
    model.eval()
    coordinates = view.bundle["coordinates"]
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch = view.batch(selected, device)
        outputs = predictions(model, batch, device)
        for i, p, y, b in zip(selected, outputs, batch.observed_hvg, batch.control_hvg):
            anchor, gene = view.bundle["keys"][i]
            rows.append(
                {"model_id": anchor, "gene": gene, **score_bag(p, y, b, coordinates)}
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def export_predictions(model, view, roles, *, device, directory, batch_size=32):
    """Fixed-checkpoint diagnostics. Only condition scores and mean effects persist."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    bundle, rows, effects = view.bundle, [], {}
    coordinates = bundle["coordinates"]
    own_indices = {g: i for i, g in enumerate(bundle["hvg_order"])}
    selected_indices = [i for role in roles for i in bundle["splits"][role]]
    correct = {}
    for start in range(0, len(selected_indices), batch_size):
        selected = selected_indices[start : start + batch_size]
        batch = view.batch(selected, device)
        outputs = predictions(model, batch, device)
        for i, p, y, basal in zip(
            selected, outputs, batch.observed_hvg, batch.control_hvg
        ):
            anchor, gene = bundle["keys"][i]
            candidates = {
                "model": p,
                "no_change": basal,
                "global_mean": basal
                + torch.as_tensor(bundle["baselines"]["global"], device=device),
            }
            if gene in bundle["baselines"]["genes"]:
                candidates["perturbation_mean"] = basal + torch.as_tensor(
                    bundle["baselines"]["genes"][gene], device=device
                )
            for method, bag in candidates.items():
                score = score_bag(bag, y, basal, coordinates, own=own_indices.get(gene))
                correct[i, method] = score
                effects[method, anchor, gene] = (
                    (bag.float().mean(0) - basal.mean(0))[coordinates].cpu().numpy(),
                    (y.mean(0) - basal.mean(0))[coordinates].cpu().numpy(),
                )
    wrong_cache = {}
    for panel_key, panel in bundle["panels"].items():
        role, anchor, panel_name = panel_key.split("/")
        if role not in roles:
            continue
        panel_identity = tuple(panel["indices"])
        cached_wrong = wrong_cache.get(panel_identity)
        wrong = (
            cached_wrong
            if cached_wrong is not None
            else {i: [] for i in panel["indices"]}
        )
        mappings = (
            panel["derangements"] if role != "train" and cached_wrong is None else []
        )
        for mapping in mappings:
            for start in range(0, len(panel["indices"]), batch_size):
                selected = panel["indices"][start : start + batch_size]
                batch = view.batch(selected, device)
                outputs = predictions(
                    model, batch, device, tuple(mapping[g] for g in batch.genes)
                )
                for i, p, y, basal in zip(
                    selected, outputs, batch.observed_hvg, batch.control_hvg
                ):
                    gene = bundle["keys"][i][1]
                    score = score_bag(
                        p, y, basal, coordinates, own=own_indices.get(gene)
                    )
                    effect = (
                        (p.float().mean(0) - basal.mean(0))[coordinates].cpu().numpy()
                    )
                    score["output_change_mse"] = float(
                        np.mean((effect - effects["model", anchor, gene][0]) ** 2)
                    )
                    wrong[i].append(score)
        wrong_cache[panel_identity] = wrong
        for i in panel["indices"]:
            gene = bundle["keys"][i][1]
            for method in ("model", "no_change", "global_mean", "perturbation_mean"):
                if (i, method) not in correct:
                    continue
                row = {
                    "role": role,
                    "model_id": anchor,
                    "gene": gene,
                    "panel": panel_name,
                    "method": method,
                    "joint_condition_holdout": bundle["joint_holdout"][i],
                    **correct[i, method],
                }
                if method == "model" and wrong[i]:
                    for metric in (
                        "response_loss",
                        "mean_delta_mse",
                        "energy_distance",
                        "without_own_response_loss",
                    ):
                        value = float(np.mean([r[metric] for r in wrong[i]]))
                        row[f"wrong_{metric}"] = value
                        row[f"identity_{metric}_advantage"] = value - row[metric]
                    row["identity_advantage"] = row["identity_response_loss_advantage"]
                    row["output_change_mse"] = float(
                        np.mean([r["output_change_mse"] for r in wrong[i]])
                    )
                    row["derangements"] = len(wrong[i])
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(directory / "conditions.parquet", index=False)
    coverage = []
    for panel_key, panel in bundle["panels"].items():
        role, anchor, name = panel_key.split("/")
        if role not in roles:
            continue
        for method in ("model", "no_change", "global_mean", "perturbation_mean"):
            available = sum((i, method) in correct for i in panel["indices"])
            coverage.append(
                {
                    "role": role,
                    "model_id": anchor,
                    "panel": name,
                    "method": method,
                    "expected": len(panel["indices"]),
                    "scored": available,
                    "missing": len(panel["indices"]) - available,
                }
            )
    pd.DataFrame(coverage).to_csv(directory / "coverage.csv", index=False)
    effect_keys = list(effects)
    np.savez_compressed(
        directory / "effects.npz",
        keys=np.asarray(effect_keys),
        predicted=np.stack([effects[k][0] for k in effect_keys]),
        observed=np.stack([effects[k][1] for k in effect_keys]),
    )
    cross_context(frame, effects, bundle).to_parquet(
        directory / "cross_context.parquet", index=False
    )
    return frame, effects


def cross_context(frame, effects, bundle):
    from itertools import combinations

    rows = []
    allowed = {tuple(bundle["keys"][i]) for i in bundle["splits"]["val"]}
    seen = {bundle["keys"][i][1] for i in bundle["splits"]["train"]}
    allowed.update(
        tuple(bundle["keys"][i])
        for i in bundle["splits"]["external"]
        if bundle["keys"][i][1] in seen
    )
    for method in frame.method.unique():
        for a, b in combinations([*bundle["anchors"], bundle["external"]], 2):
            genes = {g for anchor, g in allowed if anchor == a} & {
                g for anchor, g in allowed if anchor == b
            }
            for gene in sorted(genes):
                if (method, a, gene) not in effects or (method, b, gene) not in effects:
                    continue
                pa, ya = effects[method, a, gene]
                pb, yb = effects[method, b, gene]
                p, y = pa - pb, ya - yb
                correlation = (
                    float(np.corrcoef(p, y)[0, 1])
                    if np.std(p) > 0 and np.std(y) > 0
                    else np.nan
                )
                rows.append(
                    {
                        "method": method,
                        "anchor_a": a,
                        "anchor_b": b,
                        "gene": gene,
                        "effect_difference_mse": float(np.mean((p - y) ** 2)),
                        "effect_difference_pearson": correlation,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "anchor_a",
            "anchor_b",
            "gene",
            "effect_difference_mse",
            "effect_difference_pearson",
        ],
    )
