"""Aligned-row GeneEffect and response evaluation for validation and testing."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.data.batches import DependencyBatch
from src.data.datasets import make_evaluation_loaders
from src.data.prepared import PreparedInputs
from src.eval.metrics import _unit_pearson, _unit_spearman
from src.eval.response import aggregate_response, response_rows


@dataclass
class EvalResult:
    metrics: dict[str, float | int | None]
    predictions: pd.DataFrame
    per_line: pd.DataFrame
    per_gene: pd.DataFrame
    response: pd.DataFrame


def compose_metrics(
    geneeffect: Mapping[str, float | int | None],
    response: Mapping[str, float | int | None],
    *,
    response_weight: float,
    prefix: str,
) -> dict[str, float | int | None]:
    """Prefix scalars and add the held-out objectives without replay dilution."""
    if not math.isfinite(response_weight) or response_weight < 0:
        raise ValueError("response_weight must be finite and non-negative")
    loss = float(geneeffect["geneeffect_loss"])
    response_loss = float(response["response_loss"])
    total = loss + response_weight * response_loss
    if not all(math.isfinite(value) for value in (loss, response_loss, total)):
        raise ValueError("evaluation losses must be finite")
    metrics = {**geneeffect, **response, "total_loss": total}
    return {
        f"{prefix}_{key}": (
            None if value is not None and not math.isfinite(value) else value
        )
        for key, value in metrics.items()
    }


def _dependency_rows(model: nn.Module, batch: DependencyBatch) -> list[dict]:
    prediction = model(batch.conditions).delta_hat.float()
    if prediction.shape != batch.residual.shape:
        raise ValueError("GeneEffect predictions and targets must have equal shape")
    valid = batch.valid
    if not all(
        bool(torch.isfinite(value[valid]).all())
        for value in (prediction, batch.residual, batch.gene_mean)
    ):
        raise ValueError("non-finite GeneEffect prediction or target on scored rows")
    values = (
        torch.stack(
            (batch.residual.float(), batch.gene_mean.float(), prediction), dim=1
        )
        .detach()
        .cpu()
        .tolist()
    )
    # Keep one dictionary per loader row until Accelerate has trimmed its tail.
    return [
        {
            "model_id": model_id,
            "gene_symbol": gene,
            "gene_effect": residual + mean if keep else float("nan"),
            "residual": residual if keep else float("nan"),
            "geneeffect_prediction": pred + mean if keep else float("nan"),
            "residual_prediction": pred if keep else float("nan"),
        }
        for model_id, gene, (residual, mean, pred), keep in zip(
            batch.conditions.model_ids,
            batch.conditions.genes,
            values,
            valid.cpu().tolist(),
            strict=True,
        )
    ]


def _gather_batch_rows(action: Callable[[], list[dict]], accelerator) -> list[dict]:
    """Propagate rank-local inference failures before the row collective."""
    rows, error = [], None
    try:
        rows = action()
    except Exception as exc:
        if accelerator is None or accelerator.num_processes == 1:
            raise
        error = f"{type(exc).__name__}: {exc}"
    if accelerator is not None and accelerator.num_processes > 1:
        errors = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(errors, error)
        if any(value is not None for value in errors):
            raise RuntimeError(f"evaluation failed on a rank: {errors}")
    if accelerator is not None:
        return accelerator.gather_for_metrics(rows, use_gather_object=True)
    return rows


def _correlation_details(
    frame: pd.DataFrame,
    units: Sequence[str],
    *,
    unit_col: str,
    truth_col: str,
    pred_col: str,
    residual_errors: bool = False,
) -> pd.DataFrame:
    rows = []
    groups = {key: group for key, group in frame.groupby(unit_col, sort=False)}
    for unit in units:
        group = groups.get(unit, frame.iloc[:0])
        truth = group[truth_col].to_numpy(dtype=float)
        prediction = group[pred_col].to_numpy(dtype=float)
        row = {
            unit_col: unit,
            "valid_pairs": int((np.isfinite(truth) & np.isfinite(prediction)).sum()),
            "pearson": _unit_pearson(truth, prediction),
            "spearman": _unit_spearman(truth, prediction),
        }
        if residual_errors:
            valid = np.isfinite(truth) & np.isfinite(prediction)
            target, predicted = truth[valid], prediction[valid]
            target_sd = float(target.std(ddof=0)) if len(target) >= 2 else math.nan
            prediction_sd = (
                float(predicted.std(ddof=0)) if len(target) >= 2 else math.nan
            )
            error = predicted - target
            row.update(
                target_sd=target_sd,
                prediction_sd=prediction_sd,
                sd_ratio=prediction_sd / target_sd if target_sd > 0 else math.nan,
                rmse=float(np.sqrt(np.mean(error**2))) if len(error) else math.nan,
                mae=float(np.mean(np.abs(error))) if len(error) else math.nan,
            )
        rows.append(row)
    columns = [unit_col, "valid_pairs", "pearson", "spearman"]
    if residual_errors:
        columns += ["target_sd", "prediction_sd", "sd_ratio", "rmse", "mae"]
    return pd.DataFrame(rows, columns=columns)


def aggregate_geneeffect(
    frame: pd.DataFrame,
    *,
    model_ids: Sequence[str],
    genes: Sequence[str],
    variable_genes: Sequence[str],
) -> tuple[dict[str, float | int | None], pd.DataFrame, pd.DataFrame]:
    """Derive pair errors and both correlation axes from the gathered rows."""
    if frame.empty:
        raise ValueError("GeneEffect evaluation has no valid pairs")
    if frame.duplicated(["model_id", "gene_symbol"]).any():
        raise ValueError("duplicate GeneEffect rows after metric gathering")
    if not set(frame.model_id).issubset(model_ids) or not set(
        frame.gene_symbol
    ).issubset(genes):
        raise ValueError("GeneEffect rows fall outside the fixed evaluation universe")
    valid = np.isfinite(frame["residual"].to_numpy(dtype=float))
    scored = frame.loc[valid]
    if scored.empty:
        raise ValueError("GeneEffect evaluation has no valid pairs")
    if not np.isfinite(
        scored[
            ["gene_effect", "geneeffect_prediction", "residual_prediction"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("non-finite GeneEffect prediction or target on scored rows")
    pred = torch.tensor(scored.residual_prediction.to_numpy(), dtype=torch.float32)
    truth = torch.tensor(scored.residual.to_numpy(), dtype=torch.float32)
    error = pred - truth
    loss = float(F.huber_loss(pred, truth, reduction="sum", delta=1.0) / len(scored))
    if not math.isfinite(loss):
        raise ValueError("GeneEffect evaluation loss must be finite")
    possible = len(model_ids) * len(genes)
    metrics: dict[str, float | int | None] = {
        "geneeffect_loss": loss,
        "geneeffect_rmse": float(error.square().mean().sqrt()),
        "geneeffect_mae": float(error.abs().mean()),
        "geneeffect_valid_pairs": len(scored),
        "geneeffect_possible_pairs": possible,
        "geneeffect_missing_pairs": possible - len(scored),
        "geneeffect_coverage": len(scored) / possible,
    }
    per_line = _correlation_details(
        frame,
        model_ids,
        unit_col="model_id",
        truth_col="gene_effect",
        pred_col="geneeffect_prediction",
    )
    per_gene = _correlation_details(
        frame,
        variable_genes,
        unit_col="gene_symbol",
        truth_col="residual",
        pred_col="residual_prediction",
        residual_errors=True,
    )
    for table, domain, axis in (
        (per_line, "geneeffect", "per_line"),
        (per_gene, "residual", "per_gene"),
    ):
        for correlation in ("pearson", "spearman"):
            defined = table[correlation].dropna()
            key = f"{domain}_{correlation}"
            metrics[f"{key}_macro_{axis}"] = (
                float(defined.mean()) if len(defined) else None
            )
            metrics[f"{key}_{axis}_scored"] = len(defined)
            metrics[f"{key}_{axis}_undefined"] = len(table) - len(defined)
    for name in ("target_sd", "prediction_sd", "sd_ratio", "rmse", "mae"):
        defined = per_gene[name].dropna()
        metrics[f"residual_{name}_macro_per_gene"] = (
            float(defined.mean()) if len(defined) else None
        )
        metrics[f"residual_{name}_per_gene_scored"] = len(defined)
        metrics[f"residual_{name}_per_gene_undefined"] = len(per_gene) - len(defined)
    return metrics, per_line, per_gene


@contextmanager
def _evaluation_mode(model: nn.Module, device: torch.device):
    """Diagnostics cannot advance training RNG streams or change module modes."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    mps_rng = torch.mps.get_rng_state() if device.type == "mps" else None
    modes = [(module, module.training) for module in model.modules()]
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)
        for module, training in modes:
            module.training = training


def evaluate_model(
    model: nn.Module,
    inputs: PreparedInputs,
    config: Mapping[str, Any],
    *,
    split: str,
    accelerator=None,
) -> EvalResult:
    """Evaluate once over fixed rows, preserving module modes on exit."""
    if split not in {"train", "val", "test"}:
        raise ValueError("evaluation split must be train, val or test")
    dependency, response_loader = make_evaluation_loaders(
        inputs, config, split, accelerator
    )
    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
    device = (
        accelerator.device
        if accelerator is not None
        else next(model.parameters()).device
    )
    dependency_rows, heldout_rows = [], []
    with _evaluation_mode(model, device):
        for batch in dependency:
            with accelerator.autocast() if accelerator is not None else nullcontext():
                dependency_rows.extend(
                    _gather_batch_rows(
                        lambda: _dependency_rows(model, batch.to(device)),
                        accelerator,
                    )
                )
        for batch in response_loader if response_loader is not None else ():
            with accelerator.autocast() if accelerator is not None else nullcontext():
                heldout_rows.extend(
                    _gather_batch_rows(
                        lambda: response_rows(unwrapped, batch.to(device)),
                        accelerator,
                    )
                )
    predictions = pd.DataFrame(dependency_rows)
    response = pd.DataFrame(
        heldout_rows,
        columns=[
            "model_id",
            "gene_symbol",
            "mean_delta_mse",
            "energy_distance",
        ],
    )
    geneeffect, per_line, per_gene = aggregate_geneeffect(
        predictions,
        model_ids=(
            inputs.split.supervised_train
            if split == "train"
            else getattr(inputs.split, split)
        ),
        genes=inputs.genes,
        variable_genes=[gene for gene in inputs.genes if gene in inputs.variable_genes],
    )
    metrics = (
        {f"train_eval_{key}": value for key, value in geneeffect.items()}
        if split == "train"
        else compose_metrics(
            geneeffect,
            aggregate_response(response, inputs.response_anchors),
            response_weight=float(config["train"]["response_weight"]),
            prefix=split,
        )
    )
    return EvalResult(metrics, predictions, per_line, per_gene, response)
