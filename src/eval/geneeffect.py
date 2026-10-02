"""GeneEffect evaluation of the joint model on one split.

Targets are residuals centred on the fold-fit gene mean (the prepared ``residual``
column); absolute predictions add the fixed training mean to the predicted residual.
Correlations of a constant prediction are undefined (NaN, counted), never zero.
"""

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.data.batches import DependencyBatch
from src.data.datasets import make_evaluation_loader, split_lines
from src.data.prepared import PreparedInputs
from src.eval.metrics import _unit_pearson, _unit_spearman


@dataclass
class EvalResult:
    metrics: dict[str, float | int | None]
    predictions: pd.DataFrame
    per_line: pd.DataFrame
    per_gene: pd.DataFrame
    # Kept for the readout and Tx1 GMM-ridge evaluators, which pass it positionally.
    response: pd.DataFrame = field(default_factory=pd.DataFrame)


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
    groups = dict(tuple(frame.groupby(unit_col, sort=False)))
    for unit in units:
        group = groups.get(unit, frame.iloc[:0])
        truth = group[truth_col].to_numpy(dtype=float)
        prediction = group[pred_col].to_numpy(dtype=float)
        valid = np.isfinite(truth) & np.isfinite(prediction)
        row = {
            unit_col: unit,
            "valid_pairs": int(valid.sum()),
            "pearson": _unit_pearson(truth, prediction),
            "spearman": _unit_spearman(truth, prediction),
        }
        if residual_errors:
            target, predicted = truth[valid], prediction[valid]
            enough = len(target) >= 2
            target_sd = float(target.std()) if enough else math.nan
            prediction_sd = float(predicted.std()) if enough else math.nan
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
    """Pair errors, absolute per-line and residual per-variable-gene correlations.

    ``frame`` holds ``model_id``, ``gene_symbol``, ``gene_effect``, ``residual``,
    ``geneeffect_prediction`` and ``residual_prediction``. Undefined correlations
    stay NaN in the tables, are left out of the macro means and are counted.
    """
    if frame.duplicated(["model_id", "gene_symbol"]).any():
        raise ValueError("duplicate GeneEffect rows")
    scored = frame.loc[np.isfinite(frame["residual"].to_numpy(dtype=float))]
    if scored.empty:
        raise ValueError("GeneEffect evaluation has no valid pairs")
    pred = torch.tensor(scored.residual_prediction.to_numpy(), dtype=torch.float32)
    truth = torch.tensor(scored.residual.to_numpy(), dtype=torch.float32)
    error = pred - truth
    possible = len(model_ids) * len(genes)
    metrics: dict[str, float | int | None] = {
        "geneeffect_loss": float(F.huber_loss(pred, truth, delta=1.0)),
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


def _rows(model: nn.Module, batch: DependencyBatch) -> list[dict]:
    prediction = model(batch.conditions).delta_hat.float()
    values = torch.stack(
        (batch.gene_effect, batch.residual, batch.gene_mean, prediction), dim=1
    )
    return [
        {
            "model_id": model_id,
            "gene_symbol": gene,
            "gene_effect": gene_effect,
            "residual": residual,
            "geneeffect_prediction": predicted + mean,
            "residual_prediction": predicted,
        }
        for model_id, gene, (gene_effect, residual, mean, predicted) in zip(
            batch.conditions.model_ids,
            batch.conditions.genes,
            values.cpu().tolist(),
            strict=True,
        )
    ]


def evaluate_model(
    model: nn.Module,
    inputs: PreparedInputs,
    config: Mapping[str, Any],
    *,
    split: str,
    accelerator=None,
) -> EvalResult:
    """Score every labelled row of ``split`` once, leaving training state untouched.

    Metrics are prefixed ``{split}_``; the training-split diagnostic uses
    ``train_eval_`` so it never collides with per-update training losses. Module
    modes and the torch RNG are restored on exit.
    """
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    device = next(model.parameters()).device
    loader = make_evaluation_loader(inputs, config, split, accelerator)
    modes = [(module, module.training) for module in model.modules()]
    devices = [device.index or 0] if device.type == "cuda" else []
    rows: list[dict] = []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            for batch in loader:
                with accelerator.autocast() if accelerator else nullcontext():
                    batch_rows = _rows(model, batch.to(device))
                if accelerator is not None:
                    batch_rows = accelerator.gather_for_metrics(
                        batch_rows, use_gather_object=True
                    )
                rows.extend(batch_rows)
    finally:
        for module, training in modes:
            module.training = training
    predictions = pd.DataFrame(rows)
    metrics, per_line, per_gene = aggregate_geneeffect(
        predictions,
        model_ids=split_lines(inputs, split),
        genes=inputs.genes,
        variable_genes=[gene for gene in inputs.genes if gene in inputs.variable_genes],
    )
    prefix = "train_eval" if split == "train" else split
    return EvalResult(
        {f"{prefix}_{key}": value for key, value in metrics.items()},
        predictions,
        per_line,
        per_gene,
    )
