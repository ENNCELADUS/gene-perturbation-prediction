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
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from src.data.datasets import DependencyDataset
from src.data.geneeffect import DEPENDENCY_THRESHOLD
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


def _aupr_lift(frame: pd.DataFrame, units: Sequence[str]) -> pd.Series:
    """Per-gene dependent-line AUPR minus prevalence; NaN without both classes.

    A gene is scored on its rows with a finite GeneEffect and prediction; a line is
    dependent when ``gene_effect < DEPENDENCY_THRESHOLD`` and ranks higher the lower
    its predicted GeneEffect. A constant prediction scores exactly 0.
    """
    groups = dict(tuple(frame.groupby("gene_symbol", sort=False)))
    lift = {}
    for gene in units:
        group = groups.get(gene, frame.iloc[:0])
        truth = group["gene_effect"].to_numpy(dtype=float)
        prediction = group["geneeffect_prediction"].to_numpy(dtype=float)
        valid = np.isfinite(truth) & np.isfinite(prediction)
        dependent = truth[valid] < DEPENDENCY_THRESHOLD
        positives = int(dependent.sum())
        if positives == 0 or positives == len(dependent):
            lift[gene] = math.nan
            continue
        lift[gene] = float(
            average_precision_score(dependent, -prediction[valid])
            - positives / len(dependent)
        )
    return pd.Series(lift, dtype=float)


def aggregate_geneeffect(
    frame: pd.DataFrame,
    *,
    model_ids: Sequence[str],
    genes: Sequence[str],
    variable_genes: Sequence[str],
    selective_genes: Sequence[str],
) -> tuple[dict[str, float | int | None], pd.DataFrame, pd.DataFrame]:
    """Pair errors, absolute per-line, residual per-variable-gene and selective metrics.

    ``frame`` holds ``model_id``, ``gene_symbol``, ``gene_effect``, ``residual``,
    ``geneeffect_prediction`` and ``residual_prediction``. Undefined correlations
    stay NaN in the tables, are left out of the macro means and are counted.

    ``per_gene`` covers the union of the variable and selective genes in ``genes``
    order, with boolean ``variable`` and ``selective`` columns and the gene's
    dependent-line ``aupr_lift``. Every ``residual_*`` metric uses the variable rows
    only; ``selective_spearman`` (residual Spearman across lines) and
    ``selective_aupr_lift`` are macro means over the selective rows.
    """
    variable, selective = set(variable_genes), set(selective_genes)
    if not (variable | selective) <= set(genes):
        raise ValueError("variable or selective genes outside the gene order")
    tabled = [gene for gene in genes if gene in variable | selective]
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
        tabled,
        unit_col="gene_symbol",
        truth_col="residual",
        pred_col="residual_prediction",
        residual_errors=True,
    )
    per_gene["variable"] = per_gene["gene_symbol"].isin(variable)
    per_gene["selective"] = per_gene["gene_symbol"].isin(selective)
    per_gene["aupr_lift"] = per_gene["gene_symbol"].map(
        _aupr_lift(frame, [gene for gene in tabled if gene in selective])
    )
    variable_rows = per_gene.loc[per_gene["variable"]]
    selective_rows = per_gene.loc[per_gene["selective"]]
    for name, column in (
        ("selective_spearman", "spearman"),
        ("selective_aupr_lift", "aupr_lift"),
    ):
        defined = selective_rows[column].dropna()
        metrics[name] = float(defined.mean()) if len(defined) else None
        metrics[f"{name}_scored"] = len(defined)
        metrics[f"{name}_undefined"] = len(selective_rows) - len(defined)
    for table, domain, axis in (
        (per_line, "geneeffect", "per_line"),
        (variable_rows, "residual", "per_gene"),
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
        defined = variable_rows[name].dropna()
        metrics[f"residual_{name}_macro_per_gene"] = (
            float(defined.mean()) if len(defined) else None
        )
        metrics[f"residual_{name}_per_gene_scored"] = len(defined)
        metrics[f"residual_{name}_per_gene_undefined"] = len(variable_rows) - len(
            defined
        )
    return metrics, per_line, per_gene


def _predict(
    model: nn.Module,
    dataset: DependencyDataset,
    batch_size: int,
    accelerator,
) -> tuple[np.ndarray, np.ndarray]:
    """This rank's row positions and float32 residual predictions.

    Batches are consecutive ``batch_size`` rows in dataset order; rank ``r`` of
    ``W`` takes batches ``r, r + W, ...``.
    """
    rank, world = (
        (0, 1)
        if accelerator is None
        else (accelerator.process_index, accelerator.num_processes)
    )
    positions, predictions = [], []
    for start in range(rank * batch_size, len(dataset), world * batch_size):
        rows = range(start, min(start + batch_size, len(dataset)))
        batch = dataset.collate(rows)
        with accelerator.autocast() if accelerator else nullcontext():
            predicted = model(batch.conditions).delta_hat.float()
        positions.append(np.asarray(rows, dtype=np.int64))
        predictions.append(predicted.cpu().numpy())
    if not positions:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    return np.concatenate(positions), np.concatenate(predictions)


def _assemble(
    gathered: Sequence[tuple[np.ndarray, np.ndarray]], rows: int
) -> np.ndarray:
    """Every rank's predictions placed at their row positions, each row exactly once."""
    positions = np.concatenate([rank_positions for rank_positions, _ in gathered])
    if not np.array_equal(np.sort(positions), np.arange(rows)):
        raise RuntimeError("ranks did not predict every evaluation row exactly once")
    predicted = np.empty(rows, dtype=np.float32)
    predicted[positions] = np.concatenate([values for _, values in gathered])
    return predicted


def _prediction_frame(
    dataset: DependencyDataset, predicted: np.ndarray
) -> pd.DataFrame:
    """One row per dataset row; float32 targets and predictions widened to float64."""
    mean = dataset.gene_mean.cpu().numpy().astype(np.float64)
    residual_prediction = predicted.astype(np.float64)
    return pd.DataFrame(
        {
            "model_id": dataset.model_ids,
            "gene_symbol": dataset.genes,
            "gene_effect": dataset.gene_effect.cpu().numpy().astype(np.float64),
            "residual": dataset.residual.cpu().numpy().astype(np.float64),
            "geneeffect_prediction": residual_prediction + mean,
            "residual_prediction": residual_prediction,
        }
    )


def evaluate_model(
    model: nn.Module,
    inputs: PreparedInputs,
    config: Mapping[str, Any],
    *,
    split: str,
    accelerator=None,
    lines: Sequence[str] | None = None,
) -> EvalResult:
    """Score every labelled row of ``split`` once, leaving training state untouched.

    ``lines`` restricts scoring to a subset of the split's lines. Metrics are
    prefixed ``{split}_``; the training-split diagnostic uses ``train_eval_`` so it
    never collides with per-update training losses. Module modes and the torch
    RNG are restored on exit.

    Under several processes each rank predicts its share of the batches, rank
    zero gathers the predictions with their row positions, builds the tables and
    metrics alone and broadcasts the metrics. Every rank returns the metrics; only
    rank zero's result carries the tables, the others carry empty frames.
    """
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    device = next(model.parameters()).device
    dataset = DependencyDataset(inputs, split, device=device, lines=lines)
    modes = [(module, module.training) for module in model.modules()]
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            local = _predict(
                model, dataset, config["train"]["dependency_batch_size"], accelerator
            )
    finally:
        for module, training in modes:
            module.training = training
    distributed = accelerator is not None and accelerator.num_processes > 1
    gathered = [local]
    if distributed:
        gathered = (
            [None] * accelerator.num_processes if accelerator.is_main_process else None
        )
        torch.distributed.gather_object(local, gathered, dst=0)
    prefix = "train_eval" if split == "train" else split
    payload: list[Any] = [None]
    failure: Exception | None = None
    empty = pd.DataFrame()
    predictions, per_line, per_gene = empty, empty, empty
    if not distributed or accelerator.is_main_process:
        try:
            predictions = _prediction_frame(dataset, _assemble(gathered, len(dataset)))
            metrics, per_line, per_gene = aggregate_geneeffect(
                predictions,
                model_ids=dataset.lines,
                genes=inputs.genes,
                variable_genes=[
                    gene for gene in inputs.genes if gene in inputs.variable_genes
                ],
                selective_genes=[
                    gene for gene in inputs.genes if gene in inputs.selective_genes
                ],
            )
            payload[0] = {f"{prefix}_{key}": value for key, value in metrics.items()}
        except Exception as error:  # the other ranks must not wait at the broadcast
            failure = error
            payload[0] = f"{type(error).__name__}: {error}"
    if distributed:
        torch.distributed.broadcast_object_list(payload, src=0)
    if failure is not None:
        raise failure
    if isinstance(payload[0], str):
        raise RuntimeError(f"rank zero evaluation failed: {payload[0]}")
    return EvalResult(payload[0], predictions, per_line, per_gene)
