"""GeneEffect evaluation of the joint model on one split.

Targets are residuals centred on the fold-fit gene mean (the prepared ``residual``
column); absolute predictions add the fixed training mean to the predicted residual.
Correlations of a constant prediction are undefined (NaN, counted), never zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from src.data.datasets import DependencyDataset
from src.data.labels import DEPENDENCY_THRESHOLD
from src.data.prepared import PreparedInputs
from src.eval.metrics import unit_pearson, unit_spearman, paired_line_bootstrap

if TYPE_CHECKING:
    from src.context_prior.targets import Definitions

#: The metrics ``score`` keeps, and the seed of every paired line bootstrap.
METRICS = (
    "selective_spearman",
    "selective_aupr_lift",
    "residual_pearson_macro_per_gene",
    "geneeffect_loss",
    "residual_sd_ratio_macro_per_gene",
)
BOOTSTRAP_SEED = 0


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
            "pearson": unit_pearson(truth, prediction),
            "spearman": unit_spearman(truth, prediction),
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
    """This rank's row positions and float32 residual predictions of the stack
    (prior offset plus head).

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
            predicted = model(batch.conditions).float() + batch.prior
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


# --- wide predictions in residual-SD units ------------------------------------------
#
# A prediction table is lines x genes in units of the per-gene residual SD, centred
# on the training gene mean of ``definitions`` (the scorer re-centres every
# prediction on it); the truth table is ``(y - mu_g) / sigma_g`` on the same axes.


def long_frame(
    prediction: pd.DataFrame, truth: pd.DataFrame, definitions: Definitions
) -> pd.DataFrame:
    """Finite-label rows in GeneEffect units, as ``aggregate_geneeffect`` reads them."""
    lines, genes = list(prediction.index), list(prediction.columns)
    scale = definitions.residual_scale.loc[genes].to_numpy()
    residual = truth.loc[lines, genes].to_numpy() * scale
    predicted = prediction.to_numpy() * scale
    keep = np.isfinite(residual)
    line_index, gene_index = np.nonzero(keep)
    means = definitions.gene_means.loc[genes].to_numpy()[gene_index]
    return pd.DataFrame(
        {
            "model_id": np.asarray(lines)[line_index],
            "gene_symbol": np.asarray(genes)[gene_index],
            "residual": residual[keep],
            "residual_prediction": predicted[keep],
            "gene_effect": residual[keep] + means,
            "geneeffect_prediction": predicted[keep] + means,
        }
    )


def score(
    prediction: pd.DataFrame, truth: pd.DataFrame, definitions: Definitions
) -> dict:
    frame = long_frame(prediction, truth, definitions)
    metrics, _, _ = aggregate_geneeffect(
        frame,
        model_ids=list(prediction.index),
        genes=list(definitions.genes),
        variable_genes=list(definitions.variable),
        selective_genes=list(definitions.selective),
    )
    return {key: metrics[key] for key in METRICS}


def gain(
    better: pd.DataFrame,
    reference: pd.DataFrame,
    truth: pd.DataFrame,
    definitions: Definitions,
    repeats: int,
    genes: Sequence[str] | None = None,
) -> dict:
    """Selective-Spearman difference and its 95% paired line-bootstrap interval,
    over ``genes`` (default: every selective gene)."""
    genes = list(definitions.selective if genes is None else genes)
    result = paired_line_bootstrap(
        long_frame(better.loc[:, genes], truth, definitions),
        long_frame(reference.loc[:, genes], truth, definitions),
        genes,
        repeats=repeats,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "difference": float(result["difference"]),
        "interval": [float(value) for value in result["interval"]],
    }


def line_spearmans(
    prediction: pd.DataFrame, truth: pd.DataFrame, genes: Sequence[str]
) -> np.ndarray:
    """Each line's Spearman across ``genes`` between its predicted and measured
    residuals (scoring units), over the genes labelled in that line: does the model
    know which genes this line needs more or less than usual? NaN where undefined
    (fewer than ``MIN_OBSERVATIONS`` genes, or a side constant across them)."""
    predicted = prediction.loc[:, list(genes)].to_numpy(dtype=np.float64)
    measured = truth.loc[prediction.index, list(genes)].to_numpy(dtype=np.float64)
    return np.array(
        [unit_spearman(measured[i], predicted[i]) for i in range(len(predicted))]
    )


def line_spearman_gain(
    better: np.ndarray, reference: np.ndarray, *, repeats: int
) -> dict[str, Any]:
    """Per-line residual Spearman difference, ``better`` minus ``reference`` (each
    line's value, aligned), over the lines defined on both sides, with a 95%
    paired line bootstrap (the same resample for both)."""
    both = np.isfinite(better) & np.isfinite(reference)
    difference = better[both] - reference[both]
    if not both.any():
        return {"difference": None, "interval": [None, None], "lines": 0}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(difference), (repeats, len(difference)))
    resampled = difference[draws].mean(axis=1)
    return {
        "difference": float(difference.mean()),
        "interval": [float(q) for q in np.quantile(resampled, [0.025, 0.975])],
        "lines": int(both.sum()),
    }


def wide_accuracy(
    prediction: pd.DataFrame,
    truth: pd.DataFrame,
    definitions: Definitions,
    *,
    panel: Sequence[str] | None = None,
) -> dict[str, float | int | None]:
    """The stage-one metrics of a wide prediction on its lines (PIE stage-one
    design §4.1): the per-line residual Spearman over the selective genes (the
    selection criterion, protocol §6: the mean over lines of ``line_spearmans``);
    selective Spearman; within-gene R² pooled over the selective
    genes (``1 - SSE / sum truth^2``, the truth centred on the training gene mean,
    which scores 0); selective AUPR lift; residual Pearson; per-line absolute
    Pearson; RMSE; the calibration slope over the selective genes. With ``panel``,
    also the selective Spearman over the selective genes in it."""
    from src.eval.metrics import calibration_slope

    lines, genes = list(prediction.index), list(definitions.genes)
    prediction, truth = prediction.loc[lines, genes], truth.loc[lines, genes]
    metrics, _, per_gene = aggregate_geneeffect(
        long_frame(prediction, truth, definitions),
        model_ids=lines,
        genes=genes,
        variable_genes=list(definitions.variable),
        selective_genes=list(definitions.selective),
    )
    selective = list(definitions.selective)
    predicted = prediction[selective].to_numpy(dtype=np.float64)
    measured = truth[selective].to_numpy(dtype=np.float64)
    finite = np.isfinite(measured)
    if not np.isfinite(predicted[finite]).all():
        raise ValueError("a prediction is missing where the truth is finite")
    error = np.where(finite, predicted - measured, 0.0)
    total = float(np.square(np.where(finite, measured, 0.0)).sum())
    per_line = line_spearmans(prediction, truth, selective)
    defined = per_line[np.isfinite(per_line)]
    out: dict[str, float | int | None] = {
        "lines": len(lines),
        "line_spearman": float(defined.mean()) if len(defined) else None,
        "line_spearman_scored": len(defined),
        "selective_spearman": metrics["selective_spearman"],
        "selective_spearman_scored": metrics["selective_spearman_scored"],
        "within_gene_r2": 1.0 - float(np.square(error).sum()) / total,
        "aupr_lift": metrics["selective_aupr_lift"],
        "residual_pearson": metrics["residual_pearson_macro_per_gene"],
        "absolute_pearson": metrics["geneeffect_pearson_macro_per_line"],
        "rmse": metrics["geneeffect_rmse"],
        "calibration_slope": calibration_slope(predicted, measured),
    }
    if panel is not None:
        inside = per_gene.loc[
            per_gene["selective"] & per_gene["gene_symbol"].isin(set(panel)),
            "spearman",
        ].dropna()
        out["selective_spearman_panel"] = float(inside.mean()) if len(inside) else None
        out["selective_spearman_panel_scored"] = len(inside)
    return out


def defined_genes(
    prediction: pd.DataFrame, truth: pd.DataFrame, genes: Sequence[str]
) -> list[str]:
    """``genes`` whose Spearman across the prediction's lines is defined: at least
    ``MIN_OBSERVATIONS`` labelled lines, with neither the truth nor the prediction
    constant over them (a row that predicts the base for a gene is undefined there)."""
    from src.eval.metrics import MIN_OBSERVATIONS

    lines = list(prediction.index)
    predicted = prediction.loc[lines, list(genes)].to_numpy(dtype=np.float64)
    measured = truth.loc[lines, list(genes)].to_numpy(dtype=np.float64)
    finite = np.isfinite(measured) & np.isfinite(predicted)
    kept = []
    for column, gene in enumerate(genes):
        rows = finite[:, column]
        x, y = predicted[rows, column], measured[rows, column]
        if rows.sum() >= MIN_OBSERVATIONS and np.ptp(x) > 0 and np.ptp(y) > 0:
            kept.append(gene)
    return kept


#: Metrics that rank or correlate lines within a gene. Out of fold, each fold's lines
#: are predicted around their own base ``m_g`` without the fold, which moves with
#: the fold's labels (a fold of dependent lines leaves a higher base), so pooling
#: folds makes these anti-correlated with the truth by construction: they are scored
#: within each fold and averaged over folds.
WITHIN_FOLD_METRICS = (
    "selective_spearman",
    "selective_spearman_panel",
    "aupr_lift",
    "residual_pearson",
    "calibration_slope",
)


def fold_accuracy(
    prediction: pd.DataFrame,
    truth: pd.DataFrame,
    definitions: Definitions,
    folds: Mapping[str, int],
    *,
    panel: Sequence[str] | None = None,
) -> dict[str, Any]:
    """``wide_accuracy`` of an out-of-fold prediction: the metrics of
    ``WITHIN_FOLD_METRICS`` scored within each fold's lines and averaged over the
    folds where they are defined; the per-line residual Spearman, within-gene R²,
    RMSE and the per-line Pearson pooled over every line (they compare each line
    with its own truth). ``per_fold``
    keeps each fold's scores and line count."""
    lines = list(prediction.index)
    out: dict[str, Any] = wide_accuracy(prediction, truth, definitions, panel=panel)
    groups: dict[int, list[str]] = {}
    for line in lines:
        groups.setdefault(int(folds[line]), []).append(line)
    per_fold = {
        k: wide_accuracy(
            prediction.loc[members], truth.loc[members], definitions, panel=panel
        )
        for k, members in sorted(groups.items())
    }
    for key in WITHIN_FOLD_METRICS:
        if key not in out:
            continue
        values = [
            float(scores[key])
            for scores in per_fold.values()
            if scores[key] is not None and np.isfinite(float(scores[key]))
        ]
        out[key] = float(np.mean(values)) if values else None
    for key in ("selective_spearman_scored", "selective_spearman_panel_scored"):
        if key in out:
            out[key] = min(int(scores[key]) for scores in per_fold.values())
    out["per_fold"] = {str(k): scores for k, scores in per_fold.items()}
    return out
