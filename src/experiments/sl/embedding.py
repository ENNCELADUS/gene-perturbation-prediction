"""The gene-knowledge, cell-set model on the multi-study SL benchmark (SL protocol
§18; docs/specs/2026-11-19-embedding-transfer-plan.md). One model
(``src/model/sl_embedding.py``), two stages:

1. Stage one, pretraining: the GeneEffect residual ``(GeneEffect - m_g) / sigma_g`` of
   the fit lines, which are the labelled training-side lines without any SL label (and
   not of an SL line's patient). ``m_g`` and ``sigma_g`` are fitted on them. The epoch
   kept has the lowest loss on the SL training lines with GeneEffect, which are out of
   sample. Thus the model reads every SL line, training or held out, out of sample.
2. Stage two: the pair head on the frozen stage-one model, then the whole model with
   the GeneEffect loss kept at each weight of ``stage_two.geneeffect_weights``. Rows
   are the SL training lines' genetic screens with both classes, drawn as the benchmark
   weighs them (each line equal, then its studies, a study's screens, a screen's two
   classes). Validation recall in the top 10% selects the epoch and the phase; test is
   scored once.

A line with cells reads its mean token (bridged pseudo-bulk, as the SL backbone) and
one token for each cell (its cached Tx1-3B embedding): a random subset at each training
step (or none, at ``cells.dropout``), all of them at scoring. A bulk-only line has the
mean token alone. ``cell_embeddings`` names the cache of the cells' embeddings
(``<ModelID>/embeddings.npy``): Tx1-3B's, or another frozen model's (CellNet's,
``hpc/cellnet/embed_lines.py``). The run stops if a line with cells is not in it. With
``cell_embeddings: null`` no line reads a cell: the same model on the mean token alone.

Outputs: ``scores/embedding.parquet`` for the benchmark harness (``extra_scores``),
``predictions.parquet`` (the model's GeneEffect of the SL lines), ``best.pt``,
``metrics.json`` and ``results.md``."""

from __future__ import annotations

import argparse
import copy
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F

from src.data.io import jsonable, write_json, write_text
from src.eval.accuracy import wide_accuracy
from src.experiments.config import read_yaml, require_keys
from src.experiments.run_dir import bind
from src.experiments.sl.benchmark_baselines import (
    HEADLINE,
    LIST_KEYS,
    METRICS,
    line_records,
    list_records,
)
from src.experiments.sl.conditional import KEYS, recall_top10, soft_weights
from src.linear_model.targets import Definitions, residual_frame
from src.model.sl_embedding import GeneLineModel
from src.training.stage_one_trainer import masked_mse

log = logging.getLogger(__name__)

SCORER = "embedding"
#: One line's mean token, cell tokens and the cells' mask (``one_line``).
Tokens = tuple[Tensor, Tensor | None, Tensor | None]
SECTIONS = {
    "model": {"hidden", "head_hidden"},
    "cells": {"per_step", "dropout"},
    "stage_one": {"max_epochs", "patience", "learning_rate", "weight_decay",
                  "lines_per_step", "genes_per_step"},
    "stage_two": {"max_epochs", "patience", "steps_per_epoch", "lines_per_step",
                  "rows_per_line", "learning_rate", "backbone_learning_rate",
                  "weight_decay", "geneeffect_weights"},
    "eval": {"rows_per_chunk", "genes_per_chunk"},
}  # fmt: skip


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "output_root", "task", "linear_model", "labels", "split",
         "gene_sources", "cell_embeddings", "top_k", *SECTIONS},
        "config",
    )  # fmt: skip
    if config["kind"] != "sl_embedding":
        raise ValueError(f"{path} is a {config['kind']} config")
    for name, keys in SECTIONS.items():
        require_keys(config[name], keys, name)
    if not config["gene_sources"]:
        raise ValueError("gene_sources: one source or more")
    return config


@dataclass
class Data:
    """Everything the model reads, on one device. ``lines`` orders the line-level
    tensors; a line's cells are ``cells[offsets[i] : offsets[i + 1]]`` (none for a
    bulk-only line). ``targets`` (lines x genes) holds the GeneEffect residual of the
    fit lines and of the early-stop lines only, NaN elsewhere. ``rows`` are the
    benchmark's rows with ``line`` (position in ``lines``), ``a`` and ``b`` (positions
    in ``genes``; -1 for an anchor without a gene vector)."""

    genes: list[str]
    lines: list[str]
    expression: Tensor
    cells: Tensor
    offsets: Tensor
    targets: Tensor
    fit: Tensor
    stop: Tensor
    rows: pd.DataFrame
    sources: dict[str, tuple[Tensor, Tensor]]
    m_g: np.ndarray
    sigma_g: np.ndarray

    @property
    def device(self) -> torch.device:
        return self.expression.device


def build_model(config: Mapping[str, Any], data: Data) -> GeneLineModel:
    return GeneLineModel(
        data.sources,
        expression_dim=data.expression.shape[1],
        cell_dim=data.cells.shape[1],
        hidden=int(config["model"]["hidden"]),
        head_hidden=int(config["model"]["head_hidden"]),
    ).to(data.device)


def cell_batch(
    data: Data,
    lines: Tensor,
    per_line: int | None,
    generator: torch.Generator | None,
    dropout: float = 0.0,
) -> tuple[Tensor | None, Tensor | None]:
    """The padded cells of ``lines`` (lines, k, cell_dim) and their mask: all of a
    line's cells, or ``per_line`` of them drawn without replacement; with
    ``dropout``, a line loses all its cells at that rate. (None, None) when no line
    of the block has a cell left."""
    lines = lines.cpu()
    starts = data.offsets[lines].tolist()
    counts = (data.offsets[lines + 1] - data.offsets[lines]).tolist()
    picks = []
    for start, count in zip(starts, counts):
        if dropout and torch.rand((), generator=generator).item() < dropout:
            count = 0
        if per_line is None or count <= per_line:
            picks.append(start + torch.arange(count))
        else:
            picks.append(start + torch.randperm(count, generator=generator)[:per_line])
    width = max(len(pick) for pick in picks)
    if width == 0:
        return None, None
    index = torch.zeros(len(picks), width, dtype=torch.long)
    mask = torch.zeros(len(picks), width, dtype=torch.bool)
    for i, pick in enumerate(picks):
        index[i, : len(pick)] = pick
        mask[i, : len(pick)] = True
    return data.cells[index.to(data.device)], mask.to(data.device)


def line_tokens(
    model: GeneLineModel,
    data: Data,
    lines: Tensor,
    per_line: int | None,
    generator: torch.Generator | None,
    dropout: float = 0.0,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """The mean tokens, the cell tokens and the cells' mask of ``lines``."""
    cells, mask = cell_batch(data, lines, per_line, generator, dropout)
    mean, cells = model.tokens(data.expression[lines.to(data.device)], cells)
    return mean, cells, mask


def one_line(
    model: GeneLineModel,
    data: Data,
    line: int,
    per_line: int | None,
    generator: torch.Generator | None,
    dropout: float = 0.0,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """One line's mean token (width,), cell tokens (k, width) and mask (k,)."""
    mean, cells, mask = line_tokens(
        model, data, torch.tensor([line]), per_line, generator, dropout
    )
    return (
        mean[0],
        None if cells is None else cells[0],
        None if mask is None else mask[0],
    )


def dependency_loss(
    model: GeneLineModel,
    data: Data,
    genes: Tensor,
    lines: Tensor,
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> Tensor:
    """The GeneEffect loss of a block of genes by fit lines (it has a label)."""
    cells = config["cells"]
    tokens = line_tokens(
        model, data, lines, int(cells["per_step"]), generator, float(cells["dropout"])
    )
    z = model.dependency(model.gene_vectors()[genes], *tokens)
    return masked_mse(z, data.targets[lines][:, genes])


@torch.no_grad()
def predict_dependency(
    model: GeneLineModel, data: Data, lines: Tensor, config: Mapping[str, Any]
) -> Tensor:
    """``z`` (lines x genes, on the CPU) with every cell of each line."""
    u = model.gene_vectors()[:-1]
    chunk = int(config["eval"]["genes_per_chunk"])
    out = torch.empty(len(lines), len(u))
    for i, line in enumerate(lines.tolist()):
        tokens = line_tokens(model, data, torch.tensor([line]), None, None)
        for start in range(0, len(u), chunk):
            out[i, start : start + chunk] = model.dependency(
                u[start : start + chunk], *tokens
            )[0].cpu()
    return out


def stop_loss(model: GeneLineModel, data: Data, config: Mapping[str, Any]) -> float:
    z = predict_dependency(model, data, data.stop, config)
    return float(masked_mse(z, data.targets[data.stop].cpu()))


def blocks(count: int, size: int, generator: torch.Generator) -> list[Tensor]:
    return list(torch.split(torch.randperm(count, generator=generator), size))


def fit_dependency(
    model: GeneLineModel,
    data: Data,
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> list[dict[str, Any]]:
    """Stage one. Leaves ``model`` at the epoch with the lowest loss on the SL
    training lines (epoch 0, the untrained model, is a candidate)."""
    train = config["stage_one"]
    optimizer = torch.optim.AdamW(
        model.backbone(), lr=train["learning_rate"], weight_decay=train["weight_decay"]
    )
    best = stop_loss(model, data, config)
    state, bad = copy.deepcopy(model.state_dict()), 0
    history = [{"epoch": 0, "train_loss": None, "stop_loss": best}]
    for epoch in range(1, int(train["max_epochs"]) + 1):
        losses = []
        lines = [data.fit[b] for b in blocks(len(data.fit), train["lines_per_step"],
                                            generator)]  # fmt: skip
        genes = blocks(len(data.genes), train["genes_per_step"], generator)
        steps = [(g, c) for g in genes for c in lines]
        for index in torch.randperm(len(steps), generator=generator).tolist():
            block, fit_lines = steps[index]
            block = block.to(data.device)
            if not torch.isfinite(data.targets[fit_lines][:, block]).any():
                continue
            loss = dependency_loss(model, data, block, fit_lines, config, generator)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        held = stop_loss(model, data, config)
        history.append(
            {"epoch": epoch, "train_loss": float(torch.stack(losses).mean()),
             "stop_loss": held}
        )  # fmt: skip
        log.info(
            "stage one, epoch %d: training loss %.4f, loss on the SL training lines "
            "%.4f (epoch 0: %.4f)",
            epoch, history[-1]["train_loss"], held, history[0]["stop_loss"],
        )  # fmt: skip
        if held < best:
            best, state, bad = held, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= int(train["patience"]):
                break
    model.load_state_dict(state)
    return history


def training_rows(rows: pd.DataFrame) -> np.ndarray:
    """The rows the pair head trains on: the training lines' genetic screens with
    both classes."""
    classes = rows.groupby(LIST_KEYS)["label"].transform("nunique")
    return (
        (rows["split"] == "train")
        & (rows["condition"] == HEADLINE)
        & (classes == 2)
    ).to_numpy()  # fmt: skip


@dataclass
class PairRows:
    """Row-level tensors on the device, and each training line's rows with the
    benchmark's weights as draw rates."""

    line: Tensor
    a: Tensor
    b: Tensor
    combinatorial: Tensor
    label: Tensor
    draws: dict[int, tuple[Tensor, Tensor]]  # line -> (row positions, rates), CPU


def pair_rows(data: Data) -> PairRows:
    rows, device = data.rows, data.device
    keep = training_rows(rows)
    chosen = rows[keep]
    positive, negative = soft_weights(chosen, chosen["label"].to_numpy(float))
    weight = pd.Series(positive + negative, index=np.flatnonzero(keep))
    draws = {
        int(line): (torch.tensor(at.index.to_numpy()), torch.tensor(at.to_numpy()))
        for line, at in weight.groupby(chosen["line"].to_numpy())
    }
    return PairRows(
        line=torch.tensor(rows["line"].to_numpy(), device=device),
        a=torch.tensor(rows["a"].to_numpy(), device=device),
        b=torch.tensor(rows["b"].to_numpy(), device=device),
        combinatorial=torch.tensor(
            (rows["design"] != "anchor").to_numpy(), device=device
        ),
        label=torch.tensor(rows["label"].to_numpy(np.float32), device=device),
        draws=draws,
    )


def pair_loss(
    model: GeneLineModel,
    data: Data,
    pairs: PairRows,
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> Tensor:
    """Binary cross-entropy of one step: ``lines_per_step`` training lines drawn
    evenly, ``rows_per_line`` rows of each drawn at the benchmark's weights."""
    train, cells = config["stage_two"], config["cells"]
    u = model.gene_vectors()
    lines = sorted(pairs.draws)
    losses = []
    for pick in torch.randint(
        len(lines), (int(train["lines_per_step"]),), generator=generator
    ).tolist():
        line = lines[pick]
        positions, rates = pairs.draws[line]
        at = positions[
            torch.multinomial(
                rates, int(train["rows_per_line"]), replacement=True,
                generator=generator,
            )
        ].to(data.device)  # fmt: skip
        tokens = one_line(
            model, data, line, int(cells["per_step"]), generator,
            float(cells["dropout"]),
        )  # fmt: skip
        logits = model.pair(
            u[pairs.a[at]], u[pairs.b[at]], pairs.combinatorial[at], *tokens
        )
        losses.append(F.binary_cross_entropy_with_logits(logits, pairs.label[at]))
    return torch.stack(losses).mean()


@torch.no_grad()
def pair_scores(
    model: GeneLineModel,
    data: Data,
    pairs: PairRows,
    rows: np.ndarray,
    config: Mapping[str, Any],
    tokens_of: Callable[[int], Tokens] | None = None,
) -> np.ndarray:
    """The SL logit of each row position in ``rows``, with every cell of its line;
    ``tokens_of`` gives a line other tokens than its own (``line_input_controls``)."""
    u = model.gene_vectors()
    chunk = int(config["eval"]["rows_per_chunk"])
    at = torch.tensor(rows, device=data.device)
    out = torch.empty(len(at))
    line_of = pairs.line[at]

    def own(line: int) -> Tokens:
        return one_line(model, data, line, None, None)

    for line in line_of.unique().tolist():
        tokens = (tokens_of or own)(line)
        mine = torch.nonzero(line_of == line).squeeze(1)
        for start in range(0, len(mine), chunk):
            part = at[mine[start : start + chunk]]
            out[mine[start : start + chunk].cpu()] = model.pair(
                u[pairs.a[part]], u[pairs.b[part]], pairs.combinatorial[part], *tokens
            ).cpu()
    return out.numpy().astype(np.float64)


@torch.no_grad()
def line_input_controls(
    model: GeneLineModel, data: Data, pairs: PairRows, config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """The SL logit of every row with the line's input replaced (no refit):
    ``mean_input``, the fit lines' mean expression and no cells, so the score reads the
    genes alone; ``other_line``, the expression and cells of another benchmark line
    (the next one in a seeded order, never the line itself)."""
    every = np.arange(len(data.rows))
    mean, _ = model.tokens(torch.zeros_like(data.expression[:1]), None)
    order = np.random.default_rng(int(config["seed"])).permutation(
        np.unique(data.rows["line"])
    )
    other = dict(zip(order.tolist(), np.roll(order, 1).tolist()))
    return {
        "mean_input": pair_scores(
            model, data, pairs, every, config, lambda line: (mean[0], None, None)
        ),
        "other_line": pair_scores(
            model, data, pairs, every, config,
            lambda line: one_line(model, data, other[line], None, None),
        ),
    }  # fmt: skip


def fit_pairs(
    model: GeneLineModel,
    data: Data,
    pairs: PairRows,
    config: Mapping[str, Any],
    generator: torch.Generator,
    geneeffect_weight: float | None,
) -> tuple[dict[str, Tensor], float, list[dict[str, Any]]]:
    """Stage two from the model's present weights. ``geneeffect_weight`` None: the
    stage-one model is frozen and the head alone trains; a number: the whole model
    trains, with the GeneEffect loss at that weight. Returns the weights, validation
    recall in the top 10% and history of the best epoch; the model is left at its
    last epoch."""
    train, one = config["stage_two"], config["stage_one"]
    joint = geneeffect_weight is not None
    for parameter in model.backbone():
        parameter.requires_grad_(joint)
    groups = [{"params": list(model.head.parameters())}]
    if joint:
        groups.append({"params": model.backbone(),
                       "lr": train["backbone_learning_rate"]})  # fmt: skip
    optimizer = torch.optim.AdamW(
        groups, lr=train["learning_rate"], weight_decay=train["weight_decay"]
    )
    val = np.flatnonzero((data.rows["split"] == "val").to_numpy())
    val_rows = data.rows.iloc[val]
    best, state, bad, history = -np.inf, None, 0, []
    for epoch in range(1, int(train["max_epochs"]) + 1):
        losses = []
        for _ in range(int(train["steps_per_epoch"])):
            loss = pair_loss(model, data, pairs, config, generator)
            if joint:
                genes = torch.randperm(len(data.genes), generator=generator)
                lines = torch.randperm(len(data.fit), generator=generator)
                loss = loss + geneeffect_weight * dependency_loss(
                    model, data, genes[: one["genes_per_step"]].to(data.device),
                    data.fit[lines[: one["lines_per_step"]].to(data.device)], config,
                    generator,
                )  # fmt: skip
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        recall = recall_top10(
            val_rows, pair_scores(model, data, pairs, val, config), int(config["top_k"])
        )
        history.append(
            {"epoch": epoch, "train_loss": float(torch.stack(losses).mean()),
             "val_recall_top10": recall}
        )  # fmt: skip
        log.info(
            "stage two, %s, epoch %d: training loss %.4f, validation recall in the "
            "top 10%% %.4f",
            "whole model" if joint else "frozen stage one", epoch,
            history[-1]["train_loss"], recall,
        )  # fmt: skip
        if recall > best:
            best, state, bad = recall, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= int(train["patience"]):
                break
    for parameter in model.backbone():
        parameter.requires_grad_(True)
    if state is None:
        raise RuntimeError("no epoch had a validation recall in the top 10%")
    return state, float(best), history


def train(
    model: GeneLineModel, data: Data, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Stage one, then stage two's phases; leaves ``model`` at the phase with the
    best validation recall in the top 10% (the frozen phase wins a tie)."""
    generator = torch.Generator().manual_seed(int(config["seed"]))
    pairs = pair_rows(data)
    stage_one = fit_dependency(model, data, config, generator)
    phases = []
    frozen, score, history = fit_pairs(model, data, pairs, config, generator, None)
    phases.append({"phase": "frozen", "geneeffect_weight": None,
                   "val_recall_top10": score, "history": history})  # fmt: skip
    states = [frozen]
    for weight in config["stage_two"]["geneeffect_weights"]:
        model.load_state_dict(frozen)
        state, score, history = fit_pairs(
            model, data, pairs, config, generator, float(weight)
        )
        phases.append({"phase": "trained further", "geneeffect_weight": float(weight),
                       "val_recall_top10": score, "history": history})  # fmt: skip
        states.append(state)
    chosen = int(np.argmax([phase["val_recall_top10"] for phase in phases]))
    model.load_state_dict(states[chosen])
    every = np.arange(len(data.rows))
    return {
        "stage_one": stage_one,
        "phases": phases,
        "chosen": chosen,
        "scores": pair_scores(model, data, pairs, every, config),
        "controls": line_input_controls(model, data, pairs, config),
    }


def sl_metrics(rows: pd.DataFrame, score: np.ndarray, top_k: int) -> dict[str, Any]:
    """The benchmark's metrics of ``score``: per split, the mean over lines of each
    line's recall in the top 10%, AUROC and AUPRC over the positive rate."""
    frame = rows[[*LIST_KEYS, "label"]].assign(scorer=SCORER, score=score)
    lines = line_records(list_records(frame, top_k))
    return {
        split: {"lines": len(group)} | {m: float(group[m].mean()) for m in METRICS}
        for split, group in lines.groupby("split")
    }


def geneeffect_table(
    model: GeneLineModel, data: Data, lines: Sequence[str], config: Mapping[str, Any]
) -> pd.DataFrame:
    """The model's GeneEffect of ``lines`` (lines x genes): ``m_g + z sigma_g``."""
    position = {m: i for i, m in enumerate(data.lines)}
    z = predict_dependency(
        model, data, torch.tensor([position[m] for m in lines]), config
    ).numpy()
    return pd.DataFrame(
        data.m_g + z * data.sigma_g, index=list(lines), columns=data.genes
    )


def stage_one_metrics(
    predicted: pd.DataFrame,
    gene_effect: pd.DataFrame,
    definitions: Definitions,
    sides: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """The accepted stage-one metrics of the model's GeneEffect on each side's lines
    with GeneEffect."""
    scale, out = definitions.residual_scale, {}
    for name, lines in sides.items():
        lines = [m for m in lines if m in gene_effect.index]
        if not lines:
            out[name] = {"lines": 0}
            continue
        guess = residual_frame(predicted, lines, definitions).div(scale, axis=1)
        truth = residual_frame(gene_effect, lines, definitions).div(scale, axis=1)
        out[name] = {"lines": len(lines)} | wide_accuracy(guess, truth, definitions)
    return out


def read_sources(
    paths: Mapping[str, str], genes: Sequence[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each gene source (an npz of ``axis``, ``vectors``, ``present``) on ``genes``,
    centred on its mean over the genes it covers; an absent gene is zero."""
    from src.experiments.stage_one.resident import centred

    out = {}
    for name, path in paths.items():
        with np.load(path) as payload:
            at = {str(g): i for i, g in enumerate(payload["axis"])}
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
            listed = np.asarray(payload["present"], dtype=bool)
        index = np.array([at.get(g, -1) for g in genes])
        present = (index >= 0) & listed[index]
        aligned = np.where(present[:, None], vectors[index], 0.0)
        if not present.any():
            raise ValueError(f"{path} covers no gene of the model")
        out[name] = (centred(aligned, present, aligned[present].mean(axis=0)), present)
    return out


def read_cells(
    cache: Path | None, lines: Sequence[str], fit: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    """The cell embeddings of ``lines`` (``<cache>/<ModelID>/embeddings.npy``)
    stacked (float16) and each line's count, centred on the mean over the fit lines
    of each line's mean cell. Without a ``cache`` (the mean-token variant), no line
    has a cell."""
    if cache is None:
        return np.zeros((0, 1), np.float16), np.zeros(len(lines), dtype=np.int64)
    missing = [m for m in lines if not (Path(cache) / m / "embeddings.npy").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} lines with cells have no embeddings in {cache}: "
            f"{missing[:10]}"
        )
    parts = [np.load(Path(cache) / m / "embeddings.npy", mmap_mode="r") for m in lines]
    centre = np.mean(
        [np.asarray(part, dtype=np.float32).mean(axis=0)
         for m, part in zip(lines, parts) if m in fit],
        axis=0,
    )  # fmt: skip
    cells = np.concatenate(
        [(np.asarray(part, dtype=np.float32) - centre).astype(np.float16)
         for part in parts]
    )  # fmt: skip
    return cells, np.array([len(part) for part in parts])


def stage_one_lines(
    split: Any,
    bulk_labelled: Sequence[str],
    patient: pd.Series,
    sl_lines: set[str],
    sl_train: set[str],
    with_geneeffect: set[str],
) -> tuple[list[str], list[str], dict[str, int]]:
    """Stage one's fit lines and early-stop lines. Fit: the split's labelled
    single-cell training lines (every lineage, as the stage-one frame) and the
    labelled bulk-only lines (``bulk_labelled``, the linear model's, without its
    dropped lineages), less every line with an SL label (``sl_lines``) and every
    line of such a line's patient. Early stop: the SL training lines (``sl_train``)
    with GeneEffect. Also the counts of lines left out."""
    from src.data.splits import assert_fit_eligible

    sl_patients = set(patient.reindex(sorted(sl_lines)).dropna())
    candidates = [
        m for m in (*split.supervised_train, *bulk_labelled) if m in with_geneeffect
    ]
    mates = [m for m in candidates if m not in sl_lines and patient[m] in sl_patients]
    fit = [m for m in candidates if m not in sl_lines and m not in set(mates)]
    for model_id in fit:
        if model_id in split.all_model_ids:
            assert_fit_eligible(model_id, split)
    leaked = {*split.val, *split.test} & set(fit)
    if leaked:
        raise ValueError(f"SL held-out lines among the fit lines: {sorted(leaked)}")
    stop = [m for m in split.supervised_train if m in sl_train and m in with_geneeffect]
    if not stop:
        raise ValueError("no SL training line has GeneEffect: nothing to stop on")
    return (
        fit,
        stop,
        {
            "sl_lines_left_out": len(set(candidates) & sl_lines),
            "patient_mates_left_out": len(mates),
        },
    )


def load_data(
    config: Mapping[str, Any], device: torch.device
) -> tuple[Data, Any, dict[str, Any]]:
    """The model's inputs on the SL split, the linear-model base they come from (for
    the definitions and GeneEffect) and the counts."""
    from src.data.extra_lines import load_bulk_lines
    from src.data.splits import load_geneeffect_split
    from src.experiments.linear_model.base import base_on
    from src.experiments.linear_model.config import load_linear_model_config
    from src.experiments.sl.benchmark_baselines import benchmark_rows
    from src.experiments.sl.conditional import bridged_rows

    split_path = Path(config["split"])
    split = load_geneeffect_split(split_path)
    base = base_on(
        load_linear_model_config(Path(config["linear_model"])),
        split,
        load_bulk_lines(split_path, split),
        [*split.val, *split.test],
        other_split=True,
    )
    genes = list(base.definitions.genes)
    labels = pd.read_parquet(config["labels"])
    rows, dropped = benchmark_rows(labels, split, set(genes))
    _, bridged, _ = bridged_rows(base)
    with_cells = set(bridged.index)
    fit, stop, left_out = stage_one_lines(
        split,
        [m for m in base.labelled if m not in with_cells],
        base.models["patient_id"],
        set(labels["model_id"]),
        set(rows.loc[rows["split"] == "train", "model_id"]),
        set(base.gene_effect.index),
    )
    cell_lines = [m for m in bridged.index
                  if m in set(fit) | set(rows["model_id"])]  # fmt: skip
    bulk_lines = [m for m in fit if m not in with_cells]
    lines = [*cell_lines, *bulk_lines]
    position = {m: i for i, m in enumerate(lines)}
    expression = pd.concat([bridged.loc[cell_lines], base.bridge.bulk.loc[bulk_lines]])
    centre, scale = expression.loc[fit].mean(axis=0), expression.loc[fit].std(axis=0)
    expression = (expression - centre) / scale.where(scale > 0, 1.0)
    labelled = base.gene_effect.reindex(index=[*fit, *stop], columns=genes)
    m_g = labelled.loc[fit].mean(axis=0)
    sigma_g = labelled.loc[fit].std(axis=0, ddof=0)
    sigma_g = sigma_g.where(sigma_g > 0, 1.0)
    targets = torch.full((len(lines), len(genes)), torch.nan)
    targets[[position[m] for m in labelled.index]] = torch.tensor(
        ((labelled - m_g) / sigma_g).to_numpy(np.float32)
    )
    cache = config["cell_embeddings"]
    cells, counts = read_cells(
        None if cache is None else Path(cache), cell_lines, set(fit)
    )
    total = int(counts.sum())
    offsets = np.concatenate([[0], np.cumsum(counts), [total] * len(bulk_lines)])
    gene_at = {g: i for i, g in enumerate(genes)}
    rows = rows.assign(
        line=rows["model_id"].map(position),
        a=[gene_at.get(g, -1) for g in rows["gene_a"]],
        b=rows["gene_b"].map(gene_at),
    )
    data = Data(
        genes=genes,
        lines=lines,
        expression=torch.tensor(expression.to_numpy(np.float32), device=device),
        cells=torch.from_numpy(cells).to(device),
        offsets=torch.tensor(offsets),
        targets=targets.to(device),
        fit=torch.tensor([position[m] for m in fit], device=device),
        stop=torch.tensor([position[m] for m in stop], device=device),
        rows=rows,
        sources={
            name: (torch.from_numpy(v).to(device), torch.from_numpy(p).to(device))
            for name, (v, p) in read_sources(config["gene_sources"], genes).items()
        },
        m_g=m_g.to_numpy(np.float64),
        sigma_g=sigma_g.to_numpy(np.float64),
    )
    counts_record = {
        "fit_lines": len(fit),
        "fit_lines_with_cells": sum(m in with_cells for m in fit),
        "stop_lines": len(stop),
        **left_out,
        "cells": total,
        "genes": len(genes),
        "expression_genes": expression.shape[1],
        "gene_source_coverage": {n: int(p.sum()) for n, (_, p) in data.sources.items()},
        "rows": {s: int((rows["split"] == s).sum()) for s in ("train", "val", "test")}
        | dropped,
        "training_rows": int(training_rows(rows).sum()),
    }
    return data, base, counts_record


def write_scores(
    run_dir: Path, rows: pd.DataFrame, scores: Mapping[str, np.ndarray]
) -> None:
    """One harness score table per name: ``scores/<name>.parquet``."""
    (run_dir / "scores").mkdir(exist_ok=True)
    for name, score in scores.items():
        table = rows[KEYS].assign(score=score).drop_duplicates(KEYS)
        staging = run_dir / "scores" / f".{name}.parquet.tmp"
        table.to_parquet(staging, index=False)
        os.replace(staging, run_dir / "scores" / f"{name}.parquet")


def controls_named(controls: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{SCORER}_{name}": score for name, score in controls.items()}


def rescore(config: Mapping[str, Any], run_id: str, device: str) -> Path:
    """The line-input controls of a finished run, from its ``best.pt``; no training."""
    run_dir = Path(config["output_root"]) / config["task"] / run_id
    data, _, _ = load_data(config, torch.device(device))
    model = build_model(config, data)
    model.load_state_dict(
        torch.load(run_dir / "best.pt", map_location=device)["weights"]
    )
    controls = line_input_controls(model, data, pair_rows(data), config)
    write_scores(run_dir, data.rows, controls_named(controls))
    return run_dir


def run(config: Mapping[str, Any], run_id: str, device: str) -> Path:
    run_dir = Path(config["output_root"]) / config["task"] / run_id
    if (run_dir / "metrics.json").exists():
        raise FileExistsError(f"{run_dir} is finished; use a new --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    torch.manual_seed(int(config["seed"]))
    data, base, counts = load_data(config, torch.device(device))
    log.info("inputs: %s", counts)
    model = build_model(config, data)
    result = train(model, data, config)
    torch.save({"weights": model.state_dict(), "config": jsonable(config)},
               run_dir / "best.pt")  # fmt: skip
    rows = data.rows
    write_scores(run_dir, rows, {SCORER: result["scores"]})
    write_scores(run_dir, rows, controls_named(result["controls"]))
    sides = {s: sorted(set(rows.loc[rows["split"] == s, "model_id"]))
             for s in ("train", "val", "test")}  # fmt: skip
    labelled = [m for group in sides.values() for m in group
                if m in base.gene_effect.index]  # fmt: skip
    predicted = geneeffect_table(model, data, labelled, config)
    predicted.astype("float32").to_parquet(run_dir / "predictions.parquet")
    metrics = {
        "run_id": run_id,
        "counts": counts,
        "sl": sl_metrics(rows, result["scores"], int(config["top_k"])),
        "stage_one": stage_one_metrics(
            predicted, base.gene_effect, base.definitions, sides
        ),
        # the gene mean of the fit lines alone (z = 0): what the model starts from
        "stage_one_base": stage_one_metrics(
            predicted * 0 + data.m_g, base.gene_effect, base.definitions, sides
        ),
        "stage_one_history": result["stage_one"],
        "phases": result["phases"],
        "chosen_phase": result["chosen"],
    }
    write_json(run_dir / "metrics.json", jsonable(metrics))
    write_text(run_dir / "results.md", results_markdown(metrics))
    return run_dir


def results_markdown(metrics: Mapping[str, Any]) -> str:
    """Table 1, SL on the benchmark; Table 2, the stage-one numbers; Table 3, the
    phases of stage two; Table 4, the epochs of stage one."""
    sl, one, counts = metrics["sl"], metrics["stage_one"], metrics["counts"]
    base = metrics["stage_one_base"]
    names = {"recall_top10": "Recall in the top 10%", "auroc": "AUROC",
             "auprc_lift": "AUPRC over the positive rate"}  # fmt: skip
    keys = [("line_spearman", "per-line residual Spearman"),
            ("selective_spearman", "per-gene residual Spearman"),
            ("within_gene_r2", "residual skill score"),
            ("absolute_pearson", "whole-line Pearson")]  # fmt: skip

    def cell(value: Any) -> str:
        return "—" if value is None else f"{value:.4f}"

    def epochs(history: Sequence[Mapping[str, Any]], key: str, best) -> str:
        chosen = best(history, key=lambda r: r[key])
        return f"{chosen['epoch']} / {history[-1]['epoch']}"

    chosen = metrics["phases"][metrics["chosen_phase"]]
    lines = [
        f"# Gene-knowledge, cell-set model: {metrics['run_id']}",
        "",
        "**Table 1.** The model on the multi-study SL benchmark, mean over lines "
        f"({sl['val']['lines']} validation, {sl['test']['lines']} test; training lines "
        "in sample). Higher is better. Validation selected the epoch and the phase; "
        "test was scored once.",
        "",
        "| Metric | Training | Validation | Test |",
        "| --- | ---: | ---: | ---: |",
        *(f"| {name} | " + " | ".join(cell(sl[s][m]) for s in ("train", "val", "test"))
          + " |" for m, name in names.items()),
        "",
        "**Table 2.** Stage-one numbers on the SL lines with GeneEffect ("
        + " / ".join(str(one[s]["lines"]) for s in ("val", "test"))
        + " lines, validation / test; out of sample for stage one), as every "
        "stage-one reader is scored: the predicted GeneEffect as a residual over the "
        "training gene mean. The second row is the fit lines' gene mean alone, which "
        "the untrained model predicts. Higher is better. They decide nothing.",
        "",
        "| Model |" + "".join(f" {name}, {s} |" for _, name in keys
                              for s in ("validation", "test")),
        "| --- |" + " ---: |" * (2 * len(keys)),
        *(f"| {name} | " + " | ".join(cell(table[s].get(k)) for k, _ in keys
                                      for s in ("val", "test")) + " |"
          for name, table in (("Gene-knowledge, cell-set model", one),
                              ("Gene mean of the fit lines", base))),
        "",
        "**Table 3.** Stage two: validation recall in the top 10% of each phase at its "
        "best epoch. The chosen phase is in bold.",
        "",
        "| Phase | GeneEffect loss weight | Best epoch / epochs run | Validation |",
        "| --- | ---: | ---: | ---: |",
        *(f"| {phase['phase']} | "
          + ("—" if phase["geneeffect_weight"] is None
             else f"{phase['geneeffect_weight']:g}")
          + f" | {epochs(phase['history'], 'val_recall_top10', max)} | "
          + (f"**{phase['val_recall_top10']:.4f}**" if phase is chosen
             else f"{phase['val_recall_top10']:.4f}") + " |"
          for phase in metrics["phases"]),
        "",
        "**Table 4.** Stage one: the GeneEffect loss on the SL training lines (out of "
        "sample; lower is better). Epoch 0 is the untrained model.",
        "",
        "| Epoch | Training loss | Loss on the SL training lines |",
        "| ---: | ---: | ---: |",
        *(f"| {r['epoch']} | {cell(r['train_loss'])} | {r['stop_loss']:.4f} |"
          for r in metrics["stage_one_history"]),
        "",
        f"Stage-one fit lines: {counts['fit_lines']:,} "
        f"({counts['fit_lines_with_cells']:,} with cells); left out: "
        f"{counts['sl_lines_left_out']} lines with an SL label and "
        f"{counts['patient_mates_left_out']} lines of their patients. Early-stop "
        f"lines: {counts['stop_lines']}. Cells: {counts['cells']:,}. Genes: "
        f"{counts['genes']:,}; with each source: "
        + ", ".join(f"{n} {c:,}" for n, c in counts["gene_source_coverage"].items())
        + f". Stage-two training rows: {counts['training_rows']:,}.",
        "",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--line-input-controls",
        action="store_true",
        help="score a finished run again with each line's input replaced; no training",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    step = rescore if args.line_input_controls else run
    out = step(load_config(args.config), args.run_id, args.device)
    print(f"gene-knowledge, cell-set model: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
