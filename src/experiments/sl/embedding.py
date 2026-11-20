"""The gene-knowledge, cell-set model on the multi-study SL benchmark (SL protocol
§18; docs/specs/2026-11-19-embedding-transfer-plan.md). One model
(``src/model/sl_embedding.py``), two stages:

1. Stage one, pretraining: GeneEffect itself of the fit lines, which are the labelled
   training-side lines without any SL label (and not of an SL line's patient), on one
   scale for every gene: ``(GeneEffect - centre) / scale``, with the mean and the SD of
   all the fit lines' values (``target_scale``). Thus the untrained model predicts one
   value for every gene, and a trained epoch can lower the loss. The epoch kept has
   the lowest loss on the SL training lines with GeneEffect, which are out of sample.
   Thus the model reads every SL line, training or held out, out of sample.
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

The line term (config section ``line_term``; ``null`` for none) trains last, on the
chosen phase. The score of a row (line c, pair A, B) is

    s(c, A, B) = f(c, A, B) + D(c, A, B) - mean over r in R of D(r, A, B)

``f(c, A, B)``: the model's own score, the pair head of the chosen phase read in line
c (``pair_scores``). ``D(r, A, B)``: the line head on the pair read in line r (its
tokens) and on the gene-in-line features of A and B in line r (the line's own
expression with ``own_expression``, and the caches of ``gene_in_line``). ``R``: the SL
training lines with training rows (genetic screens with both classes); at a training
step, a draw of ``reference_lines`` of them. ``f`` and ``D`` of a line read the same
tokens. Only the line head and its input maps train, so ``f`` does not change, and
the gene pair alone cannot fill the line term. Validation recall in the top 10% of
``s`` selects the epoch (``max_epochs``, ``patience``, ``learning_rate``).

Outputs: ``scores/embedding.parquet`` for the benchmark harness (``extra_scores``);
the line-input controls ``scores/embedding_mean_input.parquet`` (the fit lines' mean
input) and ``scores/embedding_other_line.parquet`` (another line's input); with a line
term, ``scores/embedding_line_term.parquet`` (``s``) and
``scores/embedding_line_term_other_line.parquet`` (``s`` with ``f`` and ``D`` read in
another line); ``predictions.parquet`` (the model's GeneEffect of the SL lines),
``best.pt``, ``metrics.json`` and ``results.md``."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
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
LINE_TERM = {"reference_lines", "max_epochs", "patience", "learning_rate",
             "own_expression", "gene_in_line"}  # fmt: skip


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "output_root", "task", "linear_model", "labels", "split",
         "gene_sources", "cell_embeddings", "top_k", "line_term", *SECTIONS},
        "config",
    )  # fmt: skip
    if config["kind"] != "sl_embedding":
        raise ValueError(f"{path} is a {config['kind']} config")
    for name, keys in SECTIONS.items():
        require_keys(config[name], keys, name)
    if config["line_term"] is not None:
        require_keys(config["line_term"], LINE_TERM, "line_term")
    if not config["gene_sources"]:
        raise ValueError("gene_sources: one source or more")
    return config


@dataclass
class Data:
    """Everything the model reads, on one device. ``lines`` orders the line-level
    tensors; a line's cells are ``cells[offsets[i] : offsets[i + 1]]`` (none for a
    bulk-only line). ``targets`` (lines x genes) holds GeneEffect on one scale
    (``(GeneEffect - centre) / scale``) of the fit lines and of the early-stop lines
    only, NaN elsewhere. ``rows`` are the benchmark's rows with ``line`` (position in
    ``lines``), ``a`` and ``b`` (positions in ``genes``; -1 for an anchor without a
    gene vector). ``in_line`` holds each source's gene-in-line features (feature
    lines x genes + 1 x width, float16; the last column is the presence flag, the last
    gene row is zero); the feature lines are the lines with benchmark rows, and
    ``in_line_at`` gives each line's row in them (-1 for a line without rows)."""

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
    centre: float
    scale: float
    in_line: dict[str, Tensor]
    in_line_at: Tensor

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
        gene_in_line={name: t.shape[2] for name, t in data.in_line.items()},
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


def step_rows(
    data: Data, pairs: PairRows, config: Mapping[str, Any], generator: torch.Generator
) -> Iterator[tuple[int, Tensor]]:
    """The lines and row positions of one step: ``lines_per_step`` training lines
    drawn evenly, ``rows_per_line`` rows of each drawn at the benchmark's weights."""
    train = config["stage_two"]
    lines = sorted(pairs.draws)
    for pick in torch.randint(
        len(lines), (int(train["lines_per_step"]),), generator=generator
    ).tolist():
        positions, rates = pairs.draws[lines[pick]]
        yield lines[pick], positions[
            torch.multinomial(
                rates, int(train["rows_per_line"]), replacement=True,
                generator=generator,
            )
        ].to(data.device)  # fmt: skip


def pair_loss(
    model: GeneLineModel,
    data: Data,
    pairs: PairRows,
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> Tensor:
    """Binary cross-entropy of one step (``step_rows``), each line equal."""
    cells = config["cells"]
    u = model.gene_vectors()
    losses = []
    for line, at in step_rows(data, pairs, config, generator):
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


def other_lines(data: Data, config: Mapping[str, Any]) -> dict[int, int]:
    """Each benchmark line's other line: the next one in a seeded order, never the
    line itself."""
    order = np.random.default_rng(int(config["seed"])).permutation(
        np.unique(data.rows["line"])
    )
    return dict(zip(order.tolist(), np.roll(order, 1).tolist()))


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
    other = other_lines(data, config)
    return {
        "mean_input": pair_scores(
            model, data, pairs, every, config, lambda line: (mean[0], None, None)
        ),
        "other_line": pair_scores(
            model, data, pairs, every, config,
            lambda line: one_line(model, data, other[line], None, None),
        ),
    }  # fmt: skip


def fit_epochs(
    model: GeneLineModel,
    data: Data,
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    step: Callable[[], Tensor],
    scores: Callable[[np.ndarray], np.ndarray],
    epochs: int,
    patience: int,
    name: str,
) -> tuple[dict[str, Tensor], float, list[dict[str, Any]]]:
    """The epoch loop of stage two and of the line term: ``step`` gives the loss of
    one step; validation recall in the top 10% of ``scores`` (of the validation row
    positions) selects the epoch. Returns the weights, validation recall and history
    of the best epoch; the model is left at its last epoch, with every parameter
    trainable again (also on an error)."""
    val = np.flatnonzero((data.rows["split"] == "val").to_numpy())
    val_rows = data.rows.iloc[val]
    best, state, bad, history = -np.inf, None, 0, []
    try:
        for epoch in range(1, epochs + 1):
            losses = []
            for _ in range(int(config["stage_two"]["steps_per_epoch"])):
                loss = step()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
            recall = recall_top10(val_rows, scores(val), int(config["top_k"]))
            history.append(
                {"epoch": epoch, "train_loss": float(torch.stack(losses).mean()),
                 "val_recall_top10": recall}
            )  # fmt: skip
            log.info(
                "%s, epoch %d: training loss %.4f, validation recall in the top 10%% "
                "%.4f", name, epoch, history[-1]["train_loss"], recall,
            )  # fmt: skip
            if recall > best:
                best, state, bad = recall, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    finally:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    if state is None:
        raise RuntimeError("no epoch had a validation recall in the top 10%")
    return state, float(best), history


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

    def step() -> Tensor:
        loss = pair_loss(model, data, pairs, config, generator)
        if joint:
            genes = torch.randperm(len(data.genes), generator=generator)
            lines = torch.randperm(len(data.fit), generator=generator)
            loss = loss + geneeffect_weight * dependency_loss(
                model, data, genes[: one["genes_per_step"]].to(data.device),
                data.fit[lines[: one["lines_per_step"]].to(data.device)], config,
                generator,
            )  # fmt: skip
        return loss

    return fit_epochs(
        model, data, config, optimizer, step,
        lambda val: pair_scores(model, data, pairs, val, config),
        int(train["max_epochs"]), int(train["patience"]),
        f"stage two, {'whole model' if joint else 'frozen stage one'}",
    )  # fmt: skip


def line_term_of(
    model: GeneLineModel,
    data: Data,
    u: Tensor,
    a: Tensor,
    b: Tensor,
    combinatorial: Tensor,
    line: int,
    tokens: Tokens,
) -> Tensor:
    """``D`` (rows,) of the pairs (``a``, ``b``: gene positions, -1 reads the zero
    row) read in ``line``: the line's ``tokens`` and its gene-in-line features of
    the two genes."""
    feature = int(data.in_line_at[line])
    if feature < 0:
        raise ValueError(f"line {data.lines[line]} has no gene-in-line features")

    def of(genes: Tensor) -> dict[str, Tensor]:
        return {n: t[feature, genes].float() for n, t in data.in_line.items()}

    return model.line_term(u[a], u[b], combinatorial, of(a), of(b), *tokens)


def line_score(
    model: GeneLineModel,
    data: Data,
    u: Tensor,
    a: Tensor,
    b: Tensor,
    combinatorial: Tensor,
    line: int,
    tokens: Tokens,
) -> Tensor:
    """``f + D`` (rows,) of the pairs read in ``line``: the model's own score (the
    pair head, no gradient) and the line term, on the same ``tokens``."""
    with torch.no_grad():
        f = model.pair(u[a], u[b], combinatorial, *tokens)
    return f + line_term_of(model, data, u, a, b, combinatorial, line, tokens)


@torch.no_grad()
def line_term_scores(
    model: GeneLineModel,
    data: Data,
    pairs: PairRows,
    rows: np.ndarray,
    config: Mapping[str, Any],
    read: Callable[[int], int] | None = None,
) -> np.ndarray:
    """The score ``s = f(c) + D(c) - mean over R of D(r)`` of each row position in
    ``rows``, with every cell of each line: ``f(c)``, the model's own score (the
    pair head) in the row's line ``c``; ``D(c)``, the line term in ``c``; ``R``, the
    SL training lines with training rows. ``read(line)`` names the line whose tokens
    and gene-in-line features stand in for ``line`` in ``f`` and ``D``; ``R`` does
    not change with it. The mean over ``R`` depends on the pair alone, so each
    distinct pair gets it once."""
    u = model.gene_vectors()
    chunk = int(config["eval"]["rows_per_chunk"])
    at = torch.tensor(rows, device=data.device)
    unique, inverse = torch.unique(
        torch.stack([pairs.a[at], pairs.b[at], pairs.combinatorial[at].long()], 1),
        dim=0,
        return_inverse=True,
    )
    reference = [(r, one_line(model, data, r, None, None)) for r in sorted(pairs.draws)]
    shared = torch.empty(len(unique), device=data.device)
    for start in range(0, len(unique), chunk):
        a, b, combinatorial = unique[start : start + chunk].T
        combinatorial = combinatorial.bool()
        deviations = [
            line_term_of(model, data, u, a, b, combinatorial, r, tokens)
            for r, tokens in reference
        ]
        shared[start : start + chunk] = torch.stack(deviations).mean(0)
    out = -shared[inverse]
    line_of = pairs.line[at]
    for line in line_of.unique().tolist():
        source = line if read is None else read(line)
        tokens = one_line(model, data, source, None, None)
        mine = torch.nonzero(line_of == line).squeeze(1)
        for start in range(0, len(mine), chunk):
            part = mine[start : start + chunk]
            row = at[part]
            out[part] += line_score(
                model, data, u, pairs.a[row], pairs.b[row], pairs.combinatorial[row],
                source, tokens,
            )  # fmt: skip
    return out.cpu().numpy().astype(np.float64)


def line_term_loss(
    model: GeneLineModel,
    data: Data,
    pairs: PairRows,
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> Tensor:
    """Binary cross-entropy of ``s`` at one step: the rows of ``step_rows``, with
    ``f`` and ``D`` of a row's line on the same drawn tokens; ``R``,
    ``line_term.reference_lines`` SL training lines drawn with replacement, one
    draw for the step."""
    cells = config["cells"]
    per_line, dropout = int(cells["per_step"]), float(cells["dropout"])
    u = model.gene_vectors()
    lines = sorted(pairs.draws)
    drawn = torch.randint(
        len(lines), (int(config["line_term"]["reference_lines"]),), generator=generator
    ).tolist()
    reference = [
        (lines[i], one_line(model, data, lines[i], per_line, generator, dropout))
        for i in drawn
    ]
    steps = [
        (line, at, one_line(model, data, line, per_line, generator, dropout))
        for line, at in step_rows(data, pairs, config, generator)
    ]
    at = torch.cat([part for _, part, _ in steps])
    a, b, combinatorial = pairs.a[at], pairs.b[at], pairs.combinatorial[at]
    own = torch.cat([
        line_score(model, data, u, pairs.a[part], pairs.b[part],
                   pairs.combinatorial[part], line, tokens)
        for line, part, tokens in steps
    ])  # fmt: skip
    deviations = [
        line_term_of(model, data, u, a, b, combinatorial, r, tokens)
        for r, tokens in reference
    ]
    # every line has rows_per_line rows: the mean over rows weighs each line equally
    return F.binary_cross_entropy_with_logits(
        own - torch.stack(deviations).mean(0), pairs.label[at]
    )


def fit_line_term(
    model: GeneLineModel,
    data: Data,
    pairs: PairRows,
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> tuple[dict[str, Tensor], float, list[dict[str, Any]]]:
    """The line term from the model's present weights. Only the line head and the
    gene-in-line maps train, so ``f`` does not change. Returns the weights,
    validation recall in the top 10% of ``line_term_scores`` and history of the best
    epoch; the model is left at its last epoch."""
    term, train = config["line_term"], config["stage_two"]
    trained = model.line_parameters()
    ids = {id(p) for p in trained}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in ids)
    optimizer = torch.optim.AdamW(
        trained, lr=term["learning_rate"], weight_decay=train["weight_decay"]
    )
    return fit_epochs(
        model, data, config, optimizer,
        lambda: line_term_loss(model, data, pairs, config, generator),
        lambda val: line_term_scores(model, data, pairs, val, config),
        int(term["max_epochs"]), int(term["patience"]), "line term",
    )  # fmt: skip


def train(
    model: GeneLineModel, data: Data, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Stage one, then stage two's phases; picks the phase with the best validation
    recall in the top 10% (the frozen phase wins a tie). With a ``line_term``
    config, the line term then trains on that phase, and ``model`` is left at its
    best epoch."""
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
    result = {
        "stage_one": stage_one,
        "phases": phases,
        "chosen": chosen,
        "scores": pair_scores(model, data, pairs, every, config),
        "controls": line_input_controls(model, data, pairs, config),
        "line_term": None,
    }
    if config["line_term"] is None:
        return result
    state, score, history = fit_line_term(model, data, pairs, config, generator)
    model.load_state_dict(state)
    other = other_lines(data, config)
    result["line_term"] = {
        "val_recall_top10": score,
        "history": history,
        "scores": line_term_scores(model, data, pairs, every, config),
        "other_line": line_term_scores(
            model, data, pairs, every, config, other.__getitem__
        ),
    }
    return result


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
    """The model's GeneEffect of ``lines`` (lines x genes): ``centre + z scale``."""
    position = {m: i for i, m in enumerate(data.lines)}
    z = predict_dependency(
        model, data, torch.tensor([position[m] for m in lines]), config
    ).numpy()
    return pd.DataFrame(
        data.centre + z * data.scale, index=list(lines), columns=data.genes
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


def read_in_line(
    cache: Path, lines: Sequence[str], genes: Sequence[str], fit: Sequence[str]
) -> np.ndarray:
    """The gene-in-line features of ``lines`` (``<cache>/<ModelID>.npy``, genes x
    width on the axis of ``<cache>/genes.json``) on ``genes``: (lines, genes + 1,
    width + 1), float16. An entry is present when all its values are finite. Each
    column is centred and scaled (SD, ``ddof=0``) on the present entries of the
    ``fit`` lines; an absent entry is zero. The last column is the presence flag; the
    last gene row is zero. A scaled value outside the float16 range raises."""
    cache = Path(cache)
    missing = [m for m in lines if not (cache / f"{m}.npy").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} lines have no gene-in-line features in {cache}: "
            f"{missing[:10]}"
        )
    at = {g: i for i, g in enumerate(json.loads((cache / "genes.json").read_text()))}
    index = np.array([at.get(g, -1) for g in genes])
    on_axis = np.flatnonzero(index >= 0)

    def one(model_id: str) -> tuple[np.ndarray, np.ndarray]:
        values = np.load(cache / f"{model_id}.npy", mmap_mode="r")[index[on_axis]]
        values = values.astype(np.float64)
        return values, np.isfinite(values).all(axis=1)

    width = np.load(cache / f"{lines[0]}.npy", mmap_mode="r").shape[1]
    total, squares, count = np.zeros(width), np.zeros(width), 0
    for model_id in fit:
        values, present = one(model_id)
        total += values[present].sum(axis=0)
        squares += np.square(values[present]).sum(axis=0)
        count += int(present.sum())
    mean = total / count
    sd = np.sqrt(np.maximum(squares / count - mean**2, 0.0))
    sd = np.where(sd > 0, sd, 1.0)
    # filled line by line: the whole array exists only in float16
    out = np.zeros((len(lines), len(genes) + 1, width + 1), np.float16)
    for i, model_id in enumerate(lines):
        values, present = one(model_id)
        with np.errstate(over="ignore"):
            out[i, on_axis, :width] = np.where(
                present[:, None], (values - mean) / sd, 0.0
            )
        if not np.isfinite(out[i]).all():
            raise ValueError(
                f"{cache}: a scaled value of {model_id} is outside float16"
            )
        out[i, on_axis, width] = present
    return out


def own_expression(
    expression: pd.DataFrame, lines: Sequence[str], genes: Sequence[str]
) -> np.ndarray:
    """The line's own value of each gene in the scaled expression frame and the
    presence flag: (lines, genes + 1, 2), float16. A gene that is not an expression
    column is absent (zero); the last gene row is zero. A value outside the float16
    range raises."""
    frame = expression.loc[list(lines)].reindex(columns=list(genes))
    out = np.zeros((len(lines), len(genes) + 1, 2), np.float16)
    with np.errstate(over="ignore"):
        out[:, :-1, 0] = frame.fillna(0.0).to_numpy()
    if not np.isfinite(out).all():
        raise ValueError("the expression: a scaled value is outside float16")
    out[:, :-1, 1] = frame.notna().to_numpy()
    return out


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


def target_scale(labelled: pd.DataFrame) -> tuple[float, float]:
    """The mean and the SD (``ddof=0``) of every finite GeneEffect value of
    ``labelled`` (lines x genes): one centre and one scale for every gene."""
    values = labelled.to_numpy(np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()), float(values.std())


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
    target_centre, target_sd = target_scale(labelled.loc[fit])
    targets = torch.full((len(lines), len(genes)), torch.nan)
    targets[[position[m] for m in labelled.index]] = torch.tensor(
        ((labelled - target_centre) / target_sd).to_numpy(np.float32)
    )
    cache = config["cell_embeddings"]
    cells, counts = read_cells(
        None if cache is None else Path(cache), cell_lines, set(fit)
    )
    total = int(counts.sum())
    offsets = np.concatenate([[0], np.cumsum(counts), [total] * len(bulk_lines)])
    with_rows = set(rows["model_id"])
    feature_lines = [m for m in lines if m in with_rows]
    in_line = {}
    if config["line_term"] is not None:
        sl_train = set(rows.loc[rows["split"] == "train", "model_id"])
        train = [m for m in feature_lines if m in sl_train]
        if config["line_term"]["own_expression"]:
            in_line["expression"] = own_expression(expression, feature_lines, genes)
        for name, path in config["line_term"]["gene_in_line"].items():
            in_line[name] = read_in_line(Path(path), feature_lines, genes, train)
    in_line_at = torch.full((len(lines),), -1)
    in_line_at[[position[m] for m in feature_lines]] = torch.arange(len(feature_lines))
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
        centre=target_centre,
        scale=target_sd,
        in_line={n: torch.from_numpy(a).to(device) for n, a in in_line.items()},
        in_line_at=in_line_at.to(device),
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
        "in_line_coverage": {
            n: int(np.count_nonzero(a[:, :, -1])) for n, a in in_line.items()
        },
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
    """The line-input controls of a finished run, from its ``best.pt``; no training.
    With a line term, also the model and the line term on another line's input (the
    line term leaves the backbone and the pair head as they were)."""
    run_dir = Path(config["output_root"]) / config["task"] / run_id
    data, _, _ = load_data(config, torch.device(device))
    model = build_model(config, data)
    model.load_state_dict(
        torch.load(run_dir / "best.pt", map_location=device)["weights"]
    )
    pairs = pair_rows(data)
    scores = controls_named(line_input_controls(model, data, pairs, config))
    if config["line_term"] is not None:
        scores[f"{SCORER}_line_term_other_line"] = line_term_scores(
            model, data, pairs, np.arange(len(data.rows)), config,
            other_lines(data, config).__getitem__,
        )  # fmt: skip
    write_scores(run_dir, data.rows, scores)
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
    rows, top_k, line_term = data.rows, int(config["top_k"]), result["line_term"]
    write_scores(run_dir, rows, {SCORER: result["scores"]})
    write_scores(run_dir, rows, controls_named(result["controls"]))
    sl_line_term = line_record = None
    if line_term is not None:
        line_scores = {"line_term": line_term["scores"],
                       "line_term_other_line": line_term["other_line"]}  # fmt: skip
        write_scores(run_dir, rows, controls_named(line_scores))
        # the model on the mean input, with the line term, and the line term on
        # another line's input
        with_mean = {"mean_input": result["controls"]["mean_input"]} | line_scores
        sl_line_term = {k: sl_metrics(rows, v, top_k) for k, v in with_mean.items()}
        line_record = {k: line_term[k] for k in ("val_recall_top10", "history")}
    sides = {s: sorted(set(rows.loc[rows["split"] == s, "model_id"]))
             for s in ("train", "val", "test")}  # fmt: skip
    labelled = [m for group in sides.values() for m in group
                if m in base.gene_effect.index]  # fmt: skip
    predicted = geneeffect_table(model, data, labelled, config)
    predicted.astype("float32").to_parquet(run_dir / "predictions.parquet")
    metrics = {
        "run_id": run_id,
        "counts": counts,
        "sl": sl_metrics(rows, result["scores"], top_k),
        "sl_line_term": sl_line_term,
        "line_term": line_record,
        "stage_one": stage_one_metrics(
            predicted, base.gene_effect, base.definitions, sides
        ),
        # the gene mean of the fit lines alone: the reference row of the report
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
    """Table 1, SL on the benchmark; Table 2, the line term; Table 3, the stage-one
    numbers; Table 4, the phases of stage two and the line term; Table 5, the epochs
    of stage one."""
    sl, one, counts = metrics["sl"], metrics["stage_one"], metrics["counts"]
    base = metrics["stage_one_base"]
    line_sl, line_term = metrics["sl_line_term"], metrics["line_term"]
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

    def bold(value: float, best: float) -> str:
        return f"**{value:.4f}**" if value == best else f"{value:.4f}"

    line_table = ["**Table 2.** No line term in this config.", ""]
    if line_sl is not None:
        models = (("Gene-knowledge, cell-set model", sl),
                  ("The model on the mean input", line_sl["mean_input"]),
                  ("The model with the line term", line_sl["line_term"]),
                  ("The model and the line term on another line's input",
                   line_sl["line_term_other_line"]))  # fmt: skip
        columns = [(m, s) for m in names for s in ("val", "test")]
        best = {c: max(table[c[1]][c[0]] for _, table in models) for c in columns}
        line_table = [
            "**Table 2.** The line term on the multi-study SL benchmark, mean over "
            "lines. The score with the line term is the model's score in the line "
            "plus the line term's deviation from its mean over the SL training lines "
            "with training rows. The last row reads the model and the line term with "
            "another line's tokens and gene-in-line features (no refit). The model on "
            "the mean input is the context-ablated reference. Higher is better; the "
            "best of each "
            "column is in bold. Validation selected the epoch; test was scored once.",
            "",
            "| Model |" + "".join(
                f" {names[m]}, {'validation' if s == 'val' else 'test'} |"
                for m, s in columns
            ),
            "| --- |" + " ---: |" * len(columns),
            *(f"| {name} | " + " | ".join(bold(table[s][m], best[(m, s)])
                                          for m, s in columns) + " |"
              for name, table in models),
            "",
        ]  # fmt: skip
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
        *line_table,
        "**Table 3.** Stage-one numbers on the SL lines with GeneEffect ("
        + " / ".join(str(one[s]["lines"]) for s in ("val", "test"))
        + " lines, validation / test; out of sample for stage one), as every "
        "stage-one reader is scored: the predicted GeneEffect as a residual over the "
        "training gene mean. The second row is the fit lines' gene mean alone. Higher "
        "is better. They decide nothing.",
        "",
        "| Model |" + "".join(f" {name}, {s} |" for _, name in keys
                              for s in ("validation", "test")),
        "| --- |" + " ---: |" * (2 * len(keys)),
        *(f"| {name} | " + " | ".join(cell(table[s].get(k)) for k, _ in keys
                                      for s in ("val", "test")) + " |"
          for name, table in (("Gene-knowledge, cell-set model", one),
                              ("Gene mean of the fit lines", base))),
        "",
        "**Table 4.** Stage two: validation recall in the top 10% of each phase at its "
        "best epoch. The chosen phase is in bold. The line term, when the config has "
        "one, trains last on the chosen phase.",
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
        *([] if line_term is None else
          [f"| line term | — | {epochs(line_term['history'], 'val_recall_top10', max)}"
           f" | {line_term['val_recall_top10']:.4f} |"]),
        "",
        "**Table 5.** Stage one: the GeneEffect loss on the SL training lines (out of "
        "sample; lower is better). Epoch 0 is the untrained model, which predicts one "
        "value for every gene.",
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
