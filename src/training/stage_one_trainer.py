"""The stage-one training loop (PIE stage-one design §4.1; protocol §16), plain torch on
one device for ``.venv-pie``: residual-SD MSE plus ``train.dependency_weight`` times the
dependency call's cross-entropy on selective genes, AdamW with the shared schedule and
optional EMA weights (then the ones validated and saved). Validation runs before the
first update (epoch 0, the base) and after every epoch; an epoch is a ``best.pt``
candidate only when ``SELECTOR`` is defined, otherwise telemetry."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from src.data.splits import FixedSplit, assert_same_membership, membership
from src.model.losses import dependency_logits
from src.training.checkpoint import (
    TrainState,
    capture_rng_state,
    load_checkpoint,
    record_validation,
    restore_rng_state,
)
from src.training.schedule import make_scheduler

log = logging.getLogger(__name__)

#: The validation metric ``best.pt`` and early stopping follow: the per-line residual
#: Spearman over the selective genes, as stage two reads each held-out line on its own
#: (protocol §6).
SELECTOR = "val_line_spearman"


@dataclass
class TrainingData:
    """What the loop reads: ``batch(genes, lines)`` builds a ``PairBatch`` from
    panel-gene and line positions; ``targets`` (lines x panel genes, on the device)
    is NaN wherever a pair is not a training label; ``train_lines`` are the line
    positions fitted. ``m_g`` and ``sigma_g`` (the run's genes, on the device) turn a
    target back into GeneEffect for the dependency call, which ``selective`` limits to
    the selective genes."""

    batch: Callable[[Tensor, Tensor], Any]
    targets: Tensor
    train_lines: Tensor
    n_genes: int
    m_g: Tensor
    sigma_g: Tensor
    selective: Tensor


def autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def blocks(
    count: int, size: int, generator: torch.Generator, device: torch.device
) -> list[Tensor]:
    order = torch.randperm(count, generator=generator).to(device)
    return list(torch.split(order, size))


def epoch_steps(
    data: TrainingData, config: Mapping[str, Any], generator: torch.Generator
) -> Iterator[tuple[Tensor, Tensor]]:
    """Every gene block with every line block once, in a random order."""
    device = data.targets.device
    train = config["train"]
    genes = blocks(data.n_genes, train["genes_per_step"], generator, device)
    lines = [
        data.train_lines[block]
        for block in blocks(
            len(data.train_lines), train["lines_per_step"], generator, device
        )
    ]
    pairs = [(g, c) for g in range(len(genes)) for c in range(len(lines))]
    for index in torch.randperm(len(pairs), generator=generator).tolist():
        g, c = pairs[index]
        yield genes[g], lines[c]


def updates_per_epoch(data: TrainingData, config: Mapping[str, Any]) -> int:
    train = config["train"]
    return math.ceil(data.n_genes / train["genes_per_step"]) * math.ceil(
        len(data.train_lines) / train["lines_per_step"]
    )


def masked_mse(z: Tensor, target: Tensor) -> Tensor:
    """Mean squared error over the finite targets (a block has at least one)."""
    keep = torch.isfinite(target)
    error = torch.where(keep, z.float() - torch.nan_to_num(target), 0.0)
    return error.square().sum() / keep.sum()


def dependency_term(
    z: Tensor, target: Tensor, genes: Tensor, data: TrainingData
) -> Tensor:
    """The dependency call's cross-entropy over the finite targets of selective
    genes in a (genes x lines) block, in GeneEffect units; zero without such a pair.
    Masked rather than indexed, so a step's shapes are static and it never waits on
    the device."""
    keep = torch.isfinite(target) & data.selective[genes].unsqueeze(1)
    sigma = data.sigma_g[genes].unsqueeze(1)
    logit, label = dependency_logits(
        z.float() * sigma,
        torch.nan_to_num(target) * sigma,
        sigma,
        data.m_g[genes].unsqueeze(1),
    )
    loss = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    return torch.where(keep, loss, 0.0).sum() / keep.sum().clamp(min=1)


def training_step(
    model: nn.Module,
    batch: Any,
    target: Tensor,
    genes: Tensor,
    data: TrainingData,
    weight: float,
) -> tuple[Tensor, Tensor]:
    """One gathered block's loss and its dependency term: the part compiled on CUDA.
    The batch and target are gathered outside it, as a CUDA graph copies every
    array it reads on each replay (the resident deltas are many GB)."""
    with autocast(target.device):
        z, _ = model(batch)
    call = dependency_term(z, target, genes, data)
    return masked_mse(z, target) + weight * call, call


def _checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def load_strict(
    path: Path, model: nn.Module, config: Mapping[str, Any], split: FixedSplit
) -> dict[str, Any]:
    """A stage-one checkpoint's payload, its weights loaded strictly into ``model``;
    refused under another config or split membership."""
    payload = load_checkpoint(path)
    if payload["config"] != json.loads(json.dumps(config)):
        raise ValueError(f"{path} was written under another config; use a new run id")
    assert_same_membership(payload["membership"], split)
    model.load_state_dict(payload["weights"])
    return payload


@dataclass
class FitResult:
    best: Path
    state: TrainState
    history: list[dict[str, Any]]


def fit(
    model: nn.Module,
    data: TrainingData,
    config: Mapping[str, Any],
    run_dir: Path,
    split: FixedSplit,
    validate: Callable[[nn.Module], dict[str, Any]],
) -> FitResult:
    """Train ``model`` (on the targets' device), validating each epoch with
    ``validate`` (a module -> metrics with ``SELECTOR``); resumes from
    ``run_dir/last.pt`` when it exists."""
    device = data.targets.device
    train = config["train"]
    record = json.loads(json.dumps(config))
    run_dir.mkdir(parents=True, exist_ok=True)
    # On CUDA the step runs as CUDA graphs and AdamW as one fused kernel: the model is
    # small, so per-kernel launch cost, not arithmetic, sets the step time. A PIE
    # trunk stays eager: its RMSNorm breaks the graph, and compiled it gained nothing.
    compiled = device.type == "cuda" and config["model"]["trunk"] is None
    step = (
        torch.compile(training_step, mode="reduce-overhead", dynamic=False)
        if compiled
        else training_step
    )
    finite = torch.isfinite(data.targets)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train["learning_rate"],
        weight_decay=train["weight_decay"],
        fused=device.type == "cuda",
    )
    scheduler = make_scheduler(optimizer, config, updates_per_epoch(data, config))
    ema = None
    if train["ema_decay"] is not None:
        ema = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(float(train["ema_decay"])),
            use_buffers=True,
        )
    validated = model if ema is None else ema.module
    state, history = TrainState(), []
    generator = torch.Generator().manual_seed(int(config["seed"]))
    last, best = run_dir / "last.pt", run_dir / "best.pt"
    if last.is_file():
        payload = load_strict(last, model, record, split)
        if ema is not None:
            ema.load_state_dict(payload["ema"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        state = TrainState(**payload["train_state"])
        history = json.loads(payload["history"])
        generator.set_state(payload["generator"])
        restore_rng_state(payload["rng"], device)
        log.info("resumed at epoch %d", state.next_epoch)

    def save_best() -> None:
        _checkpoint(
            best,
            {
                "weights": validated.state_dict(),
                "epoch": state.best_epoch,
                "score": state.best_score,
                "config": record,
                "membership": membership(split),
            },
        )

    if last.is_file() and state.best_epoch >= 0:
        # last.pt is written before best.pt: a crash between them leaves best.pt
        # behind the state, whose weights (the last epoch's) are the best ones.
        saved = load_checkpoint(best) if best.is_file() else None
        if saved is None or (saved["epoch"], saved["score"]) != (
            state.best_epoch,
            state.best_score,
        ):
            if state.best_epoch != state.next_epoch - 1:
                raise RuntimeError(
                    f"{best} does not hold the recorded best epoch {state.best_epoch}"
                )
            save_best()
            log.info("best.pt rewritten for epoch %d", state.best_epoch)

    def save_last() -> None:
        _checkpoint(
            last,
            {
                "weights": model.state_dict(),
                "ema": None if ema is None else ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_state": asdict(state),
                "history": json.dumps(history),
                "generator": generator.get_state(),
                "rng": capture_rng_state(device),
                "config": record,
                "membership": membership(split),
            },
        )

    def judge(epoch: int, metrics: dict[str, Any]) -> None:
        """A defined score is a candidate; an undefined one (a prediction constant
        within every gene) never is, and after epoch 0 counts as a bad epoch."""
        score = metrics[SELECTOR]
        candidate = score is not None and math.isfinite(float(score))
        improved = candidate and record_validation(state, metrics, epoch, key=SELECTOR)
        if not candidate:
            state.next_epoch = epoch + 1
            state.bad_epochs += int(epoch > 0)
        history.append({"epoch": epoch, "candidate": candidate, **metrics})
        save_last()
        if improved:
            save_best()
        log.info(
            "epoch %d: %s%s",
            epoch,
            ", ".join(
                f"{key} {value:.4g}" if isinstance(value, float) else f"{key} {value}"
                for key, value in metrics.items()
                if key
                in (
                    "train_loss",
                    "train_dependency",
                    "epoch_seconds",
                    "val_line_spearman",
                    "val_selective_spearman",
                    "val_within_gene_r2",
                )
            ),
            " (best)" if improved else "",
        )

    if state.next_epoch == 0:
        judge(0, validate(validated))
    while (
        state.next_epoch <= train["max_epochs"] and state.bad_epochs < train["patience"]
    ):
        epoch = state.next_epoch
        model.train()
        started = time.monotonic()
        losses, dependency = [], []
        steps = list(epoch_steps(data, config, generator))
        # One device sync per epoch: a block without a training label is skipped.
        labelled = torch.stack(
            [finite[lines][:, genes].any() for genes, lines in steps]
        ).tolist()
        for (genes, lines), has_label in zip(steps, labelled):
            if not has_label:
                continue
            if compiled:
                torch.compiler.cudagraph_mark_step_begin()
            loss, call = step(
                model,
                data.batch(genes, lines),
                data.targets[lines][:, genes].T,
                genes,
                data,
                train["dependency_weight"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update_parameters(model)
            state.global_step += 1
            # Cloned: a CUDA graph overwrites its outputs on the next replay.
            losses.append(loss.detach().clone())
            dependency.append(call.detach().clone())
        metrics = {
            "train_loss": torch.stack(losses).double().mean().item()
            if losses
            else None,
            "train_dependency": (
                torch.stack(dependency).double().mean().item() if dependency else None
            ),
            # The updates alone, without validation: a streamed input's I/O shows here.
            "epoch_seconds": time.monotonic() - started,
        }
        metrics |= validate(validated)
        judge(epoch, metrics)
    if not best.is_file():
        raise RuntimeError(f"{run_dir}: no epoch was a candidate for best.pt")
    return FitResult(best=best, state=state, history=history)


@torch.no_grad()
def predict(
    model: nn.Module,
    batch: Callable[[Tensor, Tensor], Any],
    lines: Tensor,
    n_genes: int,
    config: Mapping[str, Any],
    viability: np.ndarray | None = None,
) -> np.ndarray:
    """``z`` (panel genes x ``lines``) in eval mode, in chunks; ``viability``, when
    given (lines x panel genes x 32, fp16, e.g. a memory map), receives V."""
    device = lines.device
    chunk_lines, chunk_genes = (
        config["eval"]["lines_per_chunk"],
        config["eval"]["genes_per_chunk"],
    )
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    out = np.empty((n_genes, len(lines)), dtype=np.float32)
    try:
        for start in range(0, len(lines), chunk_lines):
            block = lines[start : start + chunk_lines]
            for first in range(0, n_genes, chunk_genes):
                genes = torch.arange(
                    first, min(first + chunk_genes, n_genes), device=device
                )
                with autocast(device):
                    z, v = model(batch(genes, block))
                out[first : first + len(genes), start : start + len(block)] = (
                    z.float().cpu().numpy()
                )
                if viability is not None:
                    viability[
                        start : start + len(block), first : first + len(genes)
                    ] = v.float().transpose(0, 1).cpu().numpy().astype(np.float16)
    finally:
        for module, training in modes:
            module.training = training
    return out
