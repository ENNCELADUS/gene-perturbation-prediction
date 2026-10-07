"""Joint training loop: one GeneEffect objective every update, optional response replay.

The objective is ``train.objective`` (``src.model.losses``); response replay runs
every ``response_interval`` updates only when ``response_weight`` is positive. The
learning rate warms up linearly over ``warmup_epochs`` and decays by cosine to zero
at ``max_epochs``, stepped per update. Validation runs once per epoch on the
validation lines' GeneEffect rows only; ``best.pt`` and early stopping follow
``val_selective_spearman`` (higher wins). The ``train_eval_`` diagnostic scores a
fixed subset of supervised training lines of the validation split's size, so the
two curves are comparable.
"""

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

from accelerate.utils import DistributedDataParallelKwargs
import numpy as np
import torch

from src.data.batches import ResponseForwardBatch
from src.data.datasets import DependencyDataset
from src.data.prepared import PreparedInputs
from src.eval.geneeffect import evaluate_model
from src.model.losses import geneeffect_loss
from src.model.normalization import fit_startup_standardizer
from src.model.response import response_loss
from src.training.checkpoint import (
    TrainState,
    record_validation,
    restore_rng_state,
    save_checkpoint,
)
from src.training.sampling import dependency_loader, response_stream

TRAINING_DIAGNOSTIC_LINES = 27


def training_diagnostic_lines(inputs: PreparedInputs) -> tuple[str, ...]:
    """The supervised training lines scored by the per-epoch ``train_eval_`` metrics.

    ``TRAINING_DIAGNOSTIC_LINES`` lines (all of them if fewer) drawn once from the
    sorted supervised training lines with ``np.random.default_rng(0)``, returned
    sorted. Depends only on the split, so a resumed run scores the same lines.
    """
    lines = sorted(inputs.split.supervised_train)
    chosen = np.random.default_rng(0).choice(
        len(lines), size=min(TRAINING_DIAGNOSTIC_LINES, len(lines)), replace=False
    )
    return tuple(lines[index] for index in sorted(chosen))


def trains_state(model, config: Mapping[str, Any]) -> bool:
    """Whether STATE's own weights train: it is used and ``state_mode`` is trainable."""
    return model.uses_state and config["train"]["state_mode"] == "trainable"


def make_optimizer(model, config: Mapping[str, Any]) -> torch.optim.AdamW:
    """AdamW over the trained modules of the unwrapped model; the rest is frozen.

    The head always trains; the ESM2 adapter trains when the model uses STATE;
    STATE itself only when ``trains_state``. Every other module gets
    ``requires_grad`` False, so neither the optimizer nor DDP holds it; parameters
    frozen at build (STATE's unused GPT-2 token and position tables) stay out too.
    """
    train = config["train"]
    modules = {
        "state": model.backbone.state,
        "adapter": model.backbone.perturbations,
        "head": model.head,
    }
    trained = {
        "state": trains_state(model, config),
        "adapter": model.uses_state,
        "head": True,
    }
    for name, module in modules.items():
        if not trained[name]:
            module.requires_grad_(False)
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in module.parameters() if p.requires_grad],
                "lr": train[f"{name}_learning_rate"],
                "name": name,
            }
            for name, module in modules.items()
            if trained[name]
        ],
        weight_decay=train["weight_decay"],
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer, config: Mapping[str, Any], updates_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over ``warmup_epochs`` epochs of updates, then cosine to zero.

    Stepped once per update: the last warmup update runs at the full rate and the
    rate reaches zero at the end of ``max_epochs``.
    """
    train = config["train"]
    warmup = train["warmup_epochs"] * updates_per_epoch
    decay = max(1, train["max_epochs"] * updates_per_epoch - warmup)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / decay)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_update(
    model, optimizer, scheduler, dependency_batch, response_batch, config, accelerator
):
    """One forward through the wrapped model, one optimizer and scheduler step."""
    dependency_batch = dependency_batch.to(accelerator.device)
    response = None
    if response_batch is not None:
        response_batch = response_batch.to(accelerator.device)
        response = ResponseForwardBatch(
            response_batch.control_hvg, response_batch.genes
        )
    with accelerator.autocast():
        output = model(dependency_batch.conditions, response=response)
    # The stack: the prior's offset plus the head's output, in residual units.
    prediction = output.delta_hat + dependency_batch.prior
    dependency_loss = geneeffect_loss(
        prediction,
        dependency_batch.residual,
        dependency_batch.residual_scale,
        objective=config["train"]["objective"],
        gene_index=dependency_batch.conditions.gene_index,
        selective=dependency_batch.selective,
    )
    total = dependency_loss
    replay_loss = torch.zeros_like(dependency_loss)
    if response is not None:
        # Batches hold equal condition counts per anchor, so the plain mean is
        # balanced across anchors.
        replay_loss = torch.stack(
            [
                response_loss(predicted, observed, control)
                for predicted, observed, control in zip(
                    output.response_predicted,
                    response_batch.observed_hvg,
                    response_batch.control_hvg,
                    strict=True,
                )
            ]
        ).mean()
        total = total + config["train"]["response_weight"] * replay_loss
    if not torch.isfinite(total):
        raise ValueError("non-finite joint loss")
    accelerator.backward(total)
    norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise ValueError("non-finite gradient norm")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    values = torch.stack([dependency_loss, total, replay_loss]).detach()
    values = accelerator.reduce(values, reduction="mean").cpu().tolist()
    return {
        "train_geneeffect_loss": values[0],
        "train_total_loss": values[1],
        "train_response_loss": values[2] if response is not None else None,
    }


def _log(run_dir: Path, record: Mapping[str, Any], accelerator) -> None:
    if accelerator.is_main_process:
        line = json.dumps(record, allow_nan=False)
        with (run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)


def fit(
    model,
    inputs: PreparedInputs,
    config: Mapping[str, Any],
    run_dir: Path,
    accelerator,
    *,
    restored: Mapping[str, Any] | None = None,
) -> TrainState:
    """Train a fresh or restored model to early stopping or ``max_epochs``.

    ``restored`` is a loaded ``last.pt``; its optimizer, scheduler, scaler and
    per-rank RNG state continue exactly where that epoch ended.
    """
    train = config["train"]
    run_dir = Path(run_dir)
    state = TrainState() if restored is None else TrainState(**restored["train_state"])
    if restored is not None and restored["world_size"] != accelerator.num_processes:
        raise ValueError(
            "resume changes the number of processes and so the effective batch; "
            "start a new run"
        )
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
    model.to(accelerator.device)
    if restored is None:
        fit_startup_standardizer(
            model,
            inputs,
            batch_size=train["dependency_batch_size"],
            accelerator=accelerator,
        )
    dependency = DependencyDataset(inputs, "train", device=accelerator.device)
    updates_per_epoch = len(dependency_loader(dependency, config, 0, accelerator))
    network = model  # unwrapped: DDP does not forward attribute access
    optimizer = make_optimizer(model, config)
    # Never through ``accelerator.prepare``: it would step once per process.
    scheduler = make_scheduler(optimizer, config, updates_per_epoch)
    if restored is not None:
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
    # Replay-free updates, disabled head blocks and STATE's unused released
    # decoder leave parameters without gradients; DDP must discover them per step.
    if accelerator.ddp_handler is None:
        accelerator.ddp_handler = DistributedDataParallelKwargs()
    accelerator.ddp_handler.find_unused_parameters = True
    model, optimizer = accelerator.prepare(model, optimizer)
    if restored is not None:
        if accelerator.scaler is not None:
            accelerator.scaler.load_state_dict(restored["scaler"])
        restore_rng_state(
            restored["rng_states"][accelerator.process_index], accelerator.device
        )
    preprocessing = inputs.preprocessing_state()
    diagnostic_lines = training_diagnostic_lines(inputs)
    for epoch in range(state.next_epoch, train["max_epochs"]):
        if state.bad_epochs >= train["patience"]:
            break
        model.train()
        if not trains_state(network, config):
            network.backbone.state.eval()  # a frozen STATE runs without dropout
        loader = dependency_loader(dependency, config, epoch, accelerator)
        responses = response_stream(inputs, config, epoch, accelerator)
        for batch in loader:
            replay = (
                responses is not None
                and state.global_step % train["response_interval"] == 0
            )
            metrics = train_update(
                model,
                optimizer,
                scheduler,
                batch,
                next(responses) if replay else None,
                config,
                accelerator,
            )
            state.global_step += 1
            _log(
                run_dir,
                {"epoch": epoch, "global_step": state.global_step, **metrics},
                accelerator,
            )
        train_metrics = evaluate_model(
            model,
            inputs,
            config,
            split="train",
            accelerator=accelerator,
            lines=diagnostic_lines,
        ).metrics
        validation = evaluate_model(
            model, inputs, config, split="val", accelerator=accelerator
        ).metrics
        improved = record_validation(state, validation, epoch)
        _log(
            run_dir,
            {
                "epoch": epoch,
                "global_step": state.global_step,
                **train_metrics,
                **validation,
            },
            accelerator,
        )
        for name, write in (("best.pt", improved), ("last.pt", True)):
            if write:
                save_checkpoint(
                    run_dir / name,
                    model,
                    optimizer,
                    scheduler,
                    state,
                    config,
                    preprocessing,
                    accelerator,
                )
    return state
