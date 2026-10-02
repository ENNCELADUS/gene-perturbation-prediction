"""Joint training loop: GeneEffect every update, anchor response every few updates.

Validation runs once per epoch on the validation lines' GeneEffect rows only;
``best.pt`` and early stopping follow ``val_geneeffect_loss``.
"""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from accelerate.utils import DistributedDataParallelKwargs
import torch

from src.data.batches import ResponseForwardBatch
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
from src.training.sampling import make_training_loaders


def make_optimizer(model, config: Mapping[str, Any]) -> torch.optim.AdamW:
    """AdamW with one group each for STATE, the ESM2 adapter and the head."""
    train = config["train"]
    return torch.optim.AdamW(
        [
            {
                "params": list(module.parameters()),
                "lr": train[f"{name}_learning_rate"],
                "name": name,
            }
            for name, module in (
                ("state", model.backbone.state),
                ("adapter", model.backbone.perturbations),
                ("head", model.head),
            )
        ],
        weight_decay=train["weight_decay"],
    )


def train_update(
    model, optimizer, dependency_batch, response_batch, config, accelerator
):
    """One forward through the wrapped model and one optimizer update."""
    dependency_batch = dependency_batch.to(accelerator.device)
    response = None
    if response_batch is not None:
        response_batch = response_batch.to(accelerator.device)
        response = ResponseForwardBatch(
            response_batch.control_hvg, response_batch.genes
        )
    with accelerator.autocast():
        output = model(dependency_batch.conditions, response=response)
    dependency_loss = geneeffect_loss(output.delta_hat, dependency_batch.residual)
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

    ``restored`` is a loaded ``last.pt``; its optimizer, scaler and per-rank RNG
    state continue exactly where that epoch ended.
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
    optimizer = make_optimizer(model, config)
    if restored is not None:
        optimizer.load_state_dict(restored["optimizer"])
    # Replay-free updates and STATE's unused released decoder leave parameters
    # without gradients; DDP must discover them per step.
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
    for epoch in range(state.next_epoch, train["max_epochs"]):
        if state.bad_epochs >= train["patience"]:
            break
        model.train()
        loader, responses = make_training_loaders(inputs, config, epoch, accelerator)
        for batch in loader:
            replay = state.global_step % train["response_interval"] == 0
            metrics = train_update(
                model,
                optimizer,
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
            model, inputs, config, split="train", accelerator=accelerator
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
                    state,
                    config,
                    preprocessing,
                    accelerator,
                )
    return state
