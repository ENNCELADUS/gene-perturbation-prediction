"""One joint dependency/response loop with epoch-end validation and resume."""

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
from src.model.response import response_terms
from src.training.checkpoint import (
    TrainState,
    record_validation,
    restore_rng_state,
    save_checkpoint,
)
from src.training.distributed import (
    raise_rank_errors,
    require_distinct_devices,
    run_rank_zero_or_raise,
)
from src.training.sampling import make_training_loaders


def make_optimizer(model, config: Mapping[str, Any]) -> torch.optim.AdamW:
    train = config["train"]
    groups = [
        {
            "params": [p for p in module.parameters() if p.requires_grad],
            "lr": train[f"{name}_learning_rate"],
            "name": name,
        }
        for name, module in (
            ("state", model.backbone.state_adapter),
            ("adapter", model.backbone.perturbations),
            ("head", model.head),
        )
    ]
    assigned = [id(p) for group in groups for p in group["params"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != {
        id(p) for p in model.parameters() if p.requires_grad
    }:
        raise ValueError("STATE, adapter and head must partition trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=train["weight_decay"])


def train_update(
    model, optimizer, dependency_batch, response_batch, config, accelerator
):
    """One wrapped forward and one optimizer update; absent replay metrics are null."""
    error = None
    try:
        dependency_batch = dependency_batch.to(accelerator.device)
        if response_batch is not None:
            response_batch = response_batch.to(accelerator.device)
        response_input = (
            None
            if response_batch is None
            else ResponseForwardBatch(response_batch.controls_tx1, response_batch.genes)
        )
        with accelerator.autocast():
            output = model(dependency_batch.conditions, response=response_input)
        dep_loss = geneeffect_loss(
            output.delta_hat, dependency_batch.residual, dependency_batch.valid
        )
        total = dep_loss
        terms = None
        if response_batch is not None:
            terms = response_terms(output.response_predicted, response_batch)
            total = total + config["train"]["response_weight"] * (
                terms["mean_delta_mse"].mean() + terms["energy_distance"].mean()
            )
        if not bool(torch.isfinite(total)):
            raise ValueError("non-finite joint loss")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    raise_rank_errors(accelerator, "training forward/loss", error)
    accelerator.backward(total)
    norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
    raise_rank_errors(
        accelerator,
        "training gradients",
        None if bool(torch.isfinite(norm)) else "non-finite gradient norm",
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    values = torch.stack(
        [
            dep_loss.detach(),
            total.detach(),
            dep_loss.new_zeros(())
            if terms is None
            else terms["mean_delta_mse"].mean().detach(),
            dep_loss.new_zeros(())
            if terms is None
            else terms["energy_distance"].mean().detach(),
        ]
    )
    values = accelerator.reduce(values, reduction="mean").cpu().tolist()
    return {
        "train_geneeffect_loss": values[0],
        "train_total_loss": values[1],
        "train_response_mean_delta_mse": None if terms is None else values[2],
        "train_response_energy_distance": None if terms is None else values[3],
        "train_response_loss": None if terms is None else values[2] + values[3],
        "train_dependency_rows": len(dependency_batch.residual)
        * accelerator.num_processes,
        "train_response_rows": 0
        if response_batch is None
        else len(response_batch.genes) * accelerator.num_processes,
    }


def _log(run_dir, record, accelerator):
    def write():
        line = json.dumps(record, allow_nan=False)
        with (run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    run_rank_zero_or_raise(accelerator, "metrics logging", write)


def fit(
    model,
    inputs: PreparedInputs,
    config: Mapping[str, Any],
    run_dir: Path,
    accelerator,
    *,
    restored: Mapping[str, Any] | None = None,
) -> TrainState:
    """Train a constructed fresh/restored model, returning completed-epoch state.

    The caller constructs Accelerator with the configured precision and restores
    model/preprocessing through build_joint_model/load_inputs before passing saved
    checkpoint data here. Only model and optimizer go through prepare: the training
    loader already has a rank-specific DistributedSampler.
    """
    train = config["train"]
    state = TrainState() if restored is None else TrainState(**restored["train_state"])
    if restored is not None:
        if restored["world_size"] != accelerator.num_processes:
            raise ValueError("resume world size changed; start a fresh run")
        if restored["amp_state"]["mixed_precision"] != accelerator.mixed_precision:
            raise ValueError("resume mixed precision differs from checkpoint")
    require_distinct_devices(accelerator)
    run_dir = Path(run_dir)
    run_rank_zero_or_raise(
        accelerator, "run directory", lambda: run_dir.mkdir(parents=True, exist_ok=True)
    )
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
    # STATE auxiliary parameters and conditional replay require dynamic unused
    # parameter discovery. Configure this before Accelerate wraps the model.
    if accelerator.ddp_handler is None:
        accelerator.ddp_handler = DistributedDataParallelKwargs()
    accelerator.ddp_handler.find_unused_parameters = True
    accelerator.ddp_handler.static_graph = False
    model, optimizer = accelerator.prepare(model, optimizer)
    if restored is not None:
        scaler = restored["amp_state"]["scaler"]
        if scaler is not None:
            if accelerator.scaler is None:
                raise ValueError("checkpoint AMP scaler is unavailable")
            accelerator.scaler.load_state_dict(scaler)
        restore_rng_state(
            restored["rng_states"][accelerator.process_index], accelerator.device
        )
    preprocessing = inputs.preprocessing_state()
    for epoch in range(state.next_epoch, train["max_epochs"]):
        if state.bad_epochs >= train["patience"]:
            break
        model.train()
        dependency_loader, response_iterator = make_training_loaders(
            inputs, config, epoch, accelerator
        )
        for dependency_batch in dependency_loader:
            replay = state.global_step % train["response_interval"] == 0
            response_batch = next(response_iterator) if replay else None
            metrics = train_update(
                model, optimizer, dependency_batch, response_batch, config, accelerator
            )
            state.global_step += 1
            _log(
                run_dir,
                {"epoch": epoch, "global_step": state.global_step, **metrics},
                accelerator,
            )
        result = evaluate_model(
            model, inputs, config, split="val", accelerator=accelerator
        )
        improved = record_validation(state, result.metrics, epoch)
        _log(
            run_dir,
            {"epoch": epoch, "global_step": state.global_step, **result.metrics},
            accelerator,
        )
        if improved:
            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                state,
                config,
                preprocessing,
                accelerator,
            )
        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            state,
            config,
            preprocessing,
            accelerator,
        )
    return state
