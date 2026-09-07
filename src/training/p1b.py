"""Single-process response adaptation with epoch-zero selection and resume."""

import math
import os
from pathlib import Path
import torch
from collections import Counter
from src.training.checkpoint import capture_rng_state, restore_rng_state
import json


def _write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def stage2_eligible(initial, best):
    if not math.isfinite(initial) or not math.isfinite(best) or initial <= 0:
        raise ValueError("positive finite initial response loss required")
    return (initial - best) / initial >= 0.01


def save(path, value):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def fit(
    model,
    groups,
    train_epoch,
    objective,
    evaluate,
    directory,
    *,
    identity,
    resume=False,
    max_epochs=50,
    patience=5,
    train_evaluate=None,
    count_exposure=None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    model.eval()
    history = []
    state = {
        "epoch": 0,
        "step": 0,
        "best_epoch": 0,
        "bad_epochs": 0,
        "identity": identity,
    }

    def checkpoint():
        return {
            **state,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng_state(device),
            "history": history,
            "budget": {"max_epochs": max_epochs, "patience": patience},
        }

    if resume:
        restored = torch.load(
            directory / "last.pt", map_location="cpu", weights_only=False
        )
        if restored["identity"] != identity:
            raise ValueError("checkpoint input identity mismatch")
        model.load_state_dict(restored["model_state"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        restore_rng_state(restored["rng"], device)
        state = {k: restored[k] for k in (*state, "best_loss")}
        history = restored["history"]
    else:
        if (directory / "last.pt").exists():
            raise FileExistsError("run exists; use resume")
        try:
            metrics = evaluate()
            initial_loss = float(metrics["response_loss"])
            if not math.isfinite(initial_loss):
                raise ValueError("nonfinite epoch-zero validation")
        except Exception as exc:
            _write_json(
                directory / "training.json",
                {"status": "failed", "error": str(exc), **state},
            )
            raise
        state["best_loss"] = initial_loss
        history.append({"epoch": 0, "step": 0, "val": metrics})
        save(directory / "best.pt", checkpoint())
        save(directory / "last.pt", checkpoint())
    _write_json(directory / "training.json", {"status": "running", **state})
    try:
        while state["epoch"] < max_epochs and state["bad_epochs"] < patience:
            epoch = state["epoch"] + 1
            start_step = state["step"]
            exposure = Counter()
            model.eval()
            before = [[p.detach().clone() for p in group["params"]] for group in groups]
            for batch in train_epoch(epoch - 1):
                if count_exposure is not None:
                    exposure.update(count_exposure(batch))
                optimizer.zero_grad(set_to_none=True)
                loss = objective(batch)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]],
                    1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                state["step"] += 1
            metrics = evaluate()
            loss = float(metrics["response_loss"])
            if not math.isfinite(loss):
                raise ValueError("nonfinite validation loss")
            improvement = loss < state["best_loss"]
            state["epoch"] = epoch
            if improvement:
                state.update(best_loss=loss, best_epoch=epoch, bad_epochs=0)
            else:
                state["bad_epochs"] += 1
            updates = {
                g["name"]: math.sqrt(
                    sum(
                        float((p.detach() - b).square().sum())
                        for p, b in zip(g["params"], old)
                    )
                )
                for g, old in zip(groups, before)
            }
            row = {
                "epoch": epoch,
                "step": state["step"],
                "val": metrics,
                "updates": updates,
                "epoch_updates": state["step"] - start_step,
                "condition_exposures": sum(exposure.values()),
                "exposures_by_anchor": dict(exposure),
            }
            if train_evaluate is not None:
                row["train"] = train_evaluate()
            history.append(row)
            if improvement:
                save(directory / "best.pt", checkpoint())
            save(directory / "last.pt", checkpoint())
            _write_json(directory / "history.json", history)
            print(
                f"epoch={epoch} step={state['step']} val_response={loss:.8g} "
                f"patience={state['bad_epochs']}",
                flush=True,
            )
        _write_json(directory / "history.json", history)
        _write_json(directory / "training.json", {"status": "completed", **state})
    except Exception as exc:
        _write_json(
            directory / "training.json",
            {"status": "failed", "error": str(exc), **state},
        )
        raise
    return state
