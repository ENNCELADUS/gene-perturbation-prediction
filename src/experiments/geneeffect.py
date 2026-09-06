"""Compose cache-only joint training and independent checkpoint evaluation."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import TYPE_CHECKING

from src.experiments.config import load_config

if TYPE_CHECKING:
    from src.eval.geneeffect import EvalResult


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _set_status(run_dir: Path, status_key: str, status: str, **details) -> None:
    path = run_dir / "run.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    record[status_key] = {"status": status, **details}
    _write_json(path, record)


def _restore_model(saved, inputs):
    from src.model.initialization import build_joint_model

    return build_joint_model(
        saved["config"],
        inputs,
        **{
            key: saved[key]
            for key in (
                "architecture",
                "model_state",
                "projection_state",
                "normalization_state",
            )
        },
    )


def run_training(
    config_path: Path, *, run_id: str | None = None, resume: Path | None = None
) -> Path:
    """Run or resume training; testing/export is a separate command and status."""
    if (run_id is None) == (resume is None):
        raise ValueError("require exactly one of run_id and resume")
    if run_id is not None and (
        Path(run_id).name != run_id or run_id in {"", ".", ".."}
    ):
        raise ValueError("run_id must be one directory name")
    config = load_config(config_path)
    import accelerate
    from accelerate import Accelerator
    import numpy as np
    import torch
    from src.data.prepared import load_inputs
    from src.model.initialization import build_joint_model
    from src.training.checkpoint import load_checkpoint
    from src.training.distributed import raise_rank_errors, run_rank_zero_or_raise
    from src.training.trainer import fit

    saved = load_checkpoint(resume) if resume is not None else None
    if saved is not None:
        if config != saved["config"]:
            raise ValueError(
                "supplied config conflicts with checkpoint saved configuration"
            )
        config = saved["config"]
    run_dir = (
        Path(resume).parent
        if resume is not None
        else Path(config["output_root"]) / run_id
    )
    accelerator = Accelerator(mixed_precision=config["precision"])

    def initialize():
        if saved is None:
            run_dir.mkdir(parents=True, exist_ok=False)
            _write_json(
                run_dir / "run.json",
                {
                    "revision": _revision(),
                    "seeds": config["seeds"],
                    "inputs": {
                        "prepared_root": config["prepared_root"],
                        **config["paths"],
                    },
                    "config": config,
                    "environment": {
                        "python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "accelerate": accelerate.__version__,
                        "world_size": accelerator.num_processes,
                        "precision": accelerator.mixed_precision,
                        "device": str(accelerator.device),
                    },
                    "training": {"status": "running"},
                    "evaluation": {"status": "not_started"},
                },
            )
            import yaml

            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False)
            )
        else:
            _set_status(run_dir, "training", "running", resumed_from=str(resume))

    run_rank_zero_or_raise(accelerator, "initialize run", initialize)
    try:
        error = None
        try:
            # Seed BEFORE constructing a fresh model, including its new adapters.
            random.seed(config["seeds"]["train"])
            np.random.seed(config["seeds"]["train"])
            torch.manual_seed(config["seeds"]["train"])
            inputs = load_inputs(
                config, preprocessing=None if saved is None else saved["preprocessing"]
            )
            model = (
                build_joint_model(config, inputs)
                if saved is None
                else _restore_model(saved, inputs)
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if accelerator.num_processes == 1:
                raise
        raise_rank_errors(accelerator, "construct prepared model", error)
        state = fit(model, inputs, config, run_dir, accelerator, restored=saved)
        run_rank_zero_or_raise(
            accelerator,
            "complete training",
            lambda: _set_status(run_dir, "training", "completed", **asdict(state)),
        )
    except Exception as exc:
        if accelerator.is_main_process:
            _set_status(
                run_dir,
                "training",
                "failed",
                phase="training",
                error_type=type(exc).__name__,
                message=str(exc),
            )
        raise
    return run_dir


def export_evaluation(result: EvalResult, out_dir: Path) -> None:
    """Write named prediction columns and ordinary scalar/per-unit artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    for name in ("per_line", "per_gene", "response"):
        getattr(result, name).to_csv(out_dir / f"{name}.csv", index=False)
    _write_json(out_dir / "metrics.json", result.metrics)


def evaluate_checkpoint(checkpoint: Path, *, split: str) -> EvalResult:
    """Restore saved preprocessing and model; never fit or take optimizer steps."""
    if split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    from accelerate import Accelerator
    from src.data.prepared import load_inputs
    from src.eval.geneeffect import evaluate_model
    from src.training.checkpoint import load_checkpoint

    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent
    _set_status(
        run_dir, "evaluation", "running", split=split, checkpoint=str(checkpoint)
    )
    phase = "evaluation"
    try:
        saved = load_checkpoint(checkpoint)
        config = saved["config"]
        inputs = load_inputs(
            config, preprocessing=saved["preprocessing"], include_test=(split == "test")
        )
        model = _restore_model(saved, inputs)
        accelerator = Accelerator(mixed_precision=config["precision"])
        if accelerator.num_processes != 1:
            raise ValueError(
                "checkpoint evaluation must be launched as one ordinary process"
            )
        model.to(accelerator.device)
        result = evaluate_model(
            model, inputs, config, split=split, accelerator=accelerator
        )
        phase = "export"
        destination = run_dir / "evaluation" / checkpoint.stem / split
        export_evaluation(result, destination)
        _set_status(
            run_dir,
            "evaluation",
            "completed",
            split=split,
            checkpoint=str(checkpoint),
            output=str(destination),
        )
        return result
    except Exception as exc:
        _set_status(
            run_dir,
            "evaluation",
            "failed",
            phase=phase,
            split=split,
            checkpoint=str(checkpoint),
            error_type=type(exc).__name__,
            message=str(exc),
        )
        raise
