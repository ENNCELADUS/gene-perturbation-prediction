"""Joint GeneEffect training in a run directory, and checkpoint evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
from typing import TYPE_CHECKING, Any

from src.experiments.config import validate_config

if TYPE_CHECKING:
    from src.data.prepared import PreparedInputs
    from src.eval.geneeffect import EvalResult
    from src.model.geneeffect import GeneEffectE2EModel


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


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


def restore_model(
    saved: Mapping[str, Any], inputs: PreparedInputs
) -> GeneEffectE2EModel:
    """Rebuild a checkpoint's joint model; ``inputs`` use its saved preprocessing."""
    from src.model.initialization import restore_joint_model

    return restore_joint_model(saved, inputs)



def run_training(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    inputs: PreparedInputs | None = None,
) -> Path:
    """Train into ``run_dir`` and return its ``best.pt``.

    Resumes from ``run_dir/last.pt`` when present (the saved config must equal
    ``config``); returns at once when ``done.json`` exists. Works as one CPU
    process or under ``accelerate launch``. ``inputs`` replaces ``load_inputs``
    for synthetic tests.
    """
    import numpy as np
    import torch
    from accelerate import Accelerator
    import yaml

    from src.model.initialization import build_joint_model
    from src.training.checkpoint import load_checkpoint
    from src.training.trainer import fit

    config = validate_config(config)
    run_dir = Path(run_dir)
    best = run_dir / "best.pt"
    if (run_dir / "done.json").exists():
        return best
    saved = (
        load_checkpoint(run_dir / "last.pt") if (run_dir / "last.pt").exists() else None
    )
    if saved is not None and saved["config"] != config:
        raise ValueError(f"config differs from the one saved in {run_dir / 'last.pt'}")
    accelerator = Accelerator(mixed_precision=config["precision"])
    if accelerator.is_main_process and saved is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(dict(config), sort_keys=False)
        )
        _write_json(
            run_dir / "run.json",
            {
                "revision": _revision(),
                "world_size": accelerator.num_processes,
                "precision": accelerator.mixed_precision,
                "device": str(accelerator.device),
            },
        )
    # Seed before constructing the new adapter and head so every rank matches.
    random.seed(config["seeds"]["train"])
    np.random.seed(config["seeds"]["train"])
    torch.manual_seed(config["seeds"]["train"])
    if inputs is None:
        from src.data.prepared import load_inputs

        inputs = load_inputs(
            config, preprocessing=None if saved is None else saved["preprocessing"]
        )
    model = (
        build_joint_model(config, inputs)
        if saved is None
        else restore_model(saved, inputs)
    )
    state = fit(model, inputs, config, run_dir, accelerator, restored=saved)
    if accelerator.is_main_process:
        _write_json(run_dir / "done.json", asdict(state))
    if accelerator.num_processes > 1:
        torch.distributed.barrier()  # every rank returns once done.json exists
    return best


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    split: str,
    inputs: PreparedInputs | None = None,
) -> EvalResult:
    """Score a checkpoint on one split with its saved preprocessing; never fits.

    ``inputs`` replaces ``load_inputs`` for synthetic tests.
    """
    from accelerate import Accelerator

    from src.eval.geneeffect import evaluate_model
    from src.training.checkpoint import load_checkpoint

    saved = load_checkpoint(Path(checkpoint))
    config = saved["config"]
    if inputs is None:
        from src.data.prepared import load_inputs

        inputs = load_inputs(
            config, preprocessing=saved["preprocessing"], include_test=split == "test"
        )
    # The same precision as training-time validation, so the numbers agree.
    accelerator = Accelerator(mixed_precision=config["precision"])
    model = restore_model(saved, inputs).to(accelerator.device)
    return evaluate_model(model, inputs, config, split=split, accelerator=accelerator)


def export_evaluation(result: EvalResult, out_dir: Path) -> None:
    """Write predictions.parquet, metrics.json, per_line.csv and per_gene.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    result.per_line.to_csv(out_dir / "per_line.csv", index=False)
    result.per_gene.to_csv(out_dir / "per_gene.csv", index=False)
    _write_json(out_dir / "metrics.json", result.metrics)
