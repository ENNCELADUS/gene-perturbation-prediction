"""Train the joint GeneEffect model into a run directory (resumes from ``last.pt``),
as one process or under ``accelerate launch``."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.data.io import write_json
from src.experiments.joint.config import load_config, mixed_precision, validate_config
from src.experiments.run_dir import git_revision

if TYPE_CHECKING:
    from src.data.prepared import PreparedInputs


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
    import yaml
    from accelerate import Accelerator

    from src.model.initialization import build_joint_model, restore_joint_model
    from src.training.checkpoint import load_checkpoint
    from src.training.trainer import fit, training_diagnostic_lines

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
    accelerator = Accelerator(mixed_precision=mixed_precision())
    if accelerator.is_main_process and saved is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(dict(config), sort_keys=False)
        )
        write_json(
            run_dir / "run.json",
            {
                "revision": git_revision(),
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
    if accelerator.is_main_process:
        path = run_dir / "run.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        record["training_diagnostic_lines"] = list(training_diagnostic_lines(inputs))
        write_json(path, record)
    model = (
        build_joint_model(config, inputs)
        if saved is None
        else restore_joint_model(saved, inputs)
    )
    state = fit(model, inputs, config, run_dir, accelerator, restored=saved)
    if accelerator.is_main_process:
        write_json(run_dir / "done.json", asdict(state))
    if accelerator.num_processes > 1:
        torch.distributed.barrier()  # every rank returns once done.json exists
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run_training(load_config(args.config), args.run_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
