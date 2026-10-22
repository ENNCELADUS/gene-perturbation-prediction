"""Score a saved joint checkpoint on one split with its saved preprocessing, never
fitting (``hpc/run.sh test``). Writes
``<checkpoint dir>/evaluation/<checkpoint name>/<split>/``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from src.data.io import write_json
from src.experiments.joint.config import mixed_precision

if TYPE_CHECKING:
    from src.eval.accuracy import EvalResult


def evaluate_checkpoint(checkpoint: Path, *, split: str) -> EvalResult:
    from accelerate import Accelerator

    from src.eval.accuracy import evaluate_model
    from src.training.checkpoint import load_checkpoint

    saved = load_checkpoint(Path(checkpoint))
    config = saved["config"]
    from src.data.prepared import load_inputs

    inputs = load_inputs(
        config, preprocessing=saved["preprocessing"], include_test=split == "test"
    )
    # The same precision as training-time validation, so the numbers agree.
    accelerator = Accelerator(mixed_precision=mixed_precision())
    if accelerator.num_processes > 1:
        raise ValueError(
            "evaluate_checkpoint runs in one process: only rank zero holds the tables"
        )
    from src.model.initialization import restore_joint_model

    model = restore_joint_model(saved, inputs).to(accelerator.device)
    return evaluate_model(model, inputs, config, split=split, accelerator=accelerator)


def export_evaluation(result: EvalResult, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    result.per_line.to_csv(out_dir / "per_line.csv", index=False)
    result.per_gene.to_csv(out_dir / "per_gene.csv", index=False)
    write_json(out_dir / "metrics.json", result.metrics)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(args.checkpoint, split=args.split)
    export_evaluation(
        result,
        args.checkpoint.parent / "evaluation" / args.checkpoint.stem / args.split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
