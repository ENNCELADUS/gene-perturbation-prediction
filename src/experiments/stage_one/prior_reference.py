"""The default prior's reference row refitted on the stage-one frame's mean channel,
the like-for-like linear reference (PIE stage-one design Task 3; protocol §16), with
its out-of-fold predictions and the expression components the components reader
reads."""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.io import write_json, write_parquet
from src.experiments.run_dir import bind


def run(config: Mapping[str, Any], run_id: str) -> Path:
    from src.context_prior.prior import (
        PriorInputs,
        PriorSpec,
        fit_prior,
        total,
    )
    from src.context_prior.reference import load_reference
    from src.context_prior.views import expression_components, fit_expression_components
    from src.data.ladder_reader import mean_channel
    from src.eval.accuracy import fold_accuracy, wide_accuracy
    from src.experiments.prior.base import stages
    from src.experiments.stage_one.setup import load_setup

    run_dir = Path(config["output_root"]) / config["task"] / run_id
    (run_dir / "components").mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    if (run_dir / "row.json").is_file():
        print(f"{run_dir}: done", flush=True)
        return run_dir
    setup = load_setup(config["ladder_config"])
    ladder = setup.ladder
    prior = setup.prior
    reference = prior["reference"]
    frame = ladder.frame
    # The labels' guard: every frame line must be a fit-eligible training line.
    setup.targets(None)
    folds = ladder.folds
    expression = mean_channel(ladder).astype(np.float64)
    components = fit_expression_components(
        expression.loc[frame], int(prior["prior"]["components"])
    )
    inputs = PriorInputs(
        expression=expression.loc[frame],
        residual=setup.truth(frame),
        components=components,
        reference=load_reference(
            Path(prior["paths"]["reference"]),
            blocked={*ladder.role_lines("val", "test"), *setup.excluded},
        ),
        lineage=ladder.lines.loc[frame, "lineage"],
        patients=ladder.lines["patient_id"].astype(str).to_dict(),
    )
    spec = PriorSpec(
        tuple(
            stages(
                prior,
                reference["block_set"],
                reference["components_penalty"],
                reference["gene_penalty"],
            )
        )
    )
    fitted = fit_prior(spec, inputs, fit_lines=frame, encoder_lines=frame)
    held = ladder.role_lines("val", "test")
    predictions = total(fitted.predict(expression.loc[held]))
    count = int(prior["prior"]["components"])
    fold_components, parts = {}, []
    for k in sorted(set(folds.values())):
        kept = [m for m in frame if folds[m] != k]
        fold_components[k] = fit_expression_components(expression.loc[kept], count)
        fitted_k = fit_prior(
            spec,
            dataclasses.replace(inputs, components=fold_components[k]),
            fit_lines=kept,
            encoder_lines=kept,
        )
        held_k = [m for m in frame if folds[m] == k]
        parts.append(total(fitted_k.predict(expression.loc[held_k])))
    oof = pd.concat(parts, verify_integrity=True).loc[frame]
    split = {
        "val": predictions.loc[ladder.role_lines("val")],
        "test": predictions.loc[ladder.role_lines("test")],
        "oof": oof,
    }
    for name, values in split.items():
        write_parquet(
            run_dir / f"predictions_{name}.parquet", values.astype(np.float32)
        )
    lines = ladder.line_ids
    write_parquet(
        run_dir / "components" / "main.parquet",
        expression_components(components, expression.loc[lines]).astype(np.float32),
    )
    for k, fitted_components in fold_components.items():
        write_parquet(
            run_dir / "components" / f"fold{k}.parquet",
            expression_components(fitted_components, expression.loc[lines]).astype(
                np.float32
            ),
        )
    cells = ladder.role_lines("train_cells")
    scored = {
        "val": split["val"],
        "test": split["test"],
        "oof": oof.loc[cells],
    }
    write_json(
        run_dir / "row.json",
        {
            "reference_row": reference,
            "frame_lines": len(frame),
            "space_genes": int(expression.shape[1]),
            "scores": {
                name: (fold_accuracy if name == "oof" else wide_accuracy)(
                    values,
                    setup.truth(list(values.index)),
                    setup.definitions,
                    **({"folds": folds} if name == "oof" else {}),
                    panel=ladder.panel,
                )
                for name, values in scored.items()
            },
        },
    )
    print(f"{run_dir}: done", flush=True)
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    from src.experiments.stage_one.config import load_stage_one_config

    config = load_stage_one_config(args.config)
    if config["kind"] != "prior_reference":
        parser.error(f"{args.config} is a {config['kind']} config")
    run(config, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
