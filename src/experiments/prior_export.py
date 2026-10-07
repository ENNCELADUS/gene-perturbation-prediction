"""Export the reference row of a prior config for the single-cell correction.

``python -m src.experiments.prior_export CONFIG --run-id ID`` writes
``<output_root>/<run id>/export/{prior.npz, prior.json}``: the prior's residual
prediction in residual-SD units for every labelled single-cell training line out of
fold (the bridge and every stage refitted without the line's fold; extras sharing a
patient with a line of the fold are held with it) and for the validation and test
lines from the fit on every labelled training-side line. ``prior.json`` records the
run id and reference row (the identity a correction checkpoint records) and the
export's own selective-Spearman scores. Nothing is chosen here.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.context_prior.bridging import oof_bridged
from src.context_prior.folds import training_side_folds
from src.context_prior.prior import PriorSpec, crossfit, fit_prior, total
from src.experiments.config import load_prior_config
from src.experiments.context_prior import (
    SCORED,
    RunBase,
    _bind,
    _build,
    _jsonable,
    _write_json,
    load_base,
    prior_inputs,
    score,
    stages,
)

EXPORT_DIR = "export"
SPLITS = ("train", *SCORED)


def export_predictions(base: RunBase) -> dict[str, pd.DataFrame]:
    """The reference row's predictions in residual-SD units: ``train`` out of fold
    for every labelled single-cell training line, ``val`` and ``test`` from the fit
    on every labelled training-side line."""
    config = base.config
    reference = config["reference"]
    experiment = config["experiments"][reference["experiment"]]
    if experiment["kind"] != "affine":
        raise ValueError(
            "the export refits the bridge out of fold for the affine bridge only, "
            f"not {experiment['kind']!r}"
        )
    inputs = _build(base, "affine", experiment["settings"][0])
    prior = prior_inputs(base, inputs)
    spec = PriorSpec(
        tuple(
            stages(
                config,
                reference["block_set"],
                reference["components_penalty"],
                reference["gene_penalty"],
            )
        )
    )
    encoders = list(prior.expression.index)
    full = fit_prior(spec, prior, fit_lines=list(base.labelled), encoder_lines=encoders)
    bridge = base.bridge
    train = list(bridge.single_cell_train)
    folds = training_side_folds(
        bridge.folds, encoders, base.models["patient_id"].to_dict()
    )
    rows = oof_bridged(
        bridge.pseudobulk, bridge.bulk, list(bridge.paired), train, bridge.folds
    )
    queries = {
        fold: rows.loc[[m for m in train if bridge.folds[m] == fold]]
        for fold in sorted(set(bridge.folds.values()))
    }
    held = total(
        crossfit(
            spec,
            prior,
            folds=folds,
            queries=queries,
            labelled=list(base.labelled),
            encoder_lines=encoders,
        )
    )
    return {
        "train": held.loc[train],
        **{name: total(full.predict(inputs.queries[name])) for name in SCORED},
    }


def _refuse_existing(out: Path) -> None:
    if out.exists():
        raise FileExistsError(
            f"{out} exists and a correction may have recorded it; use a new --run-id"
        )


def write_export(run_dir: Path, base: RunBase, run_id: str) -> Path:
    """Write ``export/`` once; an existing export is never replaced, because a
    correction checkpoint may have recorded it."""
    out = Path(run_dir) / EXPORT_DIR
    _refuse_existing(out)
    predictions = export_predictions(base)
    genes = list(base.definitions.genes)
    frame = pd.concat([predictions[split] for split in SPLITS]).loc[:, genes]
    scores = {
        split: score(
            predictions[split],
            base.truth(list(predictions[split].index)),
            base.definitions,
        )
        for split in SPLITS
    }
    staging = out.with_name(f".{EXPORT_DIR}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    np.savez(
        staging / "prior.npz",
        values=frame.to_numpy(dtype=np.float32),
        lines=np.asarray(frame.index, dtype=str),
        genes=np.asarray(genes, dtype=str),
        residual_scale=base.definitions.residual_scale.loc[genes].to_numpy(np.float64),
    )
    _write_json(
        staging / "prior.json",
        _jsonable(
            {
                "run_id": run_id,
                "reference": dict(base.config["reference"]),
                "units": "residual SD",
                "lines": {split: list(predictions[split].index) for split in SPLITS},
                "scores": scores,
            }
        ),
    )
    os.replace(staging, out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    config = load_prior_config(args.config)
    run_dir = Path(config["output_root"]) / args.run_id
    _refuse_existing(run_dir / EXPORT_DIR)  # before the long load
    run_dir.mkdir(parents=True, exist_ok=True)
    _bind(run_dir, config)
    out = write_export(run_dir, load_base(config), args.run_id)
    print(f"prior export: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
