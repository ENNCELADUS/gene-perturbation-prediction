"""The production fit of stage one (decision 2026-11-04): the configuration the
stage-one benchmark chose (the linear context prior's reference row), fitted once on
every labelled line, with no validation set and no cross-fitting. It writes one
GeneEffect table (line x gene) for every single-cell line of the final split, which
stage two reads. A labelled line that shares a patient with an SL validation or test
line without GeneEffect is left out of the fit (still predicted), and the record
names it."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.context_prior.bridge import BridgeInputs, fit_bridge
from src.context_prior.prior import PriorSpec, fit_prior, total
from src.data.depmap import read_models
from src.data.extra_lines import ExtraLines, load_bulk_lines
from src.data.io import jsonable, write_json
from src.data.splits import FixedSplit, load_geneeffect_split
from src.experiments.joint.config import load_config
from src.experiments.prior.base import base_on, prior_inputs, stages
from src.experiments.prior.config import load_prior_config
from src.experiments.run_dir import bind

PRODUCTION_DIR = "production"


def left_out(
    sl_split: Mapping[str, Any],
    without: Sequence[str],
    patients: Mapping[str, str],
    labelled: Sequence[str],
) -> dict[str, str]:
    """Labelled lines sharing a patient with an SL validation or test line that has
    no GeneEffect (``without``), with the reason."""
    held = {
        patients[m]: m for m in (*sl_split["val"], *sl_split["test"]) if m in without
    }
    return {
        m: f"shares patient {patients[m]} with SL held-out line {held[patients[m]]}"
        for m in labelled
        if patients[m] in held
    }


def production_split(
    split: FixedSplit, extra: ExtraLines, out: Mapping[str, str]
) -> tuple[FixedSplit, ExtraLines]:
    """Every single-cell line on the training side: validation and test lines with
    GeneEffect become labelled training lines. A left-out line keeps its expression
    and gives no GeneEffect target: a single-cell one is ``unlabeled_train``, a
    bulk-only one an unlabelled encoder row."""
    lines = [*split.train, *split.val, *split.test]
    unlabelled = {*split.unlabeled_train, *out}
    return (
        FixedSplit(
            train=tuple(lines),
            val=(),
            test=(),
            unlabeled_train=tuple(m for m in lines if m in unlabelled),
        ),
        ExtraLines(
            tuple(m for m in extra.labelled if m not in out),
            (*extra.unlabelled, *(m for m in extra.labelled if m in out)),
            dict(extra.excluded),
        ),
    )


def production_table(config: Mapping[str, Any], sl_split_path: Path) -> tuple:
    """The production GeneEffect table and its record."""
    reference = config["reference"]
    experiment = config["experiments"][reference["experiment"]]
    if experiment["kind"] != "affine":
        raise ValueError(
            f"production reads the affine bridge, not {experiment['kind']}"
        )
    joint = load_config(Path(config["joint_config"]))
    split_path = Path(joint["paths"]["split"])
    split = load_geneeffect_split(split_path)
    extra = load_bulk_lines(split_path, split)
    sl_split = json.loads(Path(sl_split_path).read_text())
    patients = read_models(Path(config["paths"]["model"]))["patient_id"].to_dict()
    labelled = [*split.supervised_train, *split.val, *split.test, *extra.labelled]
    out = left_out(sl_split, split.unlabeled_train, patients, labelled)
    fit_split, fit_extra = production_split(split, extra, out)
    scored = [m for m in (*sl_split["val"], *sl_split["test"]) if m in fit_split.train]
    base = base_on(config, fit_split, fit_extra, scored)
    bridge = base.bridge
    inputs = BridgeInputs(
        expression=bridge.bulk,
        queries={},
        oof_paired=bridge.bulk.iloc[:0],
        oof_bulk=bridge.bulk.iloc[:0],
    )
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
    fitted = fit_prior(
        spec,
        prior,
        fit_lines=list(base.labelled),
        encoder_lines=list(prior.expression.index),
    )
    paired = list(bridge.paired)
    rows = fit_bridge(bridge.pseudobulk.loc[paired], bridge.bulk.loc[paired]).apply(
        bridge.pseudobulk.loc[list(fit_split.train)]
    )
    definitions = base.definitions
    genes = list(definitions.genes)
    residual = total(fitted.predict(rows)).loc[:, genes]
    table = residual * definitions.residual_scale[genes] + definitions.gene_means[genes]
    fit_lines = set(base.labelled) | set(fit_split.supervised_train)
    record = {
        "reference": dict(reference),
        "units": "GeneEffect",
        "lines": len(table),
        "fit_lines": len(base.labelled),
        "in_sample": sorted(m for m in table.index if m in fit_lines),
        "out_of_sample": sorted(m for m in table.index if m not in fit_lines),
        "left_out": out,
    }
    return table, record


def write_production(run_dir: Path, config: Mapping[str, Any], sl_split: Path) -> Path:
    """Write ``production/`` once (``predictions.parquet`` and ``production.json``)."""
    out = Path(run_dir) / PRODUCTION_DIR
    if out.exists():
        raise FileExistsError(f"{out} exists; use a new --run-id")
    table, record = production_table(config, sl_split)
    staging = out.with_name(f".{PRODUCTION_DIR}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    table.astype("float32").to_parquet(staging / "predictions.parquet")
    write_json(
        staging / "production.json",
        jsonable({**record, "sl_split": str(sl_split)}),
    )
    os.replace(staging, out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--sl-split", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    config = load_prior_config(args.config)
    run_dir = Path(config["output_root"]) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, {"config": config, "sl_split": str(args.sl_split)})
    out = write_production(run_dir, config, args.sl_split)
    print(f"stage-one production table: {out / 'predictions.parquet'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
