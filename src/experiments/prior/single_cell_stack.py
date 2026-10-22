"""Single-cell statistics stacked on the default prior: does a line's basal
single-cell distribution add to what its pseudo-bulk already gives the prior?

``python -m src.experiments.prior.single_cell_stack CONFIG --run-id ID`` reads the prior
run named by ``prior_run`` (its bound config and its ``export/``), fits each arm's
two-stage ridge (``src.context_prior.single_cell``) on what the export leaves on the
single-cell training lines (out of fold, so the stack learns at the prior's
deployment error), and adds it to the export on validation and test. Arms differ
only in the statistics they read; the ``mean`` arm (the log-space mean, close to
pseudo-bulk) is the like-for-like control for arms that add the detected fraction
and variance. Writes into ``<output_root>/<run id>/single_cell_stack/``:

- ``rows/<arm>.json``: per (own shrinkage, selected penalty), validation and test
  metrics, the selective-Spearman gain over the prior alone, and over the ``mean``
  arm at the same penalties, each with its 95% paired line-bootstrap interval;
- ``facts.json`` and ``results.md``.

It reports and chooses nothing. Nothing here is synthetic-lethality evidence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from src.context_prior.single_cell import (
    normalize_statistics,
    read_statistics,
    stack_predictions,
)
from src.data.io import write_json, write_text
from src.data.prepared import read_manifest
from src.data.prior_offsets import checked_prior, read_prior_offsets
from src.eval.accuracy import gain, score
from src.experiments.joint.config import load_config
from src.experiments.prior.base import (
    SCORED,
    RunBase,
    load_base,
)
from src.experiments.prior.config import load_stack_config
from src.experiments.run_dir import bind, number

OUT_DIR = "single_cell_stack"
CONTROL_ARM = "mean"


def _penalty(value: float) -> str:
    """A penalty as written in the config; infinite own shrinkage is pooled only."""
    return "pooled only" if value == float("inf") else f"{value:g}"


def run_arms(
    config: Mapping[str, object],
    base: RunBase,
    prior: pd.DataFrame,
    statistics: Mapping[str, pd.DataFrame],
) -> dict:
    """Rows of every arm, the prior-alone row and the facts; nothing chosen.

    ``prior`` is the export (lines x genes, residual-SD units): out of fold on
    the training lines, the full fit on validation and test.
    """
    train = list(base.bridge.single_cell_train)
    queries = {"val": list(base.bridge.val), "test": list(base.bridge.test)}
    every = [*train, *queries["val"], *queries["test"]]
    truth = base.truth(every)
    target = truth.loc[train] - prior.loc[train]
    repeats = int(config["bootstrap_repeats"])
    penalties = config["penalties"]

    prior_alone = {
        split: score(prior.loc[queries[split]], truth, base.definitions)
        for split in SCORED
    }
    arms: dict[str, dict] = {}
    predictions: dict[str, dict] = {}
    for arm, chosen in config["arms"].items():
        fitted = stack_predictions(
            statistics,
            target,
            [*queries["val"], *queries["test"]],
            chosen=list(chosen),
            own_shrinkages=[float(p) for p in penalties["own"]],
            selected_penalties=[float(p) for p in penalties["selected"]],
            count=int(config["selected_genes"]),
        )
        predictions[arm] = {
            key: prior.loc[frame.index] + frame for key, frame in fitted.items()
        }
    for arm, chosen in config["arms"].items():
        rows = []
        for (shrinkage, penalty), stacked in predictions[arm].items():
            row = {"own_shrinkage": shrinkage, "selected_penalty": penalty}
            for split in SCORED:
                lines = queries[split]
                entry = score(stacked.loc[lines], truth, base.definitions)
                entry["gain_over_prior"] = gain(
                    stacked.loc[lines],
                    prior.loc[lines],
                    truth,
                    base.definitions,
                    repeats,
                )
                control = predictions.get(CONTROL_ARM, {}).get((shrinkage, penalty))
                if arm != CONTROL_ARM and control is not None:
                    entry[f"gain_over_{CONTROL_ARM}"] = gain(
                        stacked.loc[lines],
                        control.loc[lines],
                        truth,
                        base.definitions,
                        repeats,
                    )
                row[split] = entry
            rows.append(row)
        arms[arm] = {"statistics": list(chosen), "rows": rows}
    facts = {
        "train_lines": len(train),
        "val_lines": len(queries["val"]),
        "test_lines": len(queries["test"]),
        "target_genes": len(base.definitions.genes),
        "selective_genes": len(base.definitions.selective),
        "statistic_genes": int(next(iter(statistics.values())).shape[1]),
    }
    return {"prior_alone": prior_alone, "arms": arms, "facts": facts}


def results_markdown(run_id: str, config: Mapping[str, object], result: dict) -> str:
    """Every row on validation and test; gains with their intervals."""

    def interval(entry: Mapping[str, object] | None) -> str:
        if entry is None:
            return ""
        low, high = entry["interval"]
        return f"{entry['difference']:+.4f} [{low:+.4f}, {high:+.4f}]"

    prior = result["prior_alone"]
    lines = [
        f"# Single-cell statistics stack `{run_id}`",
        "",
        f"Stacked on prior run `{config['prior_run']}`. Each arm fits a two-stage "
        "ridge (own statistics, then data-selected statistics) on what the prior's "
        "out-of-fold export leaves on the training lines, and adds it to the export. "
        "Selective Spearman and its gains are macro means over the selective genes; "
        "intervals are 95% paired line bootstraps. Nothing is chosen here; nothing "
        "here is synthetic-lethality evidence.",
        "",
        f"Prior alone: validation {number(prior['val']['selective_spearman'])}, "
        f"test {number(prior['test']['selective_spearman'])}.",
        "",
    ]
    for arm, record in result["arms"].items():
        lines += [
            f"## Arm `{arm}`: {', '.join(record['statistics'])}",
            "",
            "| Own shrinkage | Selected penalty | Validation | Gain over prior (val) "
            f"| Gain over `{CONTROL_ARM}` (val) | Test | Gain over prior (test) "
            f"| Gain over `{CONTROL_ARM}` (test) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in record["rows"]:
            cells = [_penalty(row["own_shrinkage"]), _penalty(row["selected_penalty"])]
            for split in SCORED:
                entry = row[split]
                cells += [
                    number(entry["selective_spearman"]),
                    interval(entry["gain_over_prior"]),
                    interval(entry.get(f"gain_over_{CONTROL_ARM}")),
                ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    config = load_stack_config(args.config)
    run_dir = Path(config["output_root"]) / args.run_id
    out = run_dir / OUT_DIR
    if (out / "results.md").exists():
        raise FileExistsError(f"{out} is complete; use a new --run-id")
    (out / "rows").mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)

    prior_run = Path(config["prior_run"])
    base_config = json.loads((prior_run / "run_config.json").read_text())["config"]
    base = load_base(base_config)
    definitions = base.definitions
    train = list(base.bridge.single_cell_train)
    every = [*train, *base.bridge.val, *base.bridge.test]
    export = checked_prior(
        read_prior_offsets(prior_run / "export"),
        genes=definitions.genes,
        residual_scale=definitions.residual_scale,
        lines=every,
    )
    joint = load_config(Path(base_config["joint_config"]))
    prepared = Path(joint["prepared_root"])
    panel = list(read_manifest(prepared)["common_gene_panel"])
    statistics = normalize_statistics(
        read_statistics(prepared, every, panel),
        train=train,
        measured_in=[*base.bridge.val, *base.bridge.test],
    )
    result = run_arms(config, base, export.values.loc[every], statistics)
    result["facts"]["prior_identity"] = export.identity
    for arm, record in result["arms"].items():
        write_json(out / "rows" / f"{arm}.json", record)
    write_json(
        out / "facts.json", {**result["facts"], "prior_alone": result["prior_alone"]}
    )
    write_text(out / "results.md", results_markdown(args.run_id, config, result))
    print(f"single-cell stack: {out / 'results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
