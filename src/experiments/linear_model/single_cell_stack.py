"""Single-cell statistics stacked on a linear model run's export (protocol §13): does
a line's basal single-cell distribution add to what its pseudo-bulk already gives the
linear model? The ``mean`` arm, close to pseudo-bulk, is the like-for-like control."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from src.linear_model.single_cell import (
    normalize_statistics,
    read_statistics,
    stack_predictions,
)
from src.data.io import write_json, write_text
from src.data.prepared import read_manifest
from src.data.linear_model_offsets import (
    checked_linear_model,
    read_linear_model_offsets,
)
from src.eval.accuracy import gain, score
from src.experiments.joint.config import load_config
from src.experiments.linear_model.base import SCORED, RunBase, load_base
from src.experiments.linear_model.config import load_stack_config
from src.experiments.run_dir import bind, number

OUT_DIR = "single_cell_stack"
CONTROL_ARM = "mean"


def _penalty(value: float) -> str:
    """A penalty as written in the config; infinite own shrinkage is pooled only."""
    return "pooled only" if value == float("inf") else f"{value:g}"


def run_arms(
    config: Mapping[str, object],
    base: RunBase,
    linear_model: pd.DataFrame,
    statistics: Mapping[str, pd.DataFrame],
) -> dict:
    """Rows of every arm, the row of the linear model alone and the facts; nothing
    chosen.

    ``linear_model`` is the export (lines x genes, residual-SD units): out of fold on
    the training lines, the full fit on validation and test.
    """
    train = list(base.bridge.single_cell_train)
    queries = {"val": list(base.bridge.val), "test": list(base.bridge.test)}
    every = [*train, *queries["val"], *queries["test"]]
    truth = base.truth(every)
    target = truth.loc[train] - linear_model.loc[train]
    repeats = int(config["bootstrap_repeats"])
    penalties = config["penalties"]

    linear_model_alone = {
        split: score(linear_model.loc[queries[split]], truth, base.definitions)
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
            key: linear_model.loc[frame.index] + frame for key, frame in fitted.items()
        }
    for arm, chosen in config["arms"].items():
        rows = []
        for (shrinkage, penalty), stacked in predictions[arm].items():
            row = {"own_shrinkage": shrinkage, "selected_penalty": penalty}
            for split in SCORED:
                lines = queries[split]
                entry = score(stacked.loc[lines], truth, base.definitions)
                entry["gain_over_linear_model"] = gain(
                    stacked.loc[lines],
                    linear_model.loc[lines],
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
    return {"linear_model_alone": linear_model_alone, "arms": arms, "facts": facts}


def results_markdown(run_id: str, config: Mapping[str, object], result: dict) -> str:
    """Every row on validation and test; gains with their intervals."""

    def interval(entry: Mapping[str, object] | None) -> str:
        if entry is None:
            return ""
        low, high = entry["interval"]
        return f"{entry['difference']:+.4f} [{low:+.4f}, {high:+.4f}]"

    linear_model = result["linear_model_alone"]
    lines = [
        f"# Single-cell statistics stack `{run_id}`",
        "",
        f"Stacked on linear model run `{config['linear_model_run']}`. "
        "Each arm fits a two-stage "
        "ridge (own statistics, then data-selected statistics) on what the "
        "linear model's "
        "out-of-fold export leaves on the training lines, and adds it to the export. "
        "Selective Spearman and its gains are macro means over the selective genes; "
        "intervals are 95% paired line bootstraps. Nothing is chosen here; nothing "
        "here is synthetic-lethality evidence.",
        "",
        "Linear model alone: validation "
        f"{number(linear_model['val']['selective_spearman'])}, "
        f"test {number(linear_model['test']['selective_spearman'])}.",
        "",
    ]
    for arm, record in result["arms"].items():
        lines += [
            f"## Arm `{arm}`: {', '.join(record['statistics'])}",
            "",
            "| Own shrinkage | Selected penalty | Validation "
            "| Gain over linear model (val) "
            f"| Gain over `{CONTROL_ARM}` (val) | Test "
            "| Gain over linear model (test) "
            f"| Gain over `{CONTROL_ARM}` (test) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in record["rows"]:
            cells = [_penalty(row["own_shrinkage"]), _penalty(row["selected_penalty"])]
            for split in SCORED:
                entry = row[split]
                cells += [
                    number(entry["selective_spearman"]),
                    interval(entry["gain_over_linear_model"]),
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

    linear_model_run = Path(config["linear_model_run"])
    base_config = json.loads((linear_model_run / "run_config.json").read_text())[
        "config"
    ]
    base = load_base(base_config)
    definitions = base.definitions
    train = list(base.bridge.single_cell_train)
    every = [*train, *base.bridge.val, *base.bridge.test]
    export = checked_linear_model(
        read_linear_model_offsets(linear_model_run / "export"),
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
    result["facts"]["linear_model_identity"] = export.identity
    for arm, record in result["arms"].items():
        write_json(out / "rows" / f"{arm}.json", record)
    write_json(
        out / "facts.json",
        {**result["facts"], "linear_model_alone": result["linear_model_alone"]},
    )
    write_text(out / "results.md", results_markdown(args.run_id, config, result))
    print(f"single-cell stack: {out / 'results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
