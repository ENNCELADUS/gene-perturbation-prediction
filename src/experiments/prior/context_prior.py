"""Linear context prior experiments.

``hpc/run.sh prior CONFIG [--run-id ID] [--experiments A,B]``
runs every listed experiment: for each bridge setting it builds the bridged inputs
(``src.context_prior.remedies.<kind>.build``), fits the prior for every block set
and penalty, and scores validation and test (and the bulk-input oracle on
validation). Each setting writes ``rows/<experiment>__<n>.json`` and is skipped when
it exists, so a rerun resumes; ``results.md`` tabulates every row present. A
gene-level block set gives one row per components and gene penalty pair; nothing is
chosen here. Gains are measured against the reference row the config pins.

A run directory belongs to one config (``run_config.json``); processes running
different experiments may share it, since rows are per setting.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.context_prior.bridging import bridge_diagnostics
from src.eval.accuracy import gain
from src.data.io import read_json, write_json, write_text
from src.experiments.prior.config import load_prior_config
from src.experiments.prior.base import (
    SCORED,
    Fitter,
    RunBase,
    build,
    load_base,
    reference_predictions,
    stages,
)
from src.experiments.run_dir import bind, locked, number


def run_setting(
    base: RunBase,
    experiment: Mapping[str, Any],
    setting: Mapping[str, Any],
    reference: Mapping[str, pd.DataFrame],
) -> dict:
    """Every block set and penalty of one bridge setting. ``reference`` maps
    ``val``/``test`` to the reference row's predictions (the gain baseline).

    A block set of expression components alone gives one row per components
    penalty; any other gives one row per components and gene penalty pair, since
    the components penalty sets how much residual the gene-level stages fit and
    the scale-free score barely separates the components penalties alone.
    """
    config = base.config
    inputs = build(base, experiment["kind"], setting)
    fit = Fitter(base, inputs)
    repeats = int(config["prior"]["bootstrap_repeats"])
    grid = fit.components()
    rows = []

    def add(block_set, components_penalty, gene_penalty, predictions, scores):
        print(
            f"  {block_set}, components {components_penalty:g}, gene "
            f"{'-' if gene_penalty is None else f'{gene_penalty:g}'}: validation "
            f"selective Spearman {number(scores['val']['selective_spearman'])}",
            flush=True,
        )
        gains = {
            f"{name}_gain": gain(
                predictions[name],
                reference[name],
                fit.truth[name],
                base.definitions,
                repeats,
            )
            for name in SCORED
        }
        rows.append(
            {
                "block_set": block_set,
                "components_penalty": components_penalty,
                "gene_penalty": gene_penalty,
                **scores,
                **gains,
            }
        )

    for name in experiment["block_sets"]:
        blocks = list(config["block_sets"][name])
        if blocks == ["expression_components"]:
            for penalty, (predictions, scores) in grid.items():
                add(name, penalty, None, predictions, scores)
            continue
        for components in grid:
            for value in config["penalties"]["gene"]:
                add(
                    name,
                    components,
                    float(value),
                    *fit(stages(config, name, components, value)),
                )
    diagnostics = bridge_diagnostics(
        inputs.oof_paired,
        inputs.oof_bulk,
        base.definitions.selective,
        base.reference.paralogs,
    )
    if inputs.gene_space is not None:
        diagnostics["gene_space"] = len(inputs.gene_space)
    return {"rows": rows, "diagnostics": diagnostics}


def _facts(base: RunBase) -> dict:
    return {
        "space_genes": base.bridge.bulk.shape[1],
        "filled_lines": base.filled_lines,
        "training_side_lines": len(base.bridge.bulk),
        "training_lines": len(base.labelled),
        "paired_lines": len(base.bridge.paired),
        "single_cell_train": len(base.bridge.single_cell_train),
        "selective_genes": len(base.definitions.selective),
    }


# ----------------------------------------------------------------------------
# Files and results.md
# ----------------------------------------------------------------------------


def _row_path(run_dir: Path, experiment: str, index: int) -> Path:
    return run_dir / "rows" / f"{experiment}__{index}.json"


def _label(setting: Mapping[str, Any]) -> str:
    return ", ".join(f"{key} {value}" for key, value in setting.items()) or "-"


def _penalty(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


def _gain(entry: Mapping[str, Any]) -> str:
    low, high = entry["interval"]
    return f"{number(entry['difference'])} [{number(low)}, {number(high)}]"


def _counts(entry: Mapping[str, Any]) -> str:
    return " / ".join(str(entry[str(t)]) for t in (0.3, 0.5, 0.7)) + (
        f" of {entry['total']}"
    )


def write_results(run_dir: Path) -> Path:
    """``results.md`` from every row file present: run facts, then per experiment
    a table with one row per setting x block set x penalty and a diagnostics
    table with one row per setting."""
    with locked(run_dir):
        return _write_results(run_dir)


def _write_results(run_dir: Path) -> Path:
    config = read_json(run_dir / "run_config.json")["config"]
    out = [f"# Linear context prior experiments, run {run_dir.name}", ""]
    if (run_dir / "facts.json").is_file():
        facts = read_json(run_dir / "facts.json")
        out += [
            f"Shared expression space: {facts['space_genes']} genes "
            f"({facts['filled_lines']} training lines took the training mean for "
            f"some). The prior fits {facts['training_lines']} labelled lines of "
            f"{facts['training_side_lines']} training-side lines with bulk RNA; "
            f"{facts['paired_lines']} of the {facts['single_cell_train']} labelled "
            "single-cell training lines have bulk RNA and fit the bridge. "
            f"{facts['selective_genes']} selective genes.",
            "",
        ]
    reference = config["reference"]
    out += [
        "Selective Spearman: macro mean over the selective genes of the Spearman "
        "across lines of the residual. Gain: selective Spearman minus that of the "
        f"reference row (experiment `{reference['experiment']}`, first setting, "
        f"block set `{reference['block_set']}`, components penalty "
        f"{_penalty(reference['components_penalty'])}, gene penalty "
        f"{_penalty(reference['gene_penalty'])}), with its "
        "95% paired line-bootstrap interval. Oracle: validation lines' bulk RNA as "
        "input (off-contract). A gene-level block set has a row per components "
        "and gene penalty pair. Nothing here is synthetic-lethality evidence.",
    ]
    records: dict[str, list[dict]] = {}
    for path in (run_dir / "rows").glob("*.json"):
        record = read_json(path)
        records.setdefault(record["experiment"], []).append(record)
    columns = (
        "Setting | Block set | Components penalty | Gene penalty "
        "| Val selective Spearman | Test selective Spearman | Val gain | Test gain "
        "| Val AUPR lift | Test AUPR lift | Val residual Pearson "
        "| Test residual Pearson | Oracle val selective Spearman"
    )
    for name, experiment in config["experiments"].items():
        found = sorted(records.get(name, []), key=lambda r: r["setting_index"])
        if not found:
            continue
        out += [
            "",
            f"## {name}",
            "",
            f"Remedy `{experiment['kind']}`.",
            "",
            f"| {columns} |",
            "|" + " --- |" * (columns.count("|") + 1),
        ]
        for record in found:
            for row in record["rows"]:
                cells = [
                    _label(record["setting"]),
                    row["block_set"],
                    _penalty(row["components_penalty"]),
                    _penalty(row["gene_penalty"]),
                    number(row["val"]["selective_spearman"]),
                    number(row["test"]["selective_spearman"]),
                    _gain(row["val_gain"]),
                    _gain(row["test_gain"]),
                    number(row["val"]["selective_aupr_lift"]),
                    number(row["test"]["selective_aupr_lift"]),
                    number(row["val"]["residual_pearson_macro_per_gene"]),
                    number(row["test"]["residual_pearson_macro_per_gene"]),
                    number(row["oracle"]["selective_spearman"]),
                ]
                out.append("| " + " | ".join(cells) + " |")
        out += [
            "",
            "Bridge diagnostics: per-gene Pearson across the paired lines between "
            "out-of-fold bridged pseudo-bulk and bulk; counts at or above 0.3 / 0.5 "
            "/ 0.7.",
            "",
            "| Setting | Median | Q25 | Q75 | All genes | Selective genes "
            "| Selective genes' paralogs | Gene space |",
            "|" + " --- |" * 8,
        ]
        for record in found:
            report = record["diagnostics"]
            cells = [
                _label(record["setting"]),
                number(report["median"], 3),
                number(report["q25"], 3),
                number(report["q75"], 3),
                _counts(report["all"]),
                _counts(report["selective"]),
                _counts(report["selective_paralogs"]),
                str(report["gene_space"]) if "gene_space" in report else "all",
            ]
            out.append("| " + " | ".join(cells) + " |")
    path = run_dir / "results.md"
    write_text(path, "\n".join(out) + "\n")
    return path


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--experiments", help="comma-separated experiment names (default: all)"
    )
    args = parser.parse_args(argv)
    config = load_prior_config(args.config)
    names = (
        list(config["experiments"])
        if args.experiments is None
        else list(dict.fromkeys(args.experiments.split(",")))
    )
    unknown = sorted(set(names) - set(config["experiments"]))
    if unknown:
        raise ValueError(
            f"unknown experiments {unknown}; the config names "
            f"{list(config['experiments'])}"
        )
    run_id = args.run_id or "prior_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"run id: {run_id}", flush=True)
    run_dir = Path(config["output_root"]) / run_id
    (run_dir / "rows").mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    pending = [
        (name, index, setting)
        for name in names
        for index, setting in enumerate(config["experiments"][name]["settings"])
        if not _row_path(run_dir, name, index).is_file()
    ]
    if pending:
        base = load_base(config)
        if not (run_dir / "facts.json").is_file():
            write_json(run_dir / "facts.json", _facts(base))
        print("reference row", flush=True)
        reference = reference_predictions(base)
        for name, index, setting in pending:
            experiment = config["experiments"][name]
            print(f"{name}, setting {index}: {_label(setting)}", flush=True)
            result = run_setting(base, experiment, setting, reference)
            write_json(
                _row_path(run_dir, name, index),
                {
                    "experiment": name,
                    "kind": experiment["kind"],
                    "setting_index": index,
                    "setting": setting,
                    **result,
                },
            )
            write_results(run_dir)
    print(f"results: {write_results(run_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
