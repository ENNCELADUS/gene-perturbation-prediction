"""Linear context prior experiments.

``python -m src.experiments.context_prior CONFIG [--run-id ID] [--experiments A,B]``
runs every listed experiment: for each bridge setting it builds the bridged inputs
(``src.context_prior.remedies.<kind>.build``), fits the prior for every block set
and penalty, and scores validation and test (and the bulk-input oracle on
validation). Each setting writes ``rows/<experiment>__<n>.json`` and is skipped when
it exists, so a rerun resumes; ``results.md`` tabulates every row present. A
gene-level block set gives one row per components and gene penalty pair; nothing is
chosen here except the reference row's components penalty.

A run directory belongs to one config (``run_config.json``); processes running
different experiments may share it, since rows are per setting.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.context_prior.bridging import BridgeBase, BridgeInputs, bridge_diagnostics
from src.context_prior.folds import patient_folds
from src.context_prior.prior import PriorInputs, PriorSpec, Stage, fit_prior, total
from src.context_prior.reference import Reference, load_reference
from src.context_prior.space import (
    fill_unmeasured,
    measured_genes,
    quantile_normalize,
    quantile_reference,
)
from src.context_prior.targets import Definitions, fit_definitions, residual_frame
from src.context_prior.views import fit_expression_components
from src.data.depmap import read_depmap_matrix, read_models
from src.data.extra_lines import load_extra_lines
from src.data.splits import FixedSplit, load_geneeffect_226_split
from src.eval.metrics import paired_line_bootstrap
from src.experiments.all import _number, _read_json
from src.experiments.config import load_config, load_prior_config

BOOTSTRAP_SEED = 0
METRICS = (
    "selective_spearman",
    "selective_aupr_lift",
    "residual_pearson_macro_per_gene",
    "geneeffect_loss",
    "residual_sd_ratio_macro_per_gene",
)
SCORED = ("val", "test")


# ----------------------------------------------------------------------------
# Run data
# ----------------------------------------------------------------------------


def scaled_residual(
    gene_effect: pd.DataFrame, lines: Sequence[str], definitions: Definitions
) -> pd.DataFrame:
    """Lines x genes ``(y - mu_hat) / sigma``; NaN where a label is missing.

    Raises when a line has no GeneEffect row at all: it would otherwise read as an
    all-missing line and count as zero residual in every fit.
    """
    absent = [m for m in lines if m not in gene_effect.index]
    if absent:
        raise ValueError(f"no GeneEffect row for {len(absent)} lines: {absent[:10]}")
    frame = residual_frame(gene_effect, lines, definitions)
    return frame.div(definitions.residual_scale, axis=1)


@dataclass(frozen=True)
class RunBase:
    """Everything the experiments share, keyed by ModelID.

    Attributes:
        config: The prior config.
        split: The 226-line split.
        gene_effect: GeneEffect of the split's lines and the kept labelled extra
            lines, over the definitions' genes.
        definitions: Gene order, means, selective and variable genes and residual
            SD, fitted on the labelled single-cell training lines.
        labelled: Labelled training-side lines with bulk RNA: the prior's fit lines.
        models: Every line's ``patient_id`` and ``lineage``.
        reference: Pinned reference tables.
        bridge: The quantile-normalised sources and folds the remedies start from.
        filled_lines: Training lines whose pseudo-bulk took the training mean for
            some gene of the space.
    """

    config: Mapping[str, Any]
    split: FixedSplit
    gene_effect: pd.DataFrame
    definitions: Definitions
    labelled: tuple[str, ...]
    models: pd.DataFrame
    reference: Reference
    bridge: BridgeBase
    filled_lines: int

    def truth(self, lines: Sequence[str]) -> pd.DataFrame:
        """Residual of ``lines`` in residual-SD units; reads their labels."""
        return scaled_residual(self.gene_effect, lines, self.definitions)


def load_base(config: Mapping[str, Any]) -> RunBase:
    """Read and normalise everything once.

    Test lines' bulk RNA and the excluded lines are dropped as soon as the bulk
    matrix is read; validation lines' bulk RNA feeds the oracle query alone. The
    space is the bulk genes every validation and test line measures in
    pseudo-bulk; a training line whose source lacks one takes the training mean.
    Bulk and pseudo-bulk are quantile-normalised to the training side's profile.
    """
    from src.data.prepared import read_manifest
    from src.data.pseudobulk import read_pseudobulk
    from src.experiments.prepare import prepare_pseudobulk

    joint = load_config(Path(config["joint_config"]))
    paths = config["paths"]
    split = load_geneeffect_226_split(Path(joint["paths"]["split"]))
    extra = load_extra_lines(Path(paths["extra_lines"]), split)
    models = read_models(Path(paths["model"]))
    dropped = set(config["training_side"]["exclude_lineages"])

    def kept(ids: Sequence[str]) -> list[str]:
        return [m for m in ids if models.at[m, "lineage"] not in dropped]

    bulk = read_depmap_matrix(Path(paths["bulk_expression"]))
    bulk = bulk.drop(index=bulk.index.intersection([*split.test, *extra.excluded]))
    side = [
        m
        for m in (*split.train, *kept(extra.labelled), *kept(extra.unlabelled))
        if m in bulk.index
    ]
    oracle_lines = [m for m in split.val if m in bulk.index]
    labelled = tuple(
        m for m in (*split.supervised_train, *kept(extra.labelled)) if m in bulk.index
    )
    paired = tuple(m for m in split.supervised_train if m in bulk.index)

    gene_effect = read_depmap_matrix(Path(joint["paths"]["gene_effect"]))
    needed = {*split.all_model_ids, *kept(extra.labelled)}
    gene_effect = gene_effect.loc[gene_effect.index.isin(needed)]

    prepared = Path(joint["prepared_root"])
    panel = list(read_manifest(prepared)["common_gene_panel"])
    prepare_pseudobulk(joint, list(bulk.columns))
    pseudo = read_pseudobulk(prepared)
    space = measured_genes(pseudo, list(bulk.columns), [*split.val, *split.test])
    pseudo, filled = fill_unmeasured(pseudo.loc[:, space], split.train)
    definitions = fit_definitions(gene_effect, split, panel, joint["features"])
    gene_effect = gene_effect.loc[:, list(definitions.genes)]

    # Training-side rows first, then the oracle's validation rows; nothing else.
    bulk = bulk.loc[[*side, *oracle_lines], space]
    profile = quantile_reference(bulk.iloc[: len(side)])
    normalized = quantile_normalize(bulk, profile)
    del bulk
    single_cell_train = tuple(split.supervised_train)
    bridge = BridgeBase(
        bulk=normalized.loc[side],
        oracle=normalized.loc[oracle_lines],
        pseudobulk=quantile_normalize(pseudo, profile),
        paired=paired,
        single_cell_train=single_cell_train,
        val=tuple(split.val),
        test=tuple(split.test),
        folds=patient_folds(
            list(single_cell_train),
            models["patient_id"].to_dict(),
            n_folds=int(config["prior"]["folds"]),
            seed=int(config["seed"]),
        ),
    )
    return RunBase(
        config=config,
        split=split,
        gene_effect=gene_effect,
        definitions=definitions,
        labelled=labelled,
        models=models,
        reference=load_reference(
            Path(paths["reference"]),
            blocked={*split.val, *split.test, *extra.excluded},
        ),
        bridge=bridge,
        filled_lines=int((filled > 0).sum()),
    )


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def long_frame(
    prediction: pd.DataFrame, truth: pd.DataFrame, definitions: Definitions
) -> pd.DataFrame:
    """Finite-label rows in GeneEffect units, as ``aggregate_geneeffect`` reads them."""
    lines, genes = list(prediction.index), list(prediction.columns)
    scale = definitions.residual_scale.loc[genes].to_numpy()
    residual = truth.loc[lines, genes].to_numpy() * scale
    predicted = prediction.to_numpy() * scale
    keep = np.isfinite(residual)
    line_index, gene_index = np.nonzero(keep)
    means = definitions.gene_means.loc[genes].to_numpy()[gene_index]
    return pd.DataFrame(
        {
            "model_id": np.asarray(lines)[line_index],
            "gene_symbol": np.asarray(genes)[gene_index],
            "residual": residual[keep],
            "residual_prediction": predicted[keep],
            "gene_effect": residual[keep] + means,
            "geneeffect_prediction": predicted[keep] + means,
        }
    )


def score(
    prediction: pd.DataFrame, truth: pd.DataFrame, definitions: Definitions
) -> dict:
    from src.eval.geneeffect import aggregate_geneeffect

    frame = long_frame(prediction, truth, definitions)
    metrics, _, _ = aggregate_geneeffect(
        frame,
        model_ids=list(prediction.index),
        genes=list(definitions.genes),
        variable_genes=list(definitions.variable),
        selective_genes=list(definitions.selective),
    )
    return {key: metrics[key] for key in METRICS}


def gain(
    better: pd.DataFrame,
    reference: pd.DataFrame,
    truth: pd.DataFrame,
    definitions: Definitions,
    repeats: int,
) -> dict:
    """Selective-Spearman difference and its 95% paired line-bootstrap interval."""
    genes = list(definitions.selective)
    result = paired_line_bootstrap(
        long_frame(better.loc[:, genes], truth, definitions),
        long_frame(reference.loc[:, genes], truth, definitions),
        genes,
        repeats=repeats,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "difference": float(result["difference"]),
        "interval": [float(value) for value in result["interval"]],
    }


def _score_key(value: float | None) -> float:
    """Ranking key: an undefined score is the worst."""
    return value if value is not None and math.isfinite(value) else -math.inf


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------


def prior_inputs(base: RunBase, inputs: BridgeInputs) -> PriorInputs:
    """The prior's inputs from a remedy's rows; labels of every labelled
    training-side line and every single-cell training line (for gene rows)."""
    lines = list(dict.fromkeys([*base.labelled, *base.bridge.single_cell_train]))
    return PriorInputs(
        expression=inputs.expression,
        residual=base.truth(lines),
        components=fit_expression_components(
            inputs.expression, int(base.config["prior"]["components"])
        ),
        reference=base.reference,
        lineage=base.models.loc[list(inputs.expression.index), "lineage"],
        patients=base.models["patient_id"].to_dict(),
        gene_space=inputs.gene_space,
        gene_rows=inputs.gene_rows,
    )


def _build(base: RunBase, kind: str, setting: Mapping[str, Any]) -> BridgeInputs:
    module = importlib.import_module(f"src.context_prior.remedies.{kind}")
    return module.build(base.bridge, setting)


class _Fitter:
    """Fits the prior of a list of stages on the labelled training lines and
    scores its total prediction of every query."""

    def __init__(self, base: RunBase, inputs: BridgeInputs) -> None:
        self.base = base
        self.inputs = inputs
        self.prior = prior_inputs(base, inputs)
        self.truth = {
            name: base.truth(list(query.index))
            for name, query in inputs.queries.items()
        }

    def __call__(
        self, stages: Sequence[Stage]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
        fitted = fit_prior(
            PriorSpec(tuple(stages)),
            self.prior,
            fit_lines=list(self.base.labelled),
            encoder_lines=list(self.prior.expression.index),
        )
        predictions = {
            name: total(fitted.predict(query))
            for name, query in self.inputs.queries.items()
        }
        scores = {
            name: score(frame, self.truth[name], self.base.definitions)
            for name, frame in predictions.items()
        }
        return predictions, scores

    def components(self) -> dict[float, tuple[dict, dict]]:
        """Every components penalty of the config, keyed by penalty."""
        return {
            float(p): self([Stage("expression_components", float(p))])
            for p in self.base.config["penalties"]["components"]
        }


def _best_penalty(grid: Mapping[float, tuple[dict, dict]]) -> float:
    """The components penalty with the best validation selective Spearman."""
    return max(grid, key=lambda p: _score_key(grid[p][1]["val"]["selective_spearman"]))


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
    inputs = _build(base, experiment["kind"], setting)
    fit = _Fitter(base, inputs)
    repeats = int(config["prior"]["bootstrap_repeats"])
    selected = int(config["prior"]["selected_genes"])
    grid = fit.components()
    rows = []

    def add(block_set, components_penalty, gene_penalty, predictions, scores):
        print(
            f"  {block_set}, components {components_penalty:g}, gene "
            f"{'-' if gene_penalty is None else f'{gene_penalty:g}'}: validation "
            f"selective Spearman {_number(scores['val']['selective_spearman'])}",
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
                stages = [Stage("expression_components", components)] + [
                    Stage(b, float(value), selected if b == "data_selected" else 0)
                    for b in blocks[1:]
                ]
                add(name, components, float(value), *fit(stages))
    diagnostics = bridge_diagnostics(
        inputs.oof_paired,
        inputs.oof_bulk,
        base.definitions.selective,
        base.reference.paralogs,
    )
    if inputs.gene_space is not None:
        diagnostics["gene_space"] = len(inputs.gene_space)
    return {"rows": rows, "diagnostics": diagnostics}


def reference_predictions(base: RunBase) -> dict[str, pd.DataFrame]:
    """The reference experiment's first setting, components only, at its best
    validation penalty: its ``val`` and ``test`` predictions."""
    experiment = base.config["experiments"][base.config["reference"]]
    grid = _Fitter(
        base, _build(base, experiment["kind"], experiment["settings"][0])
    ).components()
    predictions, _ = grid[_best_penalty(grid)]
    return {name: predictions[name] for name in SCORED}


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


def _jsonable(value: Any) -> Any:
    """Plain JSON values: numpy scalars become Python numbers, NaN becomes None
    (undefined, never 0) and an infinite float the string ``"inf"``."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def _write_text(path: Path, text: str) -> None:
    """Atomic, and safe when several processes write the same path."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n")


@contextmanager
def _locked(run_dir: Path) -> Iterator[None]:
    """Serialises the run directory's shared files across processes: the config
    check and its creation, and a results scan with its replacement (so an older
    scan never overwrites a newer one)."""
    with open(run_dir / ".lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _bind(run_dir: Path, config: Mapping[str, Any]) -> None:
    """A run directory belongs to one config: finished settings are skipped by
    file existence, so resuming under another would mix experiments."""
    record = _jsonable({"config": config})
    path = run_dir / "run_config.json"
    with _locked(run_dir):
        if path.is_file():
            if _read_json(path) != record:
                raise ValueError(
                    f"{run_dir} was started with a different config ({path}); "
                    "use a new --run-id"
                )
            return
        _write_json(path, record)


def _row_path(run_dir: Path, experiment: str, index: int) -> Path:
    return run_dir / "rows" / f"{experiment}__{index}.json"


def _label(setting: Mapping[str, Any]) -> str:
    return ", ".join(f"{key} {value}" for key, value in setting.items()) or "-"


def _penalty(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


def _gain(entry: Mapping[str, Any]) -> str:
    low, high = entry["interval"]
    return f"{_number(entry['difference'])} [{_number(low)}, {_number(high)}]"


def _counts(entry: Mapping[str, Any]) -> str:
    return " / ".join(str(entry[str(t)]) for t in (0.3, 0.5, 0.7)) + (
        f" of {entry['total']}"
    )


def write_results(run_dir: Path) -> Path:
    """``results.md`` from every row file present: run facts, then per experiment
    a table with one row per setting x block set x penalty and a diagnostics
    table with one row per setting."""
    with _locked(run_dir):
        return _write_results(run_dir)


def _write_results(run_dir: Path) -> Path:
    config = _read_json(run_dir / "run_config.json")["config"]
    out = [f"# Linear context prior experiments, run {run_dir.name}", ""]
    if (run_dir / "facts.json").is_file():
        facts = _read_json(run_dir / "facts.json")
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
    out += [
        "Selective Spearman: macro mean over the selective genes of the Spearman "
        "across lines of the residual. Gain: selective Spearman minus that of the "
        f"reference row (experiment `{config['reference']}`, first setting, "
        "expression components alone at their best validation penalty), with its "
        "95% paired line-bootstrap interval. Oracle: validation lines' bulk RNA as "
        "input (off-contract). A gene-level block set has a row per components "
        "and gene penalty pair. Nothing here is synthetic-lethality evidence.",
    ]
    records: dict[str, list[dict]] = {}
    for path in (run_dir / "rows").glob("*.json"):
        record = _read_json(path)
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
                    _number(row["val"]["selective_spearman"]),
                    _number(row["test"]["selective_spearman"]),
                    _gain(row["val_gain"]),
                    _gain(row["test_gain"]),
                    _number(row["val"]["selective_aupr_lift"]),
                    _number(row["test"]["selective_aupr_lift"]),
                    _number(row["val"]["residual_pearson_macro_per_gene"]),
                    _number(row["test"]["residual_pearson_macro_per_gene"]),
                    _number(row["oracle"]["selective_spearman"]),
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
                _number(report["median"], 3),
                _number(report["q25"], 3),
                _number(report["q75"], 3),
                _counts(report["all"]),
                _counts(report["selective"]),
                _counts(report["selective_paralogs"]),
                str(report["gene_space"]) if "gene_space" in report else "all",
            ]
            out.append("| " + " | ".join(cells) + " |")
    path = run_dir / "results.md"
    _write_text(path, "\n".join(out) + "\n")
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
    _bind(run_dir, config)
    pending = [
        (name, index, setting)
        for name in names
        for index, setting in enumerate(config["experiments"][name]["settings"])
        if not _row_path(run_dir, name, index).is_file()
    ]
    if pending:
        base = load_base(config)
        if not (run_dir / "facts.json").is_file():
            _write_json(run_dir / "facts.json", _facts(base))
        print("reference row", flush=True)
        reference = reference_predictions(base)
        for name, index, setting in pending:
            experiment = config["experiments"][name]
            print(f"{name}, setting {index}: {_label(setting)}", flush=True)
            result = run_setting(base, experiment, setting, reference)
            _write_json(
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
