"""The linear context prior run: learning curve, extra-lines decision, block
selection, cross-fitting and one test score.

``python -m src.experiments.context_prior CONFIG [--run-id ID] [--oracle-only]``
writes ``<output_root>/<run id>/``. With ``--oracle-only`` (the Mac) only the
learning curve and the extra-lines decision run, scored from validation lines' bulk
RNA: an off-contract upper bound, so the decision is not binding. The full run (the
H20 host) prepares pseudo-bulk, fits the bridge, scores the curve from bridged
pseudo-bulk as well, decides on the extra lines, selects blocks on validation,
cross-fits the chosen prior and scores it once on test against the existing
controls. A step whose output exists is skipped, so rerunning with the same run id
resumes; a run directory belongs to one config and mode.

Outputs: ``curve.json`` and ``decision.json`` (curve and decision);
``selection.json`` (the chosen prior and the selection log); ``oof.parquet``,
``folds.json``, ``bridge_quality.csv``, ``view_weights.parquet`` (when kept) and
``crossfit.json`` (cross-fitting and the view-weights candidate); ``baselines/``,
``predictions.parquet`` and ``metrics.json`` (validation and test scores); and
``summary.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.context_prior.bridge import Bridge, bridge_quality, fit_bridge
from src.context_prior.folds import patient_folds, patient_subset
from src.context_prior.prior import (
    BLOCKS,
    CONTEXT_BLOCKS,
    PriorInputs,
    PriorSpec,
    Stage,
    crossfit,
    fit_prior,
    total,
)
from src.context_prior.reference import load_reference
from src.context_prior.space import quantile_normalize, quantile_reference
from src.context_prior.targets import Definitions, fit_definitions, residual_frame
from src.context_prior.view_weights import ViewWeights, fit_view_weights
from src.context_prior.views import fit_expression_components
from src.data.depmap import read_depmap_matrix, read_models
from src.data.embeddings import load_esm2_embeddings
from src.data.extra_lines import load_extra_lines
from src.data.splits import FixedSplit, load_geneeffect_226_split
from src.eval.metrics import macro_gene_spearman, paired_line_bootstrap
from src.experiments.all import BASELINE_NAMES, _number, _read_json
from src.experiments.config import load_config, load_prior_config

BOOTSTRAP_SEED = 0
TX1_RIDGE = "context_pca_ridge[tx1]"
#: The curve configs; the extra-lines decision reads the second.
CURVE_CONFIGS = ("components", "components_and_selected")
DECISION_CONFIG = CURVE_CONFIGS[1]
#: Scored splits of the chosen prior, with their summary titles.
EVALUATED = {
    "val": "Validation",
    "test": "Test",
    "oracle_val": (
        "Validation, bulk input (off-contract upper bound; never a model or a "
        "comparison row)"
    ),
}


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
class RunData:
    """Everything the steps read, keyed by ModelID.

    Attributes:
        config: The prior config.
        joint: The joint config it names (split, labels, features, prepared root).
        split: The 226-line split.
        gene_effect: GeneEffect of the split's lines and the kept labelled extra
            lines, over the definitions' genes.
        definitions: Gene order, means, selective and variable genes and residual
            SD, fitted on the 170 labelled single-cell training lines.
        inputs: What the prior fits: the training side's normalised bulk and the
            residual of ``labelled``.
        labelled: Lines whose labels train the prior: every labelled training-side
            line with bulk RNA, or the single-cell training lines alone once the
            extra lines fail the decision.
        single_cell_train: Labelled single-cell training lines with bulk RNA; the
            bridge's lines and the curve's single-cell point.
        lineage: Every line's lineage, for the descriptive table.
        pseudobulk: Quantile-normalised pseudo-bulk of the 226 lines; None with
            ``oracle_only``.
        bridge: The bridge fitted on ``single_cell_train``; None with
            ``oracle_only``.
        queries: ``oracle``: validation lines' normalised bulk (off-contract);
            ``val``: bridged validation pseudo-bulk (absent with ``oracle_only``).
    """

    config: Mapping[str, Any]
    joint: Mapping[str, Any]
    split: FixedSplit
    gene_effect: pd.DataFrame
    definitions: Definitions
    inputs: PriorInputs
    labelled: tuple[str, ...]
    single_cell_train: tuple[str, ...]
    lineage: pd.Series
    pseudobulk: pd.DataFrame | None
    bridge: Bridge | None
    queries: Mapping[str, pd.DataFrame]

    def truth(self, lines: Sequence[str]) -> pd.DataFrame:
        """Residual of ``lines`` in residual-SD units; reads their labels."""
        return scaled_residual(self.gene_effect, lines, self.definitions)


def load_run_data(config: Mapping[str, Any], *, oracle_only: bool) -> RunData:
    """Read, normalise and bridge everything the steps share.

    Test lines' bulk RNA and the excluded lines are dropped as soon as the bulk
    matrix is read; validation lines' bulk RNA feeds the oracle query alone.
    """
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
    single_cell_train = tuple(m for m in split.supervised_train if m in bulk.index)

    gene_effect = read_depmap_matrix(Path(joint["paths"]["gene_effect"]))
    needed = {*split.all_model_ids, *kept(extra.labelled)}
    gene_effect = gene_effect.loc[gene_effect.index.isin(needed)]

    pseudo = None
    if oracle_only:
        panel = [g for g in gene_effect.columns if g in bulk.columns]
        space = list(bulk.columns)
    else:
        from src.data.prepared import read_manifest
        from src.data.pseudobulk import read_pseudobulk
        from src.experiments.prepare import prepare_pseudobulk

        prepared = Path(joint["prepared_root"])
        panel = list(read_manifest(prepared)["common_gene_panel"])
        prepare_pseudobulk(joint, list(bulk.columns))
        pseudo = read_pseudobulk(prepared)
        finite = np.isfinite(pseudo.loc[:, list(bulk.columns)].to_numpy()).all(axis=0)
        space = [g for g, ok in zip(bulk.columns, finite, strict=True) if ok]
    definitions = fit_definitions(gene_effect, split, panel, joint["features"])
    gene_effect = gene_effect.loc[:, list(definitions.genes)]

    # Training-side rows first, then the oracle's validation rows; nothing else.
    bulk = bulk.loc[[*side, *oracle_lines], space]
    reference_profile = quantile_reference(bulk.iloc[: len(side)])
    normalized = quantile_normalize(bulk, reference_profile)
    del bulk
    expression = normalized.loc[side]
    queries = {"oracle": normalized.loc[oracle_lines]}
    pseudobulk = bridge = None
    if pseudo is not None:
        pseudobulk = quantile_normalize(pseudo.loc[:, space], reference_profile)
        bridge = fit_bridge(
            pseudobulk.loc[list(single_cell_train)],
            expression.loc[list(single_cell_train)],
        )
        queries["val"] = bridge.apply(pseudobulk.loc[list(split.val)])
    del normalized

    inputs = PriorInputs(
        expression=expression,
        residual=scaled_residual(gene_effect, labelled, definitions),
        components=fit_expression_components(
            expression, int(config["prior"]["components"])
        ),
        reference=load_reference(
            Path(paths["reference"]),
            blocked={*split.val, *split.test, *extra.excluded},
        ),
        lineage=models.loc[side, "lineage"],
        patients=models["patient_id"].to_dict(),
    )
    return RunData(
        config=config,
        joint=joint,
        split=split,
        gene_effect=gene_effect,
        definitions=definitions,
        inputs=inputs,
        labelled=labelled,
        single_cell_train=single_cell_train,
        lineage=models["lineage"],
        pseudobulk=pseudobulk,
        bridge=bridge,
        queries=queries,
    )


def single_cell_only(data: RunData) -> RunData:
    """The prior trained on the single-cell training lines with bulk RNA alone:
    the extra lines failed the decision, so their labels leave the fit."""
    lines = data.single_cell_train
    inputs = dataclasses.replace(
        data.inputs, residual=data.inputs.residual.loc[list(lines)]
    )
    return dataclasses.replace(data, labelled=lines, inputs=inputs)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def selective_score(
    prediction: pd.DataFrame, truth: pd.DataFrame, selective: Sequence[str]
) -> float:
    """Validation selective Spearman of ``prediction``'s lines (the selector)."""
    genes = list(selective)
    lines = list(prediction.index)
    return macro_gene_spearman(
        truth.loc[lines, genes].to_numpy().T, prediction.loc[:, genes].to_numpy().T
    )


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


def paired_gain(
    data: RunData, better: pd.DataFrame, simpler: pd.DataFrame, truth: pd.DataFrame
) -> dict[str, Any]:
    """Selective-Spearman gain of ``better`` over ``simpler`` on the same lines:
    the observed difference and its 95% paired line-bootstrap interval."""
    genes = list(data.definitions.selective)
    result = paired_line_bootstrap(
        long_frame(better.loc[:, genes], truth, data.definitions),
        long_frame(simpler.loc[:, genes], truth, data.definitions),
        genes,
        repeats=int(data.config["prior"]["bootstrap_repeats"]),
        seed=BOOTSTRAP_SEED,
    )
    return {
        "difference": float(result["difference"]),
        "interval": [float(value) for value in result["interval"]],
    }


def _score_key(score: float) -> float:
    """Ranking key: an undefined score is the worst."""
    return score if math.isfinite(score) else -math.inf


def _best(fits: Sequence[tuple[float, Any, pd.DataFrame]]):
    return max(fits, key=lambda item: _score_key(item[0]))


# ----------------------------------------------------------------------------
# Learning curve and the extra-lines decision
# ----------------------------------------------------------------------------


def curve_points(data: RunData) -> list[tuple[str, tuple[str, ...]]]:
    """The single-cell training lines, seeded patient subsets, then every line."""
    curve = data.config["curve"]
    seed = int(data.config["seed"])
    points = [("single_cell_train", data.single_cell_train)]
    for size in curve["sizes"]:
        for subset in range(int(curve["subsets"])):
            lines = patient_subset(
                data.labelled, data.inputs.patients, size=int(size), seed=seed + subset
            )
            points.append((f"random_{size}_{subset}", lines))
    points.append(("all", data.labelled))
    return points


def run_curve(data: RunData) -> tuple[list[dict], dict]:
    """Rows of the learning curve and the chosen predictions of each point, config
    and input; ridge penalties are chosen on validation per input."""
    penalties = [float(p) for p in data.config["selection"]["penalties"]]
    count = int(data.config["curve"]["selected"])
    truth = data.truth(data.split.val)
    encoder = list(data.inputs.expression.index)
    names = list(data.queries)
    rows: list[dict] = []
    predictions: dict[tuple[str, str, str], pd.DataFrame] = {}

    def score(frame: pd.DataFrame) -> float:
        return selective_score(frame, truth, data.definitions.selective)

    def predict(spec: PriorSpec, lines, wanted) -> dict[str, pd.DataFrame]:
        fitted = fit_prior(spec, data.inputs, fit_lines=lines, encoder_lines=encoder)
        return {name: total(fitted.predict(data.queries[name])) for name in wanted}

    for point, lines in curve_points(data):
        print(f"learning curve: {point}, {len(lines)} lines", flush=True)
        components = {
            penalty: predict(
                PriorSpec((Stage("expression_components", penalty),)), lines, names
            )
            for penalty in penalties
        }
        chosen = {}
        for name in names:
            value, penalty, frame = _best(
                [(score(fits[name]), p, fits[name]) for p, fits in components.items()]
            )
            chosen[name] = penalty
            rows.append(
                {
                    "point": point,
                    "lines": len(lines),
                    "config": "components",
                    "input": name,
                    "penalties": [penalty],
                    "score": value,
                }
            )
            predictions[(point, "components", name)] = frame
        del components
        # One fit per pair of penalties, shared by the inputs that chose the first.
        selected: dict[str, list] = {name: [] for name in names}
        for first in sorted(set(chosen.values())):
            sharing = [name for name in names if chosen[name] == first]
            for second in penalties:
                spec = PriorSpec(
                    (
                        Stage("expression_components", first),
                        Stage("data_selected", second, selected=count),
                    )
                )
                for name, frame in predict(spec, lines, sharing).items():
                    selected[name].append((score(frame), second, frame))
        for name in names:
            value, second, frame = _best(selected[name])
            rows.append(
                {
                    "point": point,
                    "lines": len(lines),
                    "config": DECISION_CONFIG,
                    "input": name,
                    "penalties": [chosen[name], second],
                    "score": value,
                }
            )
            predictions[(point, DECISION_CONFIG, name)] = frame
    return rows, predictions


def decide(data: RunData, predictions: Mapping[tuple, pd.DataFrame]) -> dict:
    """Extra lines pass when the all-lines prior beats the single-cell-line prior
    (components plus data-selected genes) with an interval excluding zero; binding
    only on bridged input."""
    name = "val" if "val" in data.queries else "oracle"
    truth = data.truth(data.split.val)

    def compare(input_name: str) -> dict[str, Any]:
        return paired_gain(
            data,
            predictions[("all", DECISION_CONFIG, input_name)],
            predictions[("single_cell_train", DECISION_CONFIG, input_name)],
            truth,
        )

    gain = compare(name)
    decision = {
        "input": name,
        "binding": name == "val",
        "config": DECISION_CONFIG,
        "lines": {
            "all": len(data.labelled),
            "single_cell_train": len(data.single_cell_train),
        },
        "difference": gain["difference"],
        "interval": gain["interval"],
        "passes": bool(gain["interval"][0] > 0),
    }
    if name == "val":
        oracle = compare("oracle")
        decision["oracle_difference"] = oracle["difference"]
        decision["oracle_interval"] = oracle["interval"]
        decision["bridge_failing"] = bool(
            oracle["interval"][0] > 0 and not decision["passes"]
        )
    return decision


# ----------------------------------------------------------------------------
# Block selection
# ----------------------------------------------------------------------------


def select_blocks(
    evaluate: Callable[[PriorSpec], tuple[float, pd.DataFrame]],
    interval: Callable[[pd.DataFrame, pd.DataFrame, str], list[float]],
    selection: Mapping[str, Any],
) -> tuple[PriorSpec, list[dict]]:
    """Add blocks in the fixed order; keep one only if its gain interval over the
    current prior excludes zero. Penalties (and N) are chosen by point estimate;
    the partner group reuses the own-expression shrinkage. The first block is the
    base and is always kept; the reduced rank is tried last."""
    spec, current, log, shrinkage = PriorSpec(()), None, [], None
    for block in BLOCKS:
        if block in CONTEXT_BLOCKS:
            grid = [Stage(block, float(p)) for p in selection["penalties"]]
        elif block == "own_expression":
            grid = [Stage(block, float(s)) for s in selection["shrinkages"]]
        elif block == "partners":
            grid = [Stage(block, shrinkage)]
        elif block == "data_selected":
            grid = [
                Stage(block, float(p), int(n))
                for n in selection["selected"]
                for p in selection["penalties"]
            ]
        else:
            raise ValueError(f"block {block!r} has no selection grid")
        (score, prediction), stage = max(
            ((evaluate(PriorSpec((*spec.stages, s), spec.rank)), s) for s in grid),
            key=lambda item: _score_key(item[0][0]),
        )
        if block == "own_expression":
            shrinkage = stage.penalty
        gain = None if current is None else interval(prediction, current, block)
        kept = gain is None or gain[0] > 0
        log.append(
            {
                "block": block,
                "penalty": stage.penalty,
                "selected": stage.selected,
                "rank": None,
                "score": score,
                "interval": gain,
                "kept": kept,
            }
        )
        if kept:
            spec, current = PriorSpec((*spec.stages, stage), spec.rank), prediction
    if selection["rank"]:
        candidate = PriorSpec(spec.stages, int(selection["rank"]))
        score, prediction = evaluate(candidate)
        gain = interval(prediction, current, "rank")
        kept = gain[0] > 0
        log.append(
            {
                "block": "reduced_rank",
                "penalty": None,
                "selected": 0,
                "rank": candidate.rank,
                "score": score,
                "interval": gain,
                "kept": kept,
            }
        )
        if kept:
            spec = candidate
    return spec, log


def _fit_all(data: RunData, spec: PriorSpec):
    return fit_prior(
        spec,
        data.inputs,
        fit_lines=list(data.labelled),
        encoder_lines=list(data.inputs.expression.index),
    )


def _describe(spec: PriorSpec) -> str:
    stages = ", ".join(
        f"{s.block}({s.penalty:g}" + (f", N={s.selected})" if s.selected else ")")
        for s in spec.stages
    )
    return stages + (f", rank {spec.rank}" if spec.rank else "")


def run_selection(data: RunData) -> tuple[PriorSpec, list[dict]]:
    """Block-by-block selection on bridged validation pseudo-bulk."""
    truth = data.truth(data.split.val)
    query = data.queries["val"]

    def evaluate(spec: PriorSpec) -> tuple[float, pd.DataFrame]:
        print(f"selection: {_describe(spec)}", flush=True)
        prediction = total(_fit_all(data, spec).predict(query))
        score = selective_score(prediction, truth, data.definitions.selective)
        return score, prediction

    def interval(prediction, current, block):
        return paired_gain(data, prediction, current, truth)["interval"]

    return select_blocks(evaluate, interval, data.config["selection"])


def _spec_json(spec: PriorSpec) -> dict:
    return {
        "stages": [
            {"block": s.block, "penalty": s.penalty, "selected": s.selected}
            for s in spec.stages
        ],
        "rank": spec.rank,
    }


def _spec_from_json(payload: Mapping) -> PriorSpec:
    stages = tuple(
        Stage(s["block"], float(s["penalty"]), int(s["selected"]))
        for s in payload["stages"]
    )
    rank = payload["rank"]
    return PriorSpec(stages, None if rank is None else int(rank))


# ----------------------------------------------------------------------------
# Cross-fitting and the view-weights candidate
# ----------------------------------------------------------------------------


def run_crossfit(
    data: RunData, spec: PriorSpec
) -> tuple[dict[str, pd.DataFrame], pd.Series, dict[str, int]]:
    """Out-of-fold stage predictions, bridge quality and the fold of each line.

    Folds cover every training-side line plus the 170 single-cell training lines
    (the 3 without bulk RNA appear only as queries). Single-cell training lines are
    predicted from bridged pseudo-bulk, the bridge refitted without their fold;
    labelled extra lines from bulk. Bridge quality is over the out-of-fold lines
    with bulk RNA.
    """
    lines = sorted({*data.inputs.expression.index, *data.split.supervised_train})
    folds = patient_folds(
        lines,
        data.inputs.patients,
        n_folds=int(data.config["prior"]["folds"]),
        seed=int(data.config["seed"]),
    )
    single_cell = list(data.split.supervised_train)
    extras = [m for m in data.labelled if m not in set(single_cell)]
    queries, bridged_parts = {}, []
    for fold in sorted(set(folds.values())):
        bridge_lines = [m for m in data.single_cell_train if folds[m] != fold]
        bridge = fit_bridge(
            data.pseudobulk.loc[bridge_lines], data.inputs.expression.loc[bridge_lines]
        )
        held = [m for m in single_cell if folds[m] == fold]
        bridged = bridge.apply(data.pseudobulk.loc[held])
        bridged_parts.append(bridged)
        queries[fold] = pd.concat(
            [
                bridged,
                data.inputs.expression.loc[[m for m in extras if folds[m] == fold]],
            ]
        )
    stages = crossfit(
        spec,
        data.inputs,
        folds=folds,
        queries=queries,
        labelled=list(data.labelled),
        encoder_lines=list(data.inputs.expression.index),
    )
    bridged = pd.concat(bridged_parts)
    with_bulk = [m for m in bridged.index if m in data.inputs.expression.index]
    quality = bridge_quality(
        bridged.loc[with_bulk], data.inputs.expression.loc[with_bulk]
    )
    return stages, quality, folds


def run_view_weights(
    data: RunData, spec: PriorSpec, stages: Mapping[str, pd.DataFrame]
) -> tuple[ViewWeights | None, dict[str, Any]]:
    """Gene-conditioned view weights over the chosen context blocks, trained on
    every out-of-fold row; kept only if their validation gain over the plain sum
    has an interval excluding zero. Tried with two or more context blocks."""
    context = [s.block for s in spec.stages if s.block in CONTEXT_BLOCKS]
    if not data.config["selection"]["view_weights"] or len(context) < 2:
        return None, {"tried": False, "kept": False}
    print("view weights", flush=True)
    table = load_esm2_embeddings(Path(data.joint["paths"]["esm2_embeddings"]))
    rows = list(stages[context[0]].index)
    weights = fit_view_weights(
        stages, data.truth(rows), table.vectors_by_symbol, context
    )
    val_stages = _fit_all(data, spec).predict(data.queries["val"])
    gain = paired_gain(
        data,
        weights.combine(val_stages),
        total(val_stages),
        data.truth(data.split.val),
    )
    kept = bool(gain["interval"][0] > 0)
    record = {"tried": True, "blocks": context, "kept": kept, **gain}
    return (weights if kept else None), record


def write_oof(stages: Mapping[str, pd.DataFrame], path: Path) -> None:
    """Long ``block, model_id, gene_symbol, prediction_sigma`` rows, one row group
    per block, with categorical keys (about 20M rows per block at full size)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    blocks = list(stages)
    first = stages[blocks[0]]
    lines, genes = pd.Index(first.index), pd.Index(first.columns)
    temporary = path.with_name(path.name + ".tmp")
    writer = None
    try:
        for code, block in enumerate(blocks):
            frame = stages[block]
            rows = lines.get_indexer(frame.index).astype(np.int32)
            columns = genes.get_indexer(frame.columns).astype(np.int32)
            if (rows < 0).any() or (columns < 0).any():
                raise ValueError("out-of-fold stages differ in lines or genes")
            part = pd.DataFrame(
                {
                    "block": pd.Categorical.from_codes(
                        np.full(frame.size, code, dtype=np.int8), categories=blocks
                    ),
                    "model_id": pd.Categorical.from_codes(
                        np.repeat(rows, len(columns)), categories=list(lines)
                    ),
                    "gene_symbol": pd.Categorical.from_codes(
                        np.tile(columns, len(rows)), categories=list(genes)
                    ),
                    "prediction_sigma": frame.to_numpy(dtype=np.float64).ravel(),
                }
            )
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary, path)


# ----------------------------------------------------------------------------
# Validation and test
# ----------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _lineage_rows(
    data: RunData, prediction: pd.DataFrame, truth: pd.DataFrame, split_name: str
) -> list[dict]:
    """Mean per-line Pearson over selective genes (residual-SD units) by lineage."""
    genes = list(data.definitions.selective)
    x = prediction.loc[:, genes].to_numpy()
    y = truth.loc[list(prediction.index), genes].to_numpy()
    rows = []
    for i, line in enumerate(prediction.index):
        keep = np.isfinite(y[i])
        lineage = data.lineage.get(line)
        rows.append(
            {
                "lineage": lineage if isinstance(lineage, str) else "unknown",
                "pearson": _pearson(x[i, keep], y[i, keep]),
            }
        )
    grouped = pd.DataFrame(rows).groupby("lineage")["pearson"]
    return [
        {
            "split": split_name,
            "lineage": str(lineage),
            "lines": int(group.size),
            "mean": float(group.mean()),
        }
        for lineage, group in grouped
    ]


def _controls(
    data: RunData, frame: pd.DataFrame, split_name: str, run_dir: Path
) -> dict[str, Any]:
    """The control ladder on ``split_name`` (fitted once, then reused) and the
    paired bootstrap of the prior minus the Tx1 context-PCA ridge on their common
    (line, gene) keys."""
    from src.experiments.baselines import run_baselines

    baseline_dir = run_dir / "baselines" / split_name
    if not (baseline_dir / "metrics.json").is_file():
        print(f"{EVALUATED[split_name].lower()} baselines", flush=True)
        run_baselines(dict(data.joint), split=split_name, out_dir=baseline_dir)
    controls = _read_json(baseline_dir / "metrics.json")
    if TX1_RIDGE not in controls:
        raise ValueError(f"{baseline_dir}/metrics.json has no {TX1_RIDGE} method")
    keys = ["model_id", "gene_symbol"]
    ridge = pd.read_parquet(baseline_dir / "predictions.parquet")
    ridge = ridge.loc[
        ridge["method"] == TX1_RIDGE, [*keys, "residual", "residual_prediction"]
    ]
    prior = frame.merge(ridge[keys], on=keys)
    ridge = ridge.merge(prior[keys], on=keys)
    result = paired_line_bootstrap(
        prior,
        ridge,
        data.definitions.selective,
        repeats=int(data.config["prior"]["bootstrap_repeats"]),
        seed=BOOTSTRAP_SEED,
    )
    return {
        "controls": controls,
        "prior_minus_tx1_ridge": {
            "pairs": len(prior),
            "difference": float(result["difference"]),
            "interval": [float(value) for value in result["interval"]],
        },
    }


def run_test(
    data: RunData, spec: PriorSpec, weights: ViewWeights | None, run_dir: Path
) -> dict:
    """Score the chosen prior, fitted on the whole training side, on validation and
    once on test against the controls; the oracle row is validation-only."""
    from src.eval.geneeffect import aggregate_geneeffect

    fitted = _fit_all(data, spec)
    queries = {
        "val": data.queries["val"],
        "test": data.bridge.apply(data.pseudobulk.loc[list(data.split.test)]),
        "oracle_val": data.queries["oracle"],
    }
    record: dict[str, Any] = {"splits": {}, "lineages": []}
    frames = []
    for split_name, query in queries.items():
        print(f"scoring the prior: {EVALUATED[split_name].lower()}", flush=True)
        stages = fitted.predict(query)
        prediction = total(stages) if weights is None else weights.combine(stages)
        truth = data.truth(list(query.index))
        frame = long_frame(prediction, truth, data.definitions)
        metrics, _, _ = aggregate_geneeffect(
            frame,
            model_ids=list(query.index),
            genes=list(data.definitions.genes),
            variable_genes=list(data.definitions.variable),
            selective_genes=list(data.definitions.selective),
        )
        entry: dict[str, Any] = {"prior": metrics}
        if split_name != "oracle_val":
            entry.update(_controls(data, frame, split_name, run_dir))
            record["lineages"] += _lineage_rows(data, prediction, truth, split_name)
        record["splits"][split_name] = entry
        frames.append(frame.assign(split=split_name))
    pd.concat(frames, ignore_index=True).to_parquet(
        run_dir / "predictions.parquet", index=False
    )
    return record


# ----------------------------------------------------------------------------
# JSON and summary.md
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


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _bind(run_dir: Path, config: Mapping[str, Any], *, oracle_only: bool) -> None:
    """A run directory belongs to one config and mode: finished steps are skipped
    by file existence, so resuming under another would mix experiments."""
    record = _jsonable({"config": config, "oracle_only": oracle_only})
    path = run_dir / "run_config.json"
    if path.is_file():
        if _read_json(path) != record:
            raise ValueError(
                f"{run_dir} was started with a different config or mode ({path}); "
                "use a new --run-id"
            )
        return
    _write_json(path, record)


def _interval(values: Sequence[Any] | None) -> str:
    if values is None:
        return "base"
    low, high = values
    return f"[{_number(low)}, {_number(high)}]"


def _setting(entry: Mapping[str, Any]) -> str:
    if entry["block"] == "reduced_rank":
        return f"rank {entry['rank']}"
    name = (
        "shrinkage" if entry["block"] in ("own_expression", "partners") else "penalty"
    )
    text = f"{name} {entry['penalty']}"
    return text + (f", N {entry['selected']}" if entry["selected"] else "")


def _metric_table(entry: Mapping[str, Any], split_name: str) -> list[str]:
    columns = (
        ("Selective Spearman", "selective_spearman"),
        ("Selective AUPR lift", "selective_aupr_lift"),
        ("Residual Pearson (per variable gene)", "residual_pearson_macro_per_gene"),
        ("Huber", "geneeffect_loss"),
        ("SD ratio (per gene)", "residual_sd_ratio_macro_per_gene"),
    )
    lines = [
        "| Model | " + " | ".join(name for name, _ in columns) + " |",
        "| --- |" + " --- |" * len(columns),
    ]
    name = "Linear context prior"
    if split_name == "oracle_val":
        name += " (bulk input, off-contract)"
    cells = [_number(entry["prior"].get(key)) for _, key in columns]
    lines.append(f"| {name} | " + " | ".join(cells) + " |")
    controls = entry.get("controls", {})
    order = [m for m in BASELINE_NAMES if m in controls]
    order += sorted(m for m in controls if m not in BASELINE_NAMES)
    for method in order:
        metrics = controls[method]
        cells = [_number(metrics.get(f"{split_name}_{key}")) for _, key in columns]
        label = BASELINE_NAMES.get(method, method)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _decision_lines(decision: Mapping[str, Any]) -> list[str]:
    lines = decision["lines"]
    status = "binding" if decision["binding"] else "not binding (bulk input)"
    out = [
        f"All {lines['all']} labelled training-side lines minus the "
        f"{lines['single_cell_train']} single-cell training lines, expression "
        f"components plus data-selected genes, {decision['input']} input: "
        f"{_number(decision['difference'])} {_interval(decision['interval'])}. "
        f"The extra lines {'pass' if decision['passes'] else 'fail'} ({status}).",
    ]
    if "oracle_interval" in decision:
        out += [
            "",
            f"Bulk input (off-contract): {_number(decision['oracle_difference'])} "
            f"{_interval(decision['oracle_interval'])}."
            + (
                " The gain appears with bulk input but not with bridged input: the "
                "bridge is failing, and contrastive-PCA alignment is the next step."
                if decision["bridge_failing"]
                else ""
            ),
        ]
    if decision["binding"] and not decision["passes"]:
        out += ["", "The prior trains on the single-cell training lines alone."]
    return out


def write_summary(run_dir: Path) -> Path:
    """summary.md from whatever steps have finished."""
    out = [
        f"# Linear context prior, run {run_dir.name}",
        "",
        "Selective Spearman is the macro mean over the selective genes of the "
        "Spearman across lines of the residual; intervals are 95% paired line "
        "bootstraps (1,000 resamples, seed 0) of the gain. Validation chooses; the "
        "chosen prior is scored once on test. Nothing here is synthetic-lethality "
        "evidence.",
        "",
        "## Learning curve",
        "",
        "Validation selective Spearman. `oracle` scores validation lines' bulk RNA "
        "(off-contract upper bound), `val` bridged pseudo-bulk; ridge penalties are "
        "chosen on validation per input.",
        "",
        "| Point | Lines | Config | Input | Penalties | Score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in _read_json(run_dir / "curve.json"):
        penalties = ", ".join(str(p) for p in row["penalties"])
        out.append(
            f"| {row['point']} | {row['lines']} | {row['config']} | {row['input']} "
            f"| {penalties} | {_number(row['score'])} |"
        )
    out += ["", "## Extra-lines decision", ""]
    out += _decision_lines(_read_json(run_dir / "decision.json"))
    if (run_dir / "selection.json").is_file():
        selection = _read_json(run_dir / "selection.json")
        out += [
            "",
            "## Block selection",
            "",
            "| Block | Setting | Score | Gain interval | Kept |",
            "| --- | --- | --- | --- | --- |",
        ]
        out += [
            f"| {entry['block']} | {_setting(entry)} | {_number(entry['score'])} "
            f"| {_interval(entry['interval'])} | {'yes' if entry['kept'] else 'no'} |"
            for entry in selection["log"]
        ]
        out += ["", f"Chosen prior: `{json.dumps(selection['spec'])}`"]
    if (run_dir / "crossfit.json").is_file():
        record = _read_json(run_dir / "crossfit.json")
        quality, weights = record["bridge_quality"], record["view_weights"]
        out += [
            "",
            "## Cross-fitting",
            "",
            f"Bridge quality, the per-gene Pearson across out-of-fold training lines "
            f"between bridged pseudo-bulk and bulk: median "
            f"{_number(quality['median'])} over {quality['defined']} genes "
            f"({quality['undefined']} undefined).",
            "",
            "View weights: "
            + (
                f"{_number(weights['difference'])} {_interval(weights['interval'])} "
                f"over the plain sum, {'kept' if weights['kept'] else 'not kept'}."
                if weights["tried"]
                else "not tried (fewer than two context blocks, or disabled)."
            ),
        ]
    if (run_dir / "metrics.json").is_file():
        record = _read_json(run_dir / "metrics.json")
        for split_name, title in EVALUATED.items():
            entry = record["splits"][split_name]
            out += ["", f"## {title}", ""] + _metric_table(entry, split_name)
            if "prior_minus_tx1_ridge" in entry:
                gain = entry["prior_minus_tx1_ridge"]
                out += [
                    "",
                    "Selective Spearman, linear context prior minus context-PCA "
                    f"ridge (Tx1): {_number(gain['difference'])} "
                    f"{_interval(gain['interval'])} over {gain['pairs']} common "
                    "(line, gene) pairs.",
                ]
        out += [
            "",
            "## Per lineage",
            "",
            "Mean per-line Pearson over the selective genes, in residual-SD units; "
            "descriptive only.",
            "",
            "| Split | Lineage | Lines | Mean |",
            "| --- | --- | --- | --- |",
        ]
        out += [
            f"| {row['split']} | {row['lineage']} | {row['lines']} "
            f"| {_number(row['mean'], 3)} |"
            for row in record["lineages"]
        ]
    path = run_dir / "summary.md"
    path.write_text("\n".join(out) + "\n")
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
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_prior_config(args.config)
    run_id = args.run_id or "prior_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"run id: {run_id}", flush=True)
    run_dir = Path(config["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _bind(run_dir, config, oracle_only=args.oracle_only)
    last = "decision.json" if args.oracle_only else "metrics.json"
    if (run_dir / last).is_file():
        print(f"summary: {write_summary(run_dir)}", flush=True)
        return 0
    data = load_run_data(config, oracle_only=args.oracle_only)
    if not (run_dir / "decision.json").is_file():
        rows, predictions = run_curve(data)
        _write_json(run_dir / "curve.json", rows)
        _write_json(run_dir / "decision.json", decide(data, predictions))
        del predictions
    if args.oracle_only:
        print(f"summary: {write_summary(run_dir)}", flush=True)
        return 0
    if not _read_json(run_dir / "decision.json")["passes"]:
        data = single_cell_only(data)
    if not (run_dir / "selection.json").is_file():
        spec, log = run_selection(data)
        _write_json(run_dir / "selection.json", {"spec": _spec_json(spec), "log": log})
    spec = _spec_from_json(_read_json(run_dir / "selection.json")["spec"])
    if not (run_dir / "crossfit.json").is_file():
        stages, quality, folds = run_crossfit(data, spec)
        write_oof(stages, run_dir / "oof.parquet")
        _write_json(run_dir / "folds.json", folds)
        quality.rename_axis("gene_symbol").reset_index().to_csv(
            run_dir / "bridge_quality.csv", index=False
        )
        weights, weights_record = run_view_weights(data, spec, stages)
        del stages
        if weights is not None:
            weights.weights.to_parquet(run_dir / "view_weights.parquet")
        defined = quality.dropna()
        _write_json(
            run_dir / "crossfit.json",
            {
                "bridge_quality": {
                    "median": float(defined.median()) if len(defined) else None,
                    "defined": len(defined),
                    "undefined": int(quality.isna().sum()),
                },
                "view_weights": weights_record,
            },
        )
    weights = None
    if _read_json(run_dir / "crossfit.json")["view_weights"]["kept"]:
        table = pd.read_parquet(run_dir / "view_weights.parquet")
        weights = ViewWeights(tuple(table.columns), table)
    if not (run_dir / "metrics.json").is_file():
        _write_json(run_dir / "metrics.json", run_test(data, spec, weights, run_dir))
    print(f"summary: {write_summary(run_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
