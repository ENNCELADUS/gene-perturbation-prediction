"""Round zero of the modular SL backbone (SL protocol §11). Four slots: the line
encoder (bridged pseudo-bulk), the dependency reader (the default linear expression
model), the context editor (gene A's expression set to the training side's low level)
and the pair scorer. The reader is fitted on the training side of the SL split, so no
SL validation or test line's GeneEffect enters it: training lines are predicted out of
fold, validation and test lines by the full fit.

The conditional score of a pair (A, B) in line c, in residual-SD units, is

    s(c, A, B) = -[y(c with A lost, B) - y(c, B)] - [y(c with B lost, A) - y(c, A)]

and an anchor list reads the first term only (A is the anchor). "A lost" sets A's
expression to the training side's ``LOW_PERCENTILE`` percentile, and leaves it as it is
when it is already lower. The context-ablated score reads the training-side mean
profile in place of the line's. A logistic head on the training lines' SL labels reads
both terms, the predicted GeneEffect and the expression of both genes and the design;
its penalty is chosen on validation recall in the top 10%. Each score goes to
``scores/<name>.parquet`` for the benchmark harness (``extra_scores``), and the
reader's GeneEffect table to ``predictions.parquet`` (its ``stage_one``)."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.extra_lines import load_bulk_lines
from src.data.io import jsonable, write_json, write_text
from src.data.splits import load_geneeffect_split
from src.eval.accuracy import fold_accuracy, wide_accuracy
from src.experiments.config import read_yaml, require_keys
from src.experiments.linear_model.base import (
    RunBase,
    base_on,
    build,
    linear_model_inputs,
    stages,
)
from src.experiments.linear_model.config import load_linear_model_config
from src.experiments.run_dir import bind
from src.experiments.sl.benchmark_baselines import (
    HEADLINE,
    LIST_KEYS,
    benchmark_rows,
    line_records,
    list_records,
)
from src.linear_model.bridge import fit_bridge, oof_bridged
from src.linear_model.folds import training_side_folds
from src.linear_model.linear_model import (
    LOW_PERCENTILE,
    FittedLinearModel,
    LinearModelSpec,
    fit_linear_model,
    total,
)

#: A score table's key (one score per line, design and ordered pair).
KEYS = ["model_id", "design", "gene_a", "gene_b"]
SCORERS = ("conditional", "conditional_ablated", "head", "head_ablated")
FEATURES = (
    "term_ab",
    "term_ba",
    "effect_a",
    "effect_b",
    "expression_a",
    "expression_b",
    "combinatorial",
)
#: Pairs reported by name where a held-out line screened them.
NAMED = (("SMARCA4", "SMARCA2"), ("ARID1A", "ARID1B"))
#: The full fit's key among the readers (folds are 0..k-1).
FULL = -1
#: Edited genes per prediction call.
CHUNK = 512


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "output_root", "task", "linear_model", "labels", "split",
         "head_penalties", "top_k"},
        "config",
    )  # fmt: skip
    if config["kind"] != "sl_conditional":
        raise ValueError(f"{path} is a {config['kind']} config")
    return config


def drop(expression: pd.DataFrame, low: pd.Series) -> pd.DataFrame:
    """How far each gene falls when lost: ``min(low - x, 0)``."""
    return (low.loc[expression.columns] - expression).clip(upper=0.0)


def pair_responses(
    reader: FittedLinearModel, origin: pd.Series, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Per pair, the change of the reader's prediction of gene ``b`` (residual SD)
    when gene ``a``'s expression rises by one from ``origin``; NaN where ``a`` is not
    an expression gene or ``b`` is not a predicted gene. Exact for an affine reader
    only (``check_affine``)."""
    space, genes = pd.Index(origin.index), pd.Index(reader.genes)
    column = genes.get_indexer(b)
    ok = (space.get_indexer(a) >= 0) & (column >= 0)
    out = np.full(len(a), np.nan)
    edited = pd.unique(a[ok])
    base = total(reader.predict(origin.to_frame().T)).to_numpy()[0]
    for start in range(0, len(edited), CHUNK):
        chunk = pd.Index(edited[start : start + CHUNK])
        values = np.tile(origin.to_numpy(dtype=np.float64), (len(chunk), 1))
        values[np.arange(len(chunk)), space.get_indexer(chunk)] += 1.0
        change = total(reader.predict(pd.DataFrame(values, columns=space))).to_numpy()
        row = chunk.get_indexer(a)
        hit = ok & (row >= 0)
        out[hit] = change[row[hit], column[hit]] - base[column[hit]]
    return out


def check_affine(
    reader: FittedLinearModel, origin: pd.Series, row: pd.Series, gene: str
) -> None:
    """Raise unless editing ``gene`` in ``row`` moves every prediction as
    ``pair_responses`` says from ``origin``: the responses hold for every line only
    while the reader is affine in expression."""
    edited = row.copy()
    edited[gene] -= 1.0
    both = total(reader.predict(pd.DataFrame([edited, row]))).to_numpy()
    genes = np.asarray(reader.genes)
    via = -pair_responses(reader, origin, np.full(len(genes), gene), genes)
    if not np.allclose(both[0] - both[1], via, rtol=1e-5, atol=1e-8):
        raise ValueError(
            "the reader is not affine in expression: query edited rows per line"
        )


def terms(
    rows: pd.DataFrame,
    expression: pd.DataFrame,
    low: pd.Series,
    assignment: Mapping[str, int],
    readers: Mapping[int, FittedLinearModel],
    origin: pd.Series,
) -> np.ndarray:
    """Rows x 2: the change of B's predicted GeneEffect when A is lost in the row's
    line (``term_ab``) and of A's when B is lost (``term_ba``, 0 for an anchor list),
    each from the reader ``assignment`` gives the line."""
    fall = drop(expression, low)
    line = fall.index.get_indexer(rows["model_id"])
    values = fall.to_numpy(dtype=np.float64)

    def falls(gene: str) -> np.ndarray:
        column = fall.columns.get_indexer(rows[gene])
        ok = (line >= 0) & (column >= 0)
        out = np.full(len(rows), np.nan)
        out[ok] = values[line[ok], column[ok]]
        return out

    fa, fb = falls("gene_a"), falls("gene_b")
    key = rows["model_id"].map(assignment).to_numpy()
    pair = (rows["design"] != "anchor").to_numpy()
    a, b = rows["gene_a"].to_numpy(dtype=object), rows["gene_b"].to_numpy(dtype=object)
    out = np.full((len(rows), 2), np.nan)
    out[~pair, 1] = 0.0
    for k, reader in readers.items():
        chosen = key == k
        out[chosen, 0] = fa[chosen] * pair_responses(
            reader, origin, a[chosen], b[chosen]
        )
        both = chosen & pair
        out[both, 1] = fb[both] * pair_responses(reader, origin, b[both], a[both])
    return out


def gather(table: pd.DataFrame, rows: pd.DataFrame, gene: str) -> np.ndarray:
    """``table[row's line, row's gene]``; NaN where either is missing."""
    line = table.index.get_indexer(rows["model_id"])
    column = table.columns.get_indexer(rows[gene])
    ok = (line >= 0) & (column >= 0)
    out = np.full(len(rows), np.nan)
    out[ok] = table.to_numpy(dtype=np.float64)[line[ok], column[ok]]
    return out


def features(
    rows: pd.DataFrame,
    pair_terms: np.ndarray,
    effects: pd.DataFrame,
    expression: pd.DataFrame,
) -> np.ndarray:
    """The head's inputs (FEATURES). Gene A of an anchor list is one gene or drug
    for the whole list, so its own values read 0; so does a value the reader cannot
    give (a gene outside the expression space)."""
    anchor = (rows["design"] == "anchor").to_numpy()
    own_a = np.stack(
        [gather(effects, rows, "gene_a"), gather(expression, rows, "gene_a")]
    )
    own_a[:, anchor] = 0.0
    values = np.column_stack(
        [
            pair_terms,
            own_a[0],
            gather(effects, rows, "gene_b"),
            own_a[1],
            gather(expression, rows, "gene_b"),
            (~anchor).astype(np.float64),
        ]
    )
    return np.nan_to_num(values, nan=0.0)


def weights(rows: pd.DataFrame) -> np.ndarray:
    """The benchmark's weighting: every line equal, then its studies, then a study's
    screens, then the two classes of a screen; mean 1."""
    n_class = rows.groupby([*LIST_KEYS, "label"])["label"].transform("size")
    screens = rows.groupby(["model_id", "study_id"])["list_id"].transform("nunique")
    studies = rows.groupby("model_id")["study_id"].transform("nunique")
    w = 1.0 / (n_class * screens * studies).to_numpy(dtype=np.float64)
    return w / w.mean()


def head_rows(rows: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    """The training rows the head fits on: genetic screens with both classes and
    finite inputs."""
    classes = rows.groupby(LIST_KEYS)["label"].transform("nunique")
    return (
        (rows["condition"] == HEADLINE).to_numpy()
        & (classes == 2).to_numpy()
        & np.isfinite(x).all(axis=1)
    )


def recall_top10(rows: pd.DataFrame, score: np.ndarray, top_k: int) -> float:
    """The benchmark's headline (mean over lines) of ``score`` on ``rows``."""
    frame = rows[[*LIST_KEYS, "label"]].assign(scorer="head", score=score)
    lines = line_records(list_records(frame, top_k))
    return float(lines["recall_top10"].mean())


def decision(model: Any, x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x).all(axis=1)
    out = np.full(len(x), np.nan)
    if finite.any():
        out[finite] = model.decision_function(x[finite])
    return out


def fit_head(
    train: pd.DataFrame,
    x_train: np.ndarray,
    val: pd.DataFrame,
    x_val: np.ndarray,
    penalties: Sequence[float],
    top_k: int,
) -> tuple[Any, list[dict[str, float]]]:
    """A logistic head per penalty (sklearn's C is its inverse), weighted as the
    benchmark; the one with the best validation recall in the top 10%."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keep = head_rows(train, x_train)
    chosen = train[keep]
    records, best = [], None
    for penalty in penalties:
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0 / float(penalty), max_iter=2000)
        )
        model.fit(
            x_train[keep],
            chosen["label"].to_numpy(),
            logisticregression__sample_weight=weights(chosen),
        )
        recall = recall_top10(val, decision(model, x_val), top_k)
        records.append({"penalty": float(penalty), "val_recall_top10": recall})
        if best is None or recall > best[0]:
            best = (recall, model)
    return best[1], records


def readers_and_rows(
    base: RunBase,
) -> tuple[dict[int, FittedLinearModel], dict[str, int], pd.DataFrame, Any]:
    """The full fit and one fit per fold, the reader of each SL line (a labelled
    single-cell training line: its fold's; any other: the full fit), every line's
    input row (bridged pseudo-bulk; out of fold for the labelled training lines) and
    the linear model's inputs."""
    config = base.config
    reference = config["reference"]
    experiment = config["experiments"][reference["experiment"]]
    if experiment["kind"] != "affine":
        raise ValueError(
            f"the backbone reads the affine bridge, not {experiment['kind']}"
        )
    inputs = build(base, "affine", experiment["settings"][0])
    linear_model = linear_model_inputs(base, inputs)
    spec = LinearModelSpec(
        tuple(
            stages(
                config,
                reference["block_set"],
                reference["components_penalty"],
                reference["gene_penalty"],
            )
        )
    )
    encoders = list(linear_model.expression.index)
    bridge = base.bridge
    folds = training_side_folds(
        bridge.folds, encoders, base.models["patient_id"].to_dict()
    )
    readers = {
        FULL: fit_linear_model(
            spec, linear_model, fit_lines=list(base.labelled), encoder_lines=encoders
        )
    }
    for fold in sorted(set(bridge.folds.values())):
        readers[fold] = fit_linear_model(
            spec,
            linear_model,
            fit_lines=[m for m in base.labelled if folds[m] != fold],
            encoder_lines=[m for m in encoders if folds[m] != fold],
        )
    train = list(bridge.single_cell_train)
    paired = list(bridge.paired)
    others = [m for m in base.split.all_model_ids if m not in set(train)]
    rows = pd.concat(
        [
            oof_bridged(bridge.pseudobulk, bridge.bulk, paired, train, bridge.folds),
            fit_bridge(bridge.pseudobulk.loc[paired], bridge.bulk.loc[paired]).apply(
                bridge.pseudobulk.loc[others]
            ),
        ]
    )
    assignment = {
        m: (bridge.folds[m] if m in bridge.folds else FULL) for m in rows.index
    }
    return readers, assignment, rows, linear_model


def predictions(
    readers: Mapping[int, FittedLinearModel],
    assignment: Mapping[str, int],
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Each line's prediction (residual SD) by its reader."""
    parts = [
        total(reader.predict(rows.loc[[m for m in rows.index if assignment[m] == k]]))
        for k, reader in readers.items()
        if any(assignment[m] == k for m in rows.index)
    ]
    return pd.concat(parts).loc[rows.index]


def stage_one_metrics(
    base: RunBase, residual: pd.DataFrame, seed: int
) -> dict[str, Any]:
    """The accepted stage-one metrics (``wide_accuracy``) of the reader on the SL
    lines with GeneEffect: training lines out of fold, validation and test from the
    full fit, each beside the same prediction with its lines permuted."""
    rng = np.random.default_rng(seed)
    labelled = set(base.gene_effect.index)
    split = base.split
    out = {}
    for name, lines in (
        ("train", list(split.supervised_train)),
        ("val", list(split.val)),
        ("test", list(split.test)),
    ):
        lines = [m for m in lines if m in labelled]
        truth = base.truth(lines)
        frame = residual.loc[lines]
        permuted = frame.set_axis(rng.permutation(lines), axis=0).loc[lines]
        if name == "train":
            folds = {m: base.bridge.folds[m] for m in lines}
            scores = fold_accuracy(frame, truth, base.definitions, folds)
            scores.pop("per_fold")
        else:
            scores = wide_accuracy(frame, truth, base.definitions)
        out[name] = {
            "reader": scores,
            "line_permuted": wide_accuracy(permuted, truth, base.definitions),
        }
    return out


def named_pairs(rows: pd.DataFrame, score: np.ndarray) -> list[dict[str, Any]]:
    """Each held-out row of a NAMED pair: its label, conditional score and percentile
    in its list (1: ranked first)."""
    frame = rows.assign(score=score)
    frame["percentile"] = frame.groupby(LIST_KEYS)["score"].rank(pct=True)
    wanted = {frozenset(p) for p in NAMED}
    named = np.array(
        [frozenset((a, b)) in wanted for a, b in zip(frame["gene_a"], frame["gene_b"])],
        dtype=bool,
    )
    chosen = frame[frame["split"].isin(["val", "test"]).to_numpy() & named]
    columns = ["split", "model_id", "study_id", "design", "gene_a", "gene_b", "label",
               "score", "percentile"]  # fmt: skip
    return chosen[columns].to_dict("records")


def run(config: Mapping[str, Any], run_id: str) -> Path:
    run_dir = Path(config["output_root"]) / config["task"] / run_id
    if (run_dir / "metrics.json").exists():
        raise FileExistsError(f"{run_dir} is finished; use a new --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    split_path = Path(config["split"])
    split = load_geneeffect_split(split_path)
    base = base_on(
        load_linear_model_config(Path(config["linear_model"])),
        split,
        load_bulk_lines(split_path, split),
        [*split.val, *split.test],
        other_split=True,
    )
    leaked = set(base.labelled) & {*split.val, *split.test}
    if leaked:
        raise ValueError(f"SL held-out lines among the fit lines: {sorted(leaked)}")
    readers, assignment, line_rows, linear_model = readers_and_rows(base)
    encoders = linear_model.expression
    origin = encoders.mean(axis=0)
    low = pd.Series(
        np.percentile(encoders.to_numpy(dtype=np.float64), LOW_PERCENTILE, axis=0),
        index=encoders.columns,
    )
    for reader in readers.values():
        check_affine(reader, origin, line_rows.iloc[0], str(encoders.columns[0]))
    residual = predictions(readers, assignment, line_rows)
    definitions = base.definitions
    genes = list(definitions.genes)
    table = residual * definitions.residual_scale[genes] + definitions.gene_means[genes]

    rows, dropped = benchmark_rows(pd.read_parquet(config["labels"]), split, set(genes))
    ablated_rows = pd.DataFrame(
        np.tile(origin.to_numpy(), (len(line_rows), 1)),
        index=line_rows.index,
        columns=origin.index,
    )
    ablated_effect = total(readers[FULL].predict(origin.to_frame().T)).iloc[0]
    ablated_effects = pd.DataFrame(
        np.tile(ablated_effect.to_numpy(), (len(line_rows), 1)),
        index=line_rows.index,
        columns=ablated_effect.index,
    )
    held_reader = {m: FULL for m in line_rows.index}
    reading = {
        "line": (terms(rows, line_rows, low, assignment, readers, origin), residual,
                 line_rows),
        "ablated": (terms(rows, ablated_rows, low, held_reader, readers, origin),
                    ablated_effects, ablated_rows),
    }  # fmt: skip
    # a gene outside the expression space cannot be edited: its term is 0 (a tie),
    # and the row is not covered (the harness's Table 4 ranks the covered rows)
    covered = np.isfinite(reading["line"][0]).all(axis=1)
    for values in reading.values():
        np.nan_to_num(values[0], copy=False, nan=0.0)
    x = {name: features(rows, *values) for name, values in reading.items()}
    train = (rows["split"] == "train").to_numpy()
    val = (rows["split"] == "val").to_numpy()
    head, selection = fit_head(
        rows[train],
        x["line"][train],
        rows[val],
        x["line"][val],
        config["head_penalties"],
        int(config["top_k"]),
    )
    scores = {
        "conditional": -reading["line"][0].sum(axis=1),
        "conditional_ablated": -reading["ablated"][0].sum(axis=1),
        "head": decision(head, x["line"]),
        "head_ablated": decision(head, x["ablated"]),
    }
    out = run_dir / "scores"
    out.mkdir(exist_ok=True)
    for name, values in scores.items():
        frame = rows[KEYS].assign(score=values, covered=covered).drop_duplicates(KEYS)
        staging = out / f".{name}.parquet.tmp"
        frame.to_parquet(staging, index=False)
        os.replace(staging, out / f"{name}.parquet")
    table.astype("float32").to_parquet(run_dir / "predictions.parquet")
    coverage = {
        side: float(covered[(rows["split"] == side).to_numpy()].mean())
        for side in ("train", "val", "test")
    }
    metrics = {
        "run_id": run_id,
        "fit_lines": len(base.labelled),
        "rows": {side: int((rows["split"] == side).sum()) for side in ("train", "val",
                 "test")} | dropped,
        "coverage": coverage,
        "head": {"features": list(FEATURES), "selection": selection,
                 "coefficients": head[-1].coef_[0].tolist()},
        "stage_one": stage_one_metrics(base, residual, int(config["seed"])),
        "named_pairs": named_pairs(rows, scores["conditional"]),
    }  # fmt: skip
    write_json(run_dir / "metrics.json", jsonable(metrics))
    write_text(run_dir / "results.md", results_markdown(metrics))
    return run_dir


def results_markdown(metrics: Mapping[str, Any]) -> str:
    """Table 1, the reader's stage-one metrics; Table 2, the head's penalties. The SL
    rows are in the benchmark harness's report."""
    keys = [("line_spearman", "per-line residual Spearman"),
            ("selective_spearman", "per-gene residual Spearman"),
            ("within_gene_r2", "residual skill score"),
            ("calibration_slope", "calibration slope"),
            ("top_recall", "top-20 recall of a line's strongest dependencies"),
            ("predictable_genes", "genes with per-gene Spearman at or above 0.3"),
            ("absolute_pearson", "whole-line Pearson (sanity)")]  # fmt: skip
    stage_one = metrics["stage_one"]

    def cell(value: Any) -> str:
        if value is None:
            return "—"
        return f"{value:,}" if isinstance(value, int) else f"{value:.4f}"

    lines = [
        f"# SL backbone, round zero: {metrics['run_id']}",
        "",
        "**Table 1.** Stage-one metrics of the linear expression model fitted on the "
        "SL split's training side, on the SL lines with GeneEffect (training lines "
        "out of fold), beside the same prediction with its lines permuted.",
        "",
        "| Metric | Train | Train, permuted | Validation | Validation, permuted | "
        "Test | Test, permuted |",
        "| --- |" + " ---: |" * 6,
        *(f"| {name} | " + " | ".join(
            cell(stage_one[s][row].get(key))
            for s in ("train", "val", "test") for row in ("reader", "line_permuted"))
          + " |" for key, name in keys),
        "",
        "**Table 2.** The head's validation recall in the top 10% at each penalty.",
        "",
        "| Penalty | Validation recall in the top 10% |",
        "| ---: | ---: |",
        *(f"| {r['penalty']:g} | {r['val_recall_top10']:.3f} |"
          for r in metrics["head"]["selection"]),
        "",
        f"Fit lines: {metrics['fit_lines']:,}. Rows whose genes the reader can edit "
        "(train / val / test): "
        + " / ".join(f"{v:.1%}" for v in metrics["coverage"].values()) + ".",
        "",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    out = run(load_config(args.config), args.run_id)
    print(f"SL backbone: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
