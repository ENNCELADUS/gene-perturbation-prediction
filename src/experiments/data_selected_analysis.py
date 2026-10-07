"""What the data-selected stage of the reference prior carries.

``python -m src.experiments.data_selected_analysis CONFIG --run-id ID`` fits the
config's reference row (it must include data-selected genes) on the labelled
training lines as the runner does, and writes ``data_selected/`` into the run
directory:

- ``targets.csv``: per selective gene, the validation and test Spearman of the
  prior with and without its data-selected stage, and its five heaviest features.
  Stages are fitted in order, so the prior without its last stage is the prior
  fitted without it.
- ``features.csv``: per expression gene, how the stage uses it over the selective
  targets (times selected, summed absolute weight on the standardised feature,
  times it is a paralog or complex partner of the target), and over the fit lines
  the share of its variance that lineage and the expression components explain.
- ``ablation.json`` and ``summary.md``: the stage kept to, or stripped of, its top
  features (ranked by summed absolute weight, a training-side quantity), scored on
  validation and test with the gain over the reference row.

It reports and chooses nothing. Nothing here is synthetic-lethality evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.context_prior.gene_features import partner_index
from src.context_prior.prior import PriorSpec, fit_prior, total
from src.context_prior.ridge import SelectedFit
from src.eval.geneeffect import aggregate_geneeffect
from src.experiments.all import _number
from src.experiments.config import load_prior_config
from src.experiments.context_prior import (
    SCORED,
    RunBase,
    _bind,
    _build,
    _jsonable,
    _write_json,
    _write_text,
    gain,
    load_base,
    long_frame,
    prior_inputs,
    score,
    stages,
)

#: Feature counts the ablation keeps or strips.
TOP = (10, 50, 200, 1000)
#: Heaviest features listed per target, and targets per feature.
LISTED = 5


def per_gene_spearman(
    prediction: pd.DataFrame, truth: pd.DataFrame, base: RunBase
) -> pd.Series:
    """Spearman across lines of each selective gene's residual (NaN: undefined)."""
    definitions = base.definitions
    _, _, per_gene = aggregate_geneeffect(
        long_frame(prediction, truth, definitions),
        model_ids=list(prediction.index),
        genes=list(definitions.genes),
        variable_genes=list(definitions.variable),
        selective_genes=list(definitions.selective),
    )
    table = per_gene.set_index("gene_symbol")
    return table.loc[list(definitions.selective), "spearman"]


def variance_explained(
    values: np.ndarray,
    groups: np.ndarray | None = None,
    design: np.ndarray | None = None,
) -> np.ndarray:
    """Per column, the share of variance that group means (``groups``) or a least
    squares fit on ``design`` (with intercept) explains; NaN for a constant column."""
    centered = values - values.mean(axis=0)
    total_ss = (centered**2).sum(axis=0)
    if groups is not None:
        fitted = pd.DataFrame(values).groupby(groups).transform("mean").to_numpy()
    else:
        x = np.column_stack([np.ones(len(design)), design])
        coef = np.linalg.lstsq(x, values, rcond=None)[0]
        fitted = x @ coef
    explained = ((fitted - values.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total_ss > 0, explained / total_ss, np.nan)


def masked(model: SelectedFit, keep: np.ndarray) -> SelectedFit:
    """The stage with every weight on a feature outside ``keep`` (a boolean over
    expression columns) set to zero; intercepts unchanged."""
    return SelectedFit(
        model.selection, model.coef * keep[model.selection], model.intercept
    )


def analyse(base: RunBase) -> dict:
    """Fit the reference row and tabulate its data-selected stage."""
    config = base.config
    reference = config["reference"]
    experiment = config["experiments"][reference["experiment"]]
    inputs = _build(base, experiment["kind"], experiment["settings"][0])
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
    if spec.stages[-1].block != "data_selected":
        raise ValueError("the reference row must end with data-selected genes")
    fitted = fit_prior(
        spec,
        prior,
        fit_lines=list(base.labelled),
        encoder_lines=list(prior.expression.index),
    )
    block, features, model = fitted.stages[-1]
    earlier = fitted.stages[:-1]
    readable = set(fitted.space if prior.gene_space is None else prior.gene_space)
    space = [g for g in fitted.space if g in readable]  # the stage's columns
    genes = list(fitted.genes)
    selective = list(base.definitions.selective)
    rows = [genes.index(g) for g in selective]

    # Feature use over the selective targets, on the standardised features.
    weight = np.abs(model.coef[rows])
    chosen = model.selection[rows]
    selected_count = np.bincount(chosen.ravel(), minlength=len(space))
    summed = np.bincount(chosen.ravel(), weights=weight.ravel(), minlength=len(space))
    index = partner_index(genes, space, base.reference)
    partner_hits = np.zeros(len(space), dtype=int)
    for local, g in enumerate(rows):
        partners = set(index.partners[g].tolist())
        for column in chosen[local]:
            if int(column) in partners:
                partner_hits[column] += 1
    fit_rows = prior.expression.loc[list(base.labelled), space].to_numpy(
        dtype=np.float64
    )
    lineage = base.models.loc[list(base.labelled), "lineage"].to_numpy()
    components = prior.components.transform(
        prior.expression.loc[list(base.labelled)].to_numpy(dtype=np.float64)
    )
    heaviest_targets: dict[int, list[tuple[float, str]]] = {}
    for local, g in enumerate(rows):
        for column, value in zip(chosen[local], weight[local], strict=True):
            heaviest_targets.setdefault(int(column), []).append((value, genes[g]))
    features_table = pd.DataFrame(
        {
            "gene": space,
            "times_selected": selected_count,
            "summed_abs_weight": summed,
            "partner_of_target": partner_hits,
            "selective_gene": [g in set(selective) for g in space],
            "lineage_r2": variance_explained(fit_rows, groups=lineage),
            "components_r2": variance_explained(fit_rows, design=components),
            "top_targets": [
                ", ".join(
                    name
                    for _, name in sorted(heaviest_targets.get(i, []), reverse=True)[
                        :LISTED
                    ]
                )
                for i in range(len(space))
            ],
        }
    ).sort_values("summed_abs_weight", ascending=False, ignore_index=True)
    rank = {gene: i for i, gene in enumerate(features_table["gene"])}
    order = np.array([rank[g] for g in space])

    # Predictions: without the stage, with it, and with it masked.
    truth = {name: base.truth(list(inputs.queries[name].index)) for name in SCORED}
    without, full, variants = {}, {}, {}
    for name in SCORED:
        query = inputs.queries[name].loc[:, list(fitted.space)]
        rest = total(
            {
                b: pd.DataFrame(m.predict(f(query)), index=query.index, columns=genes)
                for b, f, m in earlier
            }
        )
        x = features(query)

        def stage(fit: SelectedFit, rest=rest, x=x, query=query) -> pd.DataFrame:
            return rest + pd.DataFrame(fit.predict(x), index=query.index, columns=genes)

        without[name] = rest
        full[name] = stage(model)
        for k in TOP:
            if k >= len(space):
                continue
            variants.setdefault(f"top {k} only", {})[name] = stage(
                masked(model, order < k)
            )
            variants.setdefault(f"without top {k}", {})[name] = stage(
                masked(model, order >= k)
            )
    variants = {"without the stage": without, **variants}
    repeats = int(config["prior"]["bootstrap_repeats"])
    ablation = {
        "reference": {
            name: score(full[name], truth[name], base.definitions) for name in SCORED
        },
        "variants": {
            label: {
                name: {
                    **score(frames[name], truth[name], base.definitions),
                    "gain": gain(
                        frames[name], full[name], truth[name], base.definitions, repeats
                    ),
                }
                for name in SCORED
            }
            for label, frames in variants.items()
        },
    }

    targets_table = pd.DataFrame(index=pd.Index(selective, name="gene"))
    for name in SCORED:
        with_stage = per_gene_spearman(full[name], truth[name], base)
        no_stage = per_gene_spearman(without[name], truth[name], base)
        targets_table[f"{name}_spearman"] = with_stage
        targets_table[f"{name}_without_stage"] = no_stage
        targets_table[f"{name}_gain"] = with_stage - no_stage
    targets_table["top_features"] = [
        ", ".join(space[c] for c in chosen[local][np.argsort(-weight[local])][:LISTED])
        for local in range(len(rows))
    ]
    return {
        "block": block,
        "features": features_table,
        "targets": targets_table.reset_index(),
        "ablation": ablation,
        "space_genes": len(space),
    }


def concentration(targets: pd.DataFrame) -> dict:
    """How the per-target gain spreads: targets ranked by validation gain, read on
    test, so the test share is not chosen on test."""
    defined = targets.dropna(subset=["val_gain", "test_gain"])
    ranked = defined.sort_values("val_gain", ascending=False)
    out = {
        "targets": len(defined),
        "val_positive": int((defined["val_gain"] > 0).sum()),
        "test_positive": int((defined["test_gain"] > 0).sum()),
        "val_test_gain_spearman": float(
            defined["val_gain"].rank().corr(defined["test_gain"].rank())
        ),
        "mean_gain": {s: float(defined[f"{s}_gain"].mean()) for s in SCORED},
        "test_gain_by_val_decile": [],
    }
    for decile, positions in enumerate(np.array_split(np.arange(len(ranked)), 10)):
        part = ranked.iloc[positions]
        out["test_gain_by_val_decile"].append(
            {
                "decile": decile + 1,
                "val_mean_gain": float(part["val_gain"].mean()),
                "test_mean_gain": float(part["test_gain"].mean()),
            }
        )
    return out


def _gain_text(entry: Mapping) -> str:
    low, high = entry["interval"]
    return f"{_number(entry['difference'])} [{_number(low)}, {_number(high)}]"


def summary_lines(result: Mapping, spread: Mapping, run_id: str) -> list[str]:
    ablation = result["ablation"]
    features = result["features"]
    used = features.loc[features["times_selected"] > 0]
    out = [
        f"# Data-selected genes of the reference prior, run {run_id}",
        "",
        "Gains are selective Spearman minus the reference row's (the full prior), "
        "with 95% paired line-bootstrap intervals. Features are ranked by summed "
        "absolute weight over the selective targets, on the training side.",
        "",
        "| Variant | Val selective Spearman | Test selective Spearman | Val gain "
        "| Test gain |",
        "| --- | ---: | ---: | --- | --- |",
        f"| reference | {_number(ablation['reference']['val']['selective_spearman'])} "
        f"| {_number(ablation['reference']['test']['selective_spearman'])} | — | — |",
    ]
    for label, scores in ablation["variants"].items():
        out.append(
            f"| {label} | {_number(scores['val']['selective_spearman'])} "
            f"| {_number(scores['test']['selective_spearman'])} "
            f"| {_gain_text(scores['val']['gain'])} "
            f"| {_gain_text(scores['test']['gain'])} |"
        )
    out += [
        "",
        f"Expression genes used by the stage: {len(used)} of {result['space_genes']}; "
        f"selections that are a paralog or complex partner of their target: "
        f"{int(features['partner_of_target'].sum())} of "
        f"{int(features['times_selected'].sum())}.",
        "",
        "Median share of variance over the fit lines explained by lineage: "
        f"{_number(features['lineage_r2'].head(50).median(), 3)} for the top 50 "
        f"features, {_number(used['lineage_r2'].median(), 3)} for every feature "
        f"used, {_number(features['lineage_r2'].median(), 3)} for every space "
        "gene; by the expression components: "
        f"{_number(features['components_r2'].head(50).median(), 3)}, "
        f"{_number(used['components_r2'].median(), 3)} and "
        f"{_number(features['components_r2'].median(), 3)}.",
        "",
        "Per-target gain of the stage (selective Spearman with minus without): "
        f"mean {_number(spread['mean_gain']['val'])} validation, "
        f"{_number(spread['mean_gain']['test'])} test; positive for "
        f"{spread['val_positive']} and {spread['test_positive']} of "
        f"{spread['targets']} targets; Spearman of the validation and test gains "
        f"across targets {_number(spread['val_test_gain_spearman'], 3)}.",
        "",
        "| Decile by validation gain | Val mean gain | Test mean gain |",
        "| ---: | ---: | ---: |",
    ]
    for row in spread["test_gain_by_val_decile"]:
        out.append(
            f"| {row['decile']} | {_number(row['val_mean_gain'])} "
            f"| {_number(row['test_mean_gain'])} |"
        )
    out += [
        "",
        "Top 30 features:",
        "",
        "| Gene | Times selected | Summed abs weight | Partner of target | Selective "
        "| Lineage R2 | Components R2 | Heaviest targets |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in features.head(30).itertuples():
        out.append(
            f"| {row.gene} | {row.times_selected} | {row.summed_abs_weight:.3f} "
            f"| {row.partner_of_target} | {'yes' if row.selective_gene else 'no'} "
            f"| {_number(row.lineage_r2, 3)} | {_number(row.components_r2, 3)} "
            f"| {row.top_targets} |"
        )
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
    run_dir.mkdir(parents=True, exist_ok=True)
    _bind(run_dir, config)
    out = run_dir / "data_selected"
    out.mkdir(exist_ok=True)
    result = analyse(load_base(config))
    spread = concentration(result["targets"])
    result["features"].to_csv(out / "features.csv", index=False)
    result["targets"].to_csv(out / "targets.csv", index=False)
    _write_json(
        out / "ablation.json",
        _jsonable({**result["ablation"], "concentration": spread}),
    )
    _write_text(
        out / "summary.md", "\n".join(summary_lines(result, spread, args.run_id)) + "\n"
    )
    print(f"data-selected analysis: {out / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
