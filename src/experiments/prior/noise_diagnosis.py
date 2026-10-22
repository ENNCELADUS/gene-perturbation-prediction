"""What limits the default prior: per-cell sampling, the number of training lines,
the deposit a line's cells come from, cell-state heterogeneity and the noise of
the label (protocol §14; design ``docs/specs/2026-10-12-noise-diagnosis-design.md``).

``python -m src.experiments.prior.noise_diagnosis CONFIG --run-id ID [--sections A,B]``
reads the prior run named by ``prior_run`` (its bound config and ``export/``) and
writes ``<output_root>/<run id>/noise/<section>.json`` for each section, skipping
a section whose file exists, then ``results.md`` from every section present:

- ``cell_sampling``: split-half reliability of pseudo-bulk per gene against the
  bridge's agreement with bulk (``cell_sampling_genes.csv``), and the reference
  prior queried with half of each validation and test line's cells and with
  10 to 400 of them;
- ``learning_curve``: the reference row refitted on nested patient-grouped
  fractions of the training side, each beside its own gene mean, scored on the
  line-specific part (selective Spearman, within-gene R², AUPR lift, residual
  Pearson) and the whole prediction (per-line Pearson, RMSE); at the full side,
  Cilantro-SL's stage one on the pairs it covers;
- ``deposit_confound``: the share of the out-of-fold error a training line's
  deposit explains, against permutations within lineage;
- ``deposit_swap``: out-of-fold predictions of training lines read from a second
  deposit, against two halves of their own deposit;
- ``cell_states``: cell-state features stacked on the out-of-fold export.

Every score is on the metric of the prior run; gains carry 95% paired
line-bootstrap intervals. It reports and chooses nothing; nothing here is
synthetic-lethality evidence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.context_prior.bridge import Bridge, bridge_quality, fit_bridge
from src.context_prior.bridging import oof_bridged
from src.context_prior.folds import training_side_folds
from src.context_prior.noise import (
    cluster_proportions,
    column_pearson,
    group_share,
    line_sums,
    nested_fractions,
    program_features,
    quartiles,
    r2_gain,
    spearman_brown,
)
from src.context_prior.prior import (
    FittedPrior,
    PriorInputs,
    PriorSpec,
    crossfit_readings,
    fit_prior,
    total,
)
from src.context_prior.ridge import shared_ridge
from src.data.prior_offsets import checked_prior, read_prior_offsets
from src.data.io import write_json, write_text
from src.experiments.run_dir import bind, interval, number
from src.experiments.joint.config import load_config
from src.experiments.prior.config import load_noise_config
from src.eval.accuracy import BOOTSTRAP_SEED, gain, long_frame, score
from src.experiments.prior.base import (
    SCORED,
    RunBase,
    build,
    Fitter,
    load_base,
    prior_inputs,
    stages,
)

OUT_DIR = "noise"
SECTIONS = (
    "cell_sampling",
    "learning_curve",
    "deposit_confound",
    "deposit_swap",
    "cell_states",
)
#: A reproduced export row may differ from the written one by float rounding only.
REPRODUCTION_TOLERANCE = 1e-6
#: Lines whose source lists less of the scoring set than this are named in
#: ``cell_states`` (their unlisted genes score as unexpressed).
LISTED_FLOOR = 0.9


# ----------------------------------------------------------------------------
# The prior run
# ----------------------------------------------------------------------------


@dataclass
class Diagnosis:
    """Everything the sections share.

    Attributes:
        config: The noise config.
        base: The prior run's data, as its own runner loads it.
        export: The prior run's export (lines x genes, residual-SD units): out of
            fold on the training lines, the full fit on validation and test.
        joint: The joint config the prior run reads (prepared root, split).
    """

    config: Mapping[str, Any]
    base: RunBase
    export: pd.DataFrame
    joint: Mapping[str, Any]

    @property
    def train(self) -> list[str]:
        """The labelled single-cell training lines (the export's training rows)."""
        return list(self.base.bridge.single_cell_train)

    @property
    def queries(self) -> dict[str, list[str]]:
        return {"val": list(self.base.bridge.val), "test": list(self.base.bridge.test)}

    @property
    def prepared(self) -> Path:
        return Path(self.joint["prepared_root"])

    def spec(self) -> PriorSpec:
        reference = self.base.config["reference"]
        return PriorSpec(
            tuple(
                stages(
                    self.base.config,
                    reference["block_set"],
                    reference["components_penalty"],
                    reference["gene_penalty"],
                )
            )
        )

    def reference_inputs(self) -> PriorInputs:
        """The reference row's prior inputs; its bridge must be the affine one,
        whose bridged queries this module rebuilds."""
        reference = self.base.config["reference"]
        experiment = self.base.config["experiments"][reference["experiment"]]
        if experiment["kind"] != "affine":
            raise ValueError(
                f"the diagnosis rebuilds the affine bridge, not {experiment['kind']!r}"
            )
        return prior_inputs(self.base, build(self.base, "affine", {}))

    def raw_pseudobulk(self) -> pd.DataFrame:
        from src.data.pseudobulk import read_pseudobulk

        return read_pseudobulk(self.prepared)

    def pseudobulk_genes(self) -> list[str]:
        from src.data.pseudobulk import PSEUDOBULK_DIR

        manifest = self.prepared / PSEUDOBULK_DIR / "manifest.json"
        return list(json.loads(manifest.read_text())["genes"])

    def subsamples(self) -> pd.DataFrame:
        """Cell-draw pseudo-bulk, prepared on first use (reads raw cells)."""
        from src.experiments.prepare import prepare_noise_inputs

        path = prepare_noise_inputs(
            self.joint,
            self.pseudobulk_genes(),
            cells=self.config["cell_sampling"]["cells"],
            programs=program_members(self.base, self.config),
        )
        return pd.read_parquet(path.parent / "subsamples.parquet")

    def reading(
        self, frame: pd.DataFrame, lines: Sequence[str], *, measured_only: bool = False
    ) -> pd.DataFrame:
        """``lines``' rows of a raw pseudo-bulk frame, in the prior's space. With
        ``measured_only`` a value the source does not measure stays missing
        instead of taking the training mean: agreement between two readings of a
        line must not count the same filled value twice."""
        rows = frame.loc[list(lines)]
        normalized = self.base.normalize_pseudobulk(rows)
        if measured_only:
            normalized = normalized.where(rows.loc[:, normalized.columns].notna())
        return normalized


def load_diagnosis(config: Mapping[str, Any]) -> Diagnosis:
    prior_run = Path(config["prior_run"])
    base_config = json.loads((prior_run / "run_config.json").read_text())["config"]
    base = load_base(base_config)
    every = [
        *base.bridge.single_cell_train,
        *base.bridge.val,
        *base.bridge.test,
    ]
    export = checked_prior(
        read_prior_offsets(prior_run / "export"),
        genes=base.definitions.genes,
        residual_scale=base.definitions.residual_scale,
        lines=every,
    )
    joint = load_config(Path(base_config["joint_config"]))
    return Diagnosis(config, base, export.values.loc[every], joint)


def program_members(base: RunBase, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """The config's programs and their member genes, from the pinned tables."""
    root = Path(base.config["paths"]["reference"])
    cycle = pd.read_csv(root / "cell_cycle.tsv", sep="\t")
    meta = pd.read_csv(root / "meta_programs.tsv", sep="\t")
    tables = {
        **{phase: list(group["gene"]) for phase, group in cycle.groupby("phase")},
        **{name: list(group["gene"]) for name, group in meta.groupby("program")},
    }
    unknown = [p for p in config["cell_states"]["programs"] if p not in tables]
    if unknown:
        raise ValueError(f"programs absent from the pinned tables: {unknown}")
    return {p: tables[p] for p in config["cell_states"]["programs"]}


def _reproduces(name: str, rebuilt: pd.DataFrame, written: pd.DataFrame) -> None:
    """Fail closed when a rebuilt export row differs from the written one: the
    diagnosis would otherwise score another prior than the run's."""
    difference = float(np.nanmax(np.abs(rebuilt.to_numpy() - written.to_numpy())))
    if not difference <= REPRODUCTION_TOLERANCE:
        raise ValueError(
            f"{name}: rebuilt predictions differ from the export by {difference:.3g}"
        )


@dataclass
class ReferenceFit:
    """The reference row fitted on the whole training side, and its bridge."""

    fitted: FittedPrior
    bridge: Bridge

    def predict(self, reading: pd.DataFrame) -> pd.DataFrame:
        """Predictions (residual-SD units) of pseudo-bulk rows in the space."""
        return total(self.fitted.predict(self.bridge.apply(reading)))


def reference_fit(diagnosis: Diagnosis) -> ReferenceFit:
    base = diagnosis.base
    prior = diagnosis.reference_inputs()
    fitted = fit_prior(
        diagnosis.spec(),
        prior,
        fit_lines=list(base.labelled),
        encoder_lines=list(prior.expression.index),
    )
    paired = list(base.bridge.paired)
    bridge = fit_bridge(
        base.bridge.pseudobulk.loc[paired], base.bridge.bulk.loc[paired]
    )
    fit = ReferenceFit(fitted, bridge)
    for split, lines in diagnosis.queries.items():
        _reproduces(
            split,
            fit.predict(base.bridge.pseudobulk.loc[lines]),
            diagnosis.export.loc[lines],
        )
    return fit


def _scores(
    diagnosis: Diagnosis,
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    lines: Sequence[str],
) -> dict:
    """Metrics of ``predictions`` on ``lines`` and the gain over ``reference``."""
    base = diagnosis.base
    truth = base.truth(list(lines))
    return {
        **score(predictions.loc[list(lines)], truth, base.definitions),
        "gain": gain(
            predictions.loc[list(lines)],
            reference.loc[list(lines)],
            truth,
            base.definitions,
            int(diagnosis.config["bootstrap_repeats"]),
        ),
    }


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------


def cell_sampling(diagnosis: Diagnosis, out: Path) -> dict:
    base = diagnosis.base
    subsamples = diagnosis.subsamples()
    full = diagnosis.raw_pseudobulk()
    lines = list(base.split.all_model_ids)

    def draw(
        name: str, wanted: Sequence[str], *, measured_only: bool = False
    ) -> tuple[pd.DataFrame, list[str]]:
        """The draw's rows of ``wanted``; a line without the draw reads its full
        pseudo-bulk and is returned as capped."""
        rows = subsamples.xs(name, level="draw")
        capped = [m for m in wanted if m not in rows.index]
        frame = pd.concat(
            [rows.loc[[m for m in wanted if m in rows.index]], full.loc[capped]]
        )
        reading = diagnosis.reading(frame, wanted, measured_only=measured_only)
        return reading, capped

    halved = set(subsamples.xs("half_b", level="draw").index)
    lines = [m for m in lines if m in halved]  # a one-cell line has no halves
    half_a, _ = draw("half_a", lines, measured_only=True)
    half_b, _ = draw("half_b", lines, measured_only=True)
    paired = [m for m in base.bridge.paired if m in halved]
    affine = build(base, "affine", {})
    genes = pd.DataFrame(
        {
            "split_half_r": column_pearson(half_a, half_b),
            "paired_split_half_r": column_pearson(
                half_a.loc[paired], half_b.loc[paired]
            ),
            "bridge_r": bridge_quality(
                affine.oof_paired, affine.oof_bulk.loc[affine.oof_paired.index]
            ),
        }
    )
    genes["reliability"] = spearman_brown(genes["split_half_r"])
    genes["paired_reliability"] = spearman_brown(genes["paired_split_half_r"])
    genes["selective"] = genes.index.isin(set(base.definitions.selective))
    ceiling = np.sqrt(
        genes["paired_reliability"].where(genes["paired_reliability"] > 0)
    )
    genes["bridge_over_ceiling"] = genes["bridge_r"] / ceiling
    genes.index.name = "gene"
    genes.to_csv(out / "cell_sampling_genes.csv")
    selective = genes.loc[genes["selective"]]
    reliability = {
        column: {
            "all": quartiles(genes[column]),
            "selective": quartiles(selective[column]),
        }
        for column in (
            "reliability",
            "paired_reliability",
            "bridge_r",
            "bridge_over_ceiling",
        )
    }

    fit = reference_fit(diagnosis)
    queries = diagnosis.queries
    every = [*queries["val"], *queries["test"]]
    full_predictions = diagnosis.export.loc[every]
    names = ["half_a", "half_b"] + [
        f"cells_{n}" for n in sorted(diagnosis.config["cell_sampling"]["cells"])
    ]
    draws, predicted = {}, {}
    for name in names:
        reading, capped = draw(name, every)
        predicted[name] = fit.predict(reading)
        draws[name] = {
            split: {
                **_scores(diagnosis, predicted[name], full_predictions, lines_),
                "capped_lines": len([m for m in capped if m in set(lines_)]),
            }
            for split, lines_ in queries.items()
        }
    selective = list(base.definitions.selective)
    agreement = {
        split: quartiles(
            column_pearson(
                predicted["half_a"].loc[lines_, selective],
                predicted["half_b"].loc[lines_, selective],
            )
        )
        for split, lines_ in queries.items()
    }
    return {
        "reliability": reliability,
        "draws": draws,
        "prediction_agreement_between_halves": agreement,
        "lines": len(lines),
        "paired_lines": len(paired),
    }


def _gene_mean(
    diagnosis: Diagnosis, labelled: Sequence[str], lines: Sequence[str]
) -> pd.DataFrame:
    """The context-blind part of a prior fitted on ``labelled``: per gene, the
    mean training target over those lines, a missing label counting as zero as
    the ridge's intercept reads it; the same row for every query line."""
    mean = diagnosis.base.truth(list(labelled)).fillna(0.0).mean()
    return pd.DataFrame(
        np.tile(mean.to_numpy(), (len(lines), 1)), index=list(lines), columns=mean.index
    )


def _accuracy(
    diagnosis: Diagnosis,
    predictions: pd.DataFrame,
    lines: Sequence[str],
    covered: pd.DataFrame | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Line-specific and whole-prediction metrics of ``predictions`` (residual-SD
    units) on ``lines``, over the entries ``covered`` marks when given; and the
    per-line sums of the pooled within-gene R² over the selective genes."""
    from src.eval.accuracy import aggregate_geneeffect

    base = diagnosis.base
    lines = list(lines)
    genes = list(base.definitions.genes)
    truth = base.truth(lines)
    if covered is not None:
        truth = truth.where(covered)
    predictions = predictions.loc[lines, genes]
    metrics, _, _ = aggregate_geneeffect(
        long_frame(predictions, truth, base.definitions),
        model_ids=lines,
        genes=genes,
        variable_genes=list(base.definitions.variable),
        selective_genes=list(base.definitions.selective),
    )
    selective = list(base.definitions.selective)
    sse, total = line_sums(
        predictions[selective].to_numpy(dtype=np.float64),
        truth[selective].to_numpy(dtype=np.float64),
    )
    return (
        {
            "selective_spearman": metrics["selective_spearman"],
            "selective_spearman_scored": metrics["selective_spearman_scored"],
            "within_gene_r2": float(1 - sse.sum() / total.sum()),
            "residual_pearson": metrics["residual_pearson_macro_per_gene"],
            "aupr_lift": metrics["selective_aupr_lift"],
            "absolute_pearson": metrics["geneeffect_pearson_macro_per_line"],
            "rmse": metrics["geneeffect_rmse"],
        },
        sse,
        total,
    )


def learning_curve(diagnosis: Diagnosis, out: Path) -> dict:
    """The reference row refitted on nested patient-grouped fractions of the
    training side, each beside its own gene mean (the prior's intercept on the
    fraction's labelled lines); at the full side, Cilantro-SL's stage one on the
    validation and test pairs it covers, when the config names its predictions."""
    base = diagnosis.base
    side = list(base.bridge.bulk.index)
    patients = base.models["patient_id"].astype(str).to_dict()
    fractions = nested_fractions(
        side,
        patients,
        diagnosis.config["learning_curve"]["fractions"],
        seed=int(diagnosis.config["seed"]),
    )
    reference_row = base.config["reference"]
    fits = []
    for fraction, kept in fractions.items():
        keep = set(kept)
        sc = [m for m in base.bridge.single_cell_train if m in keep]
        sub = replace(
            base,
            labelled=tuple(m for m in base.labelled if m in keep),
            bridge=replace(
                base.bridge,
                bulk=base.bridge.bulk.loc[kept],
                paired=tuple(m for m in base.bridge.paired if m in keep),
                single_cell_train=tuple(sc),
                folds={m: base.bridge.folds[m] for m in sc},
            ),
        )
        predictions, _ = Fitter(sub, build(sub, "affine", {}))(
            stages(
                base.config,
                reference_row["block_set"],
                reference_row["components_penalty"],
                reference_row["gene_penalty"],
            )
        )
        print(f"  learning curve {fraction:g}: {len(kept)} lines", flush=True)
        fits.append((fraction, sub, pd.concat([predictions[s] for s in SCORED])))
    fits.append((1.0, base, diagnosis.export))
    repeats = int(diagnosis.config["bootstrap_repeats"])
    full = {
        split: _accuracy(diagnosis, diagnosis.export, lines)
        for split, lines in diagnosis.queries.items()
    }
    rows = []
    for fraction, sub, predictions in fits:
        row = {
            "fraction": fraction,
            "training_side_lines": len(sub.bridge.bulk),
            "labelled_lines": len(sub.labelled),
            "paired_lines": len(sub.bridge.paired),
        }
        for split, lines in diagnosis.queries.items():
            prior, sse, total = _accuracy(diagnosis, predictions, lines)
            mean, mean_sse, _ = _accuracy(
                diagnosis, _gene_mean(diagnosis, sub.labelled, lines), lines
            )
            truth = base.truth(lines)
            row[split] = {
                "prior": prior,
                "gene_mean": mean,
                "gain": gain(
                    predictions.loc[lines],
                    diagnosis.export.loc[lines],
                    truth,
                    base.definitions,
                    repeats,
                ),
                "r2_over_gene_mean": r2_gain(
                    sse, mean_sse, total, repeats=repeats, seed=BOOTSTRAP_SEED
                ),
                "r2_over_full": r2_gain(
                    sse, full[split][1], total, repeats=repeats, seed=BOOTSTRAP_SEED
                ),
            }
        rows.append(row)
    path = Path(diagnosis.config["learning_curve"]["cilantro_predictions"])
    return {"rows": rows, "cilantro": cilantro_rows(diagnosis, path)}


def cilantro_rows(diagnosis: Diagnosis, path: Path) -> dict:
    """Cilantro-SL's stage one (GeneEffect predictions keyed by ``model_id`` and
    ``gene_symbol``) against the full prior and its gene mean, all three scored on
    the validation and test pairs stage one covers (a gene among the line's top
    2,048 expressed). The released stage one trained on every labelled line, so
    its rows on validation and test lines are in-sample."""
    base = diagnosis.base
    rows = pd.read_csv(path)
    if rows.duplicated(["model_id", "gene_symbol"]).any():
        raise ValueError(f"{path}: duplicate (model_id, gene_symbol) rows")
    wide = rows.pivot(index="model_id", columns="gene_symbol", values="prediction")
    genes = list(base.definitions.genes)
    repeats = int(diagnosis.config["bootstrap_repeats"])
    out = {}
    for split, lines in diagnosis.queries.items():
        absolute = wide.reindex(index=lines, columns=genes)
        cilantro = absolute.sub(base.definitions.gene_means, axis=1).div(
            base.definitions.residual_scale, axis=1
        )
        covered = cilantro.notna() & base.truth(lines).notna()
        scored = {}
        sums = {}
        for name, predictions in (
            ("cilantro", cilantro),
            ("prior", diagnosis.export.loc[lines]),
            ("gene_mean", _gene_mean(diagnosis, base.labelled, lines)),
        ):
            scored[name], sums[name], total = _accuracy(
                diagnosis, predictions.where(covered), lines, covered
            )
        selective = covered[list(base.definitions.selective)]
        out[split] = {
            **scored,
            "pairs": int(covered.to_numpy().sum()),
            "selective_pairs": int(selective.to_numpy().sum()),
            "lines": int(covered.any(axis=1).sum()),
            "cilantro_over_gene_mean": r2_gain(
                sums["cilantro"],
                sums["gene_mean"],
                total,
                repeats=repeats,
                seed=BOOTSTRAP_SEED,
            ),
            "prior_over_cilantro": r2_gain(
                sums["prior"],
                sums["cilantro"],
                total,
                repeats=repeats,
                seed=BOOTSTRAP_SEED,
            ),
        }
    return out


def _deposits(diagnosis: Diagnosis) -> pd.Series:
    """Each split line's deposit: the final split table's ``expression_source``,
    the 226's names mapped to the identity map's source keys (Tahoe-100M is one
    deposit under two names)."""
    from src.data.build.plan_second_sources import PRIMARY_KEYS

    table = pd.read_csv(Path(diagnosis.joint["paths"]["split"]).with_suffix(".csv"))
    sources = table.set_index("model_id")["expression_source"].astype(str)
    return sources.replace(PRIMARY_KEYS)


def deposit_confound(diagnosis: Diagnosis, out: Path) -> dict:
    base = diagnosis.base
    settings = diagnosis.config["deposit"]
    train = diagnosis.train
    selective = list(base.definitions.selective)
    truth = base.truth(train).loc[:, selective]
    error = truth - diagnosis.export.loc[train, selective]
    deposit = _deposits(diagnosis).loc[train]
    counts = deposit.value_counts()
    small = set(counts.index[counts < int(settings["min_lines"])])
    grouped = deposit.where(~deposit.isin(small), "other")
    lineage = base.models.loc[train, "lineage"].astype(str)
    permutations, seed = int(settings["permutations"]), int(diagnosis.config["seed"])
    shares = {
        name: {
            group: group_share(
                frame,
                labels,
                strata,
                permutations=permutations,
                seed=seed,
            )
            # Deposits are permuted within lineage, lineages among all lines.
            for group, labels, strata in (
                ("deposit", grouped, lineage),
                ("lineage", lineage, None),
            )
        }
        for name, frame in (("error", error), ("residual", truth))
    }
    squared = error.pow(2)
    overall = float(np.nanmean(squared.to_numpy()))
    per_deposit = {
        name: {
            "lines": int((grouped == name).sum()),
            "error_ms_ratio": float(np.nanmean(squared.loc[grouped == name].to_numpy()))
            / overall,
        }
        for name in sorted(set(grouped))
    }
    return {
        "shares": shares,
        "per_deposit": per_deposit,
        "pooled_as_other": sorted(small),
        "lines": len(train),
    }


def deposit_swap(diagnosis: Diagnosis, out: Path) -> dict:
    from src.experiments.prepare import prepare_second_source

    base = diagnosis.base
    path = prepare_second_source(
        diagnosis.joint,
        diagnosis.pseudobulk_genes(),
        Path(diagnosis.config["deposit"]["second_source_root"]),
    )
    second = pd.read_parquet(path.parent / "pseudobulk.parquet")
    lines = [m for m in diagnosis.train if m in second.index]
    if not lines:
        raise ValueError(
            f"no labelled training line has a second deposit in {path.parent}"
        )
    subsamples = diagnosis.subsamples()
    raw = {
        "primary": diagnosis.raw_pseudobulk(),
        "second": second,
        "half_a": subsamples.xs("half_a", level="draw"),
        "half_b": subsamples.xs("half_b", level="draw"),
    }
    readings = {name: diagnosis.reading(frame, lines) for name, frame in raw.items()}
    measured = {
        name: diagnosis.reading(frame, lines, measured_only=True)
        for name, frame in raw.items()
    }
    bridge = base.bridge
    bridged = {
        name: oof_bridged(
            bridge.pseudobulk,
            bridge.bulk,
            list(bridge.paired),
            lines,
            bridge.folds,
            reading=reading,
        )
        for name, reading in readings.items()
    }
    prior = diagnosis.reference_inputs()
    encoders = list(prior.expression.index)
    folds = training_side_folds(
        bridge.folds, encoders, base.models["patient_id"].to_dict()
    )
    queries = {
        fold: {
            name: frame.loc[[m for m in lines if bridge.folds[m] == fold]]
            for name, frame in bridged.items()
        }
        for fold in sorted({bridge.folds[m] for m in lines})
    }
    predicted = {
        name: total(blocks)
        for name, blocks in crossfit_readings(
            diagnosis.spec(),
            prior,
            folds=folds,
            queries=queries,
            labelled=list(base.labelled),
            encoder_lines=encoders,
        ).items()
    }
    _reproduces("primary", predicted["primary"].loc[lines], diagnosis.export.loc[lines])
    selective = list(base.definitions.selective)

    def agreement(left: pd.DataFrame, right: pd.DataFrame) -> dict:
        return quartiles(column_pearson(left.loc[lines], right.loc[lines]))

    return {
        "lines": len(lines),
        "deposits": {
            "primary": _deposits(diagnosis).loc[lines].value_counts().to_dict(),
        },
        "scores": {
            name: _scores(diagnosis, frame, predicted["primary"], lines)
            for name, frame in predicted.items()
        },
        "prediction_agreement": {
            "primary_second": agreement(
                predicted["primary"][selective], predicted["second"][selective]
            ),
            "half_a_half_b": agreement(
                predicted["half_a"][selective], predicted["half_b"][selective]
            ),
            "primary_half_a": agreement(
                predicted["primary"][selective], predicted["half_a"][selective]
            ),
        },
        "input_agreement": {
            "primary_second": agreement(measured["primary"], measured["second"]),
            "half_a_half_b": agreement(measured["half_a"], measured["half_b"]),
        },
    }


def _program_scores(
    diagnosis: Diagnosis, lines: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Per line, its cells' program scores and the shares of the scoring set and
    of each program its source lists."""
    diagnosis.subsamples()  # prepares the program scores alongside the draws
    from src.experiments.prepare import NOISE_DIR

    root = diagnosis.prepared / NOISE_DIR
    listed = json.loads((root / "manifest.json").read_text())["listed"]
    out = {}
    for model_id in lines:
        with np.load(root / "programs" / f"{model_id}.npz") as payload:
            out[model_id] = payload["scores"].astype(np.float64)
    return out, {m: listed[m] for m in lines}


def _context_cells(diagnosis: Diagnosis, lines: Sequence[str]) -> dict[str, np.ndarray]:
    from src.data.prepared import read_manifest, read_prepared_line

    genes = tuple(read_manifest(diagnosis.prepared)["common_gene_panel"])
    return {
        m: read_prepared_line(
            diagnosis.prepared / "lines" / f"{m}.npz", genes
        ).basal_hvg.astype(np.float32)
        for m in lines
    }


def cell_states(diagnosis: Diagnosis, out: Path) -> dict:
    base = diagnosis.base
    settings = diagnosis.config["cell_states"]
    train, queries = diagnosis.train, diagnosis.queries
    every = [*train, *queries["val"], *queries["test"]]
    programs = list(settings["programs"])
    scores, listed = _program_scores(diagnosis, every)
    fractions, means, thresholds = program_features(
        scores, train, programs, float(settings["threshold_percentile"])
    )
    arms: dict[str, pd.DataFrame] = {
        "program_fractions": fractions,
        "program_means": means,
    }
    clusters = cluster_proportions(
        _context_cells(diagnosis, every),
        train,
        components=int(settings["cluster_components"]),
        counts=settings["cluster_counts"],
        seed=int(diagnosis.config["seed"]),
    )
    arms.update({f"clusters_{k}": frame for k, frame in clusters.items()})

    target = base.truth(train) - diagnosis.export.loc[train]
    target = target.to_numpy()
    target = np.where(np.isfinite(target), target, 0.0)
    penalties = [float(p) for p in settings["penalties"]]
    predicted: dict[str, dict[float, pd.DataFrame]] = {}
    for arm, features in arms.items():
        fits = shared_ridge(features.loc[train].to_numpy(), target, penalties)
        predicted[arm] = {}
        for penalty, fit in zip(penalties, fits, strict=True):
            lines = [*queries["val"], *queries["test"]]
            predicted[arm][penalty] = diagnosis.export.loc[lines] + pd.DataFrame(
                fit.predict(features.loc[lines].to_numpy()),
                index=lines,
                columns=list(base.definitions.genes),
            )
    prior_alone = {
        split: score(diagnosis.export.loc[lines], base.truth(lines), base.definitions)
        for split, lines in queries.items()
    }
    rows = {}
    for arm in arms:
        rows[arm] = []
        for penalty in penalties:
            row = {"penalty": penalty}
            for split, lines in queries.items():
                entry = _scores(
                    diagnosis, predicted[arm][penalty], diagnosis.export, lines
                )
                if arm == "program_fractions":
                    entry["gain_over_program_means"] = _scores(
                        diagnosis,
                        predicted[arm][penalty],
                        predicted["program_means"][penalty],
                        lines,
                    )["gain"]
                row[split] = entry
            rows[arm].append(row)
    return {
        "prior_alone": prior_alone,
        "arms": rows,
        "programs": programs,
        "thresholds": dict(zip(programs, thresholds.tolist(), strict=True)),
        "features": {arm: int(frame.shape[1]) for arm, frame in arms.items()},
        # A gene a source does not list scores as unexpressed; how much each
        # line's source lists of the scoring set and of each program.
        "listed": {
            name: quartiles([listed[m][name] for m in every])
            for name in ["scoring_set", *programs]
        },
        "lines_listing_under_90_percent": sorted(
            m for m in every if listed[m]["scoring_set"] < LISTED_FLOOR
        ),
    }


RUNNERS: dict[str, Callable[[Diagnosis, Path], dict]] = {
    "cell_sampling": cell_sampling,
    "learning_curve": learning_curve,
    "deposit_confound": deposit_confound,
    "deposit_swap": deposit_swap,
    "cell_states": cell_states,
}


# ----------------------------------------------------------------------------
# results.md
# ----------------------------------------------------------------------------


def _quartiles(entry: Mapping[str, Any]) -> str:
    return (
        f"{number(entry['median'], 3)} [{number(entry['q25'], 3)}, "
        f"{number(entry['q75'], 3)}] (n={entry['count']})"
    )


def _split_cells(entry: Mapping[str, Any]) -> list[str]:
    return [number(entry["selective_spearman"]), interval(entry["gain"])]


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A markdown table; the first column left-aligned, the rest right."""
    return [
        "| " + " | ".join(header) + " |",
        "| --- |" + " ---: |" * (len(header) - 1),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _cell_sampling_md(section: Mapping[str, Any]) -> list[str]:
    halves = section["prediction_agreement_between_halves"]
    draws = [
        [
            name,
            *(
                cell
                for split in SCORED
                for cell in (*_split_cells(row[split]), str(row[split]["capped_lines"]))
            ),
        ]
        for name, row in section["draws"].items()
    ]
    return [
        "## Cell sampling",
        "",
        f"Per space gene over {section['lines']} lines: split-half reliability "
        "(Spearman-Brown), and over the paired lines "
        f"({section['paired_lines']}) the bridge's out-of-fold agreement with bulk "
        "and that agreement over the square root of their reliability: median "
        "[quartiles].",
        "",
        *_table(
            ["Quantity", "All genes", "Selective genes"],
            [
                [name, _quartiles(entry["all"]), _quartiles(entry["selective"])]
                for name, entry in section["reliability"].items()
            ],
        ),
        "",
        "The reference prior queried with draws of each validation and test line's "
        "cells; gains over the full pseudo-bulk; a capped line has fewer cells than "
        "the draw and reads them all.",
        "",
        *_table(
            ["Draw", "Val", "Val gain", "Val capped"]
            + ["Test", "Test gain", "Test capped"],
            draws,
        ),
        "",
        "Per selective gene, Pearson across lines between the two halves' "
        f"predictions: validation {_quartiles(halves['val'])}, test "
        f"{_quartiles(halves['test'])}.",
    ]


def _learning_curve_md(section: Mapping[str, Any]) -> list[str]:
    specific, whole = [], []
    for row in section["rows"]:
        lines = (
            f"{row['training_side_lines']:,} / {row['labelled_lines']:,} / "
            f"{row['paired_lines']:,}"
        )
        for split in SCORED:
            entry = row[split]
            prior, mean = entry["prior"], entry["gene_mean"]
            specific.append(
                [
                    f"{row['fraction']:g}",
                    lines,
                    split,
                    number(prior["selective_spearman"]),
                    interval(entry["gain"]),
                    number(prior["within_gene_r2"]),
                    number(mean["within_gene_r2"]),
                    interval(entry["r2_over_gene_mean"]),
                    interval(entry["r2_over_full"]),
                    number(prior["aupr_lift"]),
                    number(prior["residual_pearson"]),
                ]
            )
            whole.append(
                [
                    f"{row['fraction']:g}",
                    split,
                    number(prior["absolute_pearson"]),
                    number(mean["absolute_pearson"]),
                    number(prior["rmse"]),
                    number(mean["rmse"]),
                ]
            )
    out = [
        "## Learning curve",
        "",
        "The reference row refitted on nested patient-grouped fractions of the "
        "training side (lines: training side / labelled / paired), each beside "
        "its gene mean (the prior's intercept on the fraction's labelled lines). "
        "Within-gene R² is pooled over the selective genes against the training "
        "gene mean (which scores 0); gains are 95% paired line bootstraps.",
        "",
        "Line-specific part:",
        "",
        *_table(
            [
                "Fraction",
                "Lines",
                "Split",
                "Selective Spearman",
                "Gain over full",
                "Within-gene R²",
                "Gene mean R²",
                "R² over gene mean",
                "R² gain over full",
                "AUPR lift",
                "Residual Pearson",
            ],
            specific,
        ),
        "",
        "Whole prediction (per-line Pearson across genes; RMSE over every pair):",
        "",
        *_table(
            [
                "Fraction",
                "Split",
                "Absolute Pearson",
                "Gene mean",
                "RMSE",
                "Gene mean RMSE",
            ],
            whole,
        ),
    ]
    cilantro = section["cilantro"]
    if cilantro is None:
        return out
    rows = []
    for split in SCORED:
        entry = cilantro[split]
        for name in ("cilantro", "prior", "gene_mean"):
            scores = entry[name]
            rows.append(
                [
                    name,
                    split,
                    f"{entry['pairs']:,} ({entry['selective_pairs']:,} selective)",
                    f"{number(scores['selective_spearman'])} "
                    f"({scores['selective_spearman_scored']} genes)",
                    number(scores["within_gene_r2"]),
                    number(scores["absolute_pearson"]),
                    number(scores["rmse"]),
                ]
            )
    gains = [
        [
            split,
            interval(cilantro[split]["cilantro_over_gene_mean"]),
            interval(cilantro[split]["prior_over_cilantro"]),
        ]
        for split in SCORED
    ]
    return out + [
        "",
        "Cilantro-SL's stage one (as released, trained on every labelled line, so "
        "in-sample on these lines) against the full prior and its gene mean, all on "
        "the validation and test pairs stage one covers:",
        "",
        *_table(
            [
                "Predictor",
                "Split",
                "Pairs",
                "Selective Spearman",
                "Within-gene R²",
                "Absolute Pearson",
                "RMSE",
            ],
            rows,
        ),
        "",
        *_table(
            ["Split", "Cilantro R² over gene mean", "Prior R² over Cilantro"], gains
        ),
    ]


def _deposit_confound_md(section: Mapping[str, Any]) -> list[str]:
    shares = [
        [
            name,
            grouping,
            number(entry["observed"]),
            number(entry["null_mean"]),
            number(entry["null_q95"]),
            number(entry["p_value"], 3),
            str(entry["groups"]),
        ]
        for name, groupings in section["shares"].items()
        for grouping, entry in groupings.items()
    ]
    deposits = [
        [name, str(entry["lines"]), number(entry["error_ms_ratio"], 3)]
        for name, entry in section["per_deposit"].items()
    ]
    return [
        "## Deposit confound",
        "",
        "η² macro-averaged over the selective genes on the "
        f"{section['lines']} training lines; the null permutes deposits within "
        "lineage and lineages among all lines.",
        "",
        *_table(
            ["Of", "Grouping", "η²", "Null mean", "Null 95%", "p", "Groups"], shares
        ),
        "",
        *_table(["Deposit", "Lines", "Error mean square / all"], deposits),
    ]


def _deposit_swap_md(section: Mapping[str, Any]) -> list[str]:
    scores = [
        [name, number(entry["selective_spearman"]), interval(entry["gain"])]
        for name, entry in section["scores"].items()
    ]
    agreement = [
        [
            name,
            _quartiles(section["prediction_agreement"][name]),
            _quartiles(section["input_agreement"][name]),
        ]
        for name in ("primary_second", "half_a_half_b")
    ]
    return [
        "## Deposit swap",
        "",
        f"{section['lines']} labelled training lines with a second deposit, "
        "predicted out of fold; gains over the primary deposit's reading.",
        "",
        *_table(["Reading", "Selective Spearman", "Gain over primary"], scores),
        "",
        "Pearson across lines between two readings: predictions over the selective "
        "genes, inputs over the space genes.",
        "",
        *_table(["Readings", "Predictions", "Inputs"], agreement),
    ]


def _cell_states_md(section: Mapping[str, Any]) -> list[str]:
    prior = section["prior_alone"]
    rows = [
        [
            arm,
            f"{row['penalty']:g}",
            *(
                cell
                for split in SCORED
                for cell in (
                    *_split_cells(row[split]),
                    interval(row[split].get("gain_over_program_means")),
                )
            ),
        ]
        for arm, arm_rows in section["arms"].items()
        for row in arm_rows
    ]
    header = ["Arm", "Penalty", "Val", "Val gain", "Val over means"]
    return [
        "## Cell states",
        "",
        "Prior alone: validation "
        f"{number(prior['val']['selective_spearman'])}, test "
        f"{number(prior['test']['selective_spearman'])}. Gains over the prior "
        "alone; `program_fractions` also over `program_means` at the same penalty.",
        "",
        *_table(header + ["Test", "Test gain", "Test over means"], rows),
        "",
        "Share of the scoring set each line's source lists (an unlisted gene scores "
        f"as unexpressed): {_quartiles(section['listed']['scoring_set'])}; lines "
        f"under {LISTED_FLOOR:.0%}: "
        f"{', '.join(section['lines_listing_under_90_percent']) or 'none'}.",
    ]


SECTION_MARKDOWN: dict[str, Callable[[Mapping[str, Any]], list[str]]] = {
    "cell_sampling": _cell_sampling_md,
    "learning_curve": _learning_curve_md,
    "deposit_confound": _deposit_confound_md,
    "deposit_swap": _deposit_swap_md,
    "cell_states": _cell_states_md,
}


def results_markdown(
    run_id: str, config: Mapping[str, Any], results: Mapping[str, dict]
) -> str:
    out = [
        f"# Noise diagnosis `{run_id}`",
        "",
        f"Prior run `{config['prior_run']}`. Selective Spearman is the macro mean "
        "over the selective genes; gains are 95% paired line bootstraps. Nothing is "
        "chosen here; nothing here is synthetic-lethality evidence.",
    ]
    for name in SECTIONS:
        if name in results:
            out += ["", *SECTION_MARKDOWN[name](results[name])]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--sections", help=f"comma-separated, of {', '.join(SECTIONS)} (default: all)"
    )
    args = parser.parse_args(argv)
    config = load_noise_config(args.config)
    names = (
        SECTIONS
        if args.sections is None
        else tuple(dict.fromkeys(args.sections.split(",")))
    )
    unknown = sorted(set(names) - set(SECTIONS))
    if unknown:
        raise ValueError(f"unknown sections {unknown}; sections are {list(SECTIONS)}")
    run_dir = Path(config["output_root"]) / args.run_id
    out = run_dir / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    pending = [name for name in names if not (out / f"{name}.json").is_file()]
    if pending:
        diagnosis = load_diagnosis(config)
        for name in pending:
            print(f"section {name}", flush=True)
            write_json(out / f"{name}.json", RUNNERS[name](diagnosis, out))
    results = {
        name: json.loads((out / f"{name}.json").read_text())
        for name in SECTIONS
        if (out / f"{name}.json").is_file()
    }
    write_text(out / "results.md", results_markdown(args.run_id, config, results))
    print(f"noise diagnosis: {out / 'results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
