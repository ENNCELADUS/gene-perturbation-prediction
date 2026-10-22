"""The noise diagnosis's sections (protocol §14): per-cell sampling, the learning curve
over training lines (with Cilantro-SL's stage one beside it), the deposit confound
and swap, and cell-state features stacked on the export."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from src.context_prior.bridge import bridge_quality
from src.context_prior.bridge import oof_bridged
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
    crossfit_readings,
    total,
)
from src.context_prior.ridge import shared_ridge
from src.eval.accuracy import BOOTSTRAP_SEED, gain, long_frame, score
from src.experiments.prior.base import (
    SCORED,
    build,
    Fitter,
    stages,
)
from src.experiments.prior.diagnosis import (
    Diagnosis,
    LISTED_FLOOR,
    reference_fit,
    reproduces,
    scores_and_gain,
)


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
                **scores_and_gain(diagnosis, predicted[name], full_predictions, lines_),
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
    reproduces("primary", predicted["primary"].loc[lines], diagnosis.export.loc[lines])
    selective = list(base.definitions.selective)

    def agreement(left: pd.DataFrame, right: pd.DataFrame) -> dict:
        return quartiles(column_pearson(left.loc[lines], right.loc[lines]))

    return {
        "lines": len(lines),
        "deposits": {
            "primary": _deposits(diagnosis).loc[lines].value_counts().to_dict(),
        },
        "scores": {
            name: scores_and_gain(diagnosis, frame, predicted["primary"], lines)
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
                entry = scores_and_gain(
                    diagnosis, predicted[arm][penalty], diagnosis.export, lines
                )
                if arm == "program_fractions":
                    entry["gain_over_program_means"] = scores_and_gain(
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
