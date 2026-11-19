"""Four ways to make the SL training target less noisy, compared by one quantity: how
well the line-specific part of the target repeats between two independent
measurements (``line_specificity.analyse``: the same-line advantage and the repeat of
the line-specific part). No model is read and the benchmark's labels are not changed.

1. The continuous score in place of the SL call: the rank, and the rank's top 10% as
   a ramp (0 below it), against the call.
2. The mean of several measurements: inside Harle 2025, the mean of two replicates
   against the third; between studies, the share of training cells that two studies
   screened.
3. A weight for each pair from its measurement error: the inverse of the pair's mean
   variance in Harle's Table S4, for Harle against another study and for two Harle
   replicates.
4. One score definition for all studies: Ito 2021 and Klingbeil 2024 publish GEMINI;
   here their score is the LFC of the double loss minus the LFC of each single loss
   (the other side a cutting control), as the other large studies define theirs.

Harle's replicates also give the repeat inside one study, which the label table (one
merged score for each pair and screen) cannot give: a replicate's score is that same
difference of LFCs, from the count matrix (Table S3), and its SL calls are as many of
the lowest scores as the line has SL calls."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.io import write_json, write_text
from src.data.splits import load_geneeffect_split
from src.experiments.config import read_yaml, require_keys
from src.experiments.run_dir import bind, number
from src.experiments.sl.line_specificity import COLUMNS, PAIR, analyse, study_cells

#: Controls that cut the genome, so a single loss carries the cutting effect too.
CUTTING = ("AAVS1", "CUTTING_CONTROL")
#: Studies whose published score (GEMINI) is replaced in option 4.
SWITCHED = {"Ito2021", "Klingbeil2024"}
HARLE = "Harle2025"
#: Settings of the report: key, name, kinds shown.
SETTINGS = (
    ("between_studies", "Two studies, own score", ("rank", "tail", "call")),
    ("switched_own", "Study pairs with Ito or Klingbeil, own score (GEMINI)",
     ("rank",)),
    ("switched_common", "The same study pairs, one score definition", ("rank",)),
    ("harle_others", "Harle and another study", ("rank",)),
    ("harle_others_weighted", "Harle and another study, pairs weighted by the "
     "inverse of their variance", ("rank",)),
    ("inside_harle", "Inside Harle, two replicates", ("rank", "tail", "call")),
    ("inside_harle_weighted", "Inside Harle, two replicates, pairs weighted by the "
     "inverse of their variance", ("rank",)),
    ("inside_harle_mean_of_two", "Inside Harle, the mean of two replicates against "
     "the third", ("rank", "tail", "call")),
)  # fmt: skip


TARGETS = {"rank": "rank", "tail": "top 10% of the rank, as a ramp", "call": "SL call"}


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "output_root", "task", "labels", "split", "bootstrap",
         "min_pairs", "min_lines", "raw"},
        "config",
    )  # fmt: skip
    require_keys(config["raw"], {"ito", "klingbeil_lfc", "harle"}, "raw")
    if config["kind"] != "sl_target_options":
        raise ValueError(f"{path} is a {config['kind']} config")
    return config


def pair_key(a: pd.Series, b: pd.Series) -> np.ndarray:
    """An unordered gene pair as the study spells it, ``A|B`` with A before B."""
    a, b = a.astype(str).to_numpy(), b.astype(str).to_numpy()
    return np.where(a < b, a + "|" + b, b + "|" + a)


def interaction(lfc: pd.DataFrame, lines: Sequence[str]) -> pd.DataFrame:
    """The LFC of the double loss minus the LFC of each single loss, long over (line,
    pair). ``lfc``: one row per construct with ``gene_a``, ``gene_b`` (a control by its
    name), ``unit_a``, ``unit_b`` (what a single loss is matched on: the gene, or its
    targeted domain) and one LFC column per line. A pair's constructs are averaged; a
    construct whose single losses were not screened has no value."""
    control_a, control_b = lfc["gene_a"].isin(CUTTING), lfc["gene_b"].isin(CUTTING)
    single = pd.concat(
        [
            lfc.loc[control_b & ~control_a].set_index("unit_a")[list(lines)],
            lfc.loc[control_a & ~control_b].set_index("unit_b")[list(lines)],
        ]
    )
    single = single.groupby(level=0).mean()
    double = lfc.loc[~control_a & ~control_b]
    values = (
        double[list(lines)].to_numpy(float)
        - single.reindex(double["unit_a"]).to_numpy(float)
        - single.reindex(double["unit_b"]).to_numpy(float)
    )
    wide = pd.DataFrame(values, columns=list(lines))
    wide["pair"] = pair_key(double["gene_a"], double["gene_b"])
    long = wide.groupby("pair").mean().reset_index()
    return long.melt("pair", var_name="line", value_name="gi").dropna()


def ito_interaction(path: Path) -> pd.DataFrame:
    """Ito 2021, raw LFCs (Supplementary Table 4); a single loss pairs with AAVS1."""
    lfc = pd.read_excel(path, sheet_name="Supplementary table 4", header=3)
    lfc = lfc.rename(columns={"gene 1": "gene_a", "gene 2": "gene_b"})
    lfc[["gene_a", "gene_b"]] = lfc[["gene_a", "gene_b"]].astype(str)
    lines = [c for c in lfc.columns if c not in ("gene_a", "gene_b")]
    lfc[lines] = lfc[lines].apply(pd.to_numeric, errors="coerce")
    return interaction(lfc.assign(unit_a=lfc["gene_a"], unit_b=lfc["gene_b"]), lines)


def klingbeil_interaction(path: Path) -> pd.DataFrame:
    """Klingbeil 2024, mean LFC of each domain construct (Table S2); a single loss
    pairs the same domain with the cutting control."""
    lfc = pd.read_excel(path)
    names = {"GENE 1": "gene_a", "GENE 2": "gene_b"}
    lfc = lfc.rename(columns=names).astype({"gene_a": str, "gene_b": str})
    lines = [
        c for c in lfc.columns if c not in (*names.values(), "domain_1", "domain_2")
    ]
    lfc[lines] = lfc[lines].apply(pd.to_numeric, errors="coerce")
    return interaction(
        lfc.assign(
            unit_a=lfc["gene_a"] + ":" + lfc["domain_1"].astype(str),
            unit_b=lfc["gene_b"] + ":" + lfc["domain_2"].astype(str),
        ),
        lines,
    )


def harle_replicates(counts: pd.DataFrame) -> pd.DataFrame:
    """Harle 2025's count matrix (Table S3) as a score for each replicate, long over
    (line, replicate, pair): the mean LFC of the pair's guide pairs minus the mean LFC
    of each gene's single-loss constructs (a safe-targeting guide on the other side).
    LFC = log2((count + 1) / (control_mean + 1)) of the normalised counts."""
    samples = [c for c in counts.columns if re.fullmatch(r".+ R\d", str(c))]
    lfc = np.log2((counts[samples] + 1).div(counts["control_mean"] + 1, axis=0))
    pair, gene = counts["sorted_gene_pair"], counts["singles_target_gene"]
    double = lfc[counts["guide_type"] == "gene|gene"].groupby(pair).mean()
    single = lfc[gene.notna()].groupby([pair, gene]).mean()
    both = single.groupby(level=0).size() == 2
    gi = double - single.groupby(level=0).sum()[both].reindex(double.index)
    long = gi.reset_index().melt("sorted_gene_pair", var_name="sample", value_name="gi")
    long = long.dropna()
    genes = long["sorted_gene_pair"].str.split("|", expand=True)
    sample = long["sample"].str.rsplit(" R", n=1, expand=True)
    return pd.DataFrame(
        {"line": sample[0].to_numpy(), "replicate": sample[1].to_numpy(),
         "pair": pair_key(genes[0], genes[1]), "gi": long["gi"].to_numpy()}
    )  # fmt: skip


def with_score(labels: pd.DataFrame, study: str, gi: pd.DataFrame) -> pd.DataFrame:
    """The labels with one study's score replaced (NaN where ``gi`` has no value)."""
    rows = labels[labels["study_id"] == study]
    key = pd.MultiIndex.from_arrays(
        [rows["line"], pair_key(rows["screen_gene_a"], rows["screen_gene_b"])]
    )
    out = labels.copy()
    out.loc[rows.index, "score"] = (
        gi.set_index(["line", "pair"])["gi"].reindex(key).to_numpy()
    )
    return out


def replicate_rows(labels: pd.DataFrame, replicates: pd.DataFrame) -> pd.DataFrame:
    """Harle's replicates as label rows, each replicate a study of its own
    (``Harle2025|R1`` ...), plus the mean of replicates 1 and 2 (``|R1+R2``). The rows
    are the study's labelled rows; a replicate's SL calls are as many of its lowest
    scores as the line has SL calls."""
    rows = labels[(labels["study_id"] == HARLE) & labels["label"].notna()]
    rows = rows.assign(pair=pair_key(rows["screen_gene_a"], rows["screen_gene_b"]))
    two = replicates[replicates["replicate"].isin(["1", "2"])]
    two = two.groupby(["line", "pair"])["gi"].agg(["mean", "size"]).reset_index()
    two = two[two["size"] == 2].rename(columns={"mean": "gi"}).assign(replicate="1+R2")
    every = pd.concat([replicates, two.drop(columns="size")], ignore_index=True)
    keep = ["line", "pair", "model_id", "design", "condition", *PAIR, "label"]
    out = every.merge(rows[keep], on=["line", "pair"])
    out["study_id"] = HARLE + "|R" + out["replicate"]
    out["list_id"] = out["study_id"] + "|" + out["line"]
    calls = out.groupby("list_id")["label"].transform("sum")
    lowest = out.groupby("list_id")["gi"].rank(method="first")
    return out.assign(score=out["gi"], label=(lowest <= calls).astype(float))[COLUMNS]


def harle_weight(table: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """A pair's weight: the inverse of its mean ``variance_norm_gi`` over Harle's
    lines (Table S4), indexed by the benchmark's gene names."""
    genes = table["sorted_gene_pair"].str.split("|", expand=True)
    variance = table["variance_norm_gi"].groupby(pair_key(genes[0], genes[1])).mean()
    rows = labels[labels["study_id"] == HARLE].drop_duplicates(PAIR)
    key = pair_key(rows["screen_gene_a"], rows["screen_gene_b"])
    weight = 1.0 / variance.reindex(key).to_numpy()
    return pd.Series(weight, index=pd.MultiIndex.from_frame(rows[PAIR])).dropna()


def published_agreement(labels: pd.DataFrame, replicates: pd.DataFrame) -> float:
    """Output check: the Spearman correlation over pairs between the mean of a line's
    replicate scores and Harle's published score, mean over the lines."""
    rows = labels[(labels["study_id"] == HARLE) & labels["label"].notna()]
    rows = rows.assign(pair=pair_key(rows["screen_gene_a"], rows["screen_gene_b"]))
    mean = replicates.groupby(["line", "pair"])["gi"].mean().rename("gi").reset_index()
    both = mean.merge(rows[["line", "pair", "score"]], on=["line", "pair"])
    return float(
        both.groupby("line")[["gi", "score"]]
        .apply(lambda g: g["gi"].corr(g["score"], method="spearman"))
        .mean()
    )


def run(config: Mapping[str, Any], run_id: str, workers: int) -> Path:
    run_dir = Path(config["output_root"]) / config["task"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    split = load_geneeffect_split(Path(config["split"]))
    lines, raw = set(split.all_model_ids), config["raw"]
    labels = pd.read_parquet(
        config["labels"], columns=[*COLUMNS, "line", "screen_gene_a", "screen_gene_b"]
    )
    common_labels = with_score(labels, "Ito2021", ito_interaction(Path(raw["ito"])))
    common_labels = with_score(
        common_labels,
        "Klingbeil2024",
        klingbeil_interaction(Path(raw["klingbeil_lfc"])),
    )
    switched_rows = labels["study_id"].isin(SWITCHED) & labels["label"].notna()
    book = pd.ExcelFile(raw["harle"])
    scores = harle_replicates(book.parse("Table S3", header=4))
    replicates = replicate_rows(labels, scores)
    weight = harle_weight(book.parse("Table S4", header=4), labels)

    own, _ = study_cells(labels, lines)
    # the own score on the rows that also have the common score: the same cells
    matched, _ = study_cells(labels[common_labels["score"].notna()], lines)
    common, common_studies = study_cells(common_labels, lines)
    inside, _ = study_cells(replicates, lines)

    def switched(first: str, second: str) -> bool:
        return bool({first, second} & SWITCHED)

    def with_harle(first: str, second: str) -> bool:
        return HARLE in (first, second)

    def single(first: str, second: str) -> bool:
        return "+" not in first + second

    def mean_of_two(first: str, second: str) -> bool:
        return {first, second} == {f"{HARLE}|R1+R2", f"{HARLE}|R3"}

    settings = {
        "between_studies": analyse(own, config, workers),
        "switched_own": analyse(matched, config, workers, switched),
        "switched_common": analyse(common, config, workers, switched),
        "harle_others": analyse(own, config, workers, with_harle),
        "harle_others_weighted": analyse(own, config, workers, with_harle, weight),
        "inside_harle": analyse(inside, config, workers, single),
        "inside_harle_weighted": analyse(inside, config, workers, single, weight),
        "inside_harle_mean_of_two": analyse(inside, config, workers, mean_of_two),
    }
    train = own[own["model_id"].isin(split.train)]
    studies = train.groupby(["model_id", *PAIR])["study_id"].nunique()
    same_line = settings["between_studies"]["pooled"]["rank"]["same_line"]
    metrics = {
        "run_id": run_id,
        "workers": workers,
        "settings": settings,
        "common_score": {
            "rows": int(switched_rows.sum()),
            "rows_with_common_score": int(
                (switched_rows & common_labels["score"].notna()).sum()
            ),
            "studies": common_studies[common_studies["study_id"].isin(SWITCHED)]
            .to_dict("records"),
        },
        "harle": {"weighted_pairs": len(weight),
                  "replicate_lists": int(replicates["list_id"].nunique()),
                  "replicate_mean_against_published": published_agreement(
                      labels, scores)},
        "several_studies": {
            "training_cells": len(studies),
            "cells_with_two_studies": int((studies >= 2).sum()),
            "same_line_agreement": same_line,
            # Spearman-Brown: the agreement of a mean of two with a third measurement
            "expected_of_two": float(np.sqrt(2 / (1 + same_line)) * same_line),
        },
    }  # fmt: skip
    write_json(run_dir / "metrics.json", metrics)
    write_text(run_dir / "results.md", results_markdown(metrics))
    return run_dir


def results_markdown(metrics: Mapping[str, Any]) -> str:
    def ranged(row: Mapping[str, Any], key: str) -> str:
        bounds = row[f"{key}_interval"]
        if bounds is None:
            return number(row[key], 3)
        return f"{number(row[key], 3)} [{number(bounds[0], 3)}, {number(bounds[1], 3)}]"

    rows = []
    for key, name, kinds in SETTINGS:
        for kind in kinds:
            pooled = metrics["settings"][key]["pooled"].get(kind)
            if pooled is None:
                rows.append(f"| {name} | {kind} | 0 | | | | |")
                continue
            rows.append(
                f"| {name} | {TARGETS[kind]} | "
                f"{pooled['study_pairs']} | {number(pooled['same_line'], 3)} | "
                f"{number(pooled['other_line'], 3)} | {ranged(pooled, 'advantage')} | "
                f"{ranged(pooled, 'line_part_repeat')} |"
            )
    common, several = metrics["common_score"], metrics["several_studies"]
    return "\n".join(
        [
            f"# Options for a less noisy SL target ({metrics['run_id']})",
            "",
            "**Table 1.** How well the line-specific part of each target repeats "
            "between two independent measurements, mean over the compared pairs of "
            "studies (or of replicates). Same line / another line: correlation over "
            "the common gene pairs. Same-line advantage: that contrast with each "
            "screen's own level removed. Repeat: correlation of a pair's deviation "
            "from its mean over the shared lines. 95% bootstrap over gene pairs. "
            "Higher is better.",
            "",
            "| Setting | Target | Compared | Same line | Another line | "
            "Same-line advantage | Repeat of the line-specific part |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
            *rows,
            "",
            f"One score definition: {common['rows_with_common_score']} of "
            f"{common['rows']} labelled rows of Ito 2021 and Klingbeil 2024 have the "
            "LFC of the double loss and of both single losses.",
            "",
            "Output check: the mean of a line's replicate scores, computed here from "
            "Harle's counts, against Harle's published score: Spearman "
            f"{number(metrics['harle']['replicate_mean_against_published'], 3)} over "
            "pairs, mean over lines.",
            "",
            f"Several studies: {several['cells_with_two_studies']} of "
            f"{several['training_cells']} (line, pair) cells of the training lines "
            "were screened by two or more studies. At the same-line agreement of "
            f"two studies ({number(several['same_line_agreement'], 3)}), the mean of "
            "two agrees with a third at "
            f"{number(several['expected_of_two'], 3)} (Spearman-Brown).",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)
    run(load_config(args.config), args.run_id, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
