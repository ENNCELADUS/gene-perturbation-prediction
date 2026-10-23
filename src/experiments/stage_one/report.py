"""A stage-one report: every row's predictions (runs, the prior, the base, the ridge)
scored on validation, test and out of fold, the paired contrasts, and results.md."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import ladder_reader as reader
from src.data.io import read_json, write_json, write_text
from src.experiments.run_dir import assert_bound, interval, number
from src.experiments.stage_one.resident import finished_prior, same_ladder
from src.experiments.stage_one.ridge import cached_ridge
from src.experiments.stage_one.run import REPORTED, ROW, SPLITS, run_directory
from src.experiments.stage_one.setup import Setup, load_setup

log = logging.getLogger(__name__)


def split_scores(
    setup: Setup, split: str, frame: pd.DataFrame, truth: pd.DataFrame
) -> dict[str, Any]:
    """A row's scores on one split; out of fold, the rank metrics are scored within
    each fold and averaged (``fold_accuracy``), since each fold's base differs."""
    from src.eval.accuracy import fold_accuracy, wide_accuracy

    ladder = setup.ladder
    if split == "oof":
        return fold_accuracy(
            frame, truth, setup.definitions, ladder.folds, panel=ladder.panel
        )
    return wide_accuracy(frame, truth, setup.definitions, panel=ladder.panel)


def finished_run(
    config: Mapping[str, Any], stem: str, fold: int | None, report_ladder: str | Path
) -> Path | None:
    """A finished run's directory (its ``row.json`` exists), checked to be bound to
    this config and fold and to read the report's ladder; None when not finished."""
    run_dir = run_directory(config, stem, fold)
    if not (run_dir / ROW).is_file():
        return None
    assert_bound(run_dir, {"run": config, "fold": fold})
    same_ladder(config["data"]["ladder_config"], report_ladder, str(run_dir))
    return run_dir


def row_predictions(
    setup: Setup,
    row: Mapping[str, Any],
    ridge: Mapping[float, pd.DataFrame],
    report_ladder: str | Path,
) -> dict[str, pd.DataFrame | None]:
    """A report row's predictions (scoring units) on ``val``, ``test`` and, out of
    fold, ``oof`` (the single-cell training lines); None where a run is missing."""
    from src.experiments.stage_one.config import load_stage_one_config

    ladder = setup.ladder
    val, test = ladder.role_lines("val"), ladder.role_lines("test")
    cells = ladder.role_lines("train_cells")
    folds = sorted(set(ladder.folds.values()))
    by_fold = {k: [m for m in cells if ladder.folds[m] == k] for k in folds}
    kind = row["kind"]
    if kind == "base":
        m_g = ladder.genes["m_g"]
        oof = pd.concat(
            [
                reader.base_prediction(ladder, setup.targets(k).m_g, by_fold[k])
                for k in folds
            ]
        )
        held = reader.base_prediction(ladder, m_g, [*val, *test])
        return {"val": held.loc[val], "test": held.loc[test], "oof": oof.loc[cells]}
    if kind == "dh_ridge":
        frame = ridge[float(row["penalty"])]
        return {"val": frame.loc[val], "test": frame.loc[test], "oof": frame.loc[cells]}
    out: dict[str, pd.DataFrame | None] = {"val": None, "test": None, "oof": None}
    if kind == "prior":
        run_dir = finished_prior(row["run"], report_ladder)
        if run_dir is None:
            return out
        for split, wanted in (("val", val), ("test", test), ("oof", cells)):
            frame = pd.read_parquet(run_dir / f"predictions_{split}.parquet")
            out[split] = frame.loc[wanted]
        return out
    config = load_stage_one_config(Path(row["config"]))
    stem = Path(row["config"]).stem
    main = finished_run(config, stem, None, report_ladder)
    if main is not None:
        frame = pd.read_parquet(main / "predictions.parquet")
        out["val"], out["test"] = frame.loc[val], frame.loc[test]
    parts = []
    for k in folds:
        run_dir = finished_run(config, stem, k, report_ladder)
        if run_dir is None:
            return out
        parts.append(pd.read_parquet(run_dir / "predictions.parquet").loc[by_fold[k]])
    out["oof"] = pd.concat(parts).loc[cells]
    return out


def _contrast(
    setup: Setup,
    left: pd.DataFrame,
    right: pd.DataFrame,
    truth: pd.DataFrame,
    repeats: int,
) -> dict[str, Any]:
    """Per-line residual Spearman, selective-Spearman and within-gene R²
    differences, left minus right, with 95% paired line-bootstrap intervals (the
    same resamples for both). The per-line difference covers the lines defined on
    both sides, the selective Spearman difference the selective genes defined on
    both sides (a row predicting the base for a gene is undefined there), and
    within-gene R² every gene."""
    from src.context_prior.noise import line_sums, r2_gain
    from src.eval.accuracy import (
        BOOTSTRAP_SEED,
        defined_genes,
        gain,
        line_spearman_gain,
        line_spearmans,
    )

    definitions = setup.definitions
    selective = list(definitions.selective)
    common = set(defined_genes(left, truth, selective)) & set(
        defined_genes(right, truth, selective)
    )
    genes = [g for g in selective if g in common]
    sums = [
        line_sums(
            frame[selective].to_numpy(np.float64), truth[selective].to_numpy(np.float64)
        )
        for frame in (left, right)
    ]
    spearman = {"difference": None, "interval": [None, None]}
    if genes:
        spearman = gain(
            left.astype(np.float64),
            right.astype(np.float64),
            truth,
            definitions,
            repeats,
            genes=genes,
        )
    return {
        "line_spearman": line_spearman_gain(
            line_spearmans(left, truth, selective),
            line_spearmans(right, truth, selective),
            repeats=repeats,
        ),
        "selective_spearman": spearman | {"genes": len(genes)},
        "within_gene_r2": r2_gain(
            sums[0][0], sums[1][0], sums[0][1], repeats=repeats, seed=BOOTSTRAP_SEED
        ),
    }


def report(config_path: Path, device: str) -> Path:
    from src.experiments.stage_one.config import load_stage_one_config

    config = load_stage_one_config(config_path)
    if config["kind"] != "report":
        raise ValueError(f"{config_path} is a {config['kind']} config, not a report")
    out = Path(config["output_root"]) / config["task"] / Path(config_path).stem
    out.mkdir(parents=True, exist_ok=True)
    setup = load_setup(config["ladder_config"])
    ladder = setup.ladder
    groups = {
        "val": ladder.role_lines("val"),
        "test": ladder.role_lines("test"),
        "oof": ladder.role_lines("train_cells"),
    }
    truth = {name: setup.truth(lines) for name, lines in groups.items()}
    penalties = sorted(
        {float(r["penalty"]) for r in config["rows"] if r["kind"] == "dh_ridge"}
    )
    ridge = cached_ridge(setup, config["ladder_config"], penalties, out, device)
    predictions = {
        row["name"]: row_predictions(setup, row, ridge, config["ladder_config"])
        for row in config["rows"]
    }
    scores = {
        name: {
            split: None
            if frame is None
            else split_scores(setup, split, frame, truth[split])
            for split, frame in splits.items()
        }
        for name, splits in predictions.items()
    }
    contrasts = []
    for left, right in config["contrasts"]:
        entry = {"left": left, "right": right}
        for split in SPLITS:
            a, b = predictions[left][split], predictions[right][split]
            entry[split] = (
                None
                if a is None or b is None
                else _contrast(
                    setup, a, b, truth[split], int(config["bootstrap_repeats"])
                )
            )
        contrasts.append(entry)
    training = {}
    for row in config["rows"]:
        if row["kind"] == "run":
            run_config = load_stage_one_config(Path(row["config"]))
            run_dir = finished_run(
                run_config, Path(row["config"]).stem, None, config["ladder_config"]
            )
            training[row["name"]] = (
                None if run_dir is None else read_json(run_dir / ROW)
            )
    payload = {
        "config": config,
        "lines": {name: len(lines) for name, lines in groups.items()},
        "scores": scores,
        "contrasts": contrasts,
        "training": {
            name: None
            if record is None
            else {
                key: record[key]
                for key in ("best_epoch", "epochs", "updates", "lines_trained")
            }
            | {"train_in_sample": record["scores"]["train_in_sample"]}
            for name, record in training.items()
        },
    }
    write_json(out / "results.json", payload)
    write_text(out / "results.md", results_markdown(payload))
    log.info("report: %s", out / "results.md")
    return out


def results_markdown(payload: Mapping[str, Any]) -> str:
    config = payload["config"]
    counts = payload["lines"]
    out = [
        f"# Stage-one report: {Path(config['task'])} ({len(config['rows'])} rows)",
        "",
        "Scoring units: every prediction is GeneEffect re-centred on the protocol's "
        "377-line mean (`(m_g - mu_g) / sigma_g + z`); metrics as protocol §6 and "
        "§14 (within-gene R² pooled over the selective genes; calibration slope: "
        "measured residual on predicted, each gene's offset removed). "
        f"Validation {counts['val']} lines, test {counts['test']}, out of fold "
        f"{counts['oof']} single-cell training lines (each scored by the run without "
        "its fold; each fold's base differs, so the rank metrics are scored within "
        "each fold and averaged, and within-gene R², RMSE and per-line Pearson are "
        "pooled). The per-line residual Spearman (the selector, protocol §6) "
        "correlates each line's predicted and measured residuals across the 4,323 "
        "selective genes and averages over lines; it is pooled out of fold, as each "
        "line is compared with its own truth. "
        "Selective Spearman covers the 4,323 protocol genes (genes scored: "
        "those defined), the panel column the selective genes in the decode panel; a "
        "contrast's Spearman difference covers the genes defined on both sides. "
        "`undefined`: a prediction constant within every gene, or a run not "
        "finished. Nothing is chosen here; this is not SL evidence.",
    ]
    titles = {"val": "Validation", "test": "Test", "oof": "Out of fold (telemetry)"}
    for split, title in titles.items():
        header = " | ".join(label for _, label in REPORTED)
        out += [
            "",
            f"## {title}",
            "",
            f"| Row | {header} |",
            "|" + " --- |" * (len(REPORTED) + 1),
        ]
        for name, splits in payload["scores"].items():
            entry = splits[split]
            cells = [
                "undefined" if entry is None else number(entry[key])
                for key, _ in REPORTED
            ]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    out += [
        "",
        "## Contrasts (left minus right; 95% paired line-bootstrap intervals)",
        "",
        "| Left | Right | Val per-line Spearman | Test per-line Spearman "
        "| Val selective Spearman | Test selective Spearman "
        "| Val within-gene R² | Test within-gene R² |",
        "|" + " --- |" * 8,
    ]
    for entry in payload["contrasts"]:
        cells = [entry["left"], entry["right"]]
        for metric in ("line_spearman", "selective_spearman", "within_gene_r2"):
            for split in SPLITS:
                result = entry[split]
                text = "undefined" if result is None else interval(result[metric])
                if result is not None and metric == "selective_spearman":
                    text += f" ({result[metric]['genes']} genes)"
                cells.append(text)
        out.append("| " + " | ".join(cells) + " |")
    out += [
        "",
        "## Training (selected epoch, and the gap from training to validation)",
        "",
        "| Row | Best epoch | Epochs | Updates | Lines trained "
        "| Train per-line Spearman | Val per-line Spearman "
        "| Train selective Spearman | Val selective Spearman "
        "| Train within-gene R² | Val within-gene R² |",
        "|" + " --- |" * 11,
    ]
    for name, record in payload["training"].items():
        if record is None:
            out.append(f"| {name} |" + " not run |" * 10)
            continue
        train, val = record["train_in_sample"], payload["scores"][name]["val"]
        # Rows trained before the per-line selector (2026-10-21) did not record it.
        train_line = (
            number(train["line_spearman"])
            if "line_spearman" in train
            else "not recorded"
        )
        out.append(
            f"| {name} | {record['best_epoch']} | {record['epochs']} "
            f"| {record['updates']} "
            f"| {record['lines_trained']} | {train_line} "
            f"| {number(None if val is None else val['line_spearman'])} "
            f"| {number(train['selective_spearman'])} "
            f"| {number(None if val is None else val['selective_spearman'])} "
            f"| {number(train['within_gene_r2'])} "
            f"| {number(None if val is None else val['within_gene_r2'])} |"
        )
    return "\n".join(out) + "\n"
