"""The noise diagnosis's ``results.md``: one table set per section present."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
from src.experiments.run_dir import interval, number
from src.experiments.prior.base import (
    SCORED,
)
from src.experiments.prior.diagnosis import LISTED_FLOOR, SECTIONS


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
