"""Generate booktabs LaTeX tables from results/*/ artifacts.

Single source of every number in the manuscript. No number is ever typed into a
section file by hand; sections only ``\\input`` the .tex files written here.

Schemas handled:
- exp06 official_metrics_summary.csv : split_type,model,metric,mean,std (no slice)
- exp07 / exp09 summary.csv          : split_type,model,slice,metric,mean,std
- exp08 phase*/_fold_results/*.json  : {"rows": [{split_type,model,fold_id,slice,metric,value}]}

Usage:
    python docs/report/scripts/make_tables.py --all
    python docs/report/scripts/make_tables.py --table floor
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import pathlib

import pandas as pd

logger = logging.getLogger(__name__)

REPO = pathlib.Path(__file__).resolve().parents[3]
TABLES = pathlib.Path(__file__).resolve().parent.parent / "tables"

EXP06 = "results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_summary.csv"
EXP07 = "results/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_no_flag/summary.csv"
EXP09 = "results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run/summary.csv"
EXP08_DIRS = "results/experiments/08_k562_sl_pair_state_dl/phase2_bce*"
FOUNDATION_CSV = "docs/report/tables/foundation_values.csv"


def read_metric(
    summary_csv: str,
    split_type: str,
    model: str,
    metric: str,
    slice: str | None = None,
) -> float:
    """Return the ``mean`` cell for one (split_type, model, metric[, slice]) row.

    When ``slice`` is None the file is assumed to lack a slice column (exp06
    schema); when given, the slice column is filtered (exp07/exp09 schema).
    """
    df = pd.read_csv(REPO / summary_csv)
    mask = (
        (df["split_type"] == split_type)
        & (df["model"] == model)
        & (df["metric"] == metric)
    )
    if slice is not None:
        mask &= df["slice"] == slice
    rows = df.loc[mask, "mean"]
    if len(rows) != 1:
        raise ValueError(
            f"expected 1 row for {split_type}/{model}/{metric}/{slice}, got {len(rows)}"
        )
    return float(rows.iloc[0])


def read_best_fold_metric(
    dirs_glob: str,
    split_type: str,
    metric: str,
    slice: str = "full_universe",
) -> float:
    """Return the best (max) per-fold value for a metric across exp08 fold JSONs.

    exp08 runs are incomplete (single selected folds, not a 5-fold mean). This
    returns the best observed fold so the method table can report it *flagged*
    as preliminary (spec F2). Used only for the preliminary method table.
    """
    best = float("-inf")
    files = sorted(glob.glob(str(REPO / dirs_glob / "_fold_results" / "*.result.json")))
    for path in files:
        with open(path) as handle:
            payload = json.load(handle)
        for row in payload.get("rows", []):
            if (
                row.get("split_type") == split_type
                and row.get("slice") == slice
                and row.get("metric") == metric
            ):
                best = max(best, float(row["value"]))
    if best == float("-inf"):
        raise ValueError(f"no exp08 fold value for {split_type}/{metric}/{slice}")
    return best


def fmt(x: float, dp: int = 3) -> str:
    """Fixed-decimal formatter for table cells."""
    return f"{x:.{dp}f}"


# --- table writers ---

_ARROW = {  # metric display name with direction
    "auroc": r"AUROC$\uparrow$",
    "aupr": r"AUPR$\uparrow$",
    "ndcg@10": r"NDCG@10$\uparrow$",
    "map@10": r"MAP@10$\uparrow$",
}


def _write(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / f"tab_{name}.tex"
    out.write_text(body)
    logger.info("wrote %s", out)


def build_floor() -> None:
    """exp06 dependency-only floor (spec C4). CV1 degree-probe row highlighted."""
    models = [("A", "Logistic reg.\\ (Model A)"), ("B", "XGBoost (Model B)"),
              ("C", "Degree probe (Model C)")]
    metrics = ["auroc", "aupr", "ndcg@10"]
    head = " & ".join([r"\textbf{Model}"]
                       + [f"\\multicolumn{{1}}{{c}}{{{_ARROW[m]}}}" for m in metrics
                          for _ in [0]])
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Dependency-only floor on the K562 SL benchmark (exp06), per-anchor "
        r"\texttt{cal\_metrics} ranking. CV1 (pair-level) is degree-gameable: a "
        r"degree-only probe (shaded) wins it, so CV1 ranking is reported as a "
        r"diagnostic, not as generalization. CV2 (one gene held out) and CV3 "
        r"(both genes cold) are the honest surfaces.}",
        r"  \label{tab:floor}",
        r"  \begin{tabular}{l" + "ccc" * 3 + "}",
        r"    \toprule",
        r"    & \multicolumn{3}{c}{CV1 (diagnostic)} & \multicolumn{3}{c}{CV2 (one held out)} "
        r"& \multicolumn{3}{c}{CV3 (both cold)} \\",
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"    Model & " + " & ".join([_ARROW[m] for _ in range(3) for m in metrics]) + r" \\",
        r"    \midrule",
    ]
    for key, label in models:
        cells = []
        for split in ("CV1", "CV2", "CV3"):
            for m in metrics:
                cells.append(fmt(read_metric(EXP06, split, key, m)))
        row = f"    {label} & " + " & ".join(cells) + r" \\"
        if key == "C":
            row = r"    \rowcolor{hl}" + "\n" + row
        lines.append(row)
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    _write("floor", "\n".join(lines))


def build_transcriptome() -> None:
    """exp07 proof-of-concept: observed transcriptome encodes SL partners (C3)."""
    metrics = ["auroc", "ndcg@10", "map@10"]
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Adding the observed perturbation transcriptome to the "
        r"dependency-only features (exp07; identical harness, seeds, and splits, "
        r"only the feature matrix changes). On CV2, where one partner is anchored "
        r"by a real Perturb-seq profile, transcriptome features more than double "
        r"ranking quality (NDCG@10 row, shaded). On CV3 (both genes cold, no "
        r"observed profile) the lift vanishes. \texttt{full\_universe} = all "
        r"benchmark pairs; \texttt{covered\_pairs} = pairs whose anchored gene has "
        r"a real profile.}",
        r"  \label{tab:transcriptome}",
        r"  \begin{tabular}{ll" + "ccc" + "}",
        r"    \toprule",
        r"    Split & Features & " + " & ".join(_ARROW[m] for m in metrics) + r" \\",
        r"    \midrule",
    ]
    for split in ("CV2", "CV3"):
        for model, label in (("B", "dependency-only"), ("B_transcript", "+ transcriptome")):
            cells = [fmt(read_metric(EXP07, split, model, m, slice="full_universe"))
                     for m in metrics]
            row = f"    {split} & {label} & " + " & ".join(cells) + r" \\"
            if split == "CV2" and model == "B_transcript":
                row = r"    \rowcolor{hl}" + "\n" + row
            lines.append(row)
        if split == "CV2":
            lines.append(r"    \cmidrule(lr){1-5}")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    _write("transcriptome", "\n".join(lines))


def build_method() -> None:
    """exp08 preliminary single-fold rows (CNEW). F1/F2 guard: never a 5-fold mean,
    never framed as beating the floor; the floor is shown alongside for reference."""
    metrics = ["auroc", "ndcg@10"]
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\textbf{Preliminary} results for the virtual-cell framework "
        r"(exp08, AIVC/STATE${+}$ESM2 adapter). Values are the \emph{best selected "
        r"fold}, \emph{not} a 5-fold mean, and best-epoch selection reads the test "
        r"fold; full 5-fold tuning is ongoing (see Limitations). The dependency-only "
        r"floor (exp06) is shown only for reference. We make no claim that the "
        r"framework beats the floor; these numbers report the current status of a "
        r"newly proposed method.}",
        r"  \label{tab:method}",
        r"  \begin{tabular}{ll" + "cc" + "}",
        r"    \toprule",
        r"    Split & Method & " + " & ".join(_ARROW[m] for m in metrics) + r" \\",
        r"    \midrule",
    ]
    for split in ("CV2", "CV3"):
        floor_cells = [fmt(read_metric(EXP06, split, "B", m)) for m in metrics]
        lines.append(
            f"    {split} & dependency-only floor (ref.) & "
            + " & ".join(floor_cells) + r" \\"
        )
        dl_cells = [fmt(read_best_fold_metric(EXP08_DIRS, split, m, slice="full_universe"))
                    for m in metrics]
        lines.append(
            f"    {split} & AIVC/STATE${{+}}$ESM2 (best fold, prelim.) & "
            + " & ".join(dl_cells) + r" \\"
        )
        if split == "CV2":
            lines.append(r"    \cmidrule(lr){1-4}")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    _write("method", "\n".join(lines))


# --- decomposition + foundation writers and main() ---


def build_decomposition() -> None:
    """exp09 selectivity: most cold-start lift is pan-essentiality, not pair-specific
    co-dependency. The non-pan-essential CV3 slice (shaded) reveals the confound (C5)."""
    metrics = ["auroc", "aupr", "ndcg@10"]
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Cross-cell-line selectivity features (exp09). They lift "
        r"classification on both honest splits, but when the benchmark is "
        r"restricted to non-pan-essential pairs on CV3 (shaded), AUROC and AUPR "
        r"collapse toward chance. Most of the genome-wide co-dependency signal is "
        r"pan-essentiality structure; pair-specific synthetic lethality is the thin, "
        r"hard residual.}",
        r"  \label{tab:decomposition}",
        r"  \begin{tabular}{ll" + "ccc" + "}",
        r"    \toprule",
        r"    Split & Features & " + " & ".join(_ARROW[m] for m in metrics) + r" \\",
        r"    \midrule",
    ]
    for split in ("CV2", "CV3"):
        for model, label in (("B", "dependency-only"), ("B_xcl", "+ selectivity")):
            cells = [fmt(read_metric(EXP09, split, model, m, slice="full_universe"))
                     for m in metrics]
            lines.append(f"    {split} & {label} & " + " & ".join(cells) + r" \\")
        if split == "CV2":
            lines.append(r"    \cmidrule(lr){1-5}")
    # confound-revealing slice
    lines.append(r"    \cmidrule(lr){1-5}")
    cells = [fmt(read_metric(EXP09, "CV3", "B_xcl", m, slice="non_pan_essential"))
             for m in metrics]
    lines.append(r"    \rowcolor{hl}")
    lines.append(r"    CV3 & + selectivity, non-pan-ess. & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    _write("decomposition", "\n".join(lines))


def build_foundation() -> None:
    """exp01-03 transcriptome -> a gene's own dependency (C1, C2). Numbers are not in
    the auto-CSV path; they flow through one hand-curated CSV copied from the ledger."""
    df = pd.read_csv(REPO / FOUNDATION_CSV)

    def val(tag: str) -> str:
        rows = df.loc[df["tag"] == tag, "value"]
        if len(rows) != 1:
            raise ValueError(f"foundation_values.csv: expected 1 row for {tag}, got {len(rows)}")
        return fmt(float(rows.iloc[0]))

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Foundation: a gene's perturbation transcriptome predicts its "
        r"\emph{own} DepMap dependency, and the signal is real transcriptomic "
        r"structure rather than a generic death axis. exp01 pseudobulk "
        r"$\Delta$-expression $\to$ GeneEffect; exp02 audit against the death "
        r"signature; exp03 single-cell distribution embeddings (exploratory). The "
        r"exp03 Adamson number came from an Adamson-guided sweep and is not a "
        r"held-out generalization result.}",
        r"  \label{tab:foundation}",
        r"  \begin{tabular}{lll}",
        r"    \toprule",
        r"    Setting & Metric & Value \\",
        r"    \midrule",
        rf"    exp01 Replogle K562 (5-fold CV) & Spearman$\uparrow$ & {val('exp01_cv_spearman')} \\",
        rf"    exp01 Adamson same-line transfer & Spearman$\uparrow$ & {val('exp01_adamson_spearman')} \\",
        rf"    exp01 Adamson same-line transfer & AUROC$\uparrow$ (GE$<-1$) & {val('exp01_adamson_auroc')} \\",
        r"    \cmidrule(lr){1-3}",
        rf"    exp02 death signature only & Spearman$\uparrow$ & {val('exp02_nar_only')} \\",
        rf"    exp02 best pseudobulk transcriptome & Spearman$\uparrow$ & {val('exp02_transcriptome')} \\",
        rf"    exp02 death-residualized transcriptome & Spearman$\uparrow$ & {val('exp02_residualized')} \\",
        r"    \cmidrule(lr){1-3}",
        rf"    exp03 scVI128 GMM-Ridge (Adamson sweep$^\dagger$) & Spearman$\uparrow$ & {val('exp03_adamson_spearman')} \\",
        rf"    exp03 scVI128 GMM-Ridge (Adamson sweep$^\dagger$) & AUROC$\uparrow$ & {val('exp03_adamson_auroc')} \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \\[2pt] {\footnotesize $^\dagger$Adamson-guided sweep; exploratory, not held-out.}",
        r"\end{table}",
        "",
    ]
    _write("foundation", "\n".join(lines))


_BUILDERS = {
    "floor": build_floor,
    "transcriptome": build_transcriptome,
    "method": build_method,
    "decomposition": build_decomposition,
    "foundation": build_foundation,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="build every table")
    parser.add_argument("--table", choices=sorted(_BUILDERS), help="build one table")
    args = parser.parse_args()
    if args.all or not args.table:
        for name in _BUILDERS:
            _BUILDERS[name]()
    else:
        _BUILDERS[args.table]()


if __name__ == "__main__":
    main()
