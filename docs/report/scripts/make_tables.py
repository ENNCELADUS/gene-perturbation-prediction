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
EXP08_PHASE2_CV2 = "results/experiments/08_k562_sl_pair_state_dl/phase2_bce"
EXP08_PHASE2_CV3 = (
    "results/experiments/08_k562_sl_pair_state_dl/phase2_bce_cv2_cv3_lr3e4_ep30"
)
EXP08_PHASE3 = (
    "results/experiments/08_k562_sl_pair_state_dl/phase3_bag_sup_cv2_cv3_lr3e4_ep30"
)
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


def fmt_best(x: float, best: float, dp: int = 3) -> str:
    """Format a value and shade/bold it when it ties the column maximum."""
    cell = fmt(x, dp=dp)
    if abs(x - best) < 5e-13:
        return r"\cellcolor{hl}\textbf{" + cell + "}"
    return cell


def fmt_optional(x: float | None, dp: int = 3) -> str:
    """Format a nullable table cell."""
    if x is None:
        return "--"
    return fmt(x, dp=dp)


def mean_over_cvs(values: list[float]) -> float:
    """Simple average of per-CV means (matches the published 'Mean' column)."""
    return sum(values) / len(values)


def read_result_metrics(result_json: str) -> dict[str, float]:
    """Return full-universe metrics from one exp08 fold result JSON."""
    with open(REPO / result_json) as handle:
        payload = json.load(handle)
    out: dict[str, float] = {}
    for row in payload.get("rows", []):
        if row.get("slice") == "full_universe":
            out[str(row["metric"])] = float(row["value"])
    return out


def count_result_jsons(run_dir: str, split_type: str) -> str:
    """Return N/5 completed exp08 fold JSON count for one run/split."""
    pattern = REPO / run_dir / "_fold_results" / f"{split_type}_fold*.result.json"
    return f"{len(glob.glob(str(pattern)))}/5"


def mean_result_metric(run_dir: str, split_type: str, metric: str) -> float:
    """Return the mean of completed exp08 full-universe fold-result JSON metrics."""
    pattern = REPO / run_dir / "_fold_results" / f"{split_type}_fold*.result.json"
    values = [
        read_result_metrics(str(pathlib.Path(path).relative_to(REPO)))[metric]
        for path in sorted(glob.glob(str(pattern)))
    ]
    if not values:
        raise ValueError(f"no exp08 fold values for {run_dir}/{split_type}/{metric}")
    return mean_over_cvs(values)


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
    """exp06 dependency-only floor (spec C4). Highlight comparable column maxima."""
    models = [("A", "Logistic reg."), ("B", "XGBoost"), ("C", "Degree probe")]
    metrics = ["aupr", "ndcg@10"]
    best = {
        (split, metric): max(read_metric(EXP06, split, key, metric) for key, _ in models)
        for split in ("CV1", "CV2", "CV3")
        for metric in metrics
    }
    lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{Dependency-only floor on the K562 SL benchmark (exp06), per-anchor "
        r"\texttt{cal\_metrics} ranking. CV1 (pair-level) is degree-gameable: a "
        r"degree-only probe wins it, so CV1 ranking is reported as a "
        r"diagnostic, not as generalization. CV2 (one gene held out) and CV3 "
        r"(both genes cold) are the honest surfaces. Shaded bold cells mark the "
        r"best value in each comparable metric column.}",
        r"  \label{tab:floor}",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lcccccc}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{CV1 (diagnostic)} & \multicolumn{2}{c}{CV2 (one held out)} "
        r"& \multicolumn{2}{c}{CV3 (both cold)} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"    Model & AUPRC$\uparrow$ & NDCG@10$\uparrow$ & AUPRC$\uparrow$ "
        r"& NDCG@10$\uparrow$ & AUPRC$\uparrow$ & NDCG@10$\uparrow$ \\",
        r"    \midrule",
    ]
    for key, label in models:
        cells = []
        for split in ("CV1", "CV2", "CV3"):
            for m in metrics:
                value = read_metric(EXP06, split, key, m)
                cells.append(fmt_best(value, best[(split, m)]))
        row = f"    {label} & " + " & ".join(cells) + r" \\"
        lines.append(row)
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    _write("floor", "\n".join(lines))


def build_transcriptome() -> None:
    """exp07 proof-of-concept: observed transcriptome encodes SL partners (C3)."""
    metrics = ["auroc", "ndcg@10", "map@10"]
    models = [("B", "dependency-only"), ("B_transcript", "+ transcriptome")]
    best = {
        (split, metric): max(
            read_metric(EXP07, split, model, metric, slice="full_universe")
            for model, _ in models
        )
        for split in ("CV2", "CV3")
        for metric in metrics
    }
    lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{Adding the observed perturbation transcriptome to the "
        r"dependency-only features (exp07; identical harness, seeds, and splits, "
        r"only the feature matrix changes). On CV2, where one partner is anchored "
        r"by a real Perturb-seq profile, transcriptome features more than double "
        r"ranking quality. On CV3 (both genes cold, no "
        r"observed profile) the lift vanishes. \texttt{full\_universe} = all "
        r"benchmark pairs; \texttt{covered\_pairs} = pairs whose anchored gene has "
        r"a real profile. Shaded bold cells mark the best value within each "
        r"split and metric.}",
        r"  \label{tab:transcriptome}",
        r"  \begin{tabular}{ll" + "ccc" + "}",
        r"    \toprule",
        r"    Split & Features & " + " & ".join(_ARROW[m] for m in metrics) + r" \\",
        r"    \midrule",
    ]
    for split in ("CV2", "CV3"):
        for model, label in models:
            cells = [
                fmt_best(
                    read_metric(EXP07, split, model, m, slice="full_universe"),
                    best[(split, m)],
                )
                for m in metrics
            ]
            row = f"    {split} & {label} & " + " & ".join(cells) + r" \\"
            lines.append(row)
        if split == "CV2":
            lines.append(r"    \cmidrule(lr){1-5}")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    _write("transcriptome", "\n".join(lines))


def build_method() -> None:
    """exp08 preliminary single-fold rows (CNEW). F1/F2 guard: never a 5-fold mean,
    never framed as beating the floor; the floor is shown alongside for reference."""
    metrics = ["auroc", "aupr", "ndcg@10", "map@10"]
    selected = {
        ("CV2", "Phase 2 BCE (selected)"): (
            EXP08_PHASE2_CV2,
            "CV2_fold2.result.json",
        ),
        ("CV2", "Phase 3 bag (selected)"): (
            EXP08_PHASE3,
            "CV2_fold1.result.json",
        ),
        ("CV3", "Phase 2 BCE (selected)"): (
            EXP08_PHASE2_CV3,
            "CV3_fold1.result.json",
        ),
        ("CV3", "Phase 3 bag (selected)"): (
            EXP08_PHASE3,
            "CV3_fold0.result.json",
        ),
    }
    lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{\textbf{Preliminary} results for the virtual-cell framework (exp08,",
        r"  AIVC/STATE${+}$ESM2 adapter). Exp08 rows are selected completed folds, not",
        r"  official 5-fold means, and best-epoch selection reads the fold's test split.",
        r"  Completion counts show available result JSONs. The dependency-only floor",
        r"  (exp06 XGBoost, 5-fold mean) is the gate. No exp08 row clears the floor, and",
        r"  Phase 3 bag supervision does not improve the selected CV2/CV3 ranking metrics.}",
        r"  \label{tab:method}",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lllcccc}",
        r"    \toprule",
        r"    Split & Method & Done & " + " & ".join(_ARROW[m] for m in metrics) + r" \\",
        r"    \midrule",
    ]
    floor_cv2 = [
        read_metric(EXP06, "CV2", "B", "auroc"),
        read_metric(EXP06, "CV2", "B", "aupr"),
        read_metric(EXP06, "CV2", "B", "ndcg@10"),
        read_metric(EXP06, "CV2", "B", "map@10"),
    ]
    lines.append(
        "    CV2 & dependency-only floor (ref.) & 5/5 & "
        + " & ".join(fmt_optional(value) for value in floor_cv2)
        + r" \\"
    )
    for split, label in [
        ("CV2", "Phase 2 BCE (selected)"),
        ("CV2", "Phase 3 bag (selected)"),
    ]:
        run_dir, file_name = selected[(split, label)]
        values = read_result_metrics(f"{run_dir}/_fold_results/{file_name}")
        cells = [fmt_optional(values[m]) for m in metrics]
        lines.append(
            f"    {split} & {label} & {count_result_jsons(run_dir, split)} & "
            + " & ".join(cells)
            + r" \\"
        )
    mean_values = [
        mean_result_metric(EXP08_PHASE3, "CV2", "auroc"),
        None,
        mean_result_metric(EXP08_PHASE3, "CV2", "ndcg@10"),
        mean_result_metric(EXP08_PHASE3, "CV2", "map@10"),
    ]
    lines.append(
        "    CV2 & Phase 3 bag (5-fold mean) & "
        + count_result_jsons(EXP08_PHASE3, "CV2")
        + " & "
        + " & ".join(fmt_optional(value) for value in mean_values)
        + r" \\"
    )
    lines.append(r"    \cmidrule(lr){1-7}")
    floor_cv3 = [read_metric(EXP06, "CV3", "B", "auroc"), None,
                 read_metric(EXP06, "CV3", "B", "ndcg@10"), None]
    lines.append(
        "    CV3 & dependency-only floor (ref.) & 5/5 & "
        + " & ".join(fmt_optional(value) for value in floor_cv3)
        + r" \\"
    )
    for split, label in [
        ("CV3", "Phase 2 BCE (selected)"),
        ("CV3", "Phase 3 bag (selected)"),
    ]:
        run_dir, file_name = selected[(split, label)]
        values = read_result_metrics(f"{run_dir}/_fold_results/{file_name}")
        cells = [fmt_optional(values[m]) for m in metrics]
        lines.append(
            f"    {split} & {label} & {count_result_jsons(run_dir, split)} & "
            + " & ".join(cells)
            + r" \\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    _write("method", "\n".join(lines))


# --- decomposition + foundation writers and main() ---


def build_decomposition() -> None:
    """exp09 selectivity: most cold-start lift is pan-essentiality, not pair-specific
    co-dependency. Highlight only comparable full-universe rows within each split."""
    metrics = ["auroc", "aupr", "ndcg@10"]
    models = [("B", "dependency-only"), ("B_xcl", "+ selectivity")]
    best = {
        (split, metric): max(
            read_metric(EXP09, split, model, metric, slice="full_universe")
            for model, _ in models
        )
        for split in ("CV2", "CV3")
        for metric in metrics
    }
    lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{Cross-cell-line selectivity features (exp09). They lift "
        r"classification on both honest splits, but when the benchmark is "
        r"restricted to non-pan-essential pairs on CV3, AUROC and AUPR "
        r"collapse toward chance. Most of the genome-wide co-dependency signal is "
        r"pan-essentiality structure; pair-specific synthetic lethality is the thin, "
        r"hard residual. Shaded bold cells mark the best value within each "
        r"comparable full-universe split and metric.}",
        r"  \label{tab:decomposition}",
        r"  \begin{tabular}{ll" + "ccc" + "}",
        r"    \toprule",
        r"    Split & Features & " + " & ".join(_ARROW[m] for m in metrics) + r" \\",
        r"    \midrule",
    ]
    for split in ("CV2", "CV3"):
        for model, label in models:
            cells = [
                fmt_best(
                    read_metric(EXP09, split, model, m, slice="full_universe"),
                    best[(split, m)],
                )
                for m in metrics
            ]
            lines.append(f"    {split} & {label} & " + " & ".join(cells) + r" \\")
        if split == "CV2":
            lines.append(r"    \cmidrule(lr){1-5}")
    # confound-revealing slice
    lines.append(r"    \cmidrule(lr){1-5}")
    cells = [fmt(read_metric(EXP09, "CV3", "B_xcl", m, slice="non_pan_essential"))
             for m in metrics]
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
        r"\begin{table}[!htbp]",
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


def build_benchmark() -> None:
    """Cross-method comparison: published label-graph methods (teammate
    reproduction CSV) vs our label-free functional-signal methods (exp06/exp07),
    same K562-DepMap splits and per-anchor cal_metrics (spec: tab_benchmark)."""
    published = pd.read_csv(REPO / "docs/report/tables/benchmark_published.csv")
    col_order = ["cv1_f1", "cv1_ndcg10", "cv2_f1", "cv2_ndcg10",
                 "cv3_f1", "cv3_ndcg10", "mean_auroc", "mean_aupr",
                 "mean_f1", "mean_ndcg10"]

    def functional_row(csv: str, model: str, sl: str | None) -> list[float]:
        cells: list[float] = []
        per_metric: dict[str, list[float]] = {
            m: [] for m in ("auroc", "aupr", "f1", "ndcg@10")
        }
        for split in ("CV1", "CV2", "CV3"):
            f1 = read_metric(csv, split, model, "f1", slice=sl)
            nd = read_metric(csv, split, model, "ndcg@10", slice=sl)
            cells += [f1, nd]
            for m in per_metric:
                per_metric[m].append(read_metric(csv, split, model, m, slice=sl))
        cells += [mean_over_cvs(per_metric[m])
                  for m in ("auroc", "aupr", "f1", "ndcg@10")]
        return cells

    rows = [
        (str(r["model"]), [float(r[c]) for c in col_order])
        for _, r in published.iterrows()
    ]
    rows += [
        ("Dependency-only (exp06)", functional_row(EXP06, "B", None)),
        (r"\quad + transcriptome (exp07)",
         functional_row(EXP07, "B_transcript", "full_universe")),
    ]
    best_by_col = [max(row[1][i] for row in rows) for i in range(len(col_order))]

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Synthetic-lethality prediction on the K562 SL benchmark: "
        r"published label-graph methods (which consume the SL association matrix "
        r"as input; \citealp{cai2020ddgcn,huang2019grsmf,liu2020sl2mf,zhu2023slgnn}) versus our "
        r"label-free functional-signal methods. Same K562-DepMap Rand 1:1 splits "
        r"and identical per-anchor \texttt{cal\_metrics} ranking on both sides. "
        r"Shaded bold cells mark the best value in each metric column, including ties. Label-graph methods lead on ranking; our "
        r"functional methods use no SL labels yet stay classification-competitive, "
        r"leaving ranking as the open gap. CV1 is the degree-gameable diagnostic.}",
        r"  \label{tab:benchmark}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{l" + "cc" * 3 + "cccc}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{CV1 (diag.)} & \multicolumn{2}{c}{CV2} "
        r"& \multicolumn{2}{c}{CV3} & \multicolumn{4}{c}{Cross-CV mean} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-11}",
        r"    Method & F1$\uparrow$ & N@10$\uparrow$ & F1$\uparrow$ & N@10$\uparrow$ "
        r"& F1$\uparrow$ & N@10$\uparrow$ & AUROC$\uparrow$ & AUPR$\uparrow$ "
        r"& F1$\uparrow$ & N@10$\uparrow$ \\",
        r"    \midrule",
        r"    \multicolumn{11}{l}{\emph{Label-graph methods (consume the SL matrix as input)}} \\",
    ]
    for model, values in rows[: len(published)]:
        cells = [fmt_best(value, best_by_col[i]) for i, value in enumerate(values)]
        lines.append(f"    {model} & " + " & ".join(cells) + r" \\")
    lines.append(r"    \cmidrule(lr){1-11}")
    lines.append(r"    \multicolumn{11}{l}{\emph{Functional-signal methods (label-free)}} \\")
    for model, values in rows[len(published):]:
        cells = [fmt_best(value, best_by_col[i]) for i, value in enumerate(values)]
        lines.append(f"    {model} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}", ""]
    _write("benchmark", "\n".join(lines))


_BUILDERS = {
    "floor": build_floor,
    "transcriptome": build_transcriptome,
    "method": build_method,
    "decomposition": build_decomposition,
    "foundation": build_foundation,
    "benchmark": build_benchmark,
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
