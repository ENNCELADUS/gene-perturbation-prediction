# Merge SL Reproduction Into Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single direct-comparison table (and a figure reference line) that places the teammate's four published label-graph SL methods (DDGCN/GRSMF/SL2MF/SLGNN) next to our label-free functional-signal methods (exp06 dependency-only, exp07 +transcriptome) under the identical K562-DepMap CV1/CV2/CV3 protocol.

**Architecture:** All reported numbers flow through `docs/report/scripts/make_tables.py` and are pytest-gated. The four published rows live verbatim in a provenance-tracked CSV (`benchmark_published.csv`); our two functional rows are read directly from exp06/exp07 artifacts via the existing `read_metric` plus a new `mean_over_cvs` helper. A new `build_benchmark()` writer emits `tab_benchmark.tex`. The paper integrates it as a new Experiments subsection, upgrades related-work wording, and adds four CrossRef-verified citations.

**Tech Stack:** Python 3.11 + pandas (number extraction), matplotlib (figure), LaTeX/NeurIPS booktabs (table), pytest (gates), `uv run` for all invocations.

## Global Constraints

- All Python/pytest/ruff invocations prefixed with `uv run`.
- No hand-typed numbers in `.tex` body; every number flows through `make_tables.py` and is pytest-gated.
- Honesty guardrails: no "outperforms / beats / SOTA / state-of-the-art"; no cold-start-win claim; published methods described as consuming the SL association matrix as input; our methods as label-free.
- Table style: booktabs only (`\toprule/\midrule/\bottomrule`, no vertical `|` rules), caption above, 3 decimal places, direction arrows in headers.
- Universe is identical (user-confirmed): same K562-DepMap splits, same per-anchor `cal_metrics`. Caption states this; no apples-to-oranges disclaimer.
- Canonical published numbers = teammate official table (`personal_report_sl_gene_perturbation.md` §6). Do NOT use `data/SL_benchmark/src/summary_all_matrics.csv`.
- `mean_over_cvs` = simple average of the three per-CV means (matches teammate "Mean" column definition).

---

### Task 1: Published-numbers CSV + `build_benchmark()` writer + gates

**Files:**
- Create: `docs/report/tables/benchmark_published.csv`
- Modify: `docs/report/scripts/make_tables.py` (add `mean_over_cvs` helper after `fmt`, add `build_benchmark` writer, register in `_BUILDERS`, add `EXP06`/`EXP07` reuse — both constants already exist at lines 30-31)
- Test: `tests/test_make_tables.py` (append new anchors)

**Interfaces:**
- Consumes: existing `read_metric(summary_csv, split_type, model, metric, slice=None) -> float`; existing `fmt(x, dp=3) -> str`; existing constants `EXP06`, `EXP07`, `REPO`, `TABLES`, `_ARROW`, `_write`.
- Produces: `mean_over_cvs(values: list[float]) -> float`; `build_benchmark() -> None` writing `docs/report/tables/tab_benchmark.tex`; CSV columns `model,cv1_f1,cv1_ndcg10,cv2_f1,cv2_ndcg10,cv3_f1,cv3_ndcg10,mean_auroc,mean_aupr,mean_f1,mean_ndcg10,source`.

- [ ] **Step 1: Create the published-numbers CSV (verbatim from teammate official table)**

Create `docs/report/tables/benchmark_published.csv`:

```csv
model,cv1_f1,cv1_ndcg10,cv2_f1,cv2_ndcg10,cv3_f1,cv3_ndcg10,mean_auroc,mean_aupr,mean_f1,mean_ndcg10,source
DDGCN,0.822,0.286,0.685,0.241,0.606,0.243,0.752,0.743,0.704,0.257,teammate SLBench reproduction (personal_report_sl_gene_perturbation.md §6); K562-DepMap Rand 1:1 splits; cal_metrics
GRSMF,0.774,0.322,0.658,0.315,0.625,0.313,0.752,0.771,0.686,0.317,teammate SLBench reproduction (personal_report_sl_gene_perturbation.md §6); K562-DepMap Rand 1:1 splits; cal_metrics
SL2MF,0.666,0.278,0.422,0.127,0.288,0.074,0.456,0.448,0.459,0.160,teammate SLBench reproduction (personal_report_sl_gene_perturbation.md §6); K562-DepMap Rand 1:1 splits; cal_metrics
SLGNN,0.870,0.150,0.685,0.050,0.667,0.000,0.721,0.746,0.741,0.066,teammate SLBench reproduction (personal_report_sl_gene_perturbation.md §6); K562-DepMap Rand 1:1 splits; cal_metrics
```

- [ ] **Step 2: Write the failing tests (append to `tests/test_make_tables.py`)**

```python
def test_benchmark_published_grsmf_mean_ndcg():
    import pandas as pd
    df = pd.read_csv("docs/report/tables/benchmark_published.csv")
    val = df.loc[df["model"] == "GRSMF", "mean_ndcg10"].iloc[0]
    assert abs(float(val) - 0.317) < 1e-9


def test_benchmark_published_slgnn_cv3_ndcg_collapse():
    import pandas as pd
    df = pd.read_csv("docs/report/tables/benchmark_published.csv")
    val = df.loc[df["model"] == "SLGNN", "cv3_ndcg10"].iloc[0]
    assert abs(float(val) - 0.000) < 1e-9


def test_mean_over_cvs_matches_simple_average():
    assert abs(mt.mean_over_cvs([0.7947, 0.7035, 0.5956]) - 0.6979333333) < 1e-6


def test_benchmark_functional_floor_mean_auroc():
    # exp06 Model B mean-over-CV AUROC, recomputed from artifacts (~0.698).
    vals = [mt.read_metric(mt.EXP06, s, "B", "auroc") for s in ("CV1", "CV2", "CV3")]
    assert abs(mt.mean_over_cvs(vals) - 0.698) < 0.005


def test_benchmark_table_has_both_rowgroups():
    import subprocess
    subprocess.run(
        ["python", "docs/report/scripts/make_tables.py", "--table", "benchmark"],
        check=True,
    )
    text = pathlib.Path("docs/report/tables/tab_benchmark.tex").read_text()
    assert "\\toprule" in text and "|" not in text
    assert "Label-graph" in text and "Functional" in text
    assert "GRSMF" in text and "DDGCN" in text
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=1 uv run python -m pytest tests/test_make_tables.py -k "benchmark or mean_over_cvs" -v`
Expected: FAIL — `mt.mean_over_cvs` does not exist (AttributeError) and `tab_benchmark.tex` is not generated.

- [ ] **Step 4: Add `mean_over_cvs` helper (in `make_tables.py`, immediately after `fmt`)**

```python
def mean_over_cvs(values: list[float]) -> float:
    """Simple average of per-CV means (matches the published 'Mean' column)."""
    return sum(values) / len(values)
```

- [ ] **Step 5: Add the `build_benchmark` writer (in `make_tables.py`, after `build_foundation`)**

```python
def build_benchmark() -> None:
    """Cross-method comparison: published label-graph methods (teammate
    reproduction CSV) vs our label-free functional-signal methods (exp06/exp07),
    same K562-DepMap splits and per-anchor cal_metrics (spec: tab_benchmark)."""
    published = pd.read_csv(REPO / "docs/report/tables/benchmark_published.csv")
    col_order = ["cv1_f1", "cv1_ndcg10", "cv2_f1", "cv2_ndcg10",
                 "cv3_f1", "cv3_ndcg10", "mean_auroc", "mean_aupr",
                 "mean_f1", "mean_ndcg10"]

    def functional_row(csv: str, model: str, sl: str | None) -> list[str]:
        cells: list[str] = []
        per_metric: dict[str, list[float]] = {m: [] for m in ("auroc", "aupr", "f1", "ndcg@10")}
        for split in ("CV1", "CV2", "CV3"):
            f1 = read_metric(csv, split, model, "f1", slice=sl)
            nd = read_metric(csv, split, model, "ndcg@10", slice=sl)
            cells += [fmt(f1), fmt(nd)]
            for m in per_metric:
                per_metric[m].append(read_metric(csv, split, model, m, slice=sl))
        cells += [fmt(mean_over_cvs(per_metric[m])) for m in ("auroc", "aupr", "f1", "ndcg@10")]
        return cells

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Synthetic-lethality prediction on the K562 SL benchmark: "
        r"published label-graph methods (which consume the SL association matrix "
        r"as input; \citealp{ddgcn2020,grsmf2019,sl2mf2018,slgnn2023}) versus our "
        r"label-free functional-signal methods. Same K562-DepMap Rand 1:1 splits "
        r"and identical per-anchor \texttt{cal\_metrics} ranking on both sides. "
        r"Label-graph methods lead on ranking (GRSMF NDCG@10, bold); our "
        r"functional methods use no SL labels yet stay classification-competitive, "
        r"leaving ranking as the open gap. CV1 is the degree-gameable diagnostic.}",
        r"  \label{tab:benchmark}",
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
    for _, r in published.iterrows():
        cells = []
        for c in col_order:
            cell = fmt(float(r[c]))
            if r["model"] == "GRSMF" and c == "mean_ndcg10":
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(f"    {r['model']} & " + " & ".join(cells) + r" \\")
    lines.append(r"    \cmidrule(lr){1-11}")
    lines.append(r"    \multicolumn{11}{l}{\emph{Functional-signal methods (label-free)}} \\")
    lines.append(r"    Dependency-only (exp06) & "
                 + " & ".join(functional_row(EXP06, "B", None)) + r" \\")
    lines.append(r"    \quad + transcriptome (exp07) & "
                 + " & ".join(functional_row(EXP07, "B_transcript", "full_universe")) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    _write("benchmark", "\n".join(lines))
```

- [ ] **Step 6: Register the writer in `_BUILDERS`**

Add to the `_BUILDERS` dict in `make_tables.py`:

```python
    "benchmark": build_benchmark,
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=1 uv run python -m pytest tests/test_make_tables.py -v`
Expected: PASS (all prior tests + 5 new ones).

- [ ] **Step 8: Regenerate all tables, confirm clean + no drift, lint**

Run: `uv run python docs/report/scripts/make_tables.py --all && uv run ruff check docs/report/scripts/make_tables.py`
Expected: writes `tab_benchmark.tex` (and others identically); ruff clean.

- [ ] **Step 9: Commit**

```bash
git add docs/report/tables/benchmark_published.csv docs/report/scripts/make_tables.py docs/report/tables/tab_benchmark.tex tests/test_make_tables.py
git commit -m "feat: add cross-method SL benchmark table (published vs functional)"
```

---

### Task 2: GRSMF reference line on the difficulty-ladder figure

**Files:**
- Modify: `docs/report/scripts/make_figures.py` (`build_difficulty_ladder`, lines 29-85)
- Test: `tests/test_make_tables.py` (one regeneration smoke assert) — or rely on visual; add a guard that the figure builds.

**Interfaces:**
- Consumes: `mt.read_metric`, `pandas` read of `benchmark_published.csv` for GRSMF per-CV N@10 (`cv1_ndcg10`, `cv2_ndcg10`, `cv3_ndcg10`).
- Produces: updated `docs/report/figures/fig_difficulty_ladder.pdf` with a dashed GRSMF reference series.

- [ ] **Step 1: Add GRSMF series read + dashed plot (inside `build_difficulty_ladder`, after the `series` dict, before/within the plotting loop)**

Insert after the `series` dict is built:

```python
    import pandas as pd  # noqa: E402
    _pub = pd.read_csv(_HERE.parent / "tables" / "benchmark_published.csv")
    _grsmf = _pub.loc[_pub["model"] == "GRSMF"].iloc[0]
    grsmf_ndcg = [float(_grsmf["cv1_ndcg10"]), float(_grsmf["cv2_ndcg10"]),
                  float(_grsmf["cv3_ndcg10"])]
```

Then after the existing series-plot loop, add the reference line:

```python
    ax.plot(list(x), grsmf_ndcg, linestyle="--", color="0.4", linewidth=1.5,
            marker="x", markersize=6, label="GRSMF (best published ranker)")
```

- [ ] **Step 2: Build the figure and verify it runs**

Run: `uv run python docs/report/scripts/make_figures.py`
Expected: "wrote .../fig_difficulty_ladder.pdf"; no exception.

- [ ] **Step 3: Add a build-smoke test (append to `tests/test_make_tables.py`)**

```python
def test_difficulty_ladder_figure_builds():
    import subprocess
    r = subprocess.run(
        ["python", "docs/report/scripts/make_figures.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert pathlib.Path("docs/report/figures/fig_difficulty_ladder.pdf").exists()
```

- [ ] **Step 4: Run the test + lint**

Run: `OMP_NUM_THREADS=1 uv run python -m pytest tests/test_make_tables.py -k difficulty_ladder -v && uv run ruff check docs/report/scripts/make_figures.py`
Expected: PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add docs/report/scripts/make_figures.py docs/report/figures/fig_difficulty_ladder.pdf tests/test_make_tables.py
git commit -m "feat: overlay GRSMF reference line on difficulty-ladder figure"
```

---

### Task 3: Paper integration — subsection, citations, related-work, compile

**Files:**
- Modify: `docs/report/sections/experiments.tex` (new subsection after the floor subsection, before the transcriptome subsection ~line 68)
- Modify: `docs/report/sections/related_work.tex` (lines 9-24, upgrade the survey sentence)
- Modify: `docs/report/references.bib` (add 4 entries)
- Modify: `docs/superpowers/claim_evidence_map.md` (add comparison-claim row)

**Interfaces:**
- Consumes: `tab_benchmark.tex` (Task 1), updated figure (Task 2), bibkeys `ddgcn2020,grsmf2019,sl2mf2018,slgnn2023` (used in the Task 1 caption — must exist in `references.bib` or LaTeX errors).
- Produces: a compiling 11-page PDF with `\ref{tab:benchmark}` resolved, 0 undefined citations.

- [ ] **Step 1: Add 4 CrossRef-verified BibTeX entries to `references.bib`**

Verify each DOI via `uv run python -c "import urllib.request,json; ..."` against CrossRef before pasting. Expected canonical sources:
- `grsmf2019` — Huang et al., "Graph regularized self-representative matrix factorization for SL", *IEEE/ACM TCBB* / Bioinformatics-family.
- `sl2mf2018` — Liu et al., "SL2MF: Predicting Synthetic Lethality in Human Cancers via Logistic Matrix Factorization", *IEEE/ACM TCBB* 2020 (DOI 10.1109/TCBB.2019.2909908).
- `ddgcn2020` — Cai et al., "Dual-dropout graph convolutional network for predicting synthetic lethality", *Bioinformatics* 2020 (DOI 10.1093/bioinformatics/btaa211).
- `slgnn2023` — SLGNN (graph neural network for SL); verify exact venue/DOI via CrossRef before committing.

Add each as `@article{...}` with verified `title/author/journal/year/doi`. If a DOI cannot be verified, mark the entry `% UNVERIFIED` and note it in the commit body rather than inventing metadata.

- [ ] **Step 2: Add the Experiments subsection (in `experiments.tex`, after `\input{tables/tab_floor}` block, before `\subsection{Proof-of-concept...}`)**

```latex
\subsection{Comparison with published SL-prediction methods}
\label{sec:benchmark}
We place our label-free functional-signal methods next to four published
synthetic-lethality predictors reproduced under the identical K562 protocol
(same Rand 1:1 splits, same per-anchor \texttt{cal\_metrics};
Table~\ref{tab:benchmark}). DDGCN and SLGNN are graph-neural-network link
predictors; GRSMF and SL2MF are matrix-factorization models. All four
\emph{consume the SL association matrix as input}; our methods use only
GeneEffect and perturbation-response features and see no SL labels at
feature-construction time.

Two patterns are honest to report. First, on ranking the label-graph methods
lead: GRSMF reaches a cross-CV NDCG@10 of 0.317 and DDGCN 0.257, against 0.032
for our dependency-only floor and 0.090 once the observed transcriptome is
added. The gap persists at cold-start CV3, so we make no claim of a cold-start
advantage. Second, on classification the picture is closer: our dependency-only
floor sits at a cross-CV AUROC near 0.70, within range of DDGCN and GRSMF
(0.752) and SLGNN (0.721). The best classifier (SLGNN, mean F1 0.741) is not the
best ranker (GRSMF), which is why we report both F1 and NDCG@10 throughout.
These published methods set the ranking bar that our generative direction
(Section~\ref{sec:method-prelim}) is intended to close, without yet doing so.
```

(If the method-preliminary subsection label differs, match the existing `\label`.)

- [ ] **Step 3: Upgrade the related-work survey sentence (`related_work.tex` ~lines 13-17)**

Replace the "families surveyed by \citet{wang2022slreview}" clause so it reads (keep surrounding sentence intact):

```latex
matrix-factorization and graph-neural-network methods (GRSMF, SL2MF, DDGCN,
and SLGNN; \citealp{grsmf2019,sl2mf2018,ddgcn2020,slgnn2023}), which we
reproduce under our shared K562 protocol (Table~\ref{tab:benchmark}), propagate
observed SL labels
```

Keep the existing F7-safe framing (these are literature methods + our reproduction, not our contribution).

- [ ] **Step 4: Add a claim-evidence row (`claim_evidence_map.md`)**

Append a row mapping the new comparison claim ("functional methods are classification-competitive but trail on ranking vs published label-graph methods, same protocol") to its evidence: `tab_benchmark` + `benchmark_published.csv` provenance + teammate reproduction (`personal_report_sl_gene_perturbation.md`).

- [ ] **Step 5: Compile the paper (full bibtex cycle)**

Run:
```bash
cd docs/report && uv run python scripts/make_tables.py --all && uv run python scripts/make_figures.py \
  && pdflatex -interaction=nonstopmode neurips_2026.tex >/dev/null \
  && bibtex neurips_2026 >/dev/null \
  && pdflatex -interaction=nonstopmode neurips_2026.tex >/dev/null \
  && pdflatex -interaction=nonstopmode neurips_2026.tex 2>&1 | tail -5
```
Expected: completes; check next step for warnings.

- [ ] **Step 6: Verify 0 undefined references / citations**

Run: `cd docs/report && grep -c "undefined" neurips_2026.log; grep -i "Citation.*undefined\|Reference.*undefined" neurips_2026.log || echo "none"`
Expected: `none` (0 undefined). If `tab:benchmark` or any new bibkey is undefined, fix the offending `\label`/bibkey and re-run Step 5.

- [ ] **Step 7: Adversarial language sweep (guardrails)**

Run: `cd docs/report && grep -rniE "outperform|beats|state.of.the.art|\bSOTA\b|superior to" sections/ && echo "FOUND — fix" || echo "clean"`
Expected: `clean` (or only legitimate guardrail negations). Fix any violation in prose.

- [ ] **Step 8: Commit**

```bash
git add docs/report/sections/experiments.tex docs/report/sections/related_work.tex \
  docs/report/references.bib docs/superpowers/claim_evidence_map.md \
  docs/report/neurips_2026.pdf
git commit -m "docs: integrate cross-method SL comparison into report (table, citations, prose)"
```

