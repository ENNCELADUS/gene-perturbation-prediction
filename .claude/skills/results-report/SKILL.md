---
name: results-report
description: How to report experiment results and research status in this repo, in a chat reply, a results/<run>/summary.html run page or the results/status.html research status page. Use whenever you present numbers from a run, write or revise a run page or the status page, make result figures, or summarise a stage-one or stage-two report, even if the user only asks "how did it go", "what are the results" or "where are we".
---

# Results report

Present results as an ML paper would. Each table has a caption that names the metric, the benchmark and split, and
whether higher is better. It has one row per named model, baseline or control (controls sit in the same table), one
column per benchmark, screen or held-out line plus a mean column, and the best value in bold. Explain the metric once
in one sentence. Give validation and test as separate tables or columns; validation only selects, test is reported
once. Contrasts with bootstrap intervals go in a second table. Never use bullet dumps or arm × scorer pivots as the
main presentation, and name every row by what it is, not by an internal label.

Report in chat by default: one line on what ran (config, host, code SHA, seed), the result tables, then a short
**Reading** section of bold one-line findings, each backed by a table. Never write a new `README.md` (or any other
Markdown record) for a run, an analysis or a results folder; a run's record is its `summary.html`. The `README.md`
records written before 2026-10-29 stay as they are.

## HTML pages

- A **run page**, `results/<run>/summary.html`, records one experiment. Write it whenever a run is recorded in
  `results/` or the user asks for one.
- The **status page**, `results/status.html`, shows where the research stands across both stages. Write it only when
  the user asks.

**Location.** `results/<run>/` holds `summary.html`, `figures/*.svg` and the copied host report it reads
(`results.md`, small JSON or CSV). Both pages are committed and pushed with the change; the repo is public, so put
nothing in them that `results/` would not already hold.

**Build.** Load the `artifact-design` skill for its design rules, then write the page as a local, self-contained file.
Never publish it with the Artifact tool or upload it anywhere. Because nothing wraps it, the file carries its own
`<!doctype html>`, charset and viewport meta, `<title>` and `<style>`; colours are tokens with a dark-mode block; no
external scripts or fonts. Check it once with a headless-Chrome screenshot, fix what it shows, and stop.

**Run page story.** Follow the reader's order, not the code's: the question and the step of the plan it serves; the
benchmark (the final split with its line counts, the genes or pairs scored, the target); the rows (what each model,
baseline and control is); what ran (host port, output path, code SHA, seed, a table of steps with finish times in
UTC, and a link to the copied host report); results (the tables above, then figures); **Reading**.

**Status page story.** The task (blueprint §3, two stages); the benchmark and final split; one section per stage with
its current headline table, its **Reading** and links to the run pages (or older `README.md` records) behind it; then
what runs next. It replaces the previous status page in place; it is not a history.

**Numbers.** Every number comes from the copied host report or the run's JSON files, never retyped from memory or
from CLAUDE.md. The status page takes its tables from the run records it links.

**Figures.** Vector only. Analysis code saves `.svg` (matplotlib with `svg.fonttype: none`); if only PDFs exist,
convert with `pdftocairo -svg`. Insert with `<img src="figures/<name>.svg">` and a caption.

**Scope.** Show only what the run or project is reporting now. Retired arms and superseded runs stay out of the page.
