# Research Vault: Cell-Fate Outcome Dynamics

**Status:** blueprint established · literature review complete · decision `narrow-or-pivot` for both candidates.
**Next:** three public-data reanalyses, then a bounded feasibility study. **No production modeling is authorized.**

## The wedge

```text
The same net fitness loss can arise from completely different cellular dynamics.
```

Strong division suppression with little loss, normal division with substantial loss, early
loss followed by survivor regrowth, and transient arrest followed by recovery can all yield
the same aggregate readout. This is **demonstrated for selected drug perturbations** (Gross
2023); **its prevalence and importance remain unresolved for genetic loss-of-function.**
That gap is the program.

## The question

> **In genetic loss-of-function, is net fitness close to a sufficient statistic — or does
> it frequently conceal reproducible and consequential division / death / recovery
> dynamics?**

Two candidate research questions are carried **in parallel**. No unit is selected.

| | Candidate A — lineage / clone | Candidate B — population |
|---|---|---|
| **Question** | Does an early molecular state predict a linked lineage's division / persistence / extinction trajectory, beyond an independently measured net fitness? | Under comparable, independently measured net fitness, does the early single-cell state *distribution* carry incremental information about independently measured future population dynamics? |
| **Evidence ceiling** | **A2** (sibling/clone proxy) for anything pooled | Population-level association; never upgradable to a per-cell fate claim |

## Reading order

| Doc | Role |
|---|---|
| [`01-blueprint.md`](01-blueprint.md) | **The contract.** Wedge, candidates, estimands, hypotheses, claim boundaries, and §13 Locked Decisions. Read first. |
| [`02-acceptance-criteria.md`](02-acceptance-criteria.md) | **The bar.** What *common*, *large*, and *consequential* mean, numerically — fixed before evidence, and the criteria Study 0 and the pilots will be graded against. Not revisable to fit a result. |
| [`03-literature-review.md`](03-literature-review.md) | **What the literature showed.** L0 + Gates 1–4 verdicts, and the surviving novelty statement (§7). |
| [`04-roadmap.md`](04-roadmap.md) | **What happens next.** The decision, the two bounded steps, and the design constraints on any wet-lab work. |
| [`results/prior-internal-evidence.md`](results/prior-internal-evidence.md) | The internal numbers (exp02, exp09) the blueprint depends on. |

## Where the project stands

| Stage | Verdict |
|---|---|
| L0 — Ontology | Complete. No selection between candidates (by design). No citable `t_commit` exists for this context. |
| Gate 1 — Measurement validity | **A: `proceed`** (narrowed to the A2 ceiling). **B: `proceed`.** Neither triggers a stop. |
| Gate 2A — Lineage prevalence | **Insufficient evidence → bounded validation pilot.** Not a stop. |
| Gate 2B — Population prevalence | **Insufficient evidence → bounded validation pilot.** Not a stop. |
| Gate 3 — Prior art / novelty | **Partially-scooped.** Five novelty claims closed; a narrower statement survives. |
| Gate 4 — Prospective designs | Both candidates → **a genuine methodological opportunity.** No design supports Analysis P today. |
| **Decision** | **Both candidates: `narrow-or-pivot`.** No production modeling. No unit selection yet. |

Full verdicts and evidence: [`03-literature-review.md`](03-literature-review.md).

## Roadmap

- [ ] **Reanalysis 1** — exp02's residualization audit on **Jost 2020's titration series** (a within-gene test)
- [ ] **Reanalysis 2** — the same audit on **Dixit 2016's 13-gene cell-cycle panel**
- [ ] **Reanalysis 3** — **Nadal-Ribelles's mean-vs-variance test on Replogle 2022 raw single-cell K562** (first "distribution beyond mean" test on mammalian genetic loss-of-function)
- [ ] **Study 0** — bounded feasibility and calibration; claim ceiling written into the protocol *before* data
- [ ] *(unscheduled)* **Candidate A linkage arm** — an A2 scaffold on a CRISPR-knockout library; a different experiment at a different scale

The reanalyses can address **specificity** only. They categorically cannot address the
phenomenon question or the trajectory question — no residualization manufactures a
trajectory label the underlying assay never captured. See
[`04-roadmap.md`](04-roadmap.md) §3.

## Where things live

- **Evidence record** — the eleven review memos, with full evidence tables and `UNVERIFIED` registers: [`ideaspark_run/cell-fate-outcome-dynamics/`](../ideaspark_run/cell-fate-outcome-dynamics/). Not edited.
- **Dataset cards** — [`docs/data/`](data/). **Meeting notes** — [`docs/discussion/`](discussion/).
- **Retired dependency/SL program** — write-ups under `docs/archive/` (untracked, gitignored). Its code still runs and the reanalyses reuse it.

## Conventions

Contract docs (`01`, `02`) are **frozen**: change them in place, never by writing a new
file. Live numbers live only in `results/`, and no number enters the vault without a source
pointer. Results enter the docs only after the analysis actually runs — a planned number is
not a number.
