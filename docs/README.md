# Research Vault: Cell-Fate Outcome Dynamics

**Status:** literature-review funnel complete (2026-07-13). Decision: both candidates `narrow-or-pivot`. Next: three public-data reanalyses, then Study 0.

## The wedge

The same net fitness loss can arise from completely different cellular dynamics: strong division suppression with little loss, normal division with substantial loss, early loss followed by survivor regrowth, or transient arrest followed by recovery. This is demonstrated for selected **drug** perturbations (Gross 2023); **prevalence and importance remain unresolved for genetic loss-of-function.** That gap is what this program is about.

## Reading order

| Doc | Role |
|---|---|
| [`01-research-direction.md`](01-research-direction.md) | The contract — locked; read first. |
| [`02-significance-criteria.md`](02-significance-criteria.md) | Frozen thresholds; not revisable to fit a result. |
| [`03-review-findings.md`](03-review-findings.md) | L0 + Gates 1-4 verdicts. |
| [`04-decision-and-roadmap.md`](04-decision-and-roadmap.md) | The decision and what happens next. |
| [`results/prior-internal-evidence.md`](results/prior-internal-evidence.md) | The only internal numbers the direction depends on. |

## Status board

| Stage | Verdict | Source |
|---|---|---|
| L0 | Complete. No selection between candidates (by design). No citable `t_commit` exists for this context. | [`03` §1](03-review-findings.md#1-l0--ontology-and-latent-to-observable-maps) |
| Gate 1 | **A: `proceed`** (narrowed to the A2 ceiling). **B: `proceed`.** Neither triggers a stop. | [`03` §2](03-review-findings.md#2-gate-1--measurement-and-observation-validity) |
| Gate 2A | **Insufficient evidence -> bounded validation pilot (not a stop).** | [`03` §3](03-review-findings.md#3-gate-2a--phenomenon-prevalence-lineage-level) |
| Gate 2B | **Insufficient evidence -> bounded validation pilot (not a stop).** | [`03` §4](03-review-findings.md#4-gate-2b--phenomenon-prevalence-population-level) |
| Gate 3 | **Partially-scooped.** Five novelty claims closed; a narrower statement survives. | [`03` §5](03-review-findings.md#5-gate-3--nearest-prior-art-and-exact-novelty) |
| Gate 4 | Both candidates -> **genuine methodological opportunity.** No design supports Analysis P today. | [`03` §6](03-review-findings.md#6-gate-4--prospective-incremental-information) |
| Decision | **Both candidates: `narrow-or-pivot`.** No production modeling. No unit selection yet. | [`04`](04-decision-and-roadmap.md) |

## Roadmap

- [x] L0 -> Gate 4 -> decision (2026-07-13)
- [x] Contract revision 4 — the eight `DECISION_MEMO` §9 amendments applied
- [ ] Reanalysis 1 — exp02's residualization audit on **Jost 2020's titration series** (within-gene; cleaner than exp02's cross-gene test)
- [ ] Reanalysis 2 — the same audit on **Dixit 2016's 13-gene cell-cycle panel**
- [ ] Reanalysis 3 — **Nadal-Ribelles's mean-vs-variance test on Replogle 2022 raw single-cell K562** (first "distribution beyond mean" test on mammalian genetic LOF)
- [ ] **Study 0** — bounded feasibility & calibration; claim ceiling written into the protocol *before* data
- [ ] *(unscheduled)* Candidate A molecular linkage arm — a different experiment at a different scale

## Where things live

- The eleven raw review memos (full evidence tables, UNVERIFIED registers) at [`ideaspark_run/cell-fate-outcome-dynamics/`](../ideaspark_run/cell-fate-outcome-dynamics/).
- Dataset cards at [`docs/data/`](data/).
- Meeting notes at [`docs/discussion/`](discussion/).
- The retired dependency/SL program's write-ups under `docs/archive/` (untracked and gitignored).

## Conventions

Numbered contract docs (`01`–`04`) are revised **in place** with an appended `**Updated YYYY-MM-DD:**` line, never superseded by a new file. Live numbers live only in `results/`; no number enters the vault without a source pointer.
