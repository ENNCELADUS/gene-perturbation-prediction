# SL stage two: PARIS and viability-embedding heads on the screen benchmark

The first stage-two run (SL protocol §10). Two heads, each fitted on the 29 screened single-cell training lines,
rank held-out pairs from the Ito 2021, Klingbeil 2024 and Harle 2025 screens. The validation lines are PA-TU-8988S
and NCI-H1299, and the test lines are A549, MDA-MB-231 and SK-MES-1. The heads are:

- **The PARIS head.** Its inputs are PARIS evidence plus the arm's GeneEffect of both genes in the line.
- **The viability-embedding head** (stage-one runs only). Its input is the run's V for both genes.

There are four arms:

- the default linear prior refitted on the frame;
- FiLM on the prior and ΔE;
- FiLM on ΔE;
- measured GeneEffect, as a reference.

All runs were on the H20 container on port 30838, at seed 0.

| Step | Output on the host | Code | Finished (UTC) |
| --- | --- | --- | --- |
| PARIS, `Ginicorrected` mode, 60 workers | `outputs/sl/paris/paris_seed0/` | `186c242` | 2026-10-23 18:10 |
| stage two | `outputs/sl/stage_two/stage_two_seed2/` | `9e1c251` | 2026-10-23 18:33 |

[stage_two_results.md](stage_two_results.md) is a copy of the host's `results.md`, with every row and contrast.

**PARIS.** PARIS was fitted on 942 reference lines: the frame lines that no screen tested. It covers 4,253 screened
dependency genes against 8,810 expression genes and selected 173,809 pairs. 4,233 genes have at least one
selection. The run took 6,615 s, about 93 s per gene across 60 workers. The Python port has not been compared
against the R package, because R is not installed on either machine.

**Rows.** There are 42,519 training rows. There are 16,152 held-out rows, counting pairs in panel genes that were
also labelled in a training line of the same screen. 1,555 held-out rows lack a decoded gene, and 17 have a pair
that was never labelled in training.

`stage_two_seed1` used an earlier context-ablated PARIS control that dropped the GeneEffect inputs altogether. That
showed only that GeneEffect helps, not that the line does, so it is superseded. The control now gives every line
each gene's frame mean GeneEffect.

## AUPRC lift, mean over held-out contexts

Lift is AUPRC divided by the positive rate. The varying columns keep only pairs whose call differs between the
screen's training lines. There are three validation contexts and five test contexts.

| Arm | Scorer | Val | Test | Val varying | Test varying |
| --- | --- | ---: | ---: | ---: | ---: |
| any | pair identity | 13.25 | 9.04 | 2.12 | 2.13 |
| any | pan-essentiality | 7.61 | 4.17 | 1.61 | 1.45 |
| any | PARIS alone | 3.47 | 2.86 | 1.51 | 1.38 |
| any | PARIS head, gene-mean GeneEffect (context-ablated) | 10.03 | 6.15 | 2.18 | 1.73 |
| linear prior | PARIS head | 10.64 | 5.84 | 2.21 | 1.68 |
| FiLM on the prior and ΔE | PARIS head | 10.80 | 5.80 | 2.18 | 1.69 |
| FiLM on ΔE | PARIS head | 10.79 | 5.74 | 2.17 | 1.68 |
| measured | PARIS head | 13.13 | 6.51 | 2.27 | 1.82 |
| FiLM on the prior and ΔE | V head | 5.55 | 3.59 | 1.88 | 1.34 |
| FiLM on the prior and ΔE | V head, gene-mean V (context-ablated) | 7.99 | 7.32 | 1.86 | 1.98 |
| FiLM on ΔE | V head | 7.85 | 4.88 | 2.20 | 1.48 |
| FiLM on ΔE | V head, gene-mean V (context-ablated) | 8.44 | 7.56 | 1.94 | 1.92 |

## Contrasts on test

Each contrast is an AUPRC difference. The numbers are the mean over the five test contexts and how many contexts
have a 95% two-way dyadic bootstrap interval above or below zero.

| Arm | Contrast | Mean | Above 0 | Below 0 |
| --- | --- | ---: | ---: | ---: |
| linear prior | PARIS head − its context-ablated head | −0.026 | 0 | 0 |
| FiLM on the prior and ΔE | PARIS head − its context-ablated head | −0.029 | 0 | 0 |
| FiLM on ΔE | PARIS head − its context-ablated head | −0.027 | 0 | 0 |
| measured | PARIS head − its context-ablated head | +0.017 | 0 | 0 |
| linear prior | PARIS head − PARIS alone | +0.164 | 3 | 0 |
| linear prior | PARIS head − pair identity | −0.222 | 0 | 2 |
| FiLM on the prior and ΔE | V head − its context-ablated head | −0.265 | 0 | 4 |
| FiLM on ΔE | V head − its context-ablated head | −0.201 | 0 | 2 |

On validation, the measured arm's PARIS head beats its context-ablated head in one of three contexts. No predicted
arm does so in any context.

## Reading

- **Pair identity wins everywhere.** This control is the pair's hit rate in the same screen's training lines. It
  holds even on varying pairs.
- **Predicted GeneEffect adds nothing over the gene mean.** Each predicted arm's PARIS head equals the head that
  reads each gene's frame mean GeneEffect, within its intervals. The prior and the two FiLM rows are
  indistinguishable from one another.
- **Measured GeneEffect is about as flat.** It gains +0.017 on test, and no test interval excludes zero. Even a
  perfect stage one would barely move this head on this benchmark.
- **The V head is worse than its own ablation.** It falls below its context-ablated version, with clear losses in
  two to four of the five test contexts. The line's V hurts; per-gene V is what carries its signal.
- **What does help is the head over PARIS alone.** The gain comes from gene-level essentiality and the screen
  indicator, not from the line.

Nothing here supports a context claim (protocol §8).
