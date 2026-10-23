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

## Results

**Table 1.** AUPRC (average precision; higher is better) for ranking SL pairs in each held-out test line, one
column per screen benchmark and line. The header gives the number of pairs and SL hits. The mean is over the five
columns, and AUROC is the mean over the same columns. The best value in each column is in bold. A random
ranking's AUPRC equals the hit rate: 0.07, 0.12, 0.04, 0.05 and 0.06.

| Model | Harle 2025 A549 (340/23) | Harle 2025 SK-MES-1 (326/40) | Ito 2021 A549 (4,272/156) | Klingbeil 2024 A549 (2,224/108) | Klingbeil 2024 MDA-MB-231 (2,172/134) | Mean AUPRC | Mean AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *Controls* | | | | | | | |
| Pair identity (the pair's hit rate in the screen's training lines) | **0.735** | **0.844** | **0.320** | **0.530** | **0.480** | **0.582** | 0.886 |
| Pan-essentiality (the two genes' summed mean GeneEffect) | 0.109 | 0.226 | 0.193 | 0.335 | 0.323 | 0.237 | 0.762 |
| PARIS alone | 0.256 | 0.365 | 0.090 | 0.167 | 0.102 | 0.196 | 0.644 |
| *PARIS head, by GeneEffect input* | | | | | | | |
| Each gene's mean GeneEffect (context-ablated) | 0.367 | 0.555 | 0.248 | 0.398 | 0.358 | 0.385 | 0.842 |
| Linear prior | 0.319 | 0.490 | 0.251 | 0.378 | 0.361 | 0.360 | 0.838 |
| FiLM on the prior and ΔE | 0.313 | 0.475 | 0.242 | 0.386 | 0.368 | 0.357 | 0.837 |
| FiLM on ΔE | 0.333 | 0.514 | 0.240 | 0.364 | 0.339 | 0.358 | 0.827 |
| Measured GeneEffect (reference) | 0.404 | 0.551 | 0.296 | 0.370 | 0.392 | 0.403 | 0.838 |
| *Viability-embedding head, by stage-one run* | | | | | | | |
| FiLM on the prior and ΔE | 0.210 | 0.355 | 0.106 | 0.206 | 0.298 | 0.235 | 0.785 |
| FiLM on the prior and ΔE, each gene's mean V (context-ablated) | 0.713 | 0.829 | 0.187 | 0.388 | 0.381 | 0.500 | **0.899** |
| FiLM on ΔE | 0.261 | 0.415 | 0.182 | 0.304 | 0.363 | 0.305 | 0.818 |
| FiLM on ΔE, each gene's mean V (context-ablated) | 0.700 | 0.763 | 0.199 | 0.389 | 0.480 | 0.506 | 0.887 |

**Table 2.** Validation AUPRC, laid out as Table 1. The validation results were used only to choose each head's
epoch. Random AUPRC is 0.13, 0.015 and 0.03.

| Model | Harle 2025 NCI-H1299 (312/39) | Ito 2021 PA-TU-8988S (4,281/64) | Klingbeil 2024 NCI-H1299 (2,225/76) | Mean AUPRC | Mean AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pair identity | **0.880** | 0.308 | **0.412** | **0.534** | 0.883 |
| Pan-essentiality | 0.227 | 0.236 | 0.179 | 0.214 | 0.795 |
| PARIS alone | 0.383 | 0.071 | 0.090 | 0.181 | 0.633 |
| PARIS head, each gene's mean GeneEffect | 0.538 | 0.288 | 0.222 | 0.349 | 0.850 |
| PARIS head, linear prior | 0.509 | 0.320 | 0.220 | 0.350 | 0.849 |
| PARIS head, FiLM on the prior and ΔE | 0.504 | 0.329 | 0.217 | 0.350 | 0.844 |
| PARIS head, FiLM on ΔE | 0.504 | 0.329 | 0.217 | 0.350 | 0.843 |
| PARIS head, measured GeneEffect | 0.484 | **0.416** | 0.262 | 0.387 | 0.863 |
| V head, FiLM on the prior and ΔE | 0.386 | 0.137 | 0.151 | 0.225 | 0.832 |
| V head, FiLM on the prior and ΔE, each gene's mean V | 0.836 | 0.150 | 0.247 | 0.411 | 0.903 |
| V head, FiLM on ΔE | 0.437 | 0.212 | 0.202 | 0.284 | 0.865 |
| V head, FiLM on ΔE, each gene's mean V | 0.857 | 0.149 | 0.290 | 0.432 | **0.908** |

The full report has the same metrics on the varying pairs, those whose SL call differs between the screen's
training lines. Every scorer's lift over random falls to 1.3–2.3 there.

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
