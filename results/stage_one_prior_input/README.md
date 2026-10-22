# Stage one: FiLM reading the prior's prediction

The step decided on 2026-10-22 before the rest of Wave 2 ([plan](../../docs/specs/2026-10-17-pie-stage-one-plan.md),
protocol §16). FiLM reads the frame-refitted default prior's prediction as an input, either alone or beside ΔE or
beside line-permuted ΔE. Each row is trained on the 981 frame lines, as a main run plus five folds, with one config each
under `configs/stage_one/prior_input/`. The report is `configs/stage_one/prior_input/report_prior_input.yaml`; it places
the rows beside the prior itself and Task 3's FiLM on ΔE.

Run on the H20 container on port 30838, with outputs in `outputs/stage_one/prior_input/` there. The code is at
`0c47b7f`. The full report is [report_prior_input_results.md](report_prior_input_results.md), a copy of the host's
`report_prior_input/results.md`. Seed 0. Nothing here is SL evidence.

## What ran

| Step | Finished (UTC) | Notes |
| --- | --- | --- |
| queue | 2026-10-22 15:26 | 18 runs, two per GPU, none failed |
| report | 2026-10-22 16:28 | |

Every prior-input row stopped early and chose epoch 1 of 6. FiLM on ΔE alone chose epoch 16 of 21.

## Validation and test

These are the per-line residual Spearman (the selector) and within-gene R². Every metric is in the full report.

| Row | Val per-line | Test per-line | Val within-gene R² | Test within-gene R² |
| --- | ---: | ---: | ---: | ---: |
| default prior refitted on the frame | 0.2146 | 0.2283 | 0.0462 | 0.0474 |
| FiLM on the prior and ΔE | 0.2033 | 0.1954 | 0.0452 | 0.0335 |
| FiLM on the prior | 0.1988 | 0.2144 | 0.0373 | 0.0375 |
| FiLM on the prior and line-permuted ΔE | 0.1972 | 0.2122 | 0.0368 | 0.0366 |
| FiLM on ΔE (Task 3) | 0.1457 | 0.1531 | −0.0092 | −0.0165 |

## Contrasts

Each contrast is left minus right, on the per-line residual Spearman, with a 95% paired line-bootstrap interval.

| Contrast | Validation | Test |
| --- | --- | --- |
| prior and ΔE − prior alone (FiLM) | +0.0046 [−0.0116, +0.0202] | −0.0189 [−0.0386, −0.0016] |
| prior and ΔE − prior and permuted ΔE | +0.0062 [−0.0100, +0.0219] | −0.0167 [−0.0365, +0.0002] |
| prior and ΔE − the prior itself | −0.0113 [−0.0275, +0.0051] | −0.0329 [−0.0537, −0.0143] |
| FiLM on the prior − the prior itself | −0.0159 [−0.0219, −0.0098] | −0.0140 [−0.0184, −0.0095] |
| prior and ΔE − ΔE alone | +0.0577 [+0.0353, +0.0825] | +0.0423 [+0.0100, +0.0745] |

Reading the prior lifts FiLM from 0.146 to 0.199–0.203 on validation. No row reaches the prior itself, and ΔE adds
nothing beside the prior that line-permuted ΔE does not.
