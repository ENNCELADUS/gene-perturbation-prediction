# Cell-state telemetry

Task 8 of the [stage-one design](../../docs/specs/2026-10-16-pie-geneeffect-ladder-design.md) (§3.5; plan Wave 1B,
protocol §17, [data card](../../docs/data/cell-states.md)): frozen cell-state vocabularies fitted on the 377 labelled
single-cell training lines, projected on training and validation lines, and the frozen PIE run on each line's state
inputs, from `configs/stage_one/states/telemetry.yaml`. Run on the H20 container on port 30838; the states live under
`data/geneeffect_joint/v2/ladder/states/` there and the report in
`outputs/stage_one/state_telemetry/telemetry/` (copied here as [telemetry_results.md](telemetry_results.md), beside
its `report.json` on the host). Fitting seed 0 is the one projected. Telemetry only: no model is trained on
GeneEffect, nothing is chosen in code, and nothing here is SL evidence.

## What ran

| Step | Finished (UTC) | Notes |
| --- | --- | --- |
| fit, project, inputs | 2026-10-21 17:46 | code at `20acd2d`; started 15:00 |
| frozen PIE on the state inputs | 2026-10-22 07:32 | four shards, one per GPU, from 18:26; 1.3–1.5 s per context; the 200 pilot genes |
| assemble, report | 2026-10-22 08:59 | code at `f5fefbe` (no state-code change since `20acd2d`) |

Sets: 17 vocabularies (cell cycle; recurrent cNMF programs; scVI line-centred and absolute, HVG principal components,
Tx1 embeddings and QC values, each at K = 4, 6, 8), the cycling-weighted input, and every vocabulary but the QC states
with its memberships permuted within line at seeds 0, 1 and 2.

## Composition, confound and stability

Training lines, fitting seed 0. Deposit η² is the mean over states (the null's 95th percentile in brackets); second
deposit is the total-variation distance between a line's two deposits' masses, with the distance between different
lines in brackets (99 lines); seed match is the matched states' mean correlation across fitting seeds.

| Vocabulary | States | Deposit η² (null q95) | Largest-state η² | Second deposit (other lines) | Seed match |
| --- | ---: | --- | ---: | --- | ---: |
| cycle | 3 | 0.364 (0.085) | 0.395 | 0.159 (0.231) | 1.000 |
| programs | 17 | 0.351 (0.094) | 0.745 | 0.490 (0.801) | 0.742 |
| scVI line-centred, K 4 / 6 / 8 | 4 / 6 / 8 | 0.43 / 0.49 / 0.45 | 0.76 / 0.84 / 0.85 | 0.22–0.28 (0.34–0.47) | 0.65 / 0.51 / 0.55 |
| scVI absolute, K 4 / 6 / 8 | 4 / 6 / 8 | 0.61 / 0.66 / 0.57 | 0.85 / 0.94 / 0.96 | 0.19 / 0.98 / 0.97 (0.97–0.99) | 0.73 / 0.86 / 0.73 |
| HVG components, K 4 / 6 / 8 | 4 / 6 / 8 | 0.64 / 0.54 / 0.52 | 0.83 / 0.86 / 0.91 | 0.37–0.45 (0.59–0.74) | 0.84 / 0.71 / 0.80 |
| Tx1 embeddings, K 4 / 6 / 8 | 4 / 6 / 8 | 0.51 / 0.49 / 0.41 | 0.72 / 0.77 / 0.77 | no Tx1 on second deposits | 0.97 / 0.72 / 0.98 |
| QC values, K 4 / 6 / 8 | 4 / 6 / 8 | 0.28 / 0.22 / 0.21 | 0.38 / 0.30 / 0.31 | 0.18–0.26 (0.17–0.23) | 1.00 / 0.93 / 0.81 |

- **Deposit explains much of every vocabulary's masses**, well above the null in all 17; one state per vocabulary
  outside the cycle and QC states has η² of 0.72–0.96. The QC states are the least deposit-bound and the cycle states
  next.
- **The absolute scVI states are line identity.** A line holds one state at mass ≥ the threshold (median), and at
  K 6 and 8 a line's two deposits are as far apart (0.98, 0.97) as two different lines (0.99).
- **A line's second deposit reproduces its masses** better than another line does for the cycle, programs, line-centred
  scVI and HVG states; the QC states' second deposit is no closer than another line.
- **Identities.** The cycle states are G2/M, S and non-cycling as named. The 16 recurrent programs include interferon
  (program 4, Kinker IFN response +0.26), stress, proteasomal degradation, unfolded protein response, MYC/translation,
  epithelial senescence and three G2/M-like programs; the background program carries the most deposit (η² 0.745).
  Among the mixture vocabularies, most states' best reference program scores under +0.05, so they are not labelled
  by a known program; the HVG states at K 6 and 8 hold clear cycle states (G2/M +0.27 and +0.31).

## The frozen PIE's gap

The gap is `Σ_k π_ck Δ(x_ck, g) − Δ(x̄_c, g)`: what reading each state through PIE and mixing adds over reading the
line's mean input, over the 200 pilot genes, relative to the whole-line delta's spread across lines within a gene. The
noise floor (the set's whole-line context against the main pass) is 0.0000 in every set.

| Set | ΔE train (val) | ΔH train | own feature train (val) | permuted ΔE train, seeds 0–2 |
| --- | --- | ---: | --- | --- |
| cycle | 0.0094 (0.0073) | 0.0003 | 0.0124 (0.0067) | 0.0076–0.0077 |
| programs | 0.0073 (0.0127) | 0.0002 | 0.0378 (0.0462) | 0.0063 |
| HVG components, K 8 | 0.0135 (0.0230) | 0.0004 | 0.0135 (0.0179) | 0.0119–0.0120 |
| Tx1 embeddings, K 6 | 0.0188 (0.0107) | 0.0006 | 0.0380 (0.0075) | 0.0137–0.0139 |
| scVI line-centred, K 8 | 0.0100 (0.0168) | 0.0004 | 0.0112 (0.0119) | 0.0085 |
| scVI absolute, K 8 | 0.0008 (0.0015) | 0.0000 | 0.0011 (0.0019) | 0.0004 |
| QC values, K 8 | 0.0192 (0.0176) | 0.0011 | 0.0164 (0.0109) | no permuted set |
| cycling-weighted input | 0.0037 (0.0036) | 0.0002 | 0.0098 (0.0057) | no permuted set |

- **The gap is small.** For ΔE it is at most 0.019 of the whole-line delta's across-line spread on training lines
  (0.023 on validation), for ΔH at most 0.001; the largest gaps are the own feature's for the programs and Tx1 states
  (0.038 train). On these genes the
  frozen PIE responds to a mixture of states almost as it responds to the mixed input.
- **Real memberships move it only a little more than permuted ones.** Every vocabulary's real-membership ΔE gap is
  above its three permuted seeds, by 0.0004–0.005, and the seeds agree with each other within 0.0002.
- **The QC states give as large a gap as any biological vocabulary** (ΔE 0.019 at K 8), so a gap's size does not by
  itself mark biology.
- The Tx1 vocabularies' own-feature gap falls from 0.031–0.038 on training lines to 0.007–0.011 on validation.

## Not exercised

The re-projection check (validation lines from deposits no training line comes from, projected again) found 0 such
lines, so it compared nothing. The report's composition table holds the summary above; recurrence, the centred
log-ratio spectrum, the per-state QC shift and the per-state second-deposit Pearson are in `report.json` on the host
and not copied here.

## Reading

Every vocabulary is built, projected and read through the frozen PIE, and the inputs Task 9 needs exist for all of
them. Deposit is the largest visible structure in most vocabularies, and the absolute scVI states encode the line
rather than a state within it. The frozen PIE's state gap is one to two percent of the across-line spread of its
whole-line ΔE, close to the permuted control's and matched by the QC states: on frozen PIE, mixing states adds little
beyond the line's mean input. Under protocol §17 a gap no larger than the permuted control's is not by itself a
reason to stop; people choose which vocabularies Task 9 precomputes.
