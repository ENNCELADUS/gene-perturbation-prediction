# Bridge remedies for the linear context prior, seed 0

Run `bridge_remedies_20261004`, 2026-10-04, on the H20 container on port 30838 (worktree
`/2023533015/VCC_Project_bridge_remedies`, branch `feat/bridge-remedies`), CPU only, config
`configs/context_prior/bridge_remedies.yaml`. Plan: [bridge remedies](../../docs/specs/2026-10-04-bridge-remedies-plan.md);
protocol §10 of the [GeneEffect protocol](../../docs/03-geneeffect-protocol.md). The runner scores every remedy setting,
block set and penalty on validation and test and decides nothing. Its only automatic choice is the components penalty
under gene-level blocks, taken as the best validation point estimate for components alone. Every choice below is read
on validation. Full table: [results.md](results.md). Nothing here is synthetic-lethality evidence.

| Process (`--experiments`) | Settings | Code |
| --- | --- | --- |
| `affine,contrastive` | affine bridge (1), contrastive PCA (8) | affine bridge at `4b895f5`; contrastive at `b98a3c9` |
| `gating,noise_matched,denoise` | reliability gating (3), noise-matched fitting (1), low-rank denoising (3) | gating at 0.3 and 0.5 at `4b895f5`; the rest at `b98a3c9` |

Both processes shared one run directory and started at `4b895f5`. A Codex adversarial review then found two
problems. The contrastive alignment and the denoising components were fitted with the held fold's bulk, so those
remedies' bridge diagnostics were partly in-sample. The two processes could also race on `run_config.json` and
`results.md`. `b98a3c9` refits both remedies per fold for the out-of-fold rows and locks the run directory, and both
processes were restarted on it. The rows kept from `4b895f5` (the affine bridge, and gating at 0.3 and 0.5) compute
exactly what `b98a3c9` computes. The leak touched diagnostics only, never a validation or test score.

Training side as the [first runs](../context_prior_seed0/README.md) chose it: 953 labelled lines with bulk RNA (of
1,405 training-side lines; Lymphoid and Myeloid extras dropped); 167 of the 170 labelled single-cell training lines
have bulk RNA and fit the bridge; shared space 9,711 genes (60 training lines took the training mean for some); 3,057
selective genes, 2,396 of them in the space. The prior uses 128 expression components, five patient-grouped folds,
1,000 bootstrap resamples and 50 data-selected genes. Penalties (× n): components 0.1, 1, 10, 100, 1000; gene-level
0.1, 1, 10. For own expression and partners the gene-level penalty shrinks each gene's deviation from pooled weights
that are fitted unshrunk; for data-selected genes it is a ridge toward zero. Block sets: components alone, plus own
expression and partners (`own_and_partners`), plus data-selected genes (`selected`), plus all three (`all`).
Contrastive settings are written as pseudo-bulk-excess / bulk-excess directions removed (8/4: eight and four). A gain
is selective Spearman minus the reference row's (the affine bridge, `affine`, components alone at penalty 10), with
a 95% paired line-bootstrap interval.

## Result

**No remedy beats the reference on validation, and no gene-level block set gains over the reference on validation
under any remedy.** The reference reproduces the first runs' chosen prior to four decimals: 0.2232 on validation and
0.2227 on test. That second clause comes from the runner's components-penalty pick (next sections); the
[rerun over the full penalty grid](#rerun-over-the-full-penalty-grid) reverses it: under the affine bridge, all
three gene-level blocks gain +0.0068 on validation and +0.0125 on test, both intervals above zero.

| Remedy | Best row on validation | Val | Test | Val gain | Test gain | Oracle val |
| --- | --- | ---: | ---: | --- | --- | ---: |
| Affine bridge (reference) | components alone, penalty 10 | **0.2232** | 0.2227 | — | — | 0.2358 |
| Reliability gating (`gating`) | threshold 0.7; + own expression and partners, gene penalty 10 | 0.2200 | 0.2151 | −0.0032 [−0.0070, 0.0016] | −0.0075 [−0.0111, −0.0034] | 0.2370 |
| Contrastive PCA (`contrastive`) | 4 pseudo-bulk directions, 0 bulk; + all three, gene penalty 10 | 0.2156 | 0.2281 | −0.0075 [−0.0218, 0.0084] | +0.0055 [−0.0121, 0.0217] | 0.2361 |
| Low-rank denoising (`denoise`) | rank 64; components alone, penalty 1000 | 0.2099 | 0.2270 | −0.0133 [−0.0267, 0.0013] | +0.0043 [−0.0107, 0.0194] | 0.2355 |
| Noise-matched fitting (`noise_matched`) | + own expression and partners, gene penalty 10 | 0.2021 | 0.2008 | −0.0210 [−0.0291, −0.0122] | −0.0219 [−0.0278, −0.0145] | 0.2023 |

Gating and noise-matched fitting change only the gene-level blocks, so their components-alone row is the reference's.
The oracle reads validation lines' bulk RNA (off-contract; never a model row).

## Bridge quality

Per-gene Pearson correlation across the 167 paired lines between out-of-fold bridged pseudo-bulk and bulk; every
setting's row is in [results.md](results.md).

| Bridge | Median | Q25 | Q75 | All genes ≥ 0.5 (of 9,711) | Selective genes ≥ 0.3 / 0.5 / 0.7 (of 2,396) | Their paralogs ≥ 0.5 (of 2,022) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Affine (also gating, noise-matched) | 0.427 | 0.269 | 0.580 | 3,612 | 1,436 / 604 / 99 | 879 |
| Contrastive, 2–16 / 0 (range over four settings) | 0.583–0.599 | 0.477–0.504 | 0.684–0.690 | 6,841–7,366 | 2,250–2,324 / 1,519–1,681 / 289–337 | 1,517–1,604 |
| Contrastive, 2–16 / 4 (range over four settings) | 0.577–0.586 | 0.480–0.491 | 0.667–0.672 | 6,874–7,093 | 2,266–2,302 / 1,559–1,636 / 244–284 | 1,511–1,561 |
| Denoising, rank 16 | 0.337 | 0.226 | 0.451 | 1,632 | 1,182 / 248 / 29 | 417 |
| Denoising, rank 32 | 0.406 | 0.301 | 0.510 | 2,661 | 1,608 / 465 / 40 | 667 |
| Denoising, rank 64 | 0.462 | 0.361 | 0.559 | 3,822 | 1,906 / 730 / 53 | 917 |

The contrastive alignment lifts the median from 0.43 to 0.58–0.60. It raises the number of selective genes at or
above 0.5 from 604 to between 1,519 and 1,681. It is measured in each fold's aligned space, where the removed
directions are gone from the bulk target too, so part of that lift comes from a changed yardstick. Denoising lowers
quality at ranks 16 and 32. At rank 64 it lifts the median to 0.462 but cuts the genes at or above 0.7 from 1,058 to
452. Gating keeps the affine bridge, and its gene-level blocks read 6,877, 3,612 or 1,058 genes at thresholds 0.3,
0.5 and 0.7. **Bridge quality does not predict the score.** The contrastive settings have the best diagnostics, yet
they lower the components' validation score in all eight settings.

## The components penalty decides what the gene-level blocks do

The gene-level stages are fitted to the residual the components stage leaves, and the two predictions are summed.
Selective Spearman is scale-free, so the components' validation score moves by at most 0.0003 between penalties 10
and 1000 in any setting. Over the same range their size falls about tenfold per step: the reference's validation SD
ratio is 0.0161, 0.0018 and 0.0002. The own-expression-and-partners block's pooled weights are fitted unshrunk, so
above penalty 10 the block is fitted to nearly the whole residual and outweighs the components in the sum. Tuning on
components alone then breaks near-ties. Six settings landed on penalty 100 or 1000 by margins of 0.00001 to 0.0003
over penalty 10: contrastive 2/0, 8/0, 16/0 and 4/4, and denoising at ranks 32 and 64. The reference itself chose 10
over 1000 by 0.00003. Under those six settings, own expression and partners collapse to 0.065–0.119 on validation and
0.093–0.143 on bulk input. Data-selected genes and all three lose 0.010–0.026 against the setting's own components.
Those rows measure the tie, not the remedy.

The oracle rows separate the bridge from the blocks. Denoising fits the prior on the reference's bulk rows and reads
the reference's oracle input. Its oracle rows are therefore the reference prior on bulk input, at components penalty
1 (rank 16) and 1000 (ranks 32 and 64). Each cell gives gene penalty 0.1 / 1 / 10.

| Input, components penalty | Components alone | + own expression and partners | + data-selected genes | + all three |
| --- | ---: | --- | --- | --- |
| Bulk (oracle), 10 | 0.2358 | 0.1984 / 0.2183 / 0.2362 | 0.2212 / 0.2309 / 0.2299 | 0.2240 / 0.2360 / 0.2343 |
| Bulk (oracle), 1 | 0.2352 | 0.2424 / 0.2423 / 0.2381 | 0.2362 / 0.2457 / 0.2464 | 0.2359 / 0.2465 / 0.2477 |
| Bulk (oracle), 1000 | 0.2355 | 0.1167 / 0.1174 / 0.0980 | 0.2138 / 0.2211 / 0.2115 | 0.2188 / 0.2275 / 0.2172 |
| Affine-bridged validation, 10 | 0.2232 | 0.1800 / 0.2002 / 0.2167 | 0.1661 / 0.2006 / 0.2081 | 0.1671 / 0.2036 / 0.2123 |

- **At the reference's penalty, the gene-level blocks add nothing even on bulk input.** The best is +0.0003, and the
  bridge turns that into a loss of 0.0065 or more. They do not fail only because of the bridge.
- **At components penalty 1 they add on bulk input.** Over components alone at 1, they gain up to +0.0073 (own
  expression and partners), +0.0112 (data-selected genes) and +0.0125 (all three, gene penalty 10). That is +0.0118
  over the best components-alone bulk score. This is the first sign in these runs that gene-level blocks carry signal
  beyond the components. Oracle differences carry no interval.
- **The affine bridge with gene-level blocks at components penalty 1 was not run**, because the runner tunes that
  penalty on components alone. The only bridged rows at penalty 1 come from contrastive 8/4 and 16/4 and from
  denoising at rank 16, whose rank-16 queries strip gene-level detail.

## Remedies

### Contrastive PCA

This remedy projects out of both sources the directions along which only one source varies. The setting gives the
number of pseudo-bulk-excess and bulk-excess directions removed. The last column is the best gene-level row minus the
setting's own components row, on validation, test and bulk input, with no interval.

| Directions (pseudo-bulk / bulk) | Components penalty | Components alone, val / test | Best gene-level row on val | Val / test | Val gain | Test gain | Added over own components, val / test / bulk |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| 2 / 0 | 100 | 0.2154 / 0.2240 | all three, 10 | 0.2028 / 0.2134 | −0.0204 [−0.0386, −0.0010] | −0.0093 [−0.0274, 0.0090] | −0.0126 / −0.0107 / −0.0158 |
| 4 / 0 | 10 | 0.2118 / 0.2224 | all three, 10 | 0.2156 / 0.2281 | −0.0075 [−0.0218, 0.0084] | +0.0055 [−0.0121, 0.0217] | +0.0038 / +0.0057 / +0.0006 |
| 8 / 0 | 1000 | 0.2150 / 0.2199 | all three, 10 | 0.2029 / 0.2131 | −0.0203 [−0.0387, −0.0005] | −0.0096 [−0.0274, 0.0090] | −0.0122 / −0.0069 / −0.0184 |
| 16 / 0 | 100 | 0.2151 / 0.2198 | all three, 10 | 0.2048 / 0.2133 | −0.0184 [−0.0363, 0.0011] | −0.0094 [−0.0280, 0.0095] | −0.0103 / −0.0064 / −0.0158 |
| 2 / 4 | 10 | 0.2153 / 0.2186 | own and partners, 10 | 0.2114 / 0.2173 | −0.0117 [−0.0255, 0.0025] | −0.0054 [−0.0190, 0.0076] | −0.0039 / −0.0013 / +0.0009 |
| 4 / 4 | 100 | 0.2056 / 0.2223 | all three, 10 | 0.1835 / 0.2190 | −0.0397 [−0.0572, −0.0202] | −0.0037 [−0.0193, 0.0099] | −0.0222 / −0.0033 / −0.0154 |
| 8 / 4 | 1 | 0.2069 / 0.2228 | all three, 10 | 0.2145 / 0.2386 | −0.0087 [−0.0244, 0.0068] | +0.0159 [0.0025, 0.0279] | +0.0076 / +0.0158 / +0.0099 |
| 16 / 4 | 1 | 0.1959 / 0.2182 | all three, 10 | 0.2025 / 0.2333 | −0.0207 [−0.0390, −0.0023] | +0.0106 [−0.0037, 0.0237] | +0.0066 / +0.0151 / +0.0092 |

- **Components.** At the tuned penalty they lose on validation in all eight settings (−0.0078 to −0.0273), while
  test moves between −0.0044 and +0.0014. Removing bulk-excess directions also costs bulk input: at the tuned
  penalty the components oracle falls to 0.2152–0.2296, against 0.2358. Those directions carry signal the prior uses.
- **Gene-level blocks.** Under the affine bridge, all three lose 0.0109 against components alone on validation but
  only 0.0015 on bulk, a gap of 0.0094 that the bridge causes. Where the contrastive tuning landed on penalty 10 or
  1, the gap between what the blocks add on validation and on bulk shrinks to between −0.0048 and +0.0032. So the
  alignment removes most of what the bridge costs these blocks. Where the tuning landed on 100 or 1000, the blocks
  lose. The components' own loss keeps every row below the reference on validation.

### Reliability gating

Gene-level blocks read only genes whose out-of-fold affine bridge quality reaches the threshold.

| Threshold | Gene space | + own expression and partners (gene penalty 10), val / test | Val gain | Test gain | Best + data-selected, val / test | Best + all three, val / test |
| --- | ---: | --- | --- | --- | --- | --- |
| none (affine) | 9,711 | 0.2167 / 0.2189 | −0.0065 [−0.0115, −0.0012] | −0.0038 [−0.0072, 0.0001] | 0.2081 / 0.2198 | 0.2123 / 0.2219 |
| 0.3 | 6,877 | 0.2152 / 0.2171 | −0.0079 [−0.0130, −0.0022] | −0.0056 [−0.0093, −0.0011] | 0.2078 / 0.2168 | 0.2109 / 0.2202 |
| 0.5 | 3,612 | 0.2182 / 0.2175 | −0.0050 [−0.0088, −0.0009] | −0.0052 [−0.0080, −0.0020] | 0.2038 / 0.2129 | 0.2058 / 0.2157 |
| 0.7 | 1,058 | 0.2200 / 0.2151 | −0.0032 [−0.0070, 0.0016] | −0.0075 [−0.0111, −0.0034] | 0.1959 / 0.2012 | 0.1968 / 0.2013 |

- **Own expression and partners.** On validation, gating moves this block toward zero as the threshold rises; test
  does not follow. On bulk input it scores 0.2343, 0.2364 and 0.2370 against 0.2358 for components alone, so the
  gated blocks add nothing on bulk either.
- **Data-selected genes.** The threshold narrows what this block can choose from, which costs it more at every step.

### Noise-matched fitting

This remedy fits the gene-level blocks on the 170 single-cell training lines' out-of-fold bridged pseudo-bulk instead
of the 953 bulk lines. The affine bridge already regresses bulk on pseudo-bulk per gene, so a bridged value is a
shrunk estimate of bulk and refitting own expression on it is close to a change of units; what changes is the
partners, the data-selected genes, the pooled weight and the per-gene intercepts, now fitted on 170 lines. The smaller
line count dominates. The best row (own expression and partners, gene penalty 10) has gains of −0.0210 [−0.0291, −0.0122]
on validation and −0.0219 [−0.0278, −0.0145] on test; data-selected genes and all three lose 0.049–0.087 on validation,
and the data-selected block overfits (at gene penalty 0.1, validation SD ratio 0.64 and Huber 0.0197 against 0.0162
for the reference). Its oracle is not an upper bound: blocks fitted on bridged rows read bulk out of their range.

### Low-rank denoising

This remedy projects the bridged validation and test rows onto the leading 16, 32 or 64 components of the
standardised training bulk.

- **Components.** At ranks 16, 32 and 64, the components lose 0.0579, 0.0279 and 0.0133 [−0.0267, 0.0013] on
  validation. On test they lose 0.0257 and 0.0051 at ranks 16 and 32, and gain +0.0043 [−0.0107, 0.0194] at rank 64.
  The score climbs back toward the reference as the rank grows, because the projection removes query signal along
  with noise.
- **Gene-level blocks.** They add nothing at rank 16 (−0.0012 to −0.0005 on validation), because their features
  read a rank-16 reconstruction. At ranks 32 and 64 they sit on components penalty 1000 (see above).

## Validation and test disagreements

None of the following is chosen on test.

- **Contrastive PCA.** Every setting loses more on validation than on test. The sign is the same across all eight
  settings, which points to a property of which lines sit in validation rather than to noise in one row; this is an
  inference.
- **Gains on test only.** Contrastive 8/4 at components penalty 1 and gene penalty 10 has the only intervals above
  zero in the run, all of them on test: +0.0139 [0.0003, 0.0261] with data-selected genes and +0.0159
  [0.0025, 0.0279] with all three. On validation the same rows score −0.0108 and −0.0087, with intervals spanning
  zero. Denoising at rank 64 (test +0.0043, validation −0.0133) and the best contrastive row (test +0.0055,
  validation −0.0075) also gain on test only.
- **The reverse.** Gating at 0.7 has a validation interval that spans zero, while its test interval excludes zero
  below.

## Rerun over the full penalty grid

Run `bridge_remedies_penalty_grid_20261005`, 2026-10-05, same host and training side, code `d314b74`, config
`configs/context_prior/bridge_remedies_penalty_grid.yaml`: every gene-level block set fitted at every components
penalty (0.1–1000) and gene penalty (0.1–1000, extended from 10), for the affine bridge, contrastive 4/0 and 8/4,
and gating at 0.7. Full table: [results_penalty_grid.md](results_penalty_grid.md). The best row per block set on
validation, with its test score; gains are over the reference row (components alone at penalty 10).

| Bridge | Block set | Penalties (components, gene) | Val | Test | Val gain | Test gain | Oracle val |
| --- | --- | --- | ---: | ---: | --- | --- | ---: |
| Affine | components alone | 10, — | 0.2232 | 0.2227 | — | — | 0.2358 |
| Affine | + own expression and partners | 1, 1 | 0.2258 | 0.2254 | +0.0027 [−0.0007, 0.0062] | +0.0027 [−0.0007, 0.0059] | 0.2423 |
| Affine | + data-selected genes | 1, 10 | 0.2290 | 0.2338 | +0.0058 [0.0001, 0.0116] | +0.0111 [0.0038, 0.0174] | 0.2464 |
| Affine | **+ all three** | **1, 10** | **0.2300** | **0.2352** | **+0.0068 [0.0011, 0.0129]** | **+0.0125 [0.0049, 0.0187]** | 0.2477 |
| Contrastive 4/0 | + all three | 1, 10 | 0.2248 | 0.2372 | +0.0016 [−0.0068, 0.0098] | +0.0145 [0.0012, 0.0263] | 0.2483 |
| Contrastive 8/4 | + all three | 1, 10 | 0.2145 | 0.2386 | −0.0087 [−0.0244, 0.0068] | +0.0159 [0.0025, 0.0279] | 0.2317 |
| Gating at 0.7 | + all three | 1, 10 | 0.2264 | 0.2285 | +0.0032 [−0.0048, 0.0118] | +0.0058 [−0.0043, 0.0156] | 0.2430 |

- **Gene-level blocks survive the affine bridge** once the components penalty is searched with the gene penalty.
  At components penalty 1 the components stage removes more of the residual and the gene-level stages fit what
  is left; all three blocks then gain on validation and on test with intervals above zero. The bulk-input gain
  (+0.0119 over the best components-alone oracle) mostly carries through the bridge (+0.0068 validation, +0.0125
  test).
- **Data-selected genes carry most of it.** Own expression and partners alone add +0.0027 with intervals spanning
  zero on both splits.
- **No remedy improves on the affine bridge on validation.** Contrastive 4/0 and 8/4 gain more on test than the
  affine row but less on validation, as in the first run; gating at 0.7 lies between, inside the intervals.
- **Grid edges.** The chosen pair (1, 10) is interior on both grids. The row was chosen among 75 gene-level rows of
  the affine bridge on validation, so its validation gain is optimistic; the test gain is the unbiased estimate.

## Recommendation (my reading; the user decides)

1. **Integrate no remedy into the prior, but add the gene-level blocks.** After the full-grid rerun, the prior to
   carry forward is the affine bridge with expression components (penalty 1) plus own expression, partners and
   data-selected genes (gene penalty 10): 0.2300 validation / 0.2352 test, against 0.2232 / 0.2227 for components
   alone. No remedy's best validation row reaches it.
2. **Carry nothing forward from noise-matched fitting or low-rank denoising.** Every validation row is below the
   reference. Denoising removes components signal at every rank, and noise-matched fitting trades 953 lines for 170.
3. **Done:** the runner now reports every components and gene penalty pair (`d314b74`), and the rerun above
   answered the open question.

## Caveats

- **Seed and split.** These are seed-0 numbers on one split, with 27 validation and 27 test lines. Intervals are
  about 0.01 wide for own expression and partners and 0.03 or more for sets with data-selected genes; most
  differences between remedies' best rows sit inside them. Additions within a setting (gene-level minus that
  setting's components) have no interval.
- **Diagnostics versus scores.** Diagnostics are per-gene correlations over the 167 paired training lines, and the
  contrastive ones are measured in the aligned space. Better diagnostics did not give better scores.
- **Grid edges.** Gene penalty 10, the top of its grid, wins on validation for every affine block set, for own
  expression and partners at every gating threshold, for every noise-matched block set and for the best gene-level
  row of all eight contrastive settings; 0.1, the bottom, wins for data-selected genes and all three under denoising
  at ranks 32 and 64. The components penalty sits at 1000, its top edge, in three settings (contrastive 8/0,
  denoising at ranks 32 and 64) and never at 0.1. The components-alone score is flat from 10 to 1000, so that edge
  is a tie rather than too narrow a grid, but it decides the gene-level rows.
- **The oracle is off-contract.** It reads validation lines' bulk RNA, enters no comparison as a model, and its
  differences have no interval. Under contrastive PCA it reads aligned bulk. Under noise-matched fitting it is not an
  upper bound.
- **Provenance.** The rows come from two commits, and the fix between them changes no score.
- **Claim boundary.** This is single-gene dependency evidence on the unchanged validation and test lines, and it
  estimates no genetic interaction.

## Next steps the results point to

1. The single-cell correction on top of the prior with gene-level blocks: what Tx1, `q_sc` and STATE add.
2. Which data-selected genes carry the gain, and whether partner features help once data-selected genes are in.
