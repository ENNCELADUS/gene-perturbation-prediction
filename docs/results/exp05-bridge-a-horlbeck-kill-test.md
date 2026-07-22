# exp05 Bridge A vs Horlbeck K562 GI — mechanism kill-test (negative)

**Status:** completed 2026-07-22; **negative**. The frozen exp05 K562 backbone,
composed into a Bridge A counterfactual co-dependency score, does **not** recover
measured Horlbeck 2018 K562 genetic interactions (|Spearman| < 0.01;
AUROC(s_A -> strong-SL) approximately 0.52). This is a **development feasibility
diagnostic**, not the formal `02` section 6 MECHANISTIC verdict (which additionally
requires a Phase-0-registered `rho_min`, a frozen candidate universe, and declared
calibration disjointness). Authority: [`../01-blueprint.md`](../01-blueprint.md),
[`../02-acceptance-criteria.md`](../02-acceptance-criteria.md); plan:
[`../specs/2026-07-22-k562-mechanism-and-geneeffect-generalization-plan.md`](../specs/2026-07-22-k562-mechanism-and-geneeffect-generalization-plan.md)
(T1).

## What was tested

Whether the single-perturbation exp05 backbone, composed pairwise via **Bridge A**,
recovers *measured* K562 epistasis in its best-supported context — the precondition
for extending the composition mechanism across cell lines. A null here means the
interaction mechanism fails at home; do not extend across contexts before rethinking
composition (plan section 2.4).

**Bridge A score.** For an ordered step, the co-dependency spike is
`Delta(a->b) = c_hat[b | control_state] - c_hat[b | a-perturbed_state]`, where the
`a`-perturbed state is the *observed* Replogle K562 cell bag for `a` fed as basal.
Symmetrized: `s_A(a,b) = 0.5 * [Delta(a->b) + Delta(b->a)]`. GeneEffect is more
negative when more essential, so `Delta > 0` means `b` becomes more essential once
`a` is lost. The hypothesis under test: larger `s_A` corresponds to more
synthetic-lethal (more negative Horlbeck `gi_score`).

## Method and provenance

- **Model.** Frozen exp05 `AivcModel` checkpoint, SHA-256
  `48097722f5742a459b86ba6153dd21f145ff1a0e30dafa80061c325c2d46b811`
  (run `exp05_fixed_k562_pool_v1`). Wiring verified against the model's own
  training predictions: control-arm `c_hat[gene | control]` reproduces
  `predictions.csv` to max abs error 8.3e-17 (mean 5.2e-17).
- **Candidate universe.** 408 genes / 83,028 pairs — the exp05-ready bound
  (`exp05_observed_pair_covered`: both genes trained in the fixed pool,
  DepMap-label-qualified, and >=8 observed Replogle cells), a complete clique over
  the 408 genes, all present in the ESM2 perturbation vocabulary. Target:
  [`../data/horlbeck-2018-k562-gi.md`](../data/horlbeck-2018-k562-gi.md) frozen
  `gi_score` (1,281 strong-SL pairs at `gi_score < -3.0`; range [-13.27, +8.56]).
- **Independent-N matching.** Both arms of each `Delta` use an equal, without-
  replacement independent-cell budget `w = min(n_cells // 64, 8)` windows; sub-window
  genes (< 64 observed cells) are bootstrapped to one window and flagged. The
  a-perturbed basal is restricted to Replogle-source cells only. Three seeds
  (41/42/43) resample the panels.
- **Both pooler reference conventions** (`self`: the a-perturbed panel is its own
  reference; `control`: pooled controls) were run. **They are indistinguishable**
  (every metric differs by < 0.001), settling the reference-latent question
  empirically.
- **Compute.** 1,006,136 forward passes, one GPU. Outputs (gitignored):
  `results/experiments/05_aivc_a_to_b_to_c/bridge_a/sweep/`.

## Result

Spearman is `s_A` vs `gi_score` (SL signal => negative); AUROC is `s_A` predicting
the binary strong-SL label (SL signal => > 0.5). Values below are the `self`
convention; `control` is identical to three decimals.

| Slice | Pairs | Strong-SL | Spearman(s_A, gi) | AUROC(s_A -> strong-SL) |
| --- | --- | --- | --- | --- |
| Full universe | 83,028 | 1,281 | +0.004 (p=0.31) | 0.518 |
| Primary (both genes >= 64 cells) | 56,616 | 841 | +0.009 (p=0.03) | 0.526 |
| Cleanest (>= 64 cells and low seed-variance) | 6,795 | 78 | -0.008 (p=0.52) | 0.617 |

## Interpretation

- **No usable signal.** |Spearman| < 0.01 on every slice; the primary-slice p=0.03
  is significance-from-N (56,616 pairs), not effect size. AUROC approximately 0.52 —
  at the 0.50 chance line, and **below the single-gene GeneEffect floor** (CV2 AUROC
  0.704 in the contract). The composition adds essentially nothing over noise.
- **Not a degenerate-output artifact.** `s_A` varies (mean 0.012, std 0.079, range
  approximately [-0.12, +0.13]); the counterfactual shift is real (mean |Delta|
  approximately 0.08) and the model demonstrably responds to the a-perturbed basal
  (basal-sensitivity gate passed, deltas 0.02-0.31). The wiring is machine-exact on
  the control arm.
- **Noise-limited estimate.** Signal-to-noise approximately 1.0 (across-pair `s_A`
  spread 0.079 vs per-pair seed std 0.076): about half of each pair's `s_A` is
  cell-set sampling noise.
- **A whisper of correct-direction signal, but practically useless.** Strong-SL
  pairs have marginally higher mean `s_A` (+0.028 vs +0.018; Mann-Whitney p=0.004),
  and the lowest-noise slice reaches AUROC 0.62 — but there is **no dose-response**
  (`gi < -5` pairs, mean `s_A` +0.022, are not more enriched than `gi < -3`, +0.028),
  which argues against a real mechanistic signal.

## Verdict and scope

The frozen single-perturbation exp05 backbone, composed via Bridge A, does not carry
K562 pairwise epistasis at a useful level. Per the plan's kill-test rule, **do not
build multi-cell-line Bridge A machinery on this checkpoint; the composition
mechanism needs rethinking.** The kill-test spent a few GPU-hours to retire this
risk before the far larger cross-cell-line investment.

Claim boundaries (contract section 9): this is K562 mechanism only, a development
diagnostic (not a formal MECHANISTIC verdict), on a single-perturbation backbone used
out-of-distribution (an a-perturbed state fed as basal). It says nothing about the
Feng2024 `q(a,b)` benchmark track or cross-cell-line generalization. Bridge B (virtual
double-knockout) is not evaluable here: the checkpoint has no trained two-gene
operator.

## Reproduction

Universe: `uv run python scripts/lock_bridge_a_universe.py`. Sweep:
`scripts/bridge_a_forward.py --genes <408 universe genes> --seeds 41 42 43
--reference-convention both`. Correlation: `scripts/analyze_bridge_a_sweep.py`.
Universe provenance (manifest `candidate_universe_manifest.json`): input SHA-256
esm2_npz `b2f2b813...`, pairs_csv `8cc5d6eb...`, pool_csv `c33dbe6c...`.
