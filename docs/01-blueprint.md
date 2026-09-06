# Research Blueprint: Context-Conditioned Synthetic-Lethality Ranking

**Status:** active research contract. Supersedes the Feng2024 two-axis formulation and the
train-free SLIdR stage; neither is part of this program. T1 and T2 closed negative and are
paused. No SL result exists; historical Exp13 Stage 2 is a negative point estimate.
The new joint GeneEffect protocol has no scientific run or result yet.
**Companions:** [`02-literature-review.md`](02-literature-review.md) fixes the prior-art
boundary · [`03-experiment-protocol.md`](03-experiment-protocol.md) is the SL-pair executable
protocol · [`joint GeneEffect design`](specs/2026-09-06-modular-joint-training-design.md)
defines current GeneEffect training · [`historical Exp13 protocol`](specs/2026-08-17-exp13-geneeffect-residual-protocol.md)
records staged runs · [`data/sl-context-screen.md`](data/sl-context-screen.md)
and [`data/cell-line-geneeffect-226.md`](data/cell-line-geneeffect-226.md) are the benchmarks.

## 1. Task

Given a cancer cell line described only by its **basal single-cell transcriptome** — no
CRISPR screen and no SL screen in that line — rank unordered gene pairs by the probability
that the pair is an experimental synthetic-lethal hit there. The generalization axis is the
**cell line**; genes are not held out, and §8 states what that forbids.

```text
basal cells + perturbation gene
  -> predicted post-perturbation cells     supervised on 4 Perturb-seq lines
  -> predicted DepMap GeneEffect           supervised on many lines
  -> pair score in a held-out line         supervised on screen labels
```

Predicted expression is a supervised intermediate, not a deliverable: it is scored only
where Perturb-seq exists and elsewhere receives gradient only through the dependency loss.

## 2. Notation and Data

| Symbol | Meaning |
| --- | --- |
| $\mathcal{G}$, $e_g$ | perturbable gene universe; ESM-2 gene embedding, defined for every gene including unscreened ones |
| $X_c=\{x_c^{(i)}\}_{i=1}^{M_c}$ | basal control single cells for context $c$ |
| $\mathcal{C}_{\text{resp}}$ | contexts with genetic Perturb-seq (K562, HCT116, Jurkat, HepG2) |
| $\mathcal{C}_{\text{dep}}$ | contexts with basal cells **and** DepMap GeneEffect |
| $\mathcal{C}_{\text{sl}}$ | contexts with screen pair labels, split train/val/test |
| $\mathcal{C}_{\text{ref}}=\mathcal{C}_{\text{dep}}\setminus\mathcal{C}_{\text{sl}}$ | profile cohort; never an SL context |
| $y_{g,c}$ | DepMap 26Q1 GeneEffect; a single-gene relative growth-rate effect |
| $D_{a,b,c}\in\{0,1\}$ | experimental screen hit / non-hit for $(a,b)$ in context $c$ |

Supervision is a pyramid: four contexts teach what a perturbation *does*, tens what it
*costs*, and the SL contexts carry pair labels. GeneEffect is neither a double-knockout
measurement nor an SL label. Backbone parameters are $\omega$, dependency heads
$\theta_\mu,\theta_\delta$, SL head $\theta$; indices run $(\text{gene},\text{context})$;
superscripts $tr$ and $te$ denote the train and test side of the published split; and $\rho$
is Pearson correlation.

## 3. Model

With pooling $\Pi(B)=[\operatorname{mean}(B),\operatorname{var}(B)]$ over a cell bag $B$:

```text
z_c            = Pi(X_c)                          context vector, F_omega INPUT space
B_c            = {b_c^(i)}                        basal cells in F_omega's OUTPUT space
Bhat_{c,g}     = { F_omega(x_c^(i), e_g) }        predicted perturbed cells, output space
Delta_{g,c}    = Pi(Bhat_{c,g}) - Pi(B_c)         output-space change; never mixed with z_c
s_{g,c}        = online dispersion statistics of Bhat_{c,g} against B_c
q_sc_{g,c}     = single-cell basal statistics of gene g in context c
muhat_g        = h_mu(e_g)                        context-blind gene mean
deltahat_{g,c} = h_delta(Delta_{g,c}, s_{g,c}, q_sc_{g,c}, e_g, z_c)   5-block residual
yhat_{g,c}     = muhat_g + deltahat_{g,c}
```

**Amendment (2026-08-17).** The prior single-symbol form, `Delta_{g,c} = Pi(Bhat_{c,g}) -
z_c`, is ill-typed whenever $F_\omega$'s input and output widths differ: $z_c$ pools the
*input* space, $\hat B_{c,g}$ the *output* space (Tx1-3B 2560-d in vs. ST 2000-d out —
[`specs/2026-08-17-exp13-geneeffect-residual-protocol.md`](specs/2026-08-17-exp13-geneeffect-residual-protocol.md) §4).
$B_c$, the basal bag re-expressed in output space, is the fix; $s_{g,c}$
and $q_{sc,g,c}$ are two further declared blocks a composition may hand $h_\delta$.

The $\mu/\delta$ split is load-bearing. Because
$\operatorname{Var}_g(\mu_g)\gg\operatorname{Var}_c(\delta_{g,c})$, one head regressing raw
GeneEffect is optimized almost entirely by the context-blind term, and a model with
$\hat\delta\approx 0$ still posts a strong raw correlation. $\hat\mu_g$ collapses to
$\mu_g^{tr}$, no learned $h_\mu$, whenever a benchmark declares every scored gene
GeneEffect-covered by construction (Exp13 does; see `specs/2026-08-17-exp13-geneeffect-residual-protocol.md`).
The SL head consumes the
$\hat\delta$ profile over $\mathcal{C}_{\text{ref}}$, summarized per gene by nine frozen
statistics $\Sigma$: mean; population standard deviation; the quantiles at
$0.10,0.25,0.50,0.75,0.90$; and the fractions of the profile below $-0.5$ and below $-1.0$.
Those constants are part of the contract, not an implementation detail — leaving them to
code lets two conforming implementations build different inputs.

```text
prof_g     = Sigma( ( deltahat_{g,c'} )_{c' in C_ref} )

phi(a,b|c) = [ prof_a + prof_b , |prof_a - prof_b| ,        18  invariant
               rho(deltahat_{a,.}, deltahat_{b,.}) ,         1  invariant
               yhat_ac + yhat_bc , |yhat_ac - yhat_bc| ,     2  context
               psi_min(yhat_ac, yhat_bc) ,                   1  context
               muhat_a + muhat_b , |muhat_a - muhat_b| ]     2  invariant

s(a,b|c)   = sigmoid( f_theta( phi(a,b|c) ) )
```

Twenty-four dimensions, three of which vary with $c$; every block is invariant under
$a\leftrightarrow b$. Supplying $(\hat\delta,\hat\mu)$ rather than $(\hat y,\hat\mu)$ is an
invertible reparametrization, used because co-dependency lives in the centered profile and
because it makes the pan-essentiality block separately ablatable.
$\psi_{\min}=\min(\hat y_{a,c},\hat y_{b,c})$ is the declared non-interaction null (HSA;
GeneEffect is negative-is-lethal). Score orientation is frozen before any test label: $s$ is
positive-is-SL, so $\psi$ used as a ranking score is negated.

## 4. Objective

### Current GeneEffect training

The [joint-training design](specs/2026-09-06-modular-joint-training-design.md)
governs the implemented GeneEffect path on the fixed 226-line split: 170 labeled
training lines, 27 validation lines and 27 test lines. Tx1 stays frozen; STATE,
the ESM2 adapter and residual head train together from initialization. Fit one
gene mean on training lines and predict the residual using Huber loss (delta 1).
Every fourth optimizer update also uses response supervision from equal numbers
of conditions from K562, Jurkat, HepG2 and HCT116. The response loss is mean-delta
MSE plus energy distance. The other updates use GeneEffect regression only.

Validate once at the end of every epoch. Report total loss, GeneEffect loss,
both response terms, their sum, absolute GeneEffect Pearson/Spearman per line,
variable-gene residual Pearson/Spearman per gene, RMSE, MAE and coverage.
**Early stopping and checkpoint selection minimize only `val_geneeffect_loss`.**
Response and correlation metrics remain scientific diagnostics. Training,
cell-collation and projection base seeds are 0. No head-only warmup, response
artifact seal, feature-store stage or gradient-ratio calibration is required.

### Separate SL composition proposal

The following objectives concern the unimplemented SL composition protocol,
including its predicted gene-mean block and out-of-fold fitting. They do not
override the fixed-mean joint GeneEffect objective above or authorize using the
226-line model directly for held-out SL claims.

Fit the gene mean on train-side contexts only, with at least three observations per gene:

```text
mu_g^tr     = mean_{c in C_dep^tr} y_{g,c}
delta_{g,c} = y_{g,c} - mu_g^tr

L_mu    = Huber(muhat_g, mu_g^tr) + alpha * [ 1 - Pearson_g(muhat, mu^tr) ]
L_delta = Huber_{g,c}(deltahat, delta)
          + beta * mean_{g in Gvar} [ 1 - Pearson_c(deltahat_{g,.}, delta_{g,.}) ]
L_dep   = L_mu + L_delta

Stage 1:   min_omega                        L_resp                    on C_resp
Stage 2:   warm theta_delta with omega frozen; then min L_resp + lam_dep * L_dep
Stage 3:   min_theta                        BCE( s , D )              on C_sl^tr
```
$L_{\text{resp}}$ matches predicted and observed cell bags by mean-delta MSE plus energy
distance; no cell-to-cell correspondence exists, so the loss is distributional. It stays on
in Stage 2 as an anchor, and response metrics are reported before and after Stage 2 — a
collapse means $\lambda_{\text{dep}}$ is wrong, not that the run finished.
$G_{\text{var}}$ is a pre-declared delta-variance gene set fit on train-side contexts;
without it the scale-free Pearson term gives a gene whose true $\delta$ is replicate noise
the same gradient as a genuinely context-dependent one. Stage 3 is context-balanced, each
training context contributing equally. One seed is not a multi-seed claim.

$\alpha$ is frozen on GeneEffect-only validation against a declared calibration band
**before any SL label is read**. Its Pearson term is shift- and scale-invariant, so $\alpha$
can otherwise buy correlation at the cost of scale — and that error lands entirely in
$\hat y$, hence in $\psi$, flattering the model by degrading its own baseline. Because
$\mu^{tr}$ uses train-side contexts only, it is not an affine function of any test label, so
the fold-fit-mean artifact that manufactures per-gene Spearman $+1$ cannot arise.

## 5. Split and Arms

One fixed split, published inside the benchmark and never redefined by a run. Test contexts
are absent from response training, dependency training, $\mu^{tr}$, SL-head training,
hyperparameter and checkpoint selection, standardizer fitting, calibration, and
thresholding. Every backbone-derived block of an SL training row — profile, target-context
values, and $\hat\mu$ — comes from a single model that excluded that row's context group and
refit $\mu^{tr}$ without it; mixing in-sample and out-of-fold blocks inside one feature
vector, then fitting a standardizer on the mixture, is prohibited.

- **Arm A, primary:** out-of-fold predicted profiles in, predicted profiles out.
- **Arm B, oracle:** measured DepMap profiles on the same $\mathcal{C}_{\text{ref}}$ columns.
- **Arm B-full, reference:** measured profiles over all DepMap columns — the honest ceiling.

Report $A-B$ restricted to the context block alongside the full $A-B$; only the restricted
form isolates out-of-sample GeneEffect cost, since 21 of 24 dimensions are context-invariant
in both arms.

## 6. Evaluation Contract

Metrics are numbered by the stage they score, matching §4. **Stage 1** reports response metrics before and after Stage 2 training.
Validation labels may select Stage 2 checkpoints and hyperparameters but never enter fitting; test labels enter neither fitting nor selection.

**Stage 2** needs two surfaces: only one SL test context has GeneEffect; a dependency split over $\mathcal{C}_{\text{dep}}$, disjoint from the
SL contexts and holding out at least eight, carries the per-gene across-context Spearman on
the residual over $G_{\text{var}}$; R1's ladder is its registered baseline. GeneEffect-covered
SL test contexts carry per-context cross-gene Spearman only, which cannot support a context claim.

**Stage 3** reports AUPR per test context and macro $\text{AUPR}-\text{prior}$ because raw
AUPR macros measure the eighteen-fold prior range. AUROC is secondary; coverage and class
counts come first. The de-duplicated diagnostic macro weights observable aggregate-label
clusters: PC9/HELA are one cluster but retain separate scores. Stratify every AUPR by
whether both, one, or neither endpoint was seen in SL training — only "neither" supports
an inductive claim. Uncertainty uses a
two-way dyadic bootstrap over both endpoints; pairs have no unique anchor gene, so a one-way
bootstrap would ignore dependence through the other endpoint.

Baselines, reported as per-context lift, never used to drop a context after the fact:

| ID | Baseline | Shortcut it removes |
| --- | --- | --- |
| C1 | pair identity / train-context label frequency | pair memorization |
| C1b | anchor-gene frequency $\max(r_a,r_b)$, for $r_g$ gene $g$'s positive rate over training contexts | gene-level memorization C1 misses |
| C2 | $\psi$ alone, predicted **and** measured | ranking that is only the null |
| C3 | the $\hat\mu$ block alone | pan-essentiality |
| C5 | R1's best residual predictor replacing the backbone | a backbone beating no simple prior |
| C6 | **matched context-ablated head** | a context claim with no context information |
| C4 | Arm B and Arm B-full | reference only, never a bar |

**C6 is load-bearing.** C1, C1b, C2 and C3 each remove one shortcut, so a model can win on
the 21 invariant dimensions, beat every individual baseline, and still carry no context
information. C6 is an identically trained head on the invariant dimensions with the three
context dimensions ablated, plus fixed context derangements and a permutation null. A
minimum per-context incremental $\text{AUPR}-\text{prior}$ attributable to the context block
is declared before any test label is read; distinguishability alone is not enough, and below
that margin no context claim is licensed whatever the absolute AUPR.

## 7. Leakage and Integrity

Operational rules live in [`03-experiment-protocol.md`](03-experiment-protocol.md); these
four are contract-level.

- Join contexts by DepMap ModelID through a checked-in map and fail loudly on an unmapped
  context. Never join on informal name — DepMap calls K562 `K-562`.
- Fit standardizer, calibrator, and thresholds on train-side data only. Per-context
  z-scoring of $\hat y$ is forbidden: it uses the test context's own distribution and erases
  the quantity under test.
- Use one common, label-independent pair universe for both arms. Missing scores stay
  missing — never imputed to zero, never counted as negatives.
- Qualify every held-out-context result with the Tx1 Tahoe-100M pretraining exposure.
  Task-label holdout is not representation-pretraining holdout.

## 8. Claim Boundaries

- Single-gene essentiality is not synthetic lethality; a pan-essentiality lift is not an SL
  result.
- **No result is described as an interaction.** A classifier taking $\psi$ as one of
  twenty-four inputs predicts no joint outcome and defines no $\text{joint}-\psi$ residual;
  the model-minus-$\psi$ gap is incremental label ranking. An interaction claim needs a joint
  or measured genetic-interaction quantity, which this contract does not supply.
- **No significance claim across contexts.** One split with few test contexts admits no valid
  family-wise inference over the several baselines, arms, and strata. Report per-context
  effect sizes with intervals and a predeclared minimum detectable effect.
- Genes are not held out, so no unseen-gene claim is available from this benchmark.
- The table retains no same-pair label reversal across contexts, so it cannot establish
  recovery of context-dependent reversal. Two of its contexts are one screen exploded twice,
  which makes this caveat more important, not less.
- Candidate prioritization is not experimental target validation.
- **Exp13 is scope-closed.** The 226-line cell-line GeneEffect residual benchmark
  (`data/cell-line-geneeffect-226.md`, protocol `specs/2026-08-17-exp13-geneeffect-residual-protocol.md`) is
  not evidence for cross-context SL. Reusing its single frozen backbone pass for an SL
  held-out-context claim needs an out-of-fold backbone per inner group, per
  [`03-experiment-protocol.md`](03-experiment-protocol.md) §5, which Exp13 does not build; the
  226 split and `context_screen_v2` never substitute for each other.

## 9. Current Scientific State

- **T1:** the exp05 Bridge-A evaluation did not recover Horlbeck K562 genetic interactions;
  closed negative and paused. Under this contract no gate consumes measured GI.
- **T2:** Tx1-3B-ST did not beat the registered few-shot copy-K562 baseline on the frozen
  nine-line test; closed negative and paused, superseded by Exp13's corrected composition
  (below). T2 ran under the pre-amendment §3, whose Delta-typing defect this amendment fixes.
- **R1:** the residual ladder is implemented and is the registered baseline for
  $\hat\delta$ and for control C5.
- **Benchmark:** the `context_screen_v2` row-level SL split is built; its raw-filter audit
  remains incomplete and no model has run.
- **Historical Exp13:** formal Stage 2 completed on the 226-line GeneEffect residual benchmark.
  The one-seed model test macro per-gene Spearman was 0.0225, below context-PCA ridge
  (0.0851) and nearest-line (0.0462); negative point estimate, with no positive context
  claim and never SL evidence. Scope-closed per §8
  ([result](results/exp13_stage2_full/README.md)).
- **Joint GeneEffect:** the replacement training protocol revisits the four response
  anchors throughout training and selects minimum validation GeneEffect loss.
  No GPU training or scientific evaluation has been run under this protocol.

Negative backbone results constrain the substrate; they are not SL results. A claim enters
`docs/results/` only after its evaluation completes and the evidence supports it.
