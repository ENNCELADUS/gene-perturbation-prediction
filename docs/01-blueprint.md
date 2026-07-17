# Research Blueprint: Virtual-Cell Composition for Synthetic-Lethality Discovery

**Status:** established. This is the research contract — locked.
**Type:** research direction and claim boundaries. **Not** an implementation spec.
**Supersedes:** the retired "cell-fate outcome dynamics" direction. Its evidence is preserved under `docs/archive/` and `ideaspark_run/` and is not edited. This document states what is true now; it is not a changelog.
**Companions:** [`docs/02-acceptance-criteria.md`](02-acceptance-criteria.md) (what counts as passing) · [`docs/03-literature-review.md`](03-literature-review.md) (related work — pending rewrite) · [`docs/04-roadmap.md`](04-roadmap.md) (what happens next — pending rewrite). The design that seeded this contract: [`docs/superpowers/specs/2026-07-17-virtual-cell-sl-composition-design.md`](superpowers/specs/2026-07-17-virtual-cell-sl-composition-design.md).

## 1. The Problem

```text
Rank a gene's synthetic-lethal partners for genes no SL screen or SL graph
has ever seen.
```

Synthetic lethality (SL) — disrupting either of two genes is tolerated, disrupting
both is lethal — turns a tumor's genetics into a selective target (PARP/BRCA). The
candidate space is $\sim 10^8$ pairs, screens cannot cover it, and SL is
context-specific, so **computational prioritization is a prerequisite for
experimental follow-up.**

The leading published predictors — **SLMGAE**, **KR4SL**, **KG4SL**, and the wider
Feng2024 zoo (DDGCN, GRSMF, SLGNN, SL2MF, …) — are **transductive graph /
knowledge-graph** models. Their signal is the topology of the curated SL graph.
The consequence is structural, not incidental:

- They are strong when test genes already sit in the training graph (CV1), but
  their inductive bias **degrades as genes are held out** (CV1 → CV2 → CV3).
- They **cannot score a gene with no curated SL edges** — the "beyond screened
  genes" frontier where undiscovered biology lives.

**The open problem is the inductive, graph-free, cold-start regime.** No method
wins it. This program targets exactly that regime.

## 2. Premise, Gap, and Value

**Premise (established internally).** A gene's knockout perturbation transcriptome
predicts its *own* dependency. The exp05 forward model (`src/aivc_model/`) — Arc
STATE + ESM2 protein identity, open-vocabulary — maps (K562 control state, gene
identity) → predicted response → **DepMap GeneEffect**, and ranks single-gene
dependency well. Foundation evidence: pseudobulk Δ-expression → GeneEffect reaches
5-fold Spearman ≈ 0.485 on Replogle K562; the signal is not a generic death axis
(death signature alone ≈ 0.244; death-residualized transcriptome ≈ 0.503).

**Gap.** A single-gene fitness model does not, by itself, score a *pair*. And
**single-gene GeneEffect provably cannot capture SL**: classic SL pairs are
individually near-neutral (GeneEffect ≈ 0) but jointly lethal, so a predictor that
sees only $\text{GeneEffect}(a)$ and $\text{GeneEffect}(b)$ is the archive's
"dependency-only floor" — ~0.70 AUROC, mostly pan-essentiality.

**Value.** If the forward model can be **composed** into a genuine *pairwise
interaction*, the score reads from gene *features* (protein identity + predicted
perturbation biology), not graph position — so it reaches pairs no SL graph has
seen. That inductive reach is necessary but **not, by itself, the contribution**: a
contemporaneous method (CILANTRO-SL — [`03`](03-literature-review.md) §5) already
reaches unscreened genes and beats KG4SL cold-start with a foundation-model
embedding-ablation. The contribution is the **mechanism** the prior art lacks: a
*perturbation-response-trained* virtual cell composed through an **explicit
interaction null** (§4), with the pairwise prediction **validated against measured
epistasis** (§7.2).

### 2.1 The Value Question — tested outside the benchmark label

A high benchmark rank on a curated label is not, by itself, evidence that the
mechanism is real: the label is a database curation, its negatives are unconfirmed
(§9), and topology can be gamed (CV1). The value of the composition must therefore
be answered on **two axes that curated-label ranking alone cannot supply**:

> **Value question.** (a) Does the composed pairwise score beat the label-graph
> SOTA on the **honest cold-start splits** (CV2/CV3), on the **pair-specific**
> (non-pan-essential) residual rather than on pan-essentiality? And (b) does the
> **virtual double-knockout** score correspond to **measured** genetic
> interactions, not only to the curated label?

The numeric bars for (a) and (b) are fixed **before evidence** in
[`docs/02-acceptance-criteria.md`](02-acceptance-criteria.md) and are not revisable
to fit a result.

### 2.2 The defensible novelty (post-review)

Adversarial literature review ([`03`](03-literature-review.md)) fixed what is and is
not novel. **Not novel:** graph-free / inductive SL that reaches unscreened genes and
beats KG4SL cold-start — CILANTRO-SL, RFM-SL, PARIS, and ESM4SL occupy that ground.
**Novel, and the claim this program leads with:**

> No prior work composes a *perturbation-response-trained* virtual cell into an SL
> score via an **explicit double-knockout / sequential-counterfactual interaction
> term** (against a stated additive/min null) and **validates it against measured
> genetic interactions** rather than only curated labels.

The closest prior art (CILANTRO-SL) uses a non-causal embedding-ablation as its
"knockout," a black-box classifier as its composition, and no joint perturbation and
no epistasis check — its own authors defer double-KO to future work. The program's
own retired exp08 already tried the weaker feature-transfer form and failed the
floor; the mechanism in §4 is the falsifiable difference.

## 3. Objects and Definitions

| Symbol | Meaning |
| --- | --- |
| $\mathcal{G}$ | The 9,471-gene K562 candidate universe (Feng2024, genes carrying numeric K562 DepMap GeneEffect). |
| $s(a,b) \in \mathbb{R}$ | The pairwise SL score. **Swap-invariant:** $s(a,b)=s(b,a)$. Used for both ranking and classification. |
| $F(X, g) \mapsto \hat B_g$ | exp05 forward model: control cells $X$ + gene identity $g$ → predicted response bag. |
| $h_C(\hat B_g) \mapsto \hat c_g$ | dependency head → predicted **GeneEffect** for $g$. |
| $\psi(\cdot,\cdot)$ | the **non-interaction null** (additive or min) against which an interaction is measured. |
| $D_{ab}\in\{0,1\}$ | the benchmark SL label (SynLethDB-derived, balanced Rand 1:1). A **curated adapter label**, not a validated K562 SL assay. |

**GeneEffect is a relative growth-rate effect** under an explicit
population-dynamics model (Chronos): more negative = more essential. It is **not**
a cell-death label, **not** single-cell, and **single-gene** — there is no
double-knockout GeneEffect in DepMap.

**Evaluation object.** The Feng2024 SynLethDB-derived K562 benchmark under its
per-anchor `cal_metrics` protocol: ranking (NDCG@{10,20,50}, MAP@k) + classification
(AUROC, AUPR, F1), on three splits of increasing cold-start severity — **CV1**
(pair-level holdout; genes may recur), **CV2** (one gene of each test pair unseen),
**CV3** (both genes unseen). CV1 is a **degree-gameable diagnostic** (a gene-degree
probe wins it); generalization claims come **only** from CV2/CV3.

## 4. The Mechanism and the Two Composition Bridges

Both bridges are built on the same exp05 forward model (frozen, or lightly
re-trained to accept composed perturbations) and use **no SL labels to construct
features**. They are compared **head-to-head**.

### 4.1 Bridge A — counterfactual co-dependency

Simulate loss of $a$, then ask whether $b$ becomes essential in that state.

1. Forward-simulate $a$-loss: $\hat B_a = F(X, a)$.
2. Predict $b$'s dependency **in the $a$-lost context**, using $\hat B_a$ as the
   control template: $\hat c_{b\mid a} = h_C\big(F(\hat B_a, b)\big)$.
3. Score the symmetrized dependency spike:

$$
s_A(a,b) = \tfrac{1}{2}\Big[(\hat c_{b} - \hat c_{b\mid a}) + (\hat c_{a} - \hat c_{a\mid b})\Big]
$$

GeneEffect is more negative when more essential, so positive $s_A$ means each gene
becomes *more* essential once its partner is lost — the co-dependency signature of
SL. Uses **only single-gene GeneEffect labels**; this is the "context-selective
dependency" bridge. Requires the forward model to **compose two perturbations
sequentially** (an extrapolation; see §6).

### 4.2 Bridge B — virtual double-knockout

Predict joint-knockout fitness and measure interaction against an explicit null.

1. Forward the **joint** perturbation of $a$ and $b$:
   $\hat c_{ab} = h_C\big(F(X, \{a,b\})\big)$.
2. Score the genetic-interaction residual against $\psi \in \{\hat c_a + \hat c_b,\ \min(\hat c_a,\hat c_b)\}$:

$$
s_B(a,b) = \psi\big(\hat c_a, \hat c_b\big) - \hat c_{ab}
$$

oriented so that joint-worse-than-null ⇒ higher SL. This is the most literal
"combination outcome." It has **no direct joint-fitness label** (DepMap is
single-gene), so its correctness is established by validation against **measured
epistasis** (§7.2, H3), not by the curated label alone.

### 4.3 Scoring and the label boundary

Each bridge yields a continuous, directly-rankable $s(\cdot)$ (primary use).
Optionally a light swap-invariant head may be **calibrated** on training-fold SL
labels using $s_A/s_B$ plus swap-invariant GeneEffect features — but **the SL graph
itself never enters feature construction**, preserving the inductive claim. The
zero-shot composition result is always reported separately from any calibrated head.

## 5. Hypotheses

Three hypotheses, tested separately, each with a predictive null.

### 5.1 Utility hypothesis (H1)

> The composition provides **incremental cold-start ranking** beyond the
> dependency-only floor **and** beyond the strong label-graph SOTA (**SLMGAE**,
> **KR4SL**), on CV2/CV3.

Null: $D_{ab} \perp s(a,b) \mid \text{floor features}$, on held-out genes.

### 5.2 Pair-specificity hypothesis (H2)

> The signal is **pair-specific interaction**, not pan-essentiality.

Null: the CV3 lift **vanishes on the non-pan-essential slice**. Grounding: the
retired program's decomposition found the naive cross-line lift collapses there
(CV3 AUROC 0.645 → 0.583, AUPR 0.651 → 0.490). A win that does not survive this
slice is pan-essentiality wearing SL's costume, not SL.

### 5.3 Mechanistic-realism hypothesis (H3)

> The **virtual double-knockout** score corresponds to **measured** genetic
> interactions in K562, not only to the curated label.

Null: $s_B \perp \text{measured GI}$. The measured-GI anchor is **Horlbeck 2018**
(K562 dual-CRISPRi GI map, ~222k pairs; **to be acquired**), with **Adamson 2016
UPR** (the only local combinatorial-CRISPRi set: 3 sensors + combos) as a small
*qualitative* transcriptomic check. **Not** the Jost/Replogle dual-sgRNA file — that
is single-gene knockdown-efficacy, not epistasis ([`03`](03-literature-review.md)
§3). Because Feng2024's K562 positives may themselves be Horlbeck-derived, the
Horlbeck validation must use continuous GI on pairs/genes **disjoint from the
benchmark positives** to avoid circularity. H3 is the credibility anchor that
survives even if the head-to-head ranking win (H1) is only partial.

## 6. What Is a Hypothesis, Not a Premise

None of the following may be assumed.

| Proposition | Status |
| --- | --- |
| STATE can predict **joint or sequential double-perturbation** responses | **Under test (H3), not assumed.** STATE was trained on single perturbations; the composition (§4) is an extrapolation. Its validity is gated by measured epistasis before benchmark numbers are trusted. |
| A **generated** response preserves partner information as well as an observed one | **Under test.** The archive's frozen-STATE+adapter route (exp08) landed *below* the dependency-only floor. The new bet is the *composition* (interaction), not per-gene feature transfer, which already failed. |
| A **graph-free** method can beat the SOTA on **CV1** | **False / not attempted.** CV1 is topology's home turf and is degree-gameable. The win is claimed only on CV2/CV3. |
| Curated **Rand negatives** are true non-SL pairs | **False.** They are unconfirmed non-SL; every output is candidate prioritization, never a validated SL target. |
| **GeneEffect alone** determines SL | **False.** That is the ~0.70-AUROC floor; the composition must emit an interaction, not a function of the two singles. |
| A high **benchmark rank** implies the mechanism is real | **Invalid inference.** Curated-label ranking is gameable and label-bounded; mechanistic realism is a separate claim, tested by H3. |
| **Graph-free inductive reach is itself the novelty** | **False (post-review).** CILANTRO-SL / RFM-SL / PARIS / ESM4SL already do inductive SL; the novelty is the mechanism (§2.2), not the reach. |
| A virtual double-KO **faithfully preserves synergy** | **Doubted, under test.** Independent benchmarks find foundation models *underestimate* synergy and do worse than an additive baseline on double perturbations ([`03`](03-literature-review.md) §2); the explicit null and a GenePert-style linear ablation are guards, not options. |

## 7. Success Criteria (contract level; numeric bars in `02`)

### 7.1 Primary — beat the label-graph SOTA on the honest splits

Beat the **strong** label-graph SOTA — **SLMGAE** (Feng2024 CV3 AUROC 0.790) and
**KR4SL** (Feng2024's flagged CV3 leader) — and the dependency-only floor, on **CV2
and CV3** per-anchor ranking, with the win **surviving the non-pan-essential
control** (H2). All baselines reproduced under the **identical** K562 `cal_metrics`
harness (only DDGCN/GRSMF/SL2MF/SLGNN reproduced so far; SLMGAE/KR4SL/KG4SL
pending). **KG4SL is a weak reference, not the bar** — its CV3 AUROC (0.562) sits
below the dependency floor (0.596). CV1 is reported only as the degree-gameable
diagnostic.

### 7.2 Mechanistic anchor — measured epistasis

The virtual double-knockout score $s_B$ (or $\hat c_{ab}$) must correspond to
**measured** genetic interactions — **Horlbeck 2018** K562 GI (to acquire; the
fitness-scale anchor) and **Adamson 2016 UPR** (local qualitative check) — by the
bar in `02`, on pairs disjoint from the benchmark positives (§5.3). The
Jost/Replogle dual-sgRNA file is **not** a GI dataset and is not used here.

### 7.3 Integrity constraints

- **Model selection is train-only.** Selecting the epoch/checkpoint on the test
  fold (the archive's exp08 flaw) makes a result **inadmissible**.
- **Identical harness.** The composition is evaluated under the same splits, seeds,
  and `cal_metrics` as the reproduced baselines — a true ablation, not a cross-run
  comparison.
- **Zero-shot reported separately.** The pure composition (no SL labels in
  features) is the headline; any label-calibrated head is an additional row.

## 8. Scope and Non-Goals

**Fixed:** K562 only; exp05 (STATE + ESM2) as the forward model; Feng2024 /
SynLethDB as the benchmark and metric protocol; the SL graph never enters feature
construction.

**Non-goals (out of scope for this program):**

- Multi-cell-line / context-specific SL (a future extension; co-dependency across
  real contexts).
- Any per-cell fate/death claim, or claiming SL from single-gene essentiality.
- Re-implementing the benchmark, its splits, or `cal_metrics`.
- Treating curated Rand negatives as validated non-SL.
- Aligning Norman CRISPRa (or any activation modality) to knockout labels without an
  explicit modality caveat (auxiliary only).

## 9. Claim Boundaries

Extending `CLAUDE.md`'s terminology guardrails:

- **DepMap GeneEffect is a relative growth-rate effect**, single-gene, under an
  explicit population-dynamics model. Never a cell-death label, never single-cell,
  never a double-knockout quantity.
- **SL benchmark outputs are candidate prioritization, not validated targets.** Rand
  negatives are unconfirmed non-SL.
- **Never claim SL from single-gene essentiality** — the interaction term is required
  ($\text{interaction} = \text{joint} - \psi(\text{singles})$, and the choice of
  $\psi$ is declared).
- **Generalization claims come only from CV2/CV3.** CV1 is degree-gameable and is
  reported as a diagnostic, never as evidence of cold-start generalization.
- **A pan-essentiality lift is not a synthetic-lethality result.** Report the
  non-pan-essential slice; a win that vanishes there is downgraded.
- **The virtual double-knockout is an extrapolation** of a single-perturbation
  backbone. Do not present its benchmark rank as mechanistically validated until it
  clears the measured-epistasis bar (H3).
- **A benchmark rank is not a mechanism.** Do not infer causation, fate commitment,
  mechanism, or manipulability from predictive ranking.
- **Modality/dataset roles:** Replogle (single-gene GWPS) and Adamson (UPR
  epistasis) are CRISPRi, modality-aligned with knockout; **Jost/Replogle
  dual-sgRNA is single-gene knockdown-efficacy, not epistasis**; Horlbeck is the
  measured GI anchor; Norman is CRISPRa, auxiliary only, always with the caveat.
- **A single-fold or test-fold-selected result is not a result.** Report 5-fold
  mean ± spread with a stated comparison, and never select on the test fold.

## 10. Locked Decisions

Settled. Changing any of these is a change of research program, not a refinement.

1. **The task.** Graph-free, inductive SL partner ranking in K562: learn $s(a,b)$
   with **no SL graph in the feature path**. (The inductive reach is shared with
   prior art; the contribution is the mechanism — §2.2 and #11.)
2. **The mechanism.** Compose the exp05 virtual-cell single-gene fitness model
   (response → GeneEffect) into a **pairwise interaction**. Single-gene GeneEffect
   alone is the floor, not the method.
3. **Two composition bridges, head-to-head.** Bridge A (counterfactual
   co-dependency) and Bridge B (virtual double-knockout), each with an explicit
   interaction null. Pure feature-transfer to a pair head (the archive's exp08) is
   **not** revived — it already failed.
4. **The win is claimed on CV2/CV3, never CV1.** CV1 is the degree-gameable
   diagnostic.
5. **The win must survive the non-pan-essential slice.** Pan-essentiality is not SL.
6. **Measured epistasis is the mechanistic anchor.** Bridge B is validated against
   **Horlbeck 2018** K562 GI (fitness-scale, to acquire) and **Adamson 2016 UPR**
   (local qualitative check), on pairs disjoint from the benchmark positives — **not**
   the Jost dual-sgRNA file, which is not epistasis. The north-star SOTA is **SLMGAE
   + KR4SL**, not KG4SL.
7. **The SL graph never enters feature construction.** Labels may calibrate a head;
   they may not build features. Zero-shot composition is reported separately.
8. **Model selection is train-only.** No test-fold epoch selection.
9. **Scope is K562.** Multi-cell-line / context-specific SL is a future extension.
10. **Acceptance criteria are frozen before evidence**
    ([`docs/02-acceptance-criteria.md`](02-acceptance-criteria.md)) and are not
    revisable to fit a result.
11. **The novelty is the mechanism, not the reach.** Lead with the
    perturbation-response-trained composition + explicit interaction null +
    measured-epistasis validation, differentiated from CILANTRO-SL and the retired
    exp08 (§2.2). Do not claim first-to-graph-free-inductive-SL.
