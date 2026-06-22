# Personal Reflection

## 1. What I set out to do

I started this project with a problem that is easy to state and hard to attack:
the human genome admits on the order of 10^8 candidate gene pairs, synthetic
lethality (SL) is context-specific, and combinatorial CRISPR screens cannot
cover that space. The clinical success of PARP inhibitors in BRCA-mutant tumors
told me the payoff is real; the scale told me that computational prioritization
is a prerequisite, not a luxury. So I narrowed the question to something I could
actually defend: within one well-characterized context (K562), can I rank a
gene's candidate SL partners?

The starting observation that carried the whole project was this: when a gene is
silenced, the cell's genome-wide transcriptional response — the perturbation
"shockwave" — reflects that gene's functional wiring. If that same response can
predict a gene's *own* dependency, it should also carry information about the
partners whose loss it cannot tolerate. The entire arc of my work was an attempt
to test that single idea honestly, and then to push it past the genes I could
actually measure.

## 2. The methodology I built, and why I built it that way

Looking back, the thing I am most satisfied with is not any single model — it is
the *ladder of evidence* I forced myself to climb in order, instead of jumping
straight to the deep-learning centerpiece.

- **Foundation first (exp01–exp02).** Before claiming anything about partners, I
  proved the prerequisite: pseudobulk Δ-expression regressed onto DepMap
  GeneEffect reached ~0.49 Spearman in cross-validation and transferred to
  Adamson at AUROC 0.886. Crucially, I then audited it — was the model just
  reading a generic "everything dies" viability axis? The death signature alone
  reached only 0.244, and residualizing it out left the transcriptomic signal
  essentially intact at 0.503. That audit mattered more to me than the headline
  number, because it is the difference between a real result and a comfortable
  illusion.
- **A representation as the bridge (exp03–exp05).** I moved from pseudobulk to
  single-cell set learning, where a frozen-GMM distribution regressor on scVI
  embeddings won (Adamson Spearman 0.666), and then connected the frozen STATE
  forward model. This gave me a way to turn *any* perturbation into a
  dependency-aware embedding — the capability I needed for the cold-start
  frontier.
- **A floor before a ceiling (exp06).** I built the most boring possible
  baseline first: just the two genes' DepMap GeneEffect scalars, five
  swap-invariant features, predict P(SL). It turned out to be strong (CV2 AUROC
  0.704). That floor became the bar every later, fancier model had to clear.
- **The proof-of-concept (exp07).** Holding the harness fixed and changing only
  the feature matrix, adding the observed transcriptome more than doubled CV2
  ranking quality (NDCG@10 0.042 → 0.094). This was the empirical license for
  everything that followed.
- **The generative framework (exp08) and a parallel route (exp09).** Finally the
  e2e model that *generates* the response for unscreened genes, plus a
  cross-cell-line selectivity route that needs no transcriptome at all.

Building in this order taught me that the sequence of experiments is itself an
argument. Each rung answered a question that made the next rung interpretable.

## 3. What I am proudest of: the discipline of honesty

If I had to name the single most valuable habit I practiced, it is refusing to
let the framing outrun the evidence. Several decisions encode this:

- **I treated CV1 as a diagnostic, not a result.** A degree-only probe wins CV1
  ranking (NDCG@10 0.197) yet collapses to ~0.001 on CV2/CV3. So CV1 rewards
  graph topology, not biology, and I judged every model only on CV2 and CV3 —
  the genuine generalization surfaces.
- **I scoped the label precisely.** The benchmark `D` is a SynLethDB-derived
  adapter target whose negatives are *unconfirmed* non-SL pairs, not a validated
  K562 SL assay. I never let a high-ranked pair be described as a confirmed
  target. It is candidate prioritization, full stop.
- **I marked pending as pending.** exp08's cluster runs were incomplete and its
  best folds did not yet clear the dependency-only floor. I reported it as a
  preliminary method with no superiority claim, and I did not fabricate a single
  metric to make the centerpiece look finished.
- **I kept the comparison fair.** The published methods (DDGCN, GRSMF, SL2MF,
  SLGNN) lead on ranking, and I said so plainly rather than hiding behind
  within-harness lifts. The honest causal claims came only from same-harness
  ablations (exp06 vs exp07 vs exp09).

This was uncomfortable at times — it is tempting to let the deep-learning model
be the triumphant ending. But a reflection forces the question: would I trust
this work if someone else had done it? Because I held the line on these points,
the answer is yes.

## 4. The hardest technical problems

The closed-vocabulary bottleneck was the defining engineering challenge. My local
STATE checkpoint identifies a perturbation through a one-hot lookup over a fixed
gene set, and only 16.3% of my 9,471-gene universe was in that vocabulary. For
the cold-start splits, 84% of genes would simply get no signal. My fix — freeze
the STATE backbone and replace its one-hot perturbation encoder with a trainable
adapter fed by ESM2 protein embeddings — was the moment the project went from
"interesting baseline study" to "a method." Mapping every gene into one
continuous coordinate system, regardless of whether it was ever screened, is what
gives the approach any path to held-out genes at all.

The other recurring difficulty was *leakage control*. Making CV2/CV3 valid meant
held-out genes had to be reached purely through `adapter(ESM2)` + frozen STATE,
and bag supervision had to be restricted to training genes that had a real
profile. It would have been easy to accidentally let a held-out gene see its own
observed response and quietly inflate the numbers. Designing the three-part loss
and the data flow so that this *could not* happen took more care than the model
architecture itself.

## 5. What did not work, and what I learned from it

Cold-start ranking (both genes unseen, CV3) remained unsolved by *every* signal I
tested. The observed-transcriptome lift vanished on CV3 because there is no
shockwave to read for an unscreened gene. Cross-cell-line selectivity improved
cold-start *classification* but left top-k *ranking* flat.

The decomposition experiment explained why, and this was probably my most
important negative finding: when I restricted CV3 to non-pan-essential pairs,
performance fell back toward chance. Most of the genome-wide co-dependency signal
is *pan-essentiality* — broadly essential genes are easy to flag — and
pair-specific synthetic lethality is the thin, biologically interesting residual
that hides underneath. That reframed the whole problem for me. The cold-start gap
is not a tuning issue; it is a precise scientific target: a generative model has
to learn the pair-specific signal that essentiality gravity obscures.

Learning to value a clean negative result as much as a positive one was a genuine
shift in how I think about research.

## 6. What this taught me

- **Evaluation design is the real research.** Choosing CV2/CV3 over CV1, building
  a strong floor, and auditing for confounds (the death-axis check, the
  pan-essentiality decomposition) shaped my conclusions more than any model did.
- **Simple baselines are not a formality.** The dependency-only floor was strong
  enough to embarrass a careless deep model, which is exactly why it was worth
  building first.
- **Foundation models are powerful but constrained.** A frozen backbone plus a
  small trainable adapter is data-efficient and respects a pretrained prior, but
  the closed vocabulary is a hard wall you have to engineer around deliberately.
- **Honesty is a method, not a mood.** Stating limitations openly (test-fold
  selection, single-fold status, adapter-label caveats) did not weaken the
  work — it made the parts that *do* hold up trustworthy.

## 7. Where I would go next

The natural next steps follow directly from the limitations I documented:

1. Complete cross-validated tuning of the generative framework with an evaluation
   that does *not* select on the test fold, so exp08 can be judged as a real
   comparison rather than a preliminary status.
2. Push beyond K562 — cross-lineage transfer, and eventually TCGA patient-context
   mapping from cell line to tumor dependency.
3. Supervise the model directly on the pair-specific residual that
   pan-essentiality otherwise hides, since that is where the genuinely useful SL
   signal lives.

## 8. Closing

The honest summary of this project is not "I discovered new synthetic-lethal
targets." It is: *I built and audited a leakage-controlled SL-pair ranking
pipeline, and I know exactly what each line of evidence can and cannot do right
now.* I proved the observed transcriptome encodes a gene's partners when one is
anchored, I built a generative framework that can in principle reach the 83.7% of
genes outside the foundation model's vocabulary, and I mapped the precise place
where the hard, unsolved residual lives. For a problem this large and this prone
to self-deception, knowing the boundary of my own claims feels like the most
durable thing I produced.
