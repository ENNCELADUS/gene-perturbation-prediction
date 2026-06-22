# Paper Self-Review — Transcriptome-Encoded SL Manuscript

Adversarial five-dimension review per `research-paper-writing/references/paper-review.md`.
Each item: `pass` / `needs revision` / `needs new experiment`. Known reject risks
from spec Section 8 are answered explicitly.

## 1. Contribution
- New knowledge: observed perturbation transcriptome encodes SL partners when one
  is anchored (C3); pan-essentiality decomposition isolates the hard residual (C5). **pass**
- Meaningful failure case: cold-start SL (CV3) is the real, unsolved frontier, not
  a trivial case. **pass**
- Non-obvious idea: generating the response via a frozen virtual-cell model + ESM2
  adapter to break closed vocabulary is novel for SL. **pass**
- Novelty type present: new finding (C3), new framework (CNEW), new analysis (C5). **pass**

## 2. Writing Clarity
- Reproducible method: Method §3.1–3.5 give dims, frozen/trainable split, 3-part
  loss, eval protocol; appendix points to the extraction script. **pass**
- Module motivation explicit: each Method subsection states motivation→design→advantage. **pass**
- Terminology consistent: sweep clean ("dependency/essentiality ranking" vs "SL
  candidate prioritization"; C vs D stable). **pass**

## 3. Experimental Strength
- Honest strengths + failures: CV2 lift reported as real; CV3 collapse and
  below-floor exp08 reported plainly. **pass**
- Absolute performance: exp08 preliminary and below floor — addressed by framing as
  method contribution with preliminary results (F1/F2/L1), not a win. **pass (scope-limited)**

## 4. Evaluation Completeness
- Ablations: feature-set ablation (dependency vs +transcriptome vs +selectivity) is
  the core ablation; identical harness/seeds/splits. **pass**
- Baselines/comparators: no published-method leaderboard comparison (exp10 DDGCN
  unfilled). Answered by F7 — we make no SOTA/leaderboard claim and use a strong
  internal dependency-only floor as the bar. **pass (claim scoped accordingly)**
- Metrics standard: AUROC/AUPR/F1 + NDCG/MAP@k per the benchmark's per-anchor
  protocol. **pass**

## 5. Method Design Soundness
- Realistic setting: per-anchor ranking on a real SynLethDB-derived K562 benchmark. **pass**
- Hidden defects: exp08 best-epoch selection reads the test fold — disclosed in
  Method §3.5 and Experiments §method (L2), results marked preliminary. **pass (disclosed)**
- Net benefit: the framework's value is architectural reach to OOV genes; benefit
  not yet empirically demonstrated, stated as such. **pass (honest)**

## Spec Section 8 reject-risk ledger
1. exp08 incomplete / below floor → preliminary framing, F1/F2/L1/L2. **answered**
2. exp05 AIVC Adamson void → excluded from all claims. **answered**
3. exp03 Adamson sweep contamination → L6/F3, flagged exploratory in Tab. foundation. **answered**
4. No published comparator → F7, no SOTA claim. **answered**
5. Coverage dilution → covered-pair slices noted (L5). **answered**
6. CV3 cold-start unsolved → C6/L4, stated as open frontier. **answered**
7. Label validity → benchmark-adapter language throughout (L8/F4/F5). **answered**

## Outstanding (recorded as future work, not blocking the draft)
- Complete 5-fold exp08 tuning with a non-test-fold selection protocol (`needs new experiment`).
- Cross-lineage / patient-context transfer (`needs new experiment`).
Both are listed in the Conclusion's future-work paragraph.

**Verdict:** no unresolved `needs revision` items in the prose; the two
`needs new experiment` items are out of scope for a drafting pass and are
disclosed as limitations/future work.
