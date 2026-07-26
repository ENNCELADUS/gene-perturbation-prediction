# Research Vault: Generalizable Synthetic-Lethality Discovery

**Status:** general SL research contract and claim-level acceptance criteria
established · Phase 0 effect-size/eligibility/estimator registration pending ·
related-work review complete · active experiment roadmap established · exp05
K562 backbone established · first HCT116 frozen-backbone transport audit closed
negative · Horlbeck K562 GI acquired and coverage-audited · T1 K562
Bridge-A-vs-Horlbeck mechanism kill-test completed **negative** (composition
mechanism paused for redesign) · **T2 few-shot cross-cell-line GeneEffect backbone
now the active near-term development focus, execution under way**: Phase A
(data-audit/manifest freeze, amended 2026-07-26) and Phase B (Tx1-3B basal
embeddings) **complete**; Phase C (ST response model) code-complete with a
training run in progress and no checkpoint yet; Phase D (rebuilt hybrid head)
partially built, training runner under construction; no Phase 3 result yet ·
SOTA reproduction and contextual SL model not yet complete.
**Goal:** build a virtual-cell SL discovery model that competes with the Feng2024
SOTA for genes withheld from SL-pair/graph training and separately generalizes to
cancer cell lines excluded from training.

## The problem

```text
Discover synthetic-lethal gene pairs for unseen genes and unseen cancer cell-line
contexts, without using the SL graph to construct features.
```

The program has two non-interchangeable evaluation tracks:

1. **General gene-pair benchmark:** the official Feng2024/SynLethDB-derived
   9,845-gene benchmark, compared with SLMGAE, KR4SL, and the official model zoo.
   CV2/CV3 test unseen genes.
2. **Cross-cell-line generalization:** a context-conditioned score `s(a,b | c)`
   evaluated on untouched held-out cancer cell lines.

The main Feng2024 benchmark is not cell-line-specific. It cannot establish
cross-cell-line generalization. K562 is the current virtual-cell backbone and one
mechanistic validation context, not the scope of the target model.

## The idea

Compose a perturbation-response-trained virtual cell into a genuine pairwise
interaction:

| Bridge | Mechanism |
| --- | --- |
| **A — counterfactual co-dependency** | Simulate loss of `a` in cell line `c`, then predict whether `b` becomes more essential; symmetrize both directions. |
| **B — virtual double knockout** | Predict joint fitness in `c` and subtract an explicit additive or min/HSA non-interaction null. |

The context-free component `q(a,b)` is evaluated against Feng2024. The contextual
increment `s(a,b | c) - q(a,b)` must add held-out-cell-line information beyond
gene marginals, cell-line identity, and direct context-free transfer.

## Reading order

| Doc | Role |
| --- | --- |
| [`01-blueprint.md`](01-blueprint.md) | **Contract.** General task, two evaluation axes, mechanism, scope, and locked decisions. |
| [`02-acceptance-criteria.md`](02-acceptance-criteria.md) | **Bar.** BEAT-SOTA, CELL-LINE-GENERALIZATION, SPECIFICITY, MECHANISTIC, and INTEGRITY. |
| [`03-literature-review.md`](03-literature-review.md) | **Related work.** Feng2024 SOTA, closest inductive methods, virtual-cell limitations, and measured-GI evidence. |
| [`04-roadmap.md`](04-roadmap.md) | **Active experiment plan.** Data-contract freeze through SOTA, held-out-cell-line, and measured-GI evaluation. |

## Current status

| Stage | Status |
| --- | --- |
| Contract (`01`) + acceptance criteria (`02`) | **Claim structure established; Phase 0 registrations pending.** |
| Related-work review (`03`) | **Complete.** Needs only live alignment as new evidence arrives. |
| Experiment roadmap (`04`) | **Active.** |
| Official Feng2024 contract/parity audit | Pending. |
| Reproduce SLMGAE, KR4SL, KG4SL and best official comparator | Pending. |
| Context-free composition score `q(a,b)` | Not started. |
| Multi-cell-line data-role audit | Not started. |
| Multi-cell-line virtual-cell backbone | **Active focus, execution under way.** Few-shot cross-cell-line GeneEffect backbone `F(X_c, c, g)`: Tx1-3B-conditioned ST + rebuilt hybrid head ([design](specs/2026-07-23-tx1-st-geneeffect-backbone-design.md)); fixed 28 train / 5 validation / 9 test split over Tahoe DMSO-basal DepMap lines plus the 4 Perturb-seq anchors (validation drawn only from the non-anchor training pool). Phase A + Phase B complete; Phase C code-complete, training in progress, no checkpoint yet; Phase D partially built, head-training runner under construction. No Phase 3 result yet. |
| HCT116 frozen K562-backbone transport | **Completed; negative.** No independent HCT116 GeneEffect signal; [closeout](results/exp05-hct116-frozen-backbone-transport.md). |
| Context-conditioned Bridge A / Bridge B | Not started; the K562 Bridge A **mechanism** was tested and is negative (see measured-GI row). |
| Held-out-cell-line SL evaluation | Not started; requires eligible pairwise labels. |
| Measured-GI validation | K562 Horlbeck acquired and coverage-audited; **K562 Bridge-A kill-test completed, negative** ([result](results/exp05-bridge-a-horlbeck-kill-test.md)); a non-K562 anchor remains to identify. |

## Immediate worklist

- [x] **T1** — K562 Bridge-A-vs-Horlbeck mechanism kill-test completed on the frozen
  exp05 backbone over the 83,028 covered pairs: **negative** (|Spearman| < 0.01;
  AUROC(s_A -> strong-SL) approximately 0.52, below the single-gene floor). The
  composition mechanism is paused for redesign and not extended across contexts;
  [result](results/exp05-bridge-a-horlbeck-kill-test.md).
- [ ] **T2 (active focus, execution under way)** — build the few-shot
  cross-cell-line GeneEffect backbone `F(X_c, c, g)`: fixed 28 train / 5
  validation / 9 test split (four CRISPRi Perturb-seq lines plus 24 Tahoe
  DMSO-basal∩DepMap lines train; 5 more Tahoe lines, lineage-stratified and
  drawn only from the non-anchor training pool, validate; 9 lineage-stratified
  Tahoe lines are the untouched unseen test) plus k-shot adaptation, scored on
  the differentially-essential slice (587 genes) against copy-K562 / mean /
  nearest-line / lineage-only / CCLE-bulk / pseudobulk-basal baselines with a
  few-shot curve (single-gene backbone transfer, task-data-held-out, not
  cross-cell-line SL). Phase A (amended 2026-07-26) and Phase B are complete;
  Phase C is code-complete with training in progress and no checkpoint yet;
  Phase D is partially built and its head-training runner is under
  construction; [design](specs/2026-07-23-tx1-st-geneeffect-backbone-design.md).
- [ ] Freeze and verify the official Feng2024 folds, labels, candidates, and
  `cal_metrics` contract.
- [ ] Reproduce the strong SOTA under that identical official harness.
- [ ] Audit cell-line-resolved SL/GI datasets and freeze at least two training and
  two untouched evaluation cell lines, or mark the axis not evaluable. Treat
  those as named-context evidence unless a separately powered line-level design
  supports a population claim.
- [x] Close the first HCT116 frozen-backbone transport audit as negative
  single-gene evidence; retain K562 exp05 as prior evidence only.
- [ ] Extend the forward model to the declared multi-cell-line interface.
- [ ] Build and ablate contextual Bridge A and Bridge B.
- [ ] Run formal Feng2024 CV2/CV3 evaluation and non-pan-essential controls.
- [ ] Run untouched held-out-cell-line evaluation.
- [ ] Validate measured GI in K562 and at least one non-K562 context before making
  a multi-cell-line mechanistic claim.
- [x] Acquire and provenance-check the Horlbeck K562 GI map and freeze its exp05
  coverage bound; [result](results/horlbeck-k562-exp05-coverage.md).

## Where things live

- **Benchmark:** Feng2024 `SL_benchmark` checkout at `data/SL_benchmark`.
- **Current forward-model backbone:** `src/aivc_model/`.
- **SL floors and evaluation machinery:** `src/sl_benchmark_baseline/` and
  `src/ddgcn/`.
- **Prior dependency/feature machinery:** `src/dependency_baseline/`.
- **Dataset cards:** [`data/`](data/).
- **Completed evidence:** [`results/`](results/) plus retired-program artifacts
  under `docs/archive/` and `ideaspark_run/`.

## Conventions

`01` and `02` are frozen and may only be changed in place when the user explicitly
changes the research program. The vault is a current snapshot, not a changelog.
Results enter `docs/results/` only after analysis completes. Status must agree
across this file, `04-roadmap.md`, and root `README.md`. Use plain GitHub Markdown,
relative links, no YAML frontmatter, and bold-key status lines.
