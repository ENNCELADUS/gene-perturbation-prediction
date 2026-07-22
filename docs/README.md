# Research Vault: Generalizable Synthetic-Lethality Discovery

**Status:** general SL research contract and claim-level acceptance criteria
established · Phase 0 effect-size/eligibility/estimator registration pending ·
related-work review complete · active experiment roadmap established · exp05
K562 backbone established · first HCT116 frozen-backbone transport audit closed
negative · Horlbeck K562 GI acquired and coverage-audited · immediate two-track
execution plan set (T1 K562 Bridge-A mechanism kill-test, T2 DepMap GeneEffect
generalization) · SOTA reproduction and contextual SL model not yet complete.
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
| Multi-cell-line virtual-cell backbone | Not started; current exp05 is K562-based. |
| HCT116 frozen K562-backbone transport | **Completed; negative.** No independent HCT116 GeneEffect signal; [closeout](results/exp05-hct116-frozen-backbone-transport.md). |
| Context-conditioned Bridge A / Bridge B | Not started. |
| Held-out-cell-line SL evaluation | Not started; requires eligible pairwise labels. |
| Measured-GI validation | K562 Horlbeck acquired and coverage-audited; model evaluation not started, and a non-K562 anchor remains to identify. |

## Immediate worklist

- [ ] **T1** — run the K562 Bridge-A-vs-Horlbeck mechanism kill-test on the frozen
  exp05 backbone over the covered pairs (development diagnostic, not the formal
  MECHANISTIC verdict); [plan](specs/2026-07-22-k562-mechanism-and-geneeffect-generalization-plan.md).
- [ ] **T2** — run DepMap leave-one-cell-line-out GeneEffect generalization scored
  on the differentially-essential slice (single-gene backbone transfer, not
  cross-cell-line SL); [plan](specs/2026-07-22-k562-mechanism-and-geneeffect-generalization-plan.md).
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
