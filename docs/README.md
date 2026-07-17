# Research Vault: Virtual-Cell Composition for Synthetic-Lethality Discovery

**Status:** contract + acceptance criteria established · related-work review complete ([`03`](03-literature-review.md), 2026-07-17) · **`01`/`02` repositioning pending user approval** (see `03` §6) · roadmap pending · forward model (exp05) in progress.
**Goal:** a virtual-cell SL predictor whose *mechanism* (double-KO + explicit null, validated against measured epistasis) beats the strong SOTA (**SLMGAE, KR4SL**) on the cold-start splits. The graph-free/inductive framing is shared with prior art (CILANTRO-SL); see [`03`](03-literature-review.md).

**Supersedes** the retired "cell-fate outcome dynamics" direction (evidence preserved
under `docs/archive/` and `ideaspark_run/`, not edited).

## The problem

```text
Rank a gene's synthetic-lethal partners for genes no SL screen or SL graph
has ever seen.
```

The strongest SL predictors — **KG4SL, SLMGAE**, and the wider Feng2024 zoo — are
*transductive* graph/knowledge-graph models. Their signal is SL-graph topology, so
they are strong on seen genes (CV1) but **break on unseen genes (CV2 → CV3)** and
cannot reach genes with no curated SL edges. **No method wins the inductive,
graph-free, cold-start regime.** That is the program.

## The idea

A **virtual cell** (exp05: Arc STATE + ESM2 identity, open-vocabulary) predicts a
single gene's knockout response → its **DepMap GeneEffect** (fitness), and already
ranks single-gene dependency. The bet: **compose that single-gene fitness model into
a genuine pairwise interaction** — a signal read from gene *features*, hence natively
inductive, hence able to score pairs no SL graph has seen. Single-gene GeneEffect
alone is the ~0.70-AUROC floor, not the method.

| Bridge | Mechanism |
|---|---|
| **A — counterfactual co-dependency** | simulate loss of `a`, then predict `b`'s GeneEffect in the `a`-lost state; SL = the dependency spike. |
| **B — virtual double-knockout** | forward the joint `a+b` perturbation → joint fitness; SL = interaction residual vs. an explicit additive/min null. |

## Reading order

| Doc | Role |
|---|---|
| [`01-blueprint.md`](01-blueprint.md) | **The contract.** Problem, mechanism, the two bridges, hypotheses, claim boundaries, and §10 Locked Decisions. Read first. |
| [`02-acceptance-criteria.md`](02-acceptance-criteria.md) | **The bar.** What "more powerful and accurate" means numerically — BEAT-SOTA, PAIR-SPECIFIC, MECHANISTIC, INTEGRITY. Frozen before evidence. |
| [`03-literature-review.md`](03-literature-review.md) | **Related work.** *Pending rewrite* — positioning vs. KG4SL/SLMGAE and the virtual-cell / SL-prediction landscape (the next step). Currently describes the retired program. |
| [`04-roadmap.md`](04-roadmap.md) | **What happens next.** *Pending rewrite* — the experiment plan. Currently describes the retired program. |
| [`superpowers/specs/2026-07-17-virtual-cell-sl-composition-design.md`](superpowers/specs/2026-07-17-virtual-cell-sl-composition-design.md) | The design doc that seeded this contract. |

## Where the project stands

| Stage | Status |
|---|---|
| Contract (`01`) + acceptance criteria (`02`) | **Established.** |
| Related-work review (`03`) | **Done (2026-07-17).** 4-agent review; found CILANTRO-SL (near prior art), KG4SL weak at cold-start, Jost≠epistasis. `01`/`02` repositioning proposed in `03` §6, **pending approval**. |
| Roadmap / experiment plan (`04`) | Pending (after `01`/`02` repositioning). |
| exp05 forward model (response → GeneEffect) | **In progress** (`src/aivc_model/`, branch `codex/exp05-k562-fixed-pool`). The backbone. |
| Reproduce SLMGAE, KR4SL, KG4SL under the K562 `cal_metrics` harness | Pending — SLMGAE/KR4SL are the real bars; KG4SL a weak reference. |
| Bridge A / Bridge B implementation | Not started. |
| Head-to-head benchmark eval (CV2/CV3 + non-pan-essential control) | Not started. |
| Mechanistic epistasis validation (Horlbeck 2018 + Adamson UPR) | Not started; Horlbeck to acquire. |

## Roadmap

- [x] **Related-work review** — done ([`03-literature-review.md`](03-literature-review.md); detail in [`results/literature-review-2026-07/`](results/literature-review-2026-07/)).
- [ ] **Reposition `01`/`02`** per `03` §6 (novelty, north-star, epistasis-data, Bridge-B risk) — pending user approval.
- [ ] **Reproduce the SOTA target** — SLMGAE, KR4SL, KG4SL under the identical K562 `cal_metrics` harness (SLMGAE/KG4SL ship in `data/SL_benchmark`; SLMGAE/KR4SL are the real bars, KG4SL a weak reference).
- [ ] **Finalize exp05** — single-gene response → GeneEffect (the composition backbone).
- [ ] **Build Bridge A + Bridge B** — the two composition operators.
- [ ] **Head-to-head eval** — CV2/CV3 ranking vs. SOTA, with the non-pan-essential control.
- [ ] **Mechanistic validation** — `s_B` vs. measured epistasis (Horlbeck 2018 K562 GI, to acquire; Adamson UPR local check). Jost dual-sgRNA is single-gene, not epistasis.
- [ ] **Paper** — conference/workshop (venue/deadline to confirm).

## Where things live

- **Design + specs** — [`docs/superpowers/specs/`](superpowers/specs/).
- **Retired-program evidence** — write-ups under `docs/archive/` (untracked, gitignored) and the eleven review memos under [`ideaspark_run/`](../ideaspark_run/). Not edited. The retired code (`src/dependency_baseline/`, `src/aivc_model/`, `src/sl_benchmark_baseline/`, `src/sl_dl_model/`, `src/ddgcn/`) still runs and is reused.
- **Dataset cards** — [`docs/data/`](data/). **Meeting notes** — [`docs/discussion/`](discussion/).
- **Benchmark** — the Feng2024 SL_benchmark checkout at `data/SL_benchmark` (12-model zoo incl. `kg4sl`, `slmgae`).

## Conventions

Contract docs (`01`, `02`) are **frozen**: change them in place, never by writing a
new file. Live numbers live only in `results/`, and no number enters the vault without
a source pointer. Results enter the docs only after the analysis actually runs — a
planned number is not a number. Plain GitHub markdown, relative links, no YAML
frontmatter, no wikilinks, status as **Status:** bold-key lines.
