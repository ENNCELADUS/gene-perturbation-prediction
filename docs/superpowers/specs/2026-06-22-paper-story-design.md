# Paper Story Design — Transcriptome-Encoded Synthetic Lethality via a Virtual-Cell Foundation Model

- **Date:** 2026-06-22
- **Status:** Approved (story design; not an implementation plan)
- **Venue framing:** Computational-biology journal (Bioinformatics / Cell Systems / Genome Biology style)
- **Scope:** Defines the paper *narrative*, the claims we may make, and the claims we may not. Every quantitative statement is traced to a local artifact. No new results are invented here.
- **Grounding:** Built from `docs/experiment/01–10`, `docs/presentation/`, `docs/data/`, `CONTEXT.md`, `CLAUDE.md`, and `results/experiments/` artifacts, audited by three inspection agents (code-explorer, spec-miner, mle-reviewer).

> Hard rule for anyone writing from this design: if a number is not in the Numbers Ledger (Section 6), it does not go in the paper. If a claim is in Section 5 (Forbidden), it never gets written, even under reviewer pressure.

---

## 1. Title and the close-loop arc

**Working title:**
*"The Transcriptional Shockwave Carries the Partner: a Virtual-Cell Foundation Model for Synthetic-Lethal Co-Dependency Prediction Beyond Screened Genes."*

**Headline contribution:** the AIVC/STATE + ESM2-adapter end-to-end framework (exp08) — a method that, by *generating* a gene's perturbation response, is architecturally able to score SL pairs where neither gene was ever screened. Results are explicitly **preliminary** (see L1, F1).

### The arc (method-centered, biologically closed loop)

1. **Premise (clinical hook).** Synthetic lethality is a validated route to selective cancer therapy (PARP/BRCA). Context-specific SL partners occupy a ~200M-pair space that wet-lab screens cannot cover, so predictive prioritization is required.
2. **Proof-of-concept (exp07).** The *observed* perturbation transcriptome encodes a gene's SL partners — but only for genes actually perturbed in the lab (the CV2 one-gene-anchored regime). You cannot observe a transcriptome for a gene nobody screened.
3. **The gap.** That is the cold-start wall (CV3, both genes unseen): no observed shockwave → no signal (exp07 CV3 ranking collapses).
4. **The idea (contribution).** A virtual-cell foundation model (AIVC/STATE) can *generate* the perturbation response for any gene from its protein-language-model identity (ESM2), including unscreened genes. If the observed shockwave encodes SL partners, a *predicted* shockwave should too — and it reaches unseen genes in principle.
5. **The method (exp08).** ESM2 gene embedding → trainable PertAdapter → frozen STATE backbone → predicted response bag → symmetric pair head; 3-part loss (SL BCE + token distillation + real-bag supervision). The adapter exists to break STATE's closed vocabulary (only 16.3% of the SL universe is in-vocab).
6. **Honest current state.** Preliminary: best folds do not yet clear the exp06 floor; full 5-fold tuning ongoing. Stated as the status of a newly proposed method, not a result.
7. **Decomposition closes the loop (exp09).** Why is the residual hard? Most genome-wide co-dependency signal is pan-essentiality "gravity"; pair-specific SL is the thin residual — precisely what a generative virtual-cell model must learn to reach.

### Foundation that makes the leap credible (exp01–03)

Before the leap from "predicted transcriptome → SL partner," the paper establishes that "transcriptome → dependency" holds at all: a gene's perturbation response predicts its *own* DepMap dependency (exp01), the signal is real transcriptomic structure rather than a generic death axis (exp02), and single-cell distribution embeddings carry it across datasets (exp03, exploratory).

---

## 2. Main claims (each traceable to an artifact)

Claims are stated at the strength the artifact supports — no further.

**C1 — Perturbation transcriptomes predict a gene's own dependency, and the signal is real, not generic death.**
- exp01: pseudobulk Δ-expression → DepMap GeneEffect; Replogle K562 5-fold CV Spearman ≈0.485 (pca50_ridge); same-cell-line Adamson transfer AUROC 0.886 (GE<−1.0), Spearman 0.500.
- exp02 audit: NAR death-signature alone Spearman 0.244 vs transcriptome 0.494; NAR-residualized transcriptome retains 0.503.
- Role: foundation/motivation, same-cell-line. Adamson framed as a transfer **gate**, not broad validation.

**C2 — Single-cell distribution embeddings carry dependency signal across datasets (foundation, exploratory).**
- exp03: scVI128 frozen-GMM + Ridge; Replogle CV Spearman ≈0.44; Adamson Spearman ≈0.666 / AUROC 0.911.
- Role: explicitly an Adamson-guided sweep, **not** an untouched held-out claim (see L6, F3).

**C3 (PROOF-OF-CONCEPT) — The observed perturbation transcriptome encodes a gene's SL partners, demonstrably when one partner is anchored.**
- exp07 CV2 (one gene held out), 5-fold complete, identical harness/seed as the floor: B_transcript vs B → AUROC 0.704→0.751, NDCG@10 0.042→0.094 (~2.2×), MAP@10 0.034→0.079. Covered-pair slice confirms the lift concentrates where real embeddings exist.
- Role: the empirical evidence that licenses the headline method. A real positive result; not the headline.

**C4 — A leakage-controlled dependency-only floor exists, and the easiest split is degree-gameable.**
- exp06: 5 swap-invariant GeneEffect features; CV2 XGB AUROC 0.704; degree probe wins CV1 (NDCG@10 0.197) → CV1 is topology-gameable; CV2/CV3 are the honest surfaces.
- Role: evaluation bar / rigor backbone.

**CNEW (HEADLINE — method, results preliminary) — A virtual-cell foundation-model framework (AIVC/STATE + ESM2 adapter) for SL-pair prioritization that, by generating perturbation responses, is architecturally able to score gene pairs where neither gene was ever screened.**
- exp08: ESM2 1280-d → PertAdapter → frozen STATE backbone → predicted bag → MeanStdPool → SymmetricPairHead; 3-part loss (SL BCE + token distillation + real-bag supervision). Adapter motivated by STATE closed vocab (16.3% in-vocab).
- Contribution = the framework + the OOV-breaking adapter + the 3-part loss design.
- Results: **preliminary, do not yet beat the floor** (L1, F1, F2).

**C5 — Most genome-wide co-dependency signal is pan-essentiality; pair-specific SL is the hard residual.**
- exp09: cross-line selectivity B_xcl lifts CV2 AUROC 0.704→0.742, CV3 0.596→0.645; but non-pan-essential CV3 slice drops to AUROC 0.583 / AUPR 0.490.
- Role: the analysis core; turns a confound into a finding and motivates the generative approach.

**C6 — Cold-start (both genes unseen) is the open frontier and the target of the contribution.**
- exp07 CV3 NDCG@10 0.002→0.001; exp09 CV3 NDCG@10 flat; exp08 e2e DL best folds below floor.
- Role: the problem the headline method targets; stated as an honest open problem, not a hidden failure.

---

## 3. Limitations (stated in the paper, not hidden)

- **L1 — The headline method does not yet beat its own floor.** exp08 best folds (CV2 AUROC 0.667) sit below the exp06 dependency-only floor (0.704); runs are incomplete (CV2 2/5, CV3 3/5 folds). Reported as preliminary; no 5-fold means claimed.
- **L2 — exp08 model selection touches the test fold.** Best-epoch selection reads the test fold (selection-matched to the SynLethDB protocol), so it is not a strict embedding-only ablation vs exp06. Stated openly.
- **L3 — Foundation-model vocabulary is the core bottleneck.** Local STATE checkpoint is closed-vocab one-hot; only 16.3% (1,542/9,471) of the SL universe is in-vocab. The ESM2 adapter is the proposed fix; its benefit is not yet demonstrated to beat baselines.
- **L4 — Cold-start ranking is unsolved by every signal tested.** Transcriptome (exp07), selectivity (exp09), and the DL model (exp08) all fail CV3 top-k. Honest open problem.
- **L5 — Coverage dilution.** gwps covers 64% of genes but only 51% of pairs are both-covered; full-universe ranking numbers are diluted. Covered-pair slices reported alongside.
- **L6 — exp03 transfer is Adamson-guided, not held-out.** The 0.666 number came from a sweep optimizing Adamson metrics; framed as exploratory foundation, never as held-out generalization.
- **L7 — Same-cell-line / single-cell-line scope.** K562 only; Adamson is same-cell-line transfer, not cross-lineage. No patient/in-vivo validation.
- **L8 — Label is a benchmark adapter, not a K562 SL assay.** D = SynLethDB Rand-1:1 pairs; negatives are unconfirmed non-SL. The K562-specific SL split file is absent from the checkout.

## 4. (reserved — merged into Section 3)

## 5. Forbidden claims (hard guardrails — never written, even under reviewer pressure)

- **F1 — Do NOT claim exp08/AIVC beats baselines or is state-of-the-art.** It is preliminary and currently below floor. No "outperforms," no "achieves SOTA."
- **F2 — Do NOT report exp08 single best folds as full CV results.** Always mark as selected folds, incomplete, results pending.
- **F3 — Do NOT present exp03's 0.666 Adamson number as held-out generalization.** It was an Adamson-guided sweep.
- **F4 — Do NOT equate gene essentiality with synthetic lethality**, and do NOT call DepMap GeneEffect a single-cell death label.
- **F5 — Do NOT call a Rand negative a confirmed non-SL pair**, and do NOT call any result a *validated* K562 SL target. Use "SL candidate prioritization / benchmark-adapter ranking."
- **F6 — Do NOT use CV1 numbers as evidence of unseen-gene generalization** (degree-gameable). Generalization claims come only from CV2/CV3.
- **F7 — Do NOT cite the deck's DDGCN/GRSMF/SL2MF rows as this repo's results** — they are cross-pipeline literature references, not our reproduction (exp10 full run is unfilled).
- **F8 — Do NOT align CRISPRa (Norman) with knockout dependency** without a modality caveat.
- **F9 — Do NOT claim a closed mechanistic loop** — the transcriptome→SL link is *predictive signal*, not demonstrated causal mechanism.

## 6. Numbers ledger (the only numbers allowed in the paper)

Every number traced to its source doc/artifact. Anything not here must not be cited.

### Foundation (exp01–03)
| Value | Metric | Source |
|---|---|---|
| 0.485 | exp01 Replogle CV Spearman (pca50_ridge) | `docs/experiment/01_...md` |
| 0.500 | exp01 Adamson Spearman (pca50_ridge) | `01_...md` |
| 0.886 | exp01 Adamson AUROC (GE<−1.0) | `01_...md` |
| 0.244 / 0.494 / 0.503 | exp02 Spearman: NAR-only / best pseudobulk / NAR-residualized | `02_...md` |
| 0.666 / 0.911 | exp03 scVI128 GMM Ridge Adamson Spearman / AUROC (Adamson-guided sweep) | `03_...md` |

### SL benchmark floor + signals (exp06–09)
| Value | Metric | Source |
|---|---|---|
| 0.704 / 0.732 | exp06 CV2 XGB (Model B) AUROC / AUPR | `06_...md` |
| 0.042 / 0.034 | exp06 CV2 Model B NDCG@10 / MAP@10 | `06_...md` |
| 0.596 | exp06 CV3 Model B AUROC | `06_...md` |
| 0.197 | exp06 CV1 degree-probe (Model C) NDCG@10 | `06_...md` |
| 0.751 / 0.094 / 0.079 | exp07 CV2 B_transcript AUROC / NDCG@10 / MAP@10 | `07_...md` |
| 0.630 / 0.001 | exp07 CV3 B_transcript AUROC / NDCG@10 | `07_...md` |
| 64.09% / 51.17% | exp07 gwps gene coverage / both-covered pair coverage | `07_...md` |
| 0.742 / 0.645 | exp09 CV2 / CV3 B_xcl AUROC | `09_...md` |
| 0.583 / 0.490 | exp09 CV3 B_xcl non-pan-essential AUROC / AUPR | `09_...md` |

### Method (exp08 — preliminary, single folds, mark as such)
| Value | Metric | Source |
|---|---|---|
| 16.3% (1,542/9,471) | STATE in-vocab fraction of SL universe | `08_...md`, deck slide 9 |
| 0.6667 / 0.0050 | exp08 Phase2 BCE CV2 best-fold AUROC / NDCG@10 (NOT a 5-fold mean) | `08_...md` |
| 0.5866 / 0.0015 | exp08 Phase2 BCE CV3 best-fold AUROC / NDCG@10 (NOT a 5-fold mean) | `08_...md` |

> Benchmark provenance: Feng et al., *Nature Communications* 15, 9058 (2024); SynLethDB-derived; K562-filtered 9,471-gene universe; per-anchor `cal_metrics` ranking protocol. Full official benchmark = 9,845 genes / 35,913 positive pairs.

## 7. Section-to-artifact map (figure/section plan)

| Paper section | Backed by | Artifact path |
|---|---|---|
| Intro / clinical hook | presentation deck motivation | `docs/presentation/final-presentation.md` |
| Foundation: transcriptome→dependency | exp01, exp02, exp03 | `docs/experiment/01–03_...md`; `results/experiments/03_...` |
| Benchmark + honest evaluation ladder | exp06 | `results/experiments/06_..._mvp/official_metrics_*` |
| Proof-of-concept: observed transcriptome → SL partner | exp07 | `results/experiments/07_..._augmented/augmented_no_flag/summary.csv` |
| Method (headline): AIVC/STATE + ESM2 adapter | exp08 + `src/sl_dl_model/`, `src/aivc_model/` | `results/experiments/08_..._state_dl/` (preliminary) |
| Analysis: pan-essentiality decomposition | exp09 | `results/experiments/09_..._selectivity/run/summary.csv` |
| Limitations / frontier | L1–L8, exp08/exp07 CV3 | this doc Section 3 |

## 8. Reviewer attack surface (pre-empted by design)

Ranked, from the mle-reviewer audit; each mapped to where the paper defuses it:
1. exp08 incomplete / below floor → defused by L1, L2, F1, F2 (framed as preliminary method contribution, not a win).
2. exp05 AIVC Adamson transfer void (0 feature overlap) → **excluded from claims entirely**; not cited as a transfer result.
3. exp03 Adamson sweep contamination → L6, F3 (exploratory foundation only).
4. No published-method comparator (exp10 DDGCN unfilled) → F7; we do not claim SOTA or leaderboard wins.
5. Coverage dilution → L5 (covered-pair slices reported alongside).
6. CV3 cold-start unsolved → C6/L4 (stated as the open frontier the method targets).
7. Label validity / benchmark-adapter distance → L8, F4, F5 (benchmark-adapter language throughout).

## 9. Explicitly out of scope / not claimed

- exp04 (linear predicted-B pilot) and exp05 (AIVC STATE A→B→C): implementation/contract only, **no completed metrics** — cited as pipeline context, never as results.
- exp10 (DDGCN reproduction): full-run table unfilled in docs — not cited as our result.
- Any cross-lineage, patient, or in-vivo validation: not performed.

