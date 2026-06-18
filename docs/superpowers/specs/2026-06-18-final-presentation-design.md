# Final Oral Presentation — Design Spec

**Date:** 2026-06-18
**Deliverable:** 12-slide deck + speaker notes for a 10–12 minute final oral
presentation on the K562 synthetic-lethality (SL) project.
**Output format:** Markdown outline with per-slide speaker notes. English on
slides, Chinese (中文) in speaker notes (bilingual).
**Audience:** Mixed ML + biology; depth balanced between the two.

## 1. Goal & Constraints

- **Length:** 10–12 minutes, 12 slides (~1 min/slide average; centerpiece slides
  get ~1.5 min).
- **Storyline:** A closed-loop narrative that 串联 (threads together) every
  experiment exp01–exp09, with the weight on the SL parts (exp06–09) and
  especially the end-to-end DL model (exp08).
- **Spine:** *Representation-as-bridge* — Stage 1/2 transcriptome→GeneEffect
  regression work produces a dependency-aware representation that is carried into
  the Stage 3 SL-pair task.
- **Close:** *Honest-framing callback* — return to the opening SL framing and
  restate that this is SL-pair classification/ranking on a benchmark adapter, not
  validated SL target discovery.
- **Framing of exp08:** Methods-forward capstone. exp08 has **no empirical results
  yet** — code + unit tests are complete, CV2/CV3 cluster gates are pending. The
  talk presents exp08 as the designed centerpiece and is honest about the pending
  status (marked "pending" everywhere) rather than inventing numbers.
- **exp09 has full results.** The CV1/CV2/CV3 × 5-fold official-metric run is
  complete (`results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run/summary.csv`,
  405 rows). It is presented with real numbers.
- **SOTA comparison.** The results scoreboard (Section 8) places our models
  alongside published SL-prediction methods (DDGCN, GRSMF, SL2MF) on the same
  Rand 1:1 CV1/CV2/CV3 F1 + NDCG@10 axes. All scoreboard values are reported to
  **4 decimal places**.

## 2. Narrative Structure (Approach A — chronological funnel)

Broad SL problem → narrow honest task framing → data choices → Stage 1/2
foundation (regression / representation) → Stage 3 SL-pair model ladder
(floor → observed transcriptome → e2e DL centerpiece) → parallel selectivity
route → honest close. The funnel shape motivates each next step; the
representation-as-bridge slide (slide 6) is the hinge between the regression arc
and the SL-pair arc.

**Slide budget discipline:** exp01+02 compress to one slide; exp03/04/05 compress
to one slide; exp08 gets two slides as the requested centerpiece. Nothing is
dropped, satisfying the "closed loop / 串联 all experiments" goal.

## 3. Slide-by-Slide Design

Numbers below are pulled from the experiment docs and result summaries and are the
concrete content slots. Pending experiments are flagged as pending, not fabricated.

| # | Title | Time | Asset | Core content |
| --- | --- | --- | --- | --- |
| 1 | Title / thesis | 0:30 | — | Project title; one-line thesis: rank candidate SL gene pairs in K562 by combining DepMap dependency, Perturb-seq response, and a frozen perturbation foundation model. |
| 2 | What is SL & the bottleneck | 1:00 | `SL_concept.png` | SL definition; BRCA/PARP clinical proof; ~200M pairs, context-dependent, can't brute-force → need computational ranking. |
| 3 | Honest task framing + benchmark | 1:00 | — | Feng et al. 2024 SynLethDB-derived benchmark; Rand 1:1; 9,471-gene K562 universe; CV1/CV2/CV3 = pair-held-out / one-gene-unseen / both-unseen. **SL-pair classification/ranking, NOT validated SL target discovery**; `Rand` negatives unconfirmed; CV2/CV3 are the honest surfaces. |
| 4 | Data sources & why these | 1:00 | — | DepMap CRISPRGeneEffect (Chronos, K562=ACH-000551); Replogle K562 gwps Perturb-seq (CRISPRi = LoF, modality-aligned with DepMap KO, 1.99M cells, 6,070 genes covered); ESM2-650M (held-out-gene generalization); Feng2024/SynLethDB for pair label `D`. Why K562: only line with deep Perturb-seq + DepMap. |
| 5 | Stage 1: observed transcriptome → dependency works | 1:00 | — | exp01 pseudobulk delta → PCA Ridge: Replogle CV ≈0.49 Spearman, Adamson transfer 0.50. exp02 audit: not just generic viability axis (NAR-only 0.244 vs 0.503 residualized). Signal is real and transcriptomic. |
| 6 | Stage 2: dependency-aware representation (the bridge) | 1:00 | — | Hinge slide. exp03 single-cell set learning: scVI128 + frozen-GMM distribution regression wins (Adamson Spearman 0.666, AUROC 0.911), beats attention MIL; HVG hurts. exp04 leakage-free predicted-B loop; exp05 frozen-STATE forward A→B→C. Takeaway: a STATE-based way to turn any perturbation into a dependency-aware embedding → carry into SL task. |
| 7 | Stage 3 begins: dependency-only floor (exp06) | 1:00 | — | Pair task reframe. Floor = two genes' GeneEffect scalars (5 swap-invariant features) → P(SL). CV2 XGB AUROC 0.704 / NDCG@10 0.042; CV3 collapses AUROC 0.596 / NDCG@10 0.002. Degree probe wins CV1 (NDCG@10 0.197) → CV1 gameable → judge on CV2/CV3. |
| 8 | Does observed Perturb-seq add lift? (exp07) | 1:00 | — | Augment exp06 with observed gwps response embeddings. Coverage crux: 64% per-gene but ~41% per-pair (59% hit fallback). Two tiers (mean-pool / frozen-scVI). Honest design: covered-pair diagnostic slice + with/without coverage flag. Negative result is publishable. |
| 9 | e2e DL centerpiece pt.1: problem & architecture (exp08) | 1:30 | `e2e_SL_DL.png` | Why exp08: local STATE is closed-vocab one-hot, only 16.3% of universe in-vocab (84% OOV). Fix: freeze 8-layer STATE Llama backbone, replace one-hot pert_encoder with trainable adapter fed by ESM2 (1280→328) → all 9,471 genes in one coordinate system. Arch: ESM2 → PertAdapter → frozen STATE → predicted bag → pooling → symmetric pair head. |
| 10 | e2e DL centerpiece pt.2: leakage-safe training & the bar | 1:30 | — | 3-part loss (SL BCE + adapter token-distill + real-bag supervision on covered-train genes only). Leakage rule: held-out genes reached purely via `adapter(ESM2)` + frozen STATE, never supervised → CV2/CV3 valid. Bar: beat exp06 (CV2 0.704 / CV3 0.596) with lift on covered-pair slice. **Status: code + unit tests complete; CV2/CV3 cluster gates pending.** |
| 11 | Parallel route: cross-cell-line selectivity (exp09) | 0:45 | — | No transcriptome: use DepMap across 1,208 lines — is KO(b) more lethal where gene_a is defective? Composite-OR defective call (mutation/CN/expression). Results: CV2 B_xcl AUROC 0.742 (+0.039 over B), NDCG@10 0.086 (+0.044); CV3 AUROC 0.645 (+0.050), NDCG@10 flat. Improves cold-start *classification* but not cold-start *ranking*. Non-pan-essential slice: lift shrinks but doesn't vanish on CV1/CV2; CV3 mostly essentiality-linked. |
| 12 | Closing the loop | 1:00 | — | Callback to slides 2–3. Built: an honest, leakage-controlled SL-pair ranking adapter, not target discovery. Recurring discipline: simple floors first, CV2/CV3 as the real bar, negative results count. Path to true context-specific SL: exp08 cluster results → exp09 selectivity → TCGA patient-context transfer. End on honest scope. |

## 4. Asset Inventory

- `docs/presentation/SL_concept.png` — slide 2 (SL concept figure).
- `docs/presentation/e2e_SL_DL.png` — slide 9 (exp08 architecture figure).
- **Results scoreboard table (Section 8)** — rendered as a **backup/appendix slide**
  after slide 12 (keeps the main deck at 12). The within-harness B_xcl-vs-B lift is
  also surfaced inline on slide 11; the SOTA-context caveat is mentioned on slide 12.
- No new figures required for the MVP deck. Optional later: a funnel/arc diagram
  for slide 6, and a CV1/CV2/CV3 results bar chart for slide 7.

## 5. Key Numbers (verified from docs)

- exp01: Replogle CV ≈0.49 Spearman (PCA Ridge/RF); Adamson transfer 0.50 Spearman,
  AUROC 0.886.
- exp02: NAR-score-only 0.244 vs best pseudobulk 0.494 vs NAR-residualized 0.503.
- exp03: scVI128 frozen-GMM distribution regression best Adamson Spearman 0.666,
  AUROC 0.911, AUPRC ~0.80, heldout Spearman ~0.64–0.67.
- exp06 (the bar): CV2 XGB AUROC 0.704 / AUPR 0.732 / NDCG@10 0.042; CV3 AUROC
  0.596 / NDCG@10 0.002; CV1 degree probe NDCG@10 0.197 (gameable).
- exp07: gwps coverage 64.09% per-gene → ~41% per-pair under independence.
- exp08: STATE closed-vocab one-hot, 2,024 genes; 16.3% of SL universe in-vocab
  (1,542 genes), 83.7% OOV. ESM2-650M = 1280-d → 328-d STATE token. **No results
  yet.**
- exp09: DepMap Public 26Q1, 1,208 GeneEffect lines, K562=ACH-000551; composite-OR
  clears n≥20 bar for 9,459/9,471 genes. **Full CV1/CV2/CV3 × 5-fold results
  complete.** B_xcl vs B: CV2 AUROC +0.039 (0.742 vs 0.704), NDCG@10 +0.044 (0.086
  vs 0.042); CV3 AUROC +0.050 (0.645 vs 0.596), but NDCG@10 flat (0.001 vs 0.002).
  Cross-cell-line selectivity improves classification on cold-start but does not fix
  cold-start ranking.

## 6. Honesty Guardrails (carried onto slides)

- Say "dependency prediction / essentiality ranking" for exp01–05; "SL candidate
  ranking / SL-pair link prediction" for exp06–09. Never "SL target discovery."
- `Rand` negatives are unconfirmed non-SL; the benchmark is an adapter, not a
  validated K562 SL assay.
- CV1 is degree-gameable; all model claims judged on CV2/CV3.
- exp08 is pending; present design + status honestly, do not present fabricated
  metrics.

## 8. Results Scoreboard with SOTA Comparison

A scoreboard slide/appendix places our benchmark-adapter models alongside published
SL-prediction methods on the same Rand 1:1 CV1/CV2/CV3 axes. All values to 4 decimals.
F1 = max-F1 over the PR curve; NDCG = NDCG@10 (our models, official per-anchor
protocol). Literature rows (DDGCN, GRSMF, SL2MF) are paper-reported Rand 1:1 values.

| Model | Source | CV1 F1 | CV1 NDCG | CV2 F1 | CV2 NDCG | CV3 F1 | CV3 NDCG |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DDGCN | literature | 0.9104 | 0.2159 | 0.9113 | 0.2494 | 0.9104 | 0.2470 |
| GRSMF | literature | 0.8757 | 0.5178 | 0.8905 | 0.5075 | 0.8905 | 0.5075 |
| SL2MF | literature | 0.8611 | 0.2745 | 0.4332 | 0.0052 | 0.4160 | 0.0001 |
| A (logreg) | exp06 | 0.6675 | 0.0040 | 0.6677 | 0.0048 | 0.6686 | 0.0035 |
| B (XGBoost) | exp06 | 0.7304 | 0.0505 | 0.6756 | 0.0421 | 0.6701 | 0.0024 |
| C (degree probe) | exp06 | 0.8227 | 0.1970 | 0.6667 | 0.0006 | 0.6667 | 0.0008 |
| A_xcl (logreg + selectivity) | exp09 | 0.6675 | 0.0096 | 0.6676 | 0.0118 | 0.6699 | 0.0081 |
| B_xcl (XGB + selectivity) | exp09 | 0.7436 | 0.1601 | 0.6942 | 0.0864 | 0.6727 | 0.0011 |
| exp07 (observed Perturb-seq) | exp07 | pending | pending | pending | pending | pending | pending |
| exp08 (frozen-STATE + ESM2 e2e DL) | exp08 | pending | pending | pending | pending | pending | pending |

**Honest read of the scoreboard (for speaker notes):**
- The literature methods report much higher CV1 F1 (0.86–0.91), but those use the
  full SynLethDB universe and their own splits/negatives — **not directly comparable**
  to our K562-filtered 9,471-gene universe and official per-anchor ranking. Present
  as context/orientation, not a head-to-head leaderboard.
- The honest takeaway is the **within-harness** comparison: B_xcl (exp09) lifts over
  the exp06 B floor on CV2 (NDCG 0.0864 vs 0.0421) — that is the apples-to-apples
  result. exp07/exp08 rows are explicitly pending.
- NDCG@k convention for the literature rows is paper-reported; our rows are NDCG@10.
  Flag this caveat on the slide so the comparison is not over-claimed.

## 9. Out of Scope

- Building actual slide media (PPTX/Keynote/PDF). Deliverable is the markdown
  outline + speaker notes; the user renders slides from it.
- Generating new result numbers or running any pending experiment.
- New figures beyond the two existing PNGs (optional enhancements noted only).
