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
- **Framing of exp08:** Methods-forward capstone. exp08 (and the full exp09 run)
  have **no empirical results yet** — code + unit tests are complete, CV2/CV3
  cluster gates are pending. The talk presents exp08 as the designed centerpiece
  and is honest about the pending status rather than inventing numbers.

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
| 11 | Parallel route: cross-cell-line selectivity (exp09) | 0:45 | — | No transcriptome: use DepMap across 1,208 lines — is KO(b) more lethal where gene_a is defective? Composite-OR defective call (mutation/CN/expression). Stays in GeneEffect family; non-pan-essential diagnostic slice guards against re-encoding essentiality. Breadth of approaches to the same bar. |
| 12 | Closing the loop | 1:00 | — | Callback to slides 2–3. Built: an honest, leakage-controlled SL-pair ranking adapter, not target discovery. Recurring discipline: simple floors first, CV2/CV3 as the real bar, negative results count. Path to true context-specific SL: exp08 cluster results → exp09 selectivity → TCGA patient-context transfer. End on honest scope. |

## 4. Asset Inventory

- `docs/presentation/SL_concept.png` — slide 2 (SL concept figure).
- `docs/presentation/e2e_SL_DL.png` — slide 9 (exp08 architecture figure).
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
  clears n≥20 bar for 9,459/9,471 genes. **Only a sanity run exists, not the full
  official-metric run.**

## 6. Honesty Guardrails (carried onto slides)

- Say "dependency prediction / essentiality ranking" for exp01–05; "SL candidate
  ranking / SL-pair link prediction" for exp06–09. Never "SL target discovery."
- `Rand` negatives are unconfirmed non-SL; the benchmark is an adapter, not a
  validated K562 SL assay.
- CV1 is degree-gameable; all model claims judged on CV2/CV3.
- exp08 and the full exp09 run are pending; present design + status honestly, do
  not present fabricated metrics.

## 7. Out of Scope

- Building actual slide media (PPTX/Keynote/PDF). Deliverable is the markdown
  outline + speaker notes; the user renders slides from it.
- Generating new result numbers or running any pending experiment.
- New figures beyond the two existing PNGs (optional enhancements noted only).
