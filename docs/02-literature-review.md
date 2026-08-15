# Literature Review: Related Work and Novelty

**Status:** compiled 2026-07-17 from a four-agent literature review (local `docs/literature/` vault + arXiv/bioRxiv/Semantic Scholar) for the active general SL discovery program. Retired cell-fate evidence remains in git history and `ideaspark_run/`.
**What this is:** the related-work landscape and an honest novelty adjudication for the virtual-cell SL composition program.
**Companions:** [`01-blueprint.md`](01-blueprint.md) (contract) · [`03-experiment-protocol.md`](03-experiment-protocol.md) (protocol).
**Detailed per-slice syntheses (evidence tables, full citations):** [`results/literature-review-2026-07/`](results/literature-review-2026-07/) — slice 1 (SL methods), slice 2 (virtual-cell models), slice 3 (epistasis), slice 4 (dependency + novelty).
**Verification note:** cold-start benchmark numbers verified directly from `data/SL_benchmark/src/summary_all_matrics.csv`; CILANTRO-SL verified to exist (bioRxiv `10.64898/2026.02.25.708096`). Numbers taken from a preprint's own text are attributed as such.

## 0. Headline — implications for the current contract

The review establishes four constraints on the general SL discovery program.

1. **Novelty must be repositioned.** A contemporaneous bioRxiv preprint, **CILANTRO-SL** (Hua, Haber & Ma, CMU, Feb 2026), is a near-duplicate of the "graph-free, inductive SL from a foundation-model in-silico knockout supervised by DepMap viability, beats KG4SL cold-start" framing. That framing is **no longer novel on its own**. What survives adversarial search (all four agents independently) is the *mechanism* — a perturbation-response-trained virtual cell composed via a joint double-KO and validated against measured epistasis — which [`01`](01-blueprint.md) §8 declines to pursue. §5 states the narrower claim the program does make.
2. **The Feng2024 leaderboard is no longer this program's bar.** `01` drops it: the benchmark is now the context-conditioned screen table, whose axis is the *cell line*, not the unseen gene. The verified numbers below stay as field background — they are why graph-free induction is worth attempting — but they are not a target this program shoots at, and no reproduction of SLMGAE, KR4SL or KG4SL is planned. The consequence is that **the new benchmark has no external SOTA**, so the internal control ladder C1–C6 in `01` §6 carries the burden the Feng zoo used to carry.
3. **Bridge B has a named, evidenced failure mode.** Multiple independent 2025–26 benchmarks find deep perturbation models **underestimate synergy** and, for *double* perturbations, do **worse than a naive additive baseline** (Ahlmann-Eltze 2025; Systema 2025). The virtual double-KO may systematically miss the non-additivity SL is made of. This is prior-art context, **not** an active requirement: [`01`](01-blueprint.md) builds no double-KO stage, so the finding is why that ground stays unclaimed rather than a check the program must run.
4. **Measured-GI evidence is presently context-limited.** The "Jost/Replogle dual-sgRNA" file is a *single-gene* knockdown-efficacy resource, **not** a genetic-interaction dataset. The local anchors are the K562 arm of **Horlbeck 2018** (continuous fitness GI; acquired and [coverage-audited](results/horlbeck-k562-exp05-coverage.md)) and **Adamson 2016 UPR epistasis** (tiny: 3 sensors + combos, transcriptomic-only). K562 cannot alone establish cross-cell-line mechanism. No active gate consumes any of it (§3).

## 1. Synthetic-lethality prediction and the cold-start critique

Computational SL prediction is overwhelmingly **link prediction over a curated SL graph** (SynLethDB). Three families solve the same edge-scoring problem: matrix factorization (GRSMF, SL2MF, CMFW), GNNs over the SL graph + PPI/GO (DDGCN, GCATSL, SLMGAE, MGE4SL, PTGNN), and knowledge-graph reasoners (KG4SL, SLGNN, PiLSL, NSF4SL, KR4SL, DGIB4SL). **All require the query gene to already be a node** with some relation (SL edge, PPI, GO, KG entity); none scores a gene from identity alone.

The field's unifying benchmark is **Feng et al. 2024** (*Nat. Commun.*), which re-runs 12 methods under one harness: three splits — **CV1** (pair holdout), **CV2** (one gene unseen, "semi-cold-start"), **CV3** (both genes unseen, "complete cold-start") — three negative-sampling schemes, four pos:neg ratios, and both classification and per-anchor ranking (`cal_metrics`). This program no longer runs that harness ([`01`](01-blueprint.md) §1); the splits are described here because they define the field's vocabulary. Feng2024 itself states CV1 "lacks the ability to extend … to genes unseen during training" and flags cold-start and context-specificity as open.

**Cold-start is where graph methods break — verified numbers** (Feng2024, Rand 1:1, from `summary_all_matrics.csv`):

| Model | CV2 AUROC | CV2 NDCG@10 | CV3 AUROC | CV3 NDCG@10 |
|---|---:|---:|---:|---:|
| **SLMGAE** | 0.853 | 0.101 | **0.790** | **0.039** |
| GCATSL | 0.839 | 0.122 | 0.678 | 0.002 |
| NSF4SL | 0.770 | 0.104 | 0.683 | 0.004 |
| **KG4SL** | 0.806 | 0.108 | **0.562** | **0.000** |
| DDGCN | 0.783 | 0.008 | 0.485 | 0.005 |
| SLGNN | 0.735 | 0.045 | 0.530 | 0.000 |

Pattern: **AUROC stays deceptively >0.5 into CV3 while NDCG@10 collapses to ~0** — models separate positives from negatives within a fixed test set but cannot *rank* a gene's true partners near the top of ~9,800 candidates, the operationally relevant failure. SLMGAE is the only listed model whose CV3 NDCG@10 clears 0.01. Some published aggregate ranking columns are internally invalid (for example, NDCG greater than 1), which is one reason these numbers are read here as field background rather than as a bar this program reproduces (§0.2).

**Why CV1 is a degree trap (this repo's own diagnostic, exp06):** a "degree probe" scoring a pair purely as `train_degree[a]·train_degree[b]` — zero biology — is the single **best** CV1 ranker (NDCG@10 0.197) and collapses to 0.001 on CV2. CV1 rewards topology, not biology; unseen-gene claims come only from CV2/CV3, while unseen-cell-line claims require a separate split.

**Ground truth is itself unstable:** SLKB (Gökbağ 2024) shows five SL-scoring formulas over the same raw dual-KO screens agree on only **1.21%** of their top-10% calls. Some CV2/CV3 "failure" is label disagreement, not absent signal.

**Emerging inductive methods (the real competition):** ESM4SL (Yang 2025, ESM-2 sequence embeddings — same lab as KG4SL), a zero-shot LLM screen (2026, AUROC 0.715 reconstructing real CRISPR SL screens), paralog-SL (De Kegel 2021; Flister 2025, feature-based, already inductive for paralogs), and **CILANTRO-SL** (§5). Full roster and citations: slice 1.

## 2. Virtual-cell / perturbation-response models

Forward perturbation prediction has two lineages: generative/representation (scGen, CPA, chemCPA, CellOT) and graph/knowledge-prior (GEARS). Since 2024 the center is large single-cell **foundation models**: scGPT (default baseline), and virtual-cell successors **STATE** (Arc; this program's exp05 backbone), **STACK** (Arc), **Tahoe-x1** (3B params, fine-tuned to output **DepMap CERES**), and **X-Cell** (Xaira; diffusion LM cross-attending **six** priors including **ESM-2 and DepMap**). Open-vocabulary gene identity is increasingly delegated to side-embeddings: scGenePT (text), **PerturbNet** (ESM protein sequence, for GATA1 variants), **GenePert** (linear-on-ESM2/GenePT). Double/combinatorial prediction is thinner and still keyed to GEARS' framing (scLAMBDA extends to unseen combinations at the transcriptome level).

**The load-bearing caveat for Bridge B.** Independent, adversarial 2025–26 benchmarks converge:
- **Ahlmann-Eltze, Huber & Anders 2025** (*Nat. Methods*): fine-tuned scGPT/scFoundation/GEARS beat **no** simple baseline for single perturbations; for **double** perturbations, deep models had **higher error than a naive additive baseline**.
- **Systema** (Viñas Torné 2025, *Nat. Biotechnol.*): on unseen **2-gene Norman** perturbations, matching-mean beats all models by 11% relative.
- **Systematic comparison** (Li 2025): fine-tuned FMs "**underestimate synergistic effects** in combinations" — biased toward the null in exactly the SL regime.
- **scPerturBench** (Wei 2026), **PerturBench** (Wu 2025, mode collapse), **Csendes 2025** (Train-Mean beats FMs): the simple-baseline pattern is robust.

**Implication:** a raw virtual double-KO output cannot be trusted over the additive null; the interaction is likely *compressed*. This is why (a) ψ must be explicit and (b) a **GenePert-style cheap ablation** (linear-on-ESM2 + simple combination rule) is mandatory — if STATE's machinery can't beat it, the machinery is unjustified. Two near-miss systems already own halves of Bridge A: **Tahoe-x1** (embedding→CERES) and **X-Cell** (ESM-2 + DepMap in one virtual cell) — neither chains a simulated KO into a second gene's fitness. Full table: slice 2.

## 3. Genetic interaction / epistasis: measurement, nulls, prediction

**Measured GI landscapes (what Bridge B validates against):**
- **Horlbeck 2018** (*Cell*): dual-sgRNA CRISPRi GI map with a 472-gene library design; the acquired post-QC K562 gene-level matrix has 448 genes and 100,128 unique off-diagonal pairs (+Jurkat) — a fitness-GI reference for context-matched mechanistic validation. Any overlap with SynLethDB/Feng2024 labels must be audited and excluded from calibration/evaluation overlap.
- **Norman 2019** (*Science*): K562 **CRISPRa** combinatorial Perturb-seq; the 5-subtype GI taxonomy (synergy/suppression/neomorphism/redundancy/epistasis). Auxiliary (modality caveat).
- **Adamson 2016** (*Cell*): K562 **CRISPRi** single + combinatorial UPR epistasis (ATF6/PERK/IRE1). A local but tiny transcriptomic anchor (3 sensors + 4 combos), not a fitness-GI map.
- **Jost/Replogle dual-sgRNA (GSE205310):** **NOT a GI dataset** — two guides per *single* gene for knockdown efficacy (403/150 target genes). Remove from the epistasis plan; usable only as a single-gene CRISPRi response supplement.
- **No active gate consumes any of this.** `01` narrows the contribution to context-conditioned label ranking and drops the measured-GI validation arm, so Horlbeck and Adamson remain closed T1 evidence rather than live evaluation surfaces. Reinstating a mechanism claim would require reopening that decision.

**The null (ψ) has direct precedent.** GI callers define "expected" double-perturbation phenotype as additive/multiplicative/min: **GEMINI** (Zamanighomi 2019) has a "Strong" (min/HSA) and a "Sensitive" (additive+max) mode — literally our additive-vs-min ψ choice. A 2025 benchmark (Ajmal et al.) found no universal winner (GEMINI-Sensitive won 3/5 datasets); a companion paper argues a simple z-scored regression residual is competitive. → keep ψ **simple and interpretable**, test both additive and min (itself a differentiator from CILANTRO-SL's black-box MLP composition).

**Computational GI prediction** is sparse: **GEARS** predicts transcriptomic GI subtype for unseen pairs (wrong axis — transcriptome, not fitness; closed GO graph); **DANGO** (Ma lab) predicts higher-order GI in yeast from functional networks. No method predicts a *fitness* GI residual for an unscreened pair from single-gene data, and no computational SL/GI predictor is validated against a **continuous measured** GI score. Full detail + the practical anchor recommendation: slice 3.

## 4. Transcriptome → dependency/fitness, and set learning

Three lineages predict a gene's own dependency: from **static omics** (DeepDep, Chiu 2021; RFM, Cai/Uhler 2023; Owkin benchmark 2024 — deep representations do help specifically on DepMap essentiality); from **perturbation transcriptome → viability** (Szalai 2019, >90k signature-viability pairs; WRFEN-XGBoost, r=0.83; MIX-Seq 2020; sci-Plex 2020) — the closest B→C precedents, none genetic/single-cell/same-line/DepMap-labeled; and **co-dependency** (PARIS, Benfatto 2021 — DepMap covariation as an SL signal, on observed data, not generative). The forward model pools a response **bag** to a scalar, so **set learning** is relevant: Deep Sets, Attention-MIL, Set Transformer, scMILD (validated on simulated CRISPR-perturbed cells). No MIL method corrects survivor bias — less relevant here because exp05's bag is *generated*, not observed after attrition. Full table: slice 4.

## 5. Closest prior art and novelty adjudication

Ranked by closeness, with what each does **not** do:

1. **CILANTRO-SL** (Hua, Haber & Ma, CMU; bioRxiv `10.64898/2026.02.25.708096`, Feb 2026; RECOMB 2026) — **the closest.** Two-stage, graph-free: a foundation model (**Geneformer**) does in-silico single-gene knockouts on bulk RNA-seq → CRISPR-viability-supervised "knockout-aware viability embeddings" → pairwise classifier + conformal uncertainty; evaluated zero-shot on unseen genes/pairs on SynLethDB; **reportedly beats KG4SL by +28.6% F1 and ESM4SL by +49.9% F1** on the gene-holdout split (per the preprint). Does **not**: (a) use a *perturbation-response-trained* forward model — Geneformer token-removal "does not simulate a mechanistic knockout" (their words); (b) simulate a **joint double perturbation** or a **sequential counterfactual** — it composes two *independent* single-gene embeddings via a black-box MLP; (c) use an **explicit interaction null**; (d) **validate against measured epistasis**; and it explicitly **defers combinatorial/double-KO to future work** (citing DANGO — same senior author).
2. **This repo's retired exp08 result** — its removed implementation tested an architecturally similar claim and **already measured a failure** against the dependency floor (CV2 AUROC 0.667 < 0.704; CV3 0.587 < 0.596; numbers retained here only, the exp08 evidence map is no longer in the vault). The new program must justify, in falsifiable terms, why Bridge A/B differ from that pooled-embedding pair-head.
3. **Tahoe-x1** (embedding→CERES) and **X-Cell** (ESM-2 + DepMap virtual cell) — own "Bridge A plumbing" halves; neither chains a simulated KO into a second gene's fitness, neither is SL-evaluated.
4. **GEARS** — conceptual ancestor of Bridge B (joint perturbation + GI subtypes), but transcriptome-magnitude GI, closed GO graph, never SL-DB-benchmarked.
5. **RFM-SL** (Cai/Uhler 2023, static feature attribution), **PARIS** (co-dependency on observed data), **ESM4SL** (ESM-2 sequence-only; shares our identity backbone; documents the unseen-gene collapse F1 0.914→0.453), **Large Perturbation Model** (Miladinovic 2025, impute-missing-perturbation MoA).

**Adjudication (all four agents independently).**
- *Graph-free / inductive / beats KG4SL cold-start*: **no longer novel** — CILANTRO-SL, RFM-SL, PARIS, ESM4SL occupy it; CILANTRO-SL already beats KG4SL cold-start on the SynLethDB family.
- *In-silico **double**-KO of a perturbation-response model, with an explicit null, validated against measured epistasis*: **survives** adversarial search — nobody does it (CILANTRO-SL's own authors defer it). **This program does not attempt it.** [`01`](01-blueprint.md) §8 drops the joint perturbation, the `joint − psi` residual, and the measured-GI arm, so the program does not claim that ground and must not be written as if it does.

**Defensible novelty statement, narrowed to what the contract delivers:**
> No prior work builds a context-conditioned synthetic-lethality score from a **perturbation-response-trained** dependency profile, with a **declared non-interaction null as a named baseline** and an explicit **gene-mean / context-residual decomposition** that makes pan-essentiality a controlled variable, evaluated on a **published cell-line-held-out split**. The closest prior art, CILANTRO-SL, reaches unscreened genes and beats KG4SL cold-start, but derives its embeddings from Geneformer token-removal — which its own authors say does not simulate a mechanistic knockout — composes two single-gene embeddings through an unexamined black-box MLP, and scores context-blind.

This is narrower than the claim the program previously carried, and it is **not** a mechanism claim: nothing here estimates a genetic interaction, and `01` §8 forbids the word. Lead with the context axis and the controlled confounds, not with "graph-free/inductive."

## 6. Implications carried into the contract (`01`/`02`)

1. **Lead with the context axis and the controlled confounds,** not mechanism and not graph-free induction. The mechanism ground is real but unclaimed (§5); graph-free induction is shared prior art.
2. **No external SOTA exists on the new benchmark.** Nothing in the Feng zoo runs on a context-conditioned screen table, so comparability comes from the internal ladder, not from a leaderboard. Build that ladder to carry the weight.
3. **One generalization axis.** The contract holds out cell lines, not genes, so no unseen-gene claim is available and CV2/CV3 language must not leak back into the program's own results.
4. **Pan-essentiality is the confound to beat,** not graph topology. The `mu`/`delta` split exists so it can be ablated rather than asserted away; the degree-probe lesson (§1) generalizes — a shortcut that scores well is a shortcut that will score well here too.
5. **Synergy compression stays a live caution** even without a double-KO stage: it is the documented reason a composed score can look calibrated and carry no interaction, which is why `01` refuses the word.

## 7. Required baselines and ablations (for the protocol, [`03`](03-experiment-protocol.md))

The binding ladder is C1–C6 in [`01`](01-blueprint.md) §6; this section records what the literature says each is for, plus the external ablations worth keeping.
- **C1 / C1b / C3 / C6 — memorization, pan-essentiality, and the matched context-ablated head.** The degree probe (§1) is the precedent: a shortcut with zero biology can top a benchmark, so each must be measured rather than assumed absent, and Feng-zoo experience shows single shortcuts get removed one at a time while the model keeps winning on what remains — which is what the matched C6 control closes.
- **GenePert-style cheap ablation** — linear/L2 on ESM2 (and GenePT) embeddings with a min/additive combination rule; the machinery must beat it or the machinery is unjustified. The **Tahoe-x1 essentiality head** on the response output is the obvious alternative dependency readout.
- **CILANTRO-SL** — differentiate head-to-head in prose. Reproduction is out of scope: it is evaluated on SynLethDB, which this program no longer uses, so there is no shared surface to compare on.
- **Retired with the Feng benchmark:** SLMGAE / KR4SL / KG4SL official-harness reproduction, and GEARS-as-SL. None of them scores a context-conditioned pair table.

## 8. Internal evidence update: T2 Phase F gate

The registered nine-line Phase F test was opened on 2026-07-29. Tx1-3B-ST
failed the k=10 single-gene GeneEffect backbone-transfer gate against
copy-K562 + 10 labels (`Delta rho = -0.0048`, line-bootstrap 95% CI
`[-0.0941, 0.0769]`, registered `rho_min = 0.05`). The HVG-ST attribution
control was also negative as a diagnostic (`Delta rho = 0.0326`, 95% CI
`[-0.0602, 0.1181]`). Both arms deteriorated as k increased; Tx1's k=0 output
variance was close to target variance, so constant-output collapse does not
explain the failed adapted gate. This is binding single-gene component evidence,
not pairwise SL or measured-GI evidence. The full report is
[`results/tx1-hvg-geneeffect-phase-f.md`](results/tx1-hvg-geneeffect-phase-f.md).

## 9. Key references (bibkeys → why cited)

`feng2024benchmarking` (benchmark + protocol) · `wang2022synlethdb` / `gokbag2023slkb` (labels; ground-truth instability) · `hao2021slmgae` (real cold-start bar) · `zhang2023kr4sl` (Feng-flagged CV3 leader) · `wang2021kg4sl` (named-but-weak north-star) · `long2021gcatsl`, `cai2020ddgcn`, `zhu2023slgnn`, `huang2019grsmf`, `liu2020sl2mf`, `nsf4sl2022`, `liu2022pilsl` (roster) · `dekegel2021paralog` / `flister2025paralog` (feature-based inductive; context-specificity) · **`hua2026cilantrosl`** (closest prior art) · `esm4sl2025` (ESM2-for-SL precedent) · `llmzeroshot2026` (cheap inductive alt) · `adduri2025state` (backbone) · `gandhi2025tahoex1`, `wang2026xcell` (Bridge-A near-misses) · `roohani2023gears` (Bridge-B ancestor) · `chen2024genepert` (mandatory ablation) · `yu2025perturbnet` (ESM-driven precedent) · `ahlmanneltze2025linearbaselines`, `vinastorne2025systema`, `wei2026scperturbench` (synergy-underestimation caveat) · `horlbeck2018landscape`, `norman2019manifolds`, `adamson2016upr` (measured epistasis) · `zamanighomi2019gemini` (ψ precedent) · `chiu2021deepdep`, `szalai2019cvs`, `benfatto2021paris` (dependency/co-dependency) · `zaheer2017deepsets`, `jeong2026scmild` (set learning). Full annotated lists in the four slice files.
