# Final Presentation Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a single Marp-renderable markdown deck (`docs/presentation/final-presentation.md`) of 12 slides + 1 appendix slide for a 10–12 minute final oral presentation on the K562 synthetic-lethality project, with English slide content and Chinese (中文) HTML-comment presenter notes.

**Architecture:** One markdown file. Slides separated by `---`. Each slide is built and verified in its own task so a reviewer can reject one slide without touching neighbors. Chinese speaker notes live in Marp-style `<!-- _notes` HTML comments directly under each slide. Two existing PNGs are referenced by relative path; no new media is generated. Every numeric value on a slide is verified against its source CSV/doc before the task closes.

**Tech Stack:** Markdown (Marp-flavored: `marp: true` front-matter, `---` slide breaks, `<!-- -->` presenter notes). No code, no build dependencies beyond an optional Marp render check.

## Global Constraints

These apply to every task. Values copied verbatim from `docs/superpowers/specs/2026-06-18-final-presentation-design.md`.

- **Deliverable format:** Markdown outline + speaker notes only. Do NOT build PPTX/Keynote/PDF. Do NOT run or generate new experiment results.
- **Language split:** Slide-visible text in English. Speaker notes in Chinese (中文); technical terms, model names, and metric numbers may stay in English/numerals.
- **Slide count:** 12 main slides + 1 appendix (scoreboard) = 13 `---`-separated sections after the front-matter.
- **Numeric precision:** All scoreboard values to **4 decimal places**. Inline metric mentions follow the spec's stated precision.
- **No fabricated metrics:** exp08 is **pending** everywhere. exp07 scoreboard cells are **pending**. Never invent numbers for pending experiments.
- **Honesty guardrails (must appear / must never be violated):**
  - Use "dependency prediction / essentiality ranking" for exp01–05; "SL candidate ranking / SL-pair link prediction" for exp06–09. Never write "SL target discovery" as a claim about what was built.
  - `Rand` negatives are unconfirmed non-SL; the benchmark is an adapter, not a validated K562 SL assay.
  - CV1 is degree-gameable; model claims are judged on CV2/CV3.
  - Literature SOTA rows (DDGCN/GRSMF/SL2MF) use a different universe/splits/negatives → present as context, NOT a head-to-head leaderboard.
- **Asset paths (relative to repo root):** `docs/presentation/SL_concept.png` (slide 2), `docs/presentation/e2e_SL_DL.png` (slide 9). Both already exist.
- **Source of truth for numbers:**
  - exp06: `results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_summary.csv`
  - exp09: `results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run/summary.csv`
  - exp01/02/03/07/08 inline numbers: the experiment docs under `docs/experiment/`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `docs/presentation/final-presentation.md` | The entire deck: Marp front-matter, 12 main slides, 1 appendix slide, Chinese presenter notes per slide. Built incrementally, one slide per task. |
| `docs/presentation/SL_concept.png` | Existing. Referenced by slide 2. Not modified. |
| `docs/presentation/e2e_SL_DL.png` | Existing. Referenced by slide 9. Not modified. |

The "test" for each task is a **content-correctness check**, not a unit test: confirm the slide renders as a single `---`-delimited block, every number matches its source, the honesty guardrails hold, and English-slide / Chinese-notes split is respected. Where a number is involved, the verification step greps the source CSV/doc to confirm the value before commit.

---

### Task 1: Deck scaffold + front-matter + verification helper

**Files:**
- Create: `docs/presentation/final-presentation.md`

**Interfaces:**
- Produces: a Marp markdown file with valid front-matter and a title slide (slide 1). Later tasks append `---`-separated slides after it.

- [ ] **Step 1: Confirm the two asset PNGs exist (the verification this task asserts)**

Run:
```bash
ls -la docs/presentation/SL_concept.png docs/presentation/e2e_SL_DL.png
```
Expected: both files listed, non-zero size. If either is missing, STOP and report — slides 2/9 depend on them.

- [ ] **Step 2: Write the front-matter and slide 1 (title)**

Create `docs/presentation/final-presentation.md` with exactly this content:

```markdown
---
marp: true
theme: default
paginate: true
title: From SL Promise to an Honest Ranking Model
---

# From SL Promise to an Honest Ranking Model

### Ranking candidate synthetic-lethal gene pairs in K562

Combining DepMap dependency, Perturb-seq response, and a frozen perturbation foundation model

<!-- _notes (中文):
开场一句话定位：我们做的是在 K562 细胞系里，对候选「合成致死」基因对做排序。
方法上把三种证据结合起来——DepMap 依赖性、Perturb-seq 扰动转录组、以及一个冻结的扰动基础模型 STATE。
强调这是一个「诚实的排序模型」，后面会反复回到「诚实」这个主题：这是 benchmark 上的基因对排序/分类，不是真正的 SL 靶点发现。
计时：约 30 秒，快速进入第 2 页。
-->
```

- [ ] **Step 3: Verify the file is a single valid slide block**

Run:
```bash
head -20 docs/presentation/final-presentation.md && echo "--- slide-break count ---" && grep -c '^---$' docs/presentation/final-presentation.md
```
Expected: front-matter visible; `grep -c '^---$'` returns `2` (the two front-matter fences). No content `---` break yet.

- [ ] **Step 4: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: scaffold final presentation deck with title slide"
```

---

### Task 2: Slide 2 — What is SL & the bottleneck

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append slide 2)

**Interfaces:**
- Consumes: scaffold from Task 1.
- Produces: slide 2 referencing `SL_concept.png`.

- [ ] **Step 1: Append slide 2**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## What is synthetic lethality — and why it's hard

![bg right:40% fit](SL_concept.png)

- **SL**: losing either gene alone is tolerated; losing **both** kills the cell
- One gene already inactivated in the tumor → its SL partner becomes a selective drug target
- Clinical proof: **PARP inhibitors in BRCA1/2-mutant tumors**
- The bottleneck: **~200M human gene pairs**, mostly **context-dependent** — can't brute-force experimentally
- → we need a computational way to **rank** candidate partners

<!-- _notes (中文):
先讲概念：两个基因，单独敲掉任何一个细胞都能活，但同时敲掉就死——这就是合成致死。
癌症里常常有一个基因已经被突变失活了，那么它的 SL 搭档就成了选择性药物靶点。
最经典的临床证据是 BRCA1/2 突变肿瘤里的 PARP 抑制剂。
难点在于：人类大约有两亿个基因对，而且 SL 关系高度依赖细胞背景（cell context），实验上不可能穷举。
所以需要计算方法来做候选搭档的排序。这一页为后面所有实验立motivation。
计时：约 1 分钟。
-->
```

- [ ] **Step 2: Verify slide break count and asset reference**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md && grep -n 'SL_concept.png' docs/presentation/final-presentation.md
```
Expected: `grep -c '^---$'` returns `3` (2 front-matter + 1 content break); `SL_concept.png` referenced on one line.

- [ ] **Step 3: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add slide 2 (SL concept and bottleneck)"
```

---

### Task 3: Slide 3 — Honest task framing + benchmark

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append slide 3)

**Interfaces:**
- Consumes: slide 2.
- Produces: the honesty-framing slide (referenced by the slide-12 callback).

- [ ] **Step 1: Append slide 3**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## The task we actually solve (and what we don't claim)

**Benchmark:** Feng et al. 2024, SynLethDB-derived · `Rand` 1:1 balanced · 9,471-gene K562 universe

| Split | Held out | Difficulty |
| --- | --- | --- |
| CV1 | pair-level (genes may recur) | easiest, **degree-gameable** |
| CV2 | one gene unseen | intermediate |
| CV3 | both genes unseen | cold-start, hardest |

- This is **SL-pair classification / ranking**, *not* validated SL target discovery
- `Rand` negatives are **unconfirmed** non-SL → benchmark adapter, not a K562 SL assay
- **CV2 / CV3 are the only honest generalization surfaces**

<!-- _notes (中文):
这一页是整个报告的「诚实声明」，非常重要。
我们用的是 Feng 2024 的 benchmark，来自 SynLethDB，Rand 1:1 平衡负样本，K562 过滤后是 9471 个基因的候选空间。
三种切分：CV1 是基因对层面切分，基因还会在训练集出现，最简单，而且可以被「度数」shortcut 钻空子；CV2 一个基因没见过；CV3 两个基因都没见过，是冷启动、最难。
必须反复强调：我们做的是基因对的分类/排序，不是真正的 SL 靶点发现。Rand 负样本是「未确认」的非 SL，所以这是一个 benchmark 适配器，不是 K562 的 SL 实验验证。
结论：只有 CV2/CV3 才是诚实的泛化评估面。后面所有模型都用这个标准来judge。
计时：约 1 分钟。
-->
```

- [ ] **Step 2: Verify slide break count**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md
```
Expected: `4`.

- [ ] **Step 3: Verify honesty language is present and forbidden claim is absent**

Run:
```bash
grep -in 'not validated SL target discovery\|benchmark adapter\|CV2 / CV3 are the only honest' docs/presentation/final-presentation.md
```
Expected: at least the "not validated SL target discovery" and "honest generalization" lines match. Confirm no line asserts the deck *performs* SL target discovery.

- [ ] **Step 4: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add slide 3 (honest task framing and benchmark)"
```

---

### Task 4: Slide 4 — Data sources & why these

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append slide 4)

**Interfaces:**
- Consumes: slide 3.
- Produces: data-rationale slide.

- [ ] **Step 1: Append slide 4**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## Data sources — and why each one

| Source | Role | Why |
| --- | --- | --- |
| **DepMap** CRISPRGeneEffect (Chronos) | dependency label `C` | population fitness; K562 = `ACH-000551` |
| **Replogle K562 gwps** Perturb-seq | transcriptomic response | **CRISPRi = loss-of-function**, modality-aligned with DepMap KO; 1.99M cells, 6,070 genes |
| **ESM2-650M** protein embeddings | gene identity | continuous → generalizes to **held-out** genes (1280-d) |
| **Feng 2024 / SynLethDB** | pair label `D` | the SL-pair benchmark |

**Why K562:** the only cell line with both deep Perturb-seq *and* DepMap dependency.

<!-- _notes (中文):
四个数据源，逐个讲为什么选它。
DepMap 的 CRISPRGeneEffect（Chronos 分数）是我们的依赖性标签 C，群体层面的适应度读出，K562 对应 ACH-000551。
Replogle K562 gwps 是全基因组 Perturb-seq，关键是它是 CRISPRi，也就是功能缺失，和 DepMap 的基因敲除模态一致；约 199 万个细胞，覆盖 6070 个基因。
ESM2-650M 蛋白质 embedding 提供基因身份的连续表示，这点对后面 exp08 处理没见过的基因至关重要，是 1280 维。
Feng 2024/SynLethDB 提供基因对标签 D。
为什么选 K562：它是唯一同时有深度 Perturb-seq 和 DepMap 依赖性数据的细胞系。
计时：约 1 分钟。
-->
```

- [ ] **Step 2: Verify slide break count and modality caveat**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md && grep -in 'CRISPRi = loss-of-function' docs/presentation/final-presentation.md
```
Expected: `5`; the CRISPRi modality line matches.

- [ ] **Step 3: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add slide 4 (data sources and rationale)"
```

---
### Task 5: Slide 5 — Stage 1: observed transcriptome → dependency works

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append slide 5)

**Interfaces:**
- Consumes: slide 4.
- Produces: compressed exp01+exp02 slide.

- [ ] **Step 1: Verify the exp01/exp02 numbers against the docs before writing them**

Run:
```bash
grep -n '0.485\|0.500\|0.886' docs/experiment/01_replogle_k562_pseudobulk_b_to_c_and_adamson_transfer.md
grep -n '0.244\|0.494\|0.503' docs/experiment/02_replogle_k562_viability_axis_audit.md
```
Expected: exp01 shows PCA Ridge Replogle CV `0.485`, Adamson `0.500`, AUROC `0.886`; exp02 shows NAR-only `0.244`, best pseudobulk `0.494`, NAR-residualized `0.503`. If a number differs, use the doc value, not the one below.

- [ ] **Step 2: Append slide 5**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## Stage 1 — observed transcriptome predicts dependency

**exp01:** pseudobulk Δ-expression → PCA + Ridge

- Replogle 5-fold CV ≈ **0.49** Spearman · Adamson transfer **0.50** (AUROC 0.886)

**exp02 audit:** is it just a generic cell-death / viability axis?

- NAR viability score alone **0.244** vs best pseudobulk **0.494**
- NAR-residualized transcriptome still **0.503** → signal is **real and transcriptomic**, not just "everything dies"

<!-- _notes (中文):
Stage 1 回答一个前提问题：观测到的扰动转录组到底能不能预测 DepMap 依赖性分数？
exp01：用 pseudobulk 的 delta 表达，做 PCA + Ridge，Replogle 五折交叉验证大约 0.49 Spearman，迁移到 Adamson 数据是 0.50，AUROC 0.886。说明这个桥是通的。
exp02 是一个审计：会不会模型只是学到了一个「细胞快死了 / 增殖快慢」的通用轴？
我们用 NAR 死亡signature 分数单独做，只有 0.244；而最好的 pseudobulk 是 0.494。把 NAR 轴残差掉之后，转录组仍然有 0.503。
结论：信号是真实的、转录组特异的，不是单纯的「泛死亡」效应。这给后面用转录组特征做铺垫。
计时：约 1 分钟。
-->
```

- [ ] **Step 3: Verify slide break count**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md
```
Expected: `6`.

- [ ] **Step 4: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add slide 5 (Stage 1 transcriptome to dependency)"
```

---

### Task 6: Slide 6 — Stage 2: dependency-aware representation (the bridge)

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append slide 6)

**Interfaces:**
- Consumes: slide 5.
- Produces: the hinge slide that introduces STATE — referenced by slide 9.

- [ ] **Step 1: Verify the exp03 numbers against the doc**

Run:
```bash
grep -n '0.666\|0.911' docs/experiment/03_replogle_k562_single_cell_deepsets_adamson.md | head -5
```
Expected: scVI128 frozen-GMM best Adamson Spearman `0.666`, AUROC `0.911` appear. Use doc values if different.

- [ ] **Step 2: Append slide 6**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## Stage 2 — learning a dependency-aware representation  ⟵ *the bridge*

- **exp03** single-cell set learning: scVI128 + **frozen-GMM distribution regression** wins
  - Adamson Spearman **0.666**, AUROC **0.911** — beats attention MIL; HVG hurts
- **exp04** leakage-free *predicted-B* loop (forward A→B→C, no observed test bag)
- **exp05** frozen-**STATE** forward model (Arc Institute), A→B→C pipeline

**Takeaway:** we now have a **STATE-based way to turn any perturbation into a dependency-aware embedding** → carry this into the SL-pair task.

<!-- _notes (中文):
这一页是承上启下的「桥」，很关键。
exp03：从 pseudobulk 升级到单细胞集合学习（set learning / MIL）。结论是 scVI128 加上冻结 GMM 的分布回归效果最好，Adamson Spearman 0.666，AUROC 0.911，比 attention MIL 还好；用 HVG 反而更差。
exp04：一个无泄漏的 predicted-B 流程，前向 A→B→C，不用测试基因的观测响应包。
exp05：接上 Arc Institute 的 STATE 前向模型，做 A→B→C。
最关键的 takeaway：我们现在有了一种基于 STATE 的方法，可以把任意扰动变成一个「依赖性感知」的 embedding。这个能力会被带到后面的 SL 基因对任务里——这就是 representation-as-bridge 的核心。
计时：约 1 分钟。
-->
```

- [ ] **Step 3: Verify slide break count and STATE mention**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md && grep -in 'STATE-based way to turn any perturbation' docs/presentation/final-presentation.md
```
Expected: `7`; the takeaway line matches.

- [ ] **Step 4: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add slide 6 (Stage 2 representation bridge)"
```

---

### Task 7: Slide 7 — Stage 3 begins: dependency-only floor (exp06)

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append slide 7)

**Interfaces:**
- Consumes: slide 6.
- Produces: the exp06 "bar" slide. Numbers reused on slides 10/11 and the appendix.

- [ ] **Step 1: Verify the exp06 numbers against the source CSV**

Run:
```bash
awk -F, '($2=="B"||$2=="C") && ($3=="auroc"||$3=="ndcg@10"){printf "%s,%s,%s,%.4f\n",$1,$2,$3,$4}' results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_summary.csv
```
Expected (key cells): CV2 B auroc `0.7035`→ rounds to `0.704`, CV2 B ndcg@10 `0.0421`; CV3 B auroc `0.5956`→`0.596`, CV3 B ndcg@10 `0.0024`; CV1 C ndcg@10 `0.1970`. Use these if they differ from the slide text.

- [ ] **Step 2: Append slide 7**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## Stage 3 — the dependency-only floor (exp06)

**Input:** only the two genes' DepMap GeneEffect scalars → 5 swap-invariant features → P(SL)

| Model | CV1 NDCG@10 | CV2 AUROC | CV2 NDCG@10 | CV3 AUROC | CV3 NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B (XGBoost) | 0.0505 | **0.704** | **0.042** | **0.596** | 0.002 |
| C (degree probe) | **0.197** | 0.500 | 0.001 | — | — |

- Degree probe **wins CV1** → CV1 is gameable by train-positive degree
- CV3 ≈ cold-start failure for dependency-only features
- **This is the bar every later model must beat — on CV2/CV3**

<!-- _notes (中文):
进入 Stage 3，正式切换到「基因对」任务。
exp06 是最朴素的 floor：只用两个基因的 DepMap GeneEffect 标量，构造 5 个对称特征（min/max/sum/product/|diff|），预测 P(SL)。
看表：XGBoost（模型 B）在 CV2 AUROC 0.704、NDCG@10 0.042；到 CV3 掉到 AUROC 0.596、NDCG@10 0.002，基本是冷启动失败。
模型 C 是「度数 degree probe」对照：它在 CV1 的 NDCG@10 高达 0.197，说明 CV1 可以靠训练正样本的度数被钻空子——所以 CV1 不算数。
核心信息：exp06 就是后面所有模型必须超越的「门槛」，而且必须在 CV2/CV3 上超越。
计时：约 1 分钟。
-->
```

- [ ] **Step 3: Verify slide break count**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md
```
Expected: `8`.

- [ ] **Step 4: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add slide 7 (exp06 dependency-only floor)"
```

---
### Task 8: Slide 8 — Does observed Perturb-seq add lift? (exp07)

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append slide 8)

**Interfaces:**
- Consumes: slide 7.
- Produces: exp07 slide (pending results, but methodology + coverage crux are complete).

- [ ] **Step 1: Append slide 8**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## Does observed Perturb-seq add lift over dependency-only? (exp07)

**Method:** augment exp06 (GeneEffect features) with **observed gwps response embeddings** per gene

**Coverage crux:**
- Replogle K562 gwps: **64% per-gene coverage** (6,070 / 9,471 genes)
- But for *pairs*: **~41% both-covered** under independence → 59% hit a fallback

**Two tiers:**
- Tier 1: PCA/HVG mean-pool
- Tier 2: frozen exp03 scVI representation

**Honest design:** covered-pair diagnostic slice + with/without coverage flag

**Status:** *results pending* — negative result is publishable and informative.

<!-- _notes (中文):
exp07 问：观测到的 Perturb-seq 响应能不能在 exp06 基础上带来提升？
方法：给 exp06 的 GeneEffect 特征加上每个基因的观测 gwps 响应 embedding。
覆盖率的坑：Replogle gwps 单基因覆盖是 64%（6070/9471），但这是「基因对」任务——如果两个基因独立，双覆盖只有大约 41%，剩下 59% 的对会碰到 fallback。
两层实现：Tier 1 是 PCA/HVG 均值池化；Tier 2 复用 exp03 的冻结 scVI 表示。
诚实设计：报告双覆盖对的 diagnostic slice，以及有/无覆盖标志的 ablation，确保提升不是 coverage indicator 带来的 shortcut。
状态：结果 pending。如果是负结果（没提升），也是可发表的、有信息量的——说明在 41% 覆盖率下，转录组信号不足以超越依赖性标量。
计时：约 1 分钟。
-->
```

- [ ] **Step 2: Verify slide break count and pending marker**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md && grep -in 'results pending' docs/presentation/final-presentation.md
```
Expected: `9`; "results pending" appears on this slide.

- [ ] **Step 3: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add slide 8 (exp07 observed Perturb-seq)"
```

---

### Task 9: Slide 9 — e2e DL centerpiece pt.1: problem & architecture (exp08)

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append slide 9)

**Interfaces:**
- Consumes: slide 6 (the STATE intro) and slide 8.
- Produces: first half of the exp08 centerpiece, referencing `e2e_SL_DL.png`.

- [ ] **Step 1: Verify the STATE coverage numbers from the spec before writing**

Run:
```bash
grep -n '16.3%\|83.7%\|2,024 genes\|1,542 genes' docs/superpowers/specs/2026-06-18-final-presentation-design.md
```
Expected: STATE checkpoint has 2,024 genes, 16.3% (1,542) of SL universe in-vocab, 83.7% OOV. Use spec values.

- [ ] **Step 2: Append slide 9**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## e2e DL centerpiece pt.1 — the problem & architecture (exp08)

![bg right:35% fit](e2e_SL_DL.png)

**Why exp08:** local STATE checkpoint is **closed-vocab one-hot**
- Only **16.3% of the SL universe in-vocab** (1,542 / 9,471 genes)
- 84% out-of-vocab → held-out genes get no gradient

**Fix:** freeze STATE's 8-layer Llama backbone, **replace** its one-hot `pert_encoder` with a trainable **adapter fed by ESM2** (1280 → 328)

→ all 9,471 genes land in **one coordinate system**

**Arch:** ESM2 → PertAdapter → frozen STATE → predicted bag → pooling → symmetric pair head

<!-- _notes (中文):
exp08 是这次报告的中心，两页幻灯片。第一页讲问题和架构。
为什么要做 exp08：我们本地的 STATE checkpoint 是封闭词表的 one-hot 模型——只有 16.3% 的 SL 宇宙（1542/9471 基因）在词表里，84% 是 out-of-vocab。对于 CV2/CV3 那些没见过的基因，one-hot 没法给梯度。
修复方案：冻结 STATE 的 8 层 Llama backbone，只替换它的 one-hot pert_encoder，换成一个可训练的 adapter，输入是 ESM2 蛋白质 embedding（1280 维到 328 维）。
这样所有 9471 个基因——包括没见过的——都能落在同一个坐标系里。
架构流：ESM2 → PertAdapter → 冻结 STATE → 预测响应包 → pooling → 对称基因对 head。
右边的图展示整个流程。
计时：约 1.5 分钟（exp08 两页共 3 分钟）。
-->
```

- [ ] **Step 3: Verify slide break count and asset reference**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md && grep -n 'e2e_SL_DL.png' docs/presentation/final-presentation.md
```
Expected: `10`; `e2e_SL_DL.png` referenced.

- [ ] **Step 4: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add slide 9 (exp08 pt.1 problem and architecture)"
```

---

### Task 10: Slide 10 — e2e DL centerpiece pt.2: leakage-safe training & the bar (exp08)

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append slide 10)

**Interfaces:**
- Consumes: slide 9.
- Produces: second half of exp08 centerpiece, reuses exp06 bar from slide 7.

- [ ] **Step 1: Append slide 10**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## e2e DL centerpiece pt.2 — leakage-safe training & the bar

**3-part loss:**
1. **SL BCE** (pair classification)
2. **Adapter token-distill** (align adapter output to STATE's in-vocab tokens where available)
3. **Real-bag supervision** (covered *train* genes only)

**Leakage rule** that makes CV2/CV3 valid:
- Held-out genes reached **purely** via `adapter(ESM2)` + frozen STATE
- Never supervised by their own observed response bag → genuinely unseen

**The bar to beat:** exp06 CV2 AUROC **0.704** / CV3 **0.596**, with lift concentrated on covered-pair slice

**Status:** code + unit tests complete; **CV2/CV3 cluster gates pending**

<!-- _notes (中文):
第二页讲训练和评估。
三部分 loss：SL BCE 做基因对分类；adapter token-distill 让 adapter 输出对齐 STATE 词表内的 token（在有的地方）；real-bag supervision 只用覆盖的训练基因的真实响应包。
泄漏规则（让 CV2/CV3 有效）：held-out 基因纯粹通过 adapter(ESM2) + 冻结 STATE 到达，从来不用它自己的观测响应包做监督——所以是真的没见过。
必须超越的 bar：exp06 的 CV2 AUROC 0.704 / CV3 0.596，而且提升要集中在双覆盖对的 slice 上。
状态：代码和单元测试都完成了，CV2/CV3 的集群 gate 还在 pending（诚实）。
计时：约 1.5 分钟。
-->
```

- [ ] **Step 2: Verify slide break count and pending marker**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md && grep -in 'CV2/CV3 cluster gates pending' docs/presentation/final-presentation.md
```
Expected: `11`; the pending line appears.

- [ ] **Step 3: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add slide 10 (exp08 pt.2 training and bar)"
```

---

### Task 11: Slide 11 — Parallel route: cross-cell-line selectivity (exp09)

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append slide 11)

**Interfaces:**
- Consumes: slide 10.
- Produces: exp09 slide with real results.

- [ ] **Step 1: Verify the exp09 numbers from the summary CSV**

Run:
```bash
awk -F, '($2=="B"||$2=="B_xcl") && $3=="full_universe" && ($4=="auroc"||$4=="ndcg@10"){printf "%s,%s,%s,%.4f\n",$1,$2,$4,$5}' results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run/summary.csv | grep -E 'CV2|CV3'
```
Expected (key deltas): CV2 B_xcl auroc `0.7420` vs B `0.7035` → Δ+0.038; B_xcl ndcg@10 `0.0864` vs B `0.0421` → Δ+0.044. CV3 B_xcl auroc `0.6454` vs B `0.5956` → Δ+0.050; ndcg@10 flat (`0.0011` vs `0.0024`). Use CSV values if different.

- [ ] **Step 2: Append slide 11**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## A parallel route — cross-cell-line selectivity (exp09)

**No transcriptome:** use DepMap across **1,208 lines** — is KO(b) more lethal where gene_a is defective?

**Composite-OR defective call:** mutation / CN loss / low expression

**Results:**
- CV2 B_xcl: AUROC **0.742** (+0.039 over B), NDCG@10 **0.086** (+0.044)
- CV3 B_xcl: AUROC **0.645** (+0.050), but NDCG@10 **flat** (0.001 vs 0.002)

**Read:** cross-cell-line selectivity improves cold-start **classification** but does not fix cold-start **ranking**

**Non-pan-essential slice:** lift shrinks but doesn't vanish on CV1/CV2; CV3 mostly essentiality-linked

<!-- _notes (中文):
exp09 是一条平行路线，不用转录组，纯粹用 DepMap 的跨细胞系对比。
方法：在 1208 条 GeneEffect 细胞系上，看某个基因 a 缺陷的细胞系里，敲除基因 b 是不是更致命？
用复合 OR 判断「缺陷」：突变/拷贝数缺失/低表达，只要有一个满足就算。
结果：CV2 的 B_xcl AUROC 0.742（比 B 高 0.039），NDCG@10 0.086（高 0.044）；CV3 AUROC 0.645（高 0.050），但 NDCG@10 是平的（0.001 vs 0.002）。
解读：跨细胞系 selectivity 能改善冷启动的分类，但修不好冷启动的排序。
非泛必需 slice：提升在 CV1/CV2 缩小但不消失；CV3 的提升大部分是必需性结构，不是基因对特异的 co-dependency。
这展示了面向同一个 bar 的多种方法的广度。
计时：约 0.75 分钟。
-->
```

- [ ] **Step 3: Verify slide break count**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md
```
Expected: `12`.

- [ ] **Step 4: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add slide 11 (exp09 cross-cell-line selectivity)"
```

---

### Task 12: Slide 12 — Closing the loop

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append slide 12)

**Interfaces:**
- Consumes: slide 11; references slide 3 (honesty framing).
- Produces: the final main slide with the honest-framing callback.

- [ ] **Step 1: Append slide 12**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## Closing the loop — what we built, what we didn't claim

**Built:** an **honest, leakage-controlled SL-pair ranking adapter** (classification + ranking), not validated SL target discovery

**The recurring discipline:**
- Simple floors first (exp06 dependency-only)
- CV2 / CV3 as the real bar (CV1 is degree-gameable)
- Negative results count (if exp07/08 don't beat exp06, that's a *finding*)
- Pending is marked pending (no fabricated metrics)

**Path to true context-specific SL discovery:**
- exp08 cluster results → does frozen-STATE+ESM2 adapter generalize to held-out genes?
- exp09 selectivity → does cross-cell-line evidence add robustness?
- TCGA patient-context transfer → cell line → tumor dependency mapping

<!-- _notes (中文):
最后一页，回到开头的诚实声明。
我们建了什么：一个诚实的、泄漏可控的 SL 基因对排序适配器（分类+排序），不是真正的 SL 靶点发现验证。
反复的纪律（recurring discipline）：
- 先做简单 floor（exp06 纯依赖性）；
- CV2/CV3 才是真正的 bar（CV1 可以被度数钻空子）；
- 负结果也算数（如果 exp07/08 没超过 exp06，那也是一个发现）；
- pending 就标 pending，不编数据。
通往真正的 context-specific SL 发现的路径：
- exp08 集群结果出来后，冻结 STATE+ESM2 adapter 能不能泛化到没见过的基因？
- exp09 selectivity 能不能带来鲁棒性？
- TCGA 患者上下文迁移——从细胞系到肿瘤的依赖性映射。
结束在诚实的 scope 上。
计时：约 1 分钟。
-->
```

- [ ] **Step 2: Verify slide break count and honesty callback**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md && grep -in 'honest, leakage-controlled SL-pair ranking adapter' docs/presentation/final-presentation.md
```
Expected: `13`; the callback line appears on slide 12.

- [ ] **Step 3: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add slide 12 (closing the loop)"
```

---

### Task 13: Appendix slide — SOTA results scoreboard

**Files:**
- Modify: `docs/presentation/final-presentation.md` (append appendix)

**Interfaces:**
- Consumes: slide 12.
- Produces: the scoreboard table from spec Section 8, all values to 4 decimals.

- [ ] **Step 1: Verify the scoreboard rows match the spec table exactly**

Run:
```bash
sed -n '115,127p' docs/superpowers/specs/2026-06-18-final-presentation-design.md
```
Expected: the 10-row scoreboard table (DDGCN/GRSMF/SL2MF literature + A/B/C/A_xcl/B_xcl/exp07/exp08) with its header. Transcribe verbatim.

- [ ] **Step 2: Append the appendix slide**

Append to `docs/presentation/final-presentation.md`:

```markdown
---

## Appendix — Results Scoreboard with SOTA Comparison

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

**Caveat:** Literature rows use different universe/splits/negatives — **context, not head-to-head**. Our NDCG = NDCG@10 (official per-anchor protocol). Within-harness comparison: B_xcl (exp09) lifts over B (exp06) on CV2 NDCG (0.0864 vs 0.0421).

<!-- _notes (中文):
附录页，完整的结果记分板。
文献方法（DDGCN/GRSMF/SL2MF）的 F1 很高（0.86-0.91），但它们用的是完整 SynLethDB 宇宙和它们自己的切分/负样本——不能直接比，只是提供上下文定位，不是 head-to-head leaderboard。
我们的 NDCG 是 NDCG@10（官方 per-anchor 协议）。
真正 apples-to-apples 的对比是 within-harness：B_xcl（exp09）在 CV2 NDCG 上超过 B（exp06）（0.0864 vs 0.0421）——这是同一个 harness、同一个随机状态、同一个 benchmark 的真正 ablation。
exp07/exp08 行明确标 pending。
这页不在主报告计时里，备用。
-->
```

- [ ] **Step 3: Verify slide break count and "pending" cells**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md && grep -c 'pending' docs/presentation/final-presentation.md
```
Expected: `14` breaks (2 front-matter + 12 content for 13 slides); `pending` appears 8 times (exp07 6 cells + exp08 6 cells, minus the slide 8/10 inline mentions, so total count varies — just confirm the appendix table has the two `pending` rows).

- [ ] **Step 4: Verify all scoreboard numeric cells are 4 decimals**

Run:
```bash
grep -E '0\.[0-9]{4}' docs/presentation/final-presentation.md | tail -20
```
Expected: all numeric cells in the appendix table match the 4-decimal pattern. If "pending" appears instead, that's correct for exp07/exp08 rows.

- [ ] **Step 5: Commit**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: add appendix slide (SOTA scoreboard)"
```

---

### Task 14: Full-deck verification and final commit

**Files:**
- Read: `docs/presentation/final-presentation.md` (verify the complete deck)

**Interfaces:**
- Consumes: all 13 slides (12 main + 1 appendix).
- Produces: a verified, commit-ready deck.

- [ ] **Step 1: Count total slides**

Run:
```bash
grep -c '^---$' docs/presentation/final-presentation.md
```
Expected: `14` (2 front-matter fences + 12 content breaks = 13 slides total).

- [ ] **Step 2: Verify both asset images are referenced**

Run:
```bash
grep -n 'SL_concept.png\|e2e_SL_DL.png' docs/presentation/final-presentation.md
```
Expected: `SL_concept.png` on slide 2, `e2e_SL_DL.png` on slide 9. Both lines present.

- [ ] **Step 3: Verify no forbidden "SL target discovery" claim (positive assertion)**

Run:
```bash
grep -in 'SL target discovery' docs/presentation/final-presentation.md | grep -iv 'not\|NOT'
```
Expected: no matches (only "not validated SL target discovery" or similar negations should appear). If a positive "SL target discovery" claim appears, STOP and fix.

- [ ] **Step 4: Verify all "pending" markers are present for exp07/exp08**

Run:
```bash
grep -in 'pending' docs/presentation/final-presentation.md | grep -E 'exp07|exp08|cluster gates'
```
Expected: slide 8 (exp07 results pending), slide 10 (exp08 cluster gates pending), appendix (exp07/exp08 rows pending). At least 3 matches.

- [ ] **Step 5: Spot-check Chinese notes are present**

Run:
```bash
grep -c '中文' docs/presentation/final-presentation.md
```
Expected: `13` (one per slide). If count differs, a slide is missing notes.

- [ ] **Step 6: Final commit with a summary message**

```bash
git add docs/presentation/final-presentation.md
git commit -m "docs: complete final presentation deck (12 slides + appendix, bilingual)"
```

---

## Self-Review Checklist

Run after Task 14 Step 6 (after the final commit).

**1. Spec coverage:**
- Slide 1 (title) ✓
- Slide 2 (SL concept) ✓
- Slide 3 (honest framing) ✓
- Slide 4 (data sources) ✓
- Slide 5 (Stage 1 exp01+02) ✓
- Slide 6 (Stage 2 bridge) ✓
- Slide 7 (exp06 floor) ✓
- Slide 8 (exp07) ✓
- Slide 9 (exp08 pt.1) ✓
- Slide 10 (exp08 pt.2) ✓
- Slide 11 (exp09) ✓
- Slide 12 (close) ✓
- Appendix (scoreboard) ✓

All 13 sections covered. No spec requirement dropped.

**2. Placeholder scan:**
- No "TBD", "TODO", or unfilled fields except the explicitly-marked "pending" for exp07/exp08, which is the honest status per the spec.
- All Chinese notes written in full; no "fill in speaker notes" placeholders.

**3. Type/name consistency:**
- exp06 models consistently named A (logreg), B (XGBoost), C (degree probe).
- exp09 models consistently named A_xcl, B_xcl.
- Numeric values cross-checked against source CSVs/docs in verification steps.
- Asset paths (`SL_concept.png`, `e2e_SL_DL.png`) consistent across tasks.

**4. Honesty guardrails:**
- "SL-pair classification / ranking, NOT validated SL target discovery" present (slide 3).
- exp08/exp07 marked pending everywhere (no fabricated metrics).
- Literature SOTA caveat present (appendix).
- CV2/CV3 as the honest bar stated (slides 3, 7, 12).

No issues found. Plan is ready for execution.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-18-final-presentation-deck.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**

