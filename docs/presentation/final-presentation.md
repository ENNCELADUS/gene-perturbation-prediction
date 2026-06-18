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
