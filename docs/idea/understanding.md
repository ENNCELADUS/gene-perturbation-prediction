# 1. Replogle K562 数据在回答什么问题？

你的 Replogle K562 数据是：

```text
310,385 cells × 8,563 genes
```

它的基本单位是 **单细胞**。

每个 cell 大概有几类关键信息：

```text
这个细胞来自 K562
这个细胞被哪个 guide / perturbation gene 命中
这个细胞扰动后测到的 8,563 个基因表达量
```

所以 Replogle 数据回答的是：

```text
在 K562 里 knockdown 某个 gene 后，
还被捕获到的细胞，其转录组状态变成了什么样？
```

生物意义是：它测的是 **扰动后的细胞状态**，也就是细胞面对某个 gene knockdown 后的 transcriptional response。

它不是直接告诉你这个细胞死没死。因为能进 scRNA-seq 的细胞，多数还是被捕获到、还存在的细胞。它更像是在看：

```text
细胞被打击以后，内部状态有没有出现 stress、cell cycle 改变、代谢变化、DNA damage response、apoptosis-like signal 等。
```

---

# 2. DepMap GeneEffect 文件在回答什么问题？

DepMap 的 `CRISPRGeneEffect.csv` 是一个 **cell line × gene** 的矩阵。

概念上长这样：

```text
                A1BG     A2M     AURKA    BCL2    RPL5    ...
ACH-000551      0.01    -0.03    -1.25   -0.12   -1.60   ...
ACH-000XXX     -0.02     0.04    -0.80   -0.55   -1.30   ...
ACH-000YYY      0.00    -0.01    -1.10    0.02   -1.45   ...
```

其中：

```text
行 = cell line / DepMap model ID
列 = gene
值 = knockout 这个 gene 后对该 cell line fitness 的影响
```

DepMap 官方说明里，`CRISPRGeneEffect.csv` 是来自 CRISPR knockout screens 的 gene effect estimates，并经过 Chronos 整合、copy-number correction、scaling 和 screen-quality correction。负值表示 gene loss 造成 growth defect / negative selection。([DepMap Community Forum][1])

对你来说，最重要的是这一行：

```text
ACH-000551 = K562
```

所以你会从 `CRISPRGeneEffect.csv` 里面取：

```text
K562 这一行
```

然后得到：

```text
每个 gene 在 K562 里被 knockout 后的 gene effect 值
```

---

# 3. GeneEffect 的数值是什么意思？

Gene effect 是一个连续值。

大致可以这样理解：

```text
gene_effect ≈ 0
说明 knockout 这个 gene 后，K562 细胞增殖/存活影响不大。

gene_effect < 0
说明 knockout 这个 gene 后，K562 细胞 fitness 下降。

gene_effect 越负
说明这个 gene 对 K562 越重要，loss 后越可能造成 growth inhibition / death。

gene_effect ≈ -1
通常接近 common essential genes 的强度。
```

DepMap FAQ 里也说明，gene effect 衡量 knockout gene 的 effect size，并且是相对于 non-essential 和 pan-essential genes 的分布归一化的；常用经验是 gene effect 小于 -0.5 往往表示 depletion，小于 -1 表示 strong killing。([DepMap Community Forum][2])

所以你问：

> 是对应 1967 组扰动标签的 gene effect 值吗？

**是的，逻辑上就是这样。**

你的 1967 个 matched perturbation genes，意思是：

```text
这些 gene 同时出现在：
1. Replogle K562 perturbation 数据里
2. DepMap K562 ACH-000551 的 gene effect 列里
```

因此每个 matched gene 都可以拿到一个 DepMap K562 gene effect 值。

对应关系是：

```text
Replogle:
perturbation_gene = G
→ K562 中 knockdown G 后的 post-perturbation expression profile

DepMap:
cell_line = K562 / ACH-000551
gene = G
→ K562 中 knockout G 后的 gene_effect
```

合起来就是：

```text
G 的扰动后表达谱  ↔  G 在 K562 里的 essentiality / fitness effect
```

---

# 4. 它是不是代表“每组扰动对 K562 的威胁有多大”？

**大体上可以这么理解，但要加三个限定。**

## 第一，它是 population-level 的威胁

DepMap gene effect 不是单细胞层面的“这个细胞死了没”。

它反映的是：

```text
在一个 pooled CRISPR screen 里，
knockout 某个 gene 后，
携带这个 perturbation 的细胞群体是否逐渐减少。
```

所以它是 **population fitness / survival readout**。

更准确的说法是：

```text
gene_effect 表示 loss of this gene 对 K562 细胞群体增殖和存活能力的影响。
```

而不是：

```text
某个单细胞的立即死亡概率。
```

## 第二，它是 knockout 的效应，而 Replogle 是 CRISPRi knockdown

你的 Replogle K562 essential 数据是 CRISPRi-compatible knockdown，DepMap 多数是 CRISPR knockout screen。

所以两边不是完全同一个实验。

```text
Replogle:
CRISPRi knockdown → 转录组状态

DepMap:
CRISPR KO → long-term fitness effect
```

它们都属于 loss-of-function，但强度、时间尺度、实验平台不同。

所以 DepMap gene effect 不能理解成 Replogle 这个 exact perturbation 的直接测量结果，而应该理解成：

```text
同一个 gene 在 K562 中 loss-of-function 后的外部 fitness 标签。
```

## 第三，它更多表示 essentiality / dependency，不是严格的 cell death

DepMap gene effect 很负，说明这个 gene loss 后 K562 细胞难以维持增殖或存活。

这可能来自：

```text
细胞死亡
细胞周期停滞
增殖变慢
代谢崩溃
长期竞争中被淘汰
```

所以你可以把它叫做：

```text
fitness threat
dependency strength
essentiality strength
vulnerability
```

但不要一开始就严格说成：

```text
直接死亡概率
```

更严谨的表述是：

```text
DepMap gene effect 表示该 gene loss 对 K562 fitness 的损害程度；这个损害可能体现为死亡、增殖缺陷或长期 depletion。
```

---

# 5. 表达谱如何对应到 DepMap？

关键不是“单个 cell 对应 DepMap”，而是：

```text
一组被同一个 gene perturb 的 cells
对应
DepMap 里同一个 gene 在 K562 的 gene effect
```

比如以 gene `RPL5` 为例：

## 在 Replogle 里

你会有很多个细胞被标注为：

```text
perturbation_gene = RPL5
```

这些细胞每个都有 8,563 genes 的表达量。

你可以把这些细胞理解成：

```text
K562 在 RPL5 knockdown 后的一群细胞状态样本
```

## 在 DepMap 里

你找：

```text
row = ACH-000551
column = RPL5
```

得到一个数：

```text
gene_effect(RPL5, K562)
```

这个数表示：

```text
RPL5 loss 对 K562 fitness 的影响
```

## 两者连接起来

所以你得到一条概念上的样本：

```text
RPL5 perturbation 后的 K562 转录组状态
→
RPL5 在 K562 里的 gene effect
```

对 1967 个 gene 重复这件事，就得到 1967 组：

```text
post-perturbation transcriptome ↔ essentiality / gene effect
```

这就是你的核心数据结构。

---

# 6. context 在这里是什么？

你问得很关键。

在最窄的意义上：

```text
context = K562
```

也就是：

```text
同一个 perturbation gene，在 K562 这个细胞背景下的效应。
```

为什么 context 重要？

因为 gene essentiality 是 context-specific 的。同一个 gene：

```text
在 K562 里可能很 essential
在 A549 里可能不 essential
在 HCT116 里可能中等 essential
在正常细胞里可能不同
```

所以 DepMap 的每个 gene effect 值都不是“这个 gene 普遍多危险”，而是：

```text
这个 gene 在某个具体 cell line context 下 loss 后有多危险。
```

在你的第一版数据里，context 几乎被固定住了：

```text
cell line = K562
DepMap ID = ACH-000551
cancer type / lineage = leukemia / blood lineage
genetic background = K562 的特定突变、拷贝数、表达状态
```

所以你现在不是在学：

```text
不同 context 下，同一个 gene 的 essentiality 怎么变
```

而是在学：

```text
在 K562 这个固定 context 里，
不同 gene perturbation 引起的转录组状态，
是否能对应到不同的 essentiality 强度。
```

更完整地说，context 包括：

```text
cell line：K562
lineage：myeloid / leukemia-like background
genetic background：K562 的 mutation / CNV / expression baseline
perturbation modality：CRISPRi knockdown vs CRISPR KO
实验条件：time point、culture condition、assay platform
```

但在你当前最核心的 K562-only 设置里，最主要的 context 就是：

```text
K562 cell-line context
```

---

# 7. 这一整套数据在生物学上形成什么链条？

可以画成这样：

```text
节点 A：扰动
knockdown gene G in K562

        ↓

节点 B：扰动后的转录组状态
K562 cells after perturbation G
8,563-gene expression response

        ↓

节点 C：fitness / essentiality 结果
DepMap K562 gene_effect for gene G
```

Replogle 给你的是：

```text
A → B
扰动 → 转录组变化
```

DepMap 给你的是：

```text
A → C
扰动 → fitness / essentiality 后果
```

你现在要理清楚、后面可能要学习的是：

```text
B → C
转录组状态 → fitness / essentiality 后果
```

生物意义就是：

```text
如果一个 gene loss 最终会伤害 K562，
那么在扰动后还存活、被测到的细胞中，
是否已经能看到某些“危险状态”的转录组迹象？
```

这些危险状态可能是：

```text
cell cycle collapse
DNA damage response
ribosome / translation stress
mitochondrial dysfunction
ER stress
apoptosis-related response
loss of proliferation program
```

所以你不是简单把 DepMap 贴到单细胞上，而是在构造一个因果链条的两个观测边：

```text
Replogle：看扰动后的内部状态
DepMap：看扰动后的长期生存后果
```

---

# 8. 最核心的一句话

你现在的 1967 个 matched genes 可以理解为 1967 个 K562 loss-of-function perturbation 条件：

```text
每个条件都有：
1. Replogle 给出的扰动后单细胞表达状态
2. DepMap 给出的同一 gene 在 K562 中的 fitness / essentiality 后果
```

所以，DepMap gene effect 确实可以作为这 1967 组扰动的外部标签，但它代表的是 **K562 context 下 gene loss 对细胞群体 fitness 的损害程度**，不是单细胞即时死亡标签；context 在当前阶段主要就是 **K562 这个细胞系及其遗传/转录组背景**。

[1]: https://forum.depmap.org/t/crispr-ko-screens/4010 "CRISPR KO screens - Q&A - DepMap Community Forum"
[2]: https://forum.depmap.org/t/depmap-genetic-dependencies-faq/131 "DepMap Genetic Dependencies FAQ - Q&A - DepMap Community Forum"
