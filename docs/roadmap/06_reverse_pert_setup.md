以“**in silico reverse perturbation prediction（检索/反推扰动基因）**”为起点、且能自然延展到“**靶点发现**”的一套研究 roadmap。核心原则仍然是：先找一个应用场景与容易验证的目标，用可靠数据讲一个闭环故事，而不是在小 benchmark 上刷分。

## 闭环生物故事主线

你们的故事主线可以明确写成：我们把“reverse perturbation”从一个 abstract 的检索任务，落到一个真实的药物研发 workflow 里，也就是 phenotypic screening → target deconvolution。

具体叙事是这样的：给定一个观察到的细胞状态变化（transcriptomic phenotype / expression signature），模型输出 Top-K 候选“干预因子”（gene target）。在 CRISPR Perturb-seq（Norman）里，这个因子等价于真实扰动基因（ground-truth label 来自实验元数据）；在 drug perturbation（Tahoe）里，这个因子等价于药物的 target_gene（ground-truth label 来自 drug metadata）。于是同一个 reverse task，可以在两类数据上形成闭环：前者验证“能否从表达反推致因基因”，后者验证“能否从药物引起的 phenotype 反推药物靶点”，从而把研究的实际价值讲清楚：它不是为了重建表达，而是为了 target discovery / mechanism hypothesis generation。

我们在 Norman（Perturb-seq）上验证 reverse perturbation prediction 能从 expression phenotype 中 recover perturbation genes，证明模型学到的是 perturbation fingerprint 而不是 reconstruction trick；随后在 Tahoe 的 single-target drug subset 上，把同一 reverse task 迁移到药物场景，直接以 drug metadata 的 target_gene 作为 ground truth，评估模型是否能从 drug-induced transcriptomic phenotype 中反推出真实靶点（hit@K / MRR / NDCG）。这一步把研究从“方法学检索任务”闭环到“药物靶点发现”的实际应用：它对应 phenotypic screening 后的 target deconvolution，并为未知靶点化合物或传统药物提供可计算的 Top-K 机制假设与后续验证路径。

---

## Roadmap（更新版）

### 阶段 0：Story、Problem Formulation 与 Eval Protocol（1–2 天）

这一阶段的更新重点是把“同一套 reverse formulation”在基因扰动与药物扰动之间统一起来，并且把反作弊协议写成可复用模板。

正向问题仍然是 VCC 的 forward transition：给定基线状态与已知扰动，预测扰动后状态。反向问题是：给定“前后状态对”去反推扰动 identity，本质上是一个 in silico causal identification / retrieval problem。

#### Problem formulation

1. **正向问题**
这是目前vcc的主任务：
$$
\text{given } (x, p) \;\Rightarrow\; \hat y = f(x, p)
$$
- $x$：扰动前的 cell state。用**同一批次/同一细胞类型的 control 细胞分布**来代表“基线状态”。scGPT 在 Norman 数据上就是用“所有 control 细胞表达的平均向量（1×M genes）”
- $p$：已知扰动（基因 KO / CRISPRi / drug）。可以视为离散的 gene ID（或 gene pair）。
- $\hat y$：预测的扰动后表达状态。数据集提供扰动后细胞表达 $y$（用来监督/验证 $\hat{y}$​）。

2. **反向问题（reverse perturbation prediction）**
in silico reverse perturbation prediction 则是： 给定 $(x, y)$ 反推扰动，本质是一个逆问题：根据观测到的$x$ 和 $y$ 找最可能的 $p$ 。
$$
\text{given } (x, y) \;\Rightarrow\; \hat p = g(x, y)
$$
$$
\hat p = g(x, y) = \arg\min_{p} d\!\left(f(x, p), y\right)
$$
3. **靶点发现**
$$
p^{*} = g(x_{\text{abn}}, y_{\text{desired}}) = \arg\min_{p} d\!\left(f(x_{\text{abn}}, p), y_{\text{desired}}\right)
$$

- $x_{\text{abn}}$：异常（疾病样）细胞状态
- $y_{\text{desired}}$：期望（被“治好”后的）目标细胞状态
	- 可取为健康/对照状态的代表向量，或某个功能性目标状态。
- $p$：候选干预/靶点（perturbation / target）
	- 在基因扰动场景通常是离散集合中的一个元素：某个基因的 KO/CRISPRi/CRISPRa（或基因对）。
- $f(x_{\text{abn}}, p)$：前向干预效应函数（intervention effect / transition model）。
- $d(\cdot,\cdot)$：距离/代价函数（misfit / loss）
	- 衡量干预后的预测状态与目标状态有多接近；越小表示越接近“治好”目标。

1. 路线 A：基于前向模型 $f$ 的枚举/检索（你写的 $\arg\min$ 形式）
   - 用 $f$ 对每个候选 $p\in\mathcal{P}$ 预测 $f(x_{\text{abn}},p)$。
   - 计算与目标的距离 $d(f(x_{\text{abn}},p), y_{\text{desired}})$。
   - 取最小者作为 $p^*$（或取 Top-K 最小者作为候选靶点列表）。
   - 这里的 $g$ 不是单独训练的判别器，而是“由 $f$ 与 $d$ 诱导出的求解算子”。

2. 路线 B：直接学习逆映射（判别式/监督式）$g_\theta$
   - 原本用于“反推致因基因”：给定观测后态 $y$，预测是谁造成的（例如 $\hat p = g_\theta(y)$ 或 $\hat p=g_\theta(x,y)$）。
   - 对应到“靶点发现”：把输入改成“当前状态与目标状态”，学习 $p^* \approx g_\theta(x_{\text{abn}}, y_{\text{desired}})$。
   - 数学上仍是在逼近同一个最优化算子 $g$：只是你不显式写出 $f$ 与 $d$，而是让模型从训练数据中直接学习“哪种 $p$ 能把某类状态推向某类目标”。
   - 语义差异：
     - 路线 A 的依据是“通过 $f$ 预测出来的结果与目标的匹配程度”。
     - 路线 B 的依据是“从数据中学到的状态 $\to$ 干预映射”，它隐式吸收了 $f$ 与 $d$ 的作用。

在 Norman（基因扰动）里，你们的核心 MVP 是 Reverse Genetic Perturbation Identification：输入是 perturbed expression（可选带 control baseline），输出是 perturbation gene（或 condition）的 Top-K 排序，标签来自 Perturb-seq 元数据，评价用 hit@K / MRR / NDCG 等排序指标。

在 Tahoe（药物扰动）里，完全对应的定义是 Reverse Drug Target Identification：输入是 drug-treated vs DMSO control 的 paired profile（或者 delta signature），输出是 target_gene 的 Top-K 排序，标签来自 drug metadata，并且因为你们只选 single-target subset，所以监督空间是 clean 的 single-gene label space，验证也直接。

反作弊协议需要在两条数据线上各写一条“对应版本”。在 Norman 里，报告里强调的作弊路径是“直接看被敲基因自身表达接近 0”，所以必须做 mask perturbed gene expression 的 ablation。 在 Tahoe 里不存在“敲到 0”的同构作弊，但仍然可能出现“target gene expression 本身极端变化导致 shortcut”的弱作弊，因此你们可以把同样的 masking 作为 robustness check（mask target_gene 的表达维度，再看 hit@K 是否仍然显著高于 baseline），以证明模型在学习下游 network effect，而不只是抓住一个单点信号。

交付物建议更新为两份卡片：一份 Norman Reverse-ID Eval Card（condition split、strata、mask 策略、指标、基线）；一份 Tahoe Target-ID Eval Card（data selection、ctrl pairing、overlap_ratio、label space、指标、基线）。

---

### 阶段 1：Norman 上复现 “reverse retrieval” 的 MVP（2–4 天）

这一阶段从“玩具 20-gene 子集复现”升级为“可落地的 Norman full split 复现”，并且明确你们会同时支持 scGPT-style route 与 compositional route。

数据切分采用你们已经定稿的 condition-level split（split unit 是 condition，不是 cell），并且包含 GEARS 语义的 seen/unseen 分层：single_unseen、combo_seen2、combo_seen1、combo_seen0，这个分层将直接成为你们后续泛化叙事的 backbone。

技术上，阶段 1 的目标不是做复杂建模，而是把“reverse retrieval 的端到端链路”跑通，并拿到一个可信的、可复现的表格结果：raw/PCA 最近邻、logistic regression、以及 scGPT（frozen encoder + light head 或 forward-finetune + retrieval）在同一 split 与同一指标上的对比。你们在 08_scGPT_route 里已经把 Route A（Forward + Retrieval）与 Route B1（Gene-level scoring + compositional ranking）定义得足够工程化，阶段 1 只需要做最小实现并验证 metric pipeline 正确。

交付物仍然是：hit@K、MRR、NDCG（以及 exact vs one-gene-overlap 的 relevant hit），并且必须包含 “mask perturbed gene” 的结果作为 sanity check。

---

### 阶段 2：把 Norman 的 MVP 升级为“可推广的检索系统”（1–2 周）

这一阶段的更新重点是把“系统性”写清楚：同一套 split、同一套 evaluator、同一套 anti-leak 协议，支持两条路线并进行可解释 error analysis。

第一，固定 condition split 与 strata reporting。你们已经有具体可复现的 split artifact（含条件数、细胞数、以及 test strata 分解），阶段 2 的产出要在 paper-style 图表里稳定呈现：overall 与分层（single_unseen / combo_seen0/1/2）都要报。

第二，Route A 与 Route B1 的对照要被写成“同任务、不同 inference mechanism”的比较，而不是“两个模型谁分数高”。Route A 强调 forward generator 的覆盖与 retrieval over hypotheses；Route B1 强调 compositional generalization（对 unseen combos 的自然支持）。这套 framing 你们的 08_scGPT_route 已经写好，阶段 2 需要把它落成稳定的实验矩阵。

第三，错误分析要从“数值误差”转为“生物一致性”。例如错误检索到的基因是否落在同一 pathway/GO term，是否共享上游 TF 或 signaling module。你们可以把这作为次指标，强化“不是刷 MSE，而是在做 mechanism-aware retrieval”。

交付物是一个系统卡（data/split/metrics/baselines/ablations/error analysis）加一张 seen→unseen 的泛化曲线或分层柱状图。

---

### 阶段 3：用 Tahoe 把 reverse task 落到“药物靶点发现”的真实闭环（2–3 周，重点更新）

阶段 3 的核心目标是：在不改变你们研究的 “reverse perturbation prediction” 本质前提下，把它放进一个真实的 drug discovery 场景里，回答审稿人最关心的一句话：So what？它能做什么？

你们要讲的价值点可以明确写成三层。

第一层价值是场景对齐（phenotypic screening → target deconvolution）。药物研发里存在大量 phenotype-first 的筛选：先看到细胞状态被“改善/改变”，但不知道药物打到了哪里。你们的 reverse task 正是把“表达 signature（phenotype readout）”映射回“候选干预因子（target gene）”，这与 Norman 的“从 perturbation signature 反推扰动基因”在形式上同构，只是 label 从 perturbation gene 变成 drug target_gene。你们在前面讨论里提到的“从表型出发，再回推靶点，让传统药物/复杂药物更可解释”，在 Tahoe 上可以用一个非常干净、可量化的方式落地。

第二层价值是可验证性（easy-to-verify objective）。Tahoe 的 data selection 已经为你们构造了一个“单靶点药物”的监督空间：只保留 single-target drug；把 Tahoe token_id 映射到 scGPT vocab gene id；为每个 treat 样本配对一个 DMSO control（按 cell_line_id 与 plate key，取第一条匹配 DMSO 以加速）；并用 overlap_ratio 做质量过滤，同时把 ctrl 表达对齐到 treat 的基因顺序（缺失补 0），保证输入是严格对齐的 paired profile。这样你们可以定义一个非常标准的 benchmark：输入 (treat, ctrl) 或 delta，输出 target_gene label，评价用 hit@K / MRR / NDCG，Top-10 里命中即是成功。它是“闭环”的，因为 ground truth 就在 metadata 里，不需要额外 wet lab。

第三层价值是方法可迁移（from labeled target to unlabeled mechanism）。一旦你们在 Tahoe 的 single-target subset 上证明：模型能从 drug-induced transcriptomic phenotype 里 recover 出正确 target（显著优于 PCA/nearest neighbor/linear classifier baseline，并且在 mask target gene 的鲁棒性测试下仍成立），你们就有非常自然的外延：对没有明确靶点的化合物、对 multi-target drug、对传统药物提取物，模型可以输出 Top-K targets 作为 hypothesis list，再结合结构侧的 docking/binding prediction 或 pathway enrichment 做 computational validation。这正对应你们前面讨论稿里强调的“phenotypic screening + computational validation”的闭环出口。

在实现层面，阶段 3 建议你们把“Tahoe 任务”写成对阶段 1/2 evaluator 的最小改动复用，而不是另起炉灶。

输入形式建议有两个等价版本，作为 ablation。

版本 A：paired profile 直接输入，即把 (ctrl aligned to treat genes, treat) 作为模型输入（或者拼接、或者差分再拼接），输出 target_gene 的 Top-K 排名。

版本 B：signature matching 风格，输入 delta signature（treat - ctrl），更贴近 CMAP-like 的传统表达签名检索叙事，同时也更容易解释“模型是在用 phenotype signature 推断 target”。

模型路线方面，你们可以直接复用阶段 2 的 Route B1 思想：输出 gene-level scores（这里的 label space 就是 target_gene），然后把 Top-K gene scores 作为候选靶点列表；如果你们希望和 “retrieval” 叙事更一致，也可以把每个 target_gene 的 reference signature（按药物或按样本聚合）建成一个库，对 query drug profile 做检索，最终返回 target_gene。两者都能与阶段 0 的 reverse formulation一致，只是 scoring function 不同。

最关键的交付物应当是一个“应用闭环图”加一组核心实验表格。

闭环图建议包含四个模块：drug-treated transcriptomic phenotype（输入）→ reverse model outputs Top-K target genes（机制假设）→ 与 metadata 中已知 target 的 hit@K 验证（闭环）→ 对 unknown/complex compounds 的 extension（应用前景）。

核心表格至少包含：PCA/NN baseline、logistic regression baseline、scGPT encoder + head（或其他 backbone）在同一 Tahoe 数据处理流程下的 hit@1/5/10、MRR、NDCG，并且附带 overlap_ratio 阈值与 ctrl pairing 策略的敏感性分析（例如 first-DMSO vs multi-DMSO aggregation，overlap_ratio=0.05 vs 更严格阈值）。这些细节在 dataset selection 文档里已经明确是关键变量，阶段 3 要把它变成你们 story 的可信度来源，而不是工程噪声。