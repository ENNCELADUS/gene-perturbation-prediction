以“**in silico reverse perturbation prediction（检索/反推扰动基因）**”为起点、且能自然延展到“**靶点发现**”的一套研究 roadmap。先找一个**应用场景**与**容易验证的目标**，用**可靠数据**讲一个**闭环故事**，而不是在小 benchmark 上刷分。

---

## 闭环生物故事主线

“给定一个异常细胞状态（例如炎症过强/疾病样表达谱），模型输出一个**可实验验证的候选基因干预列表（Top-K）**；用 Perturb-seq 的**真实扰动标签**与独立的已审阅证据（例如已验证调控因子列表）验证命中率与机制一致性，从而把评估对齐到‘靶点发现’而非重建误差。”这正对应你们 deep research 报告提出的闭环方向：用逆向任务直接对齐治疗靶点发现，并强调评估应以“检索到真实因子/已验证靶点”等为成功标准，而非仅低 MSE。

在 scGPT 原文里，这条叙事被明确写出：reverse prediction 可以“促进潜在治疗基因靶点的发现”，并举了“预测能让细胞从疾病状态恢复的 CRISPR 靶基因”的假想应用。

---

## Roadmap

### 阶段 0：Story and Eval metrics
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

4. 明确两层任务定义 
    第一层（核心 MVP）：Reverse Genetic Perturbation Identification（反推致因基因）。deep research 报告把它定义得非常清楚：输入是单细胞或平均后的表达向量，输出是候选扰动基因的**排序列表**，标签来自实验元数据，评价看 Top-K 命中/排序指标。  
    第二层（靶点发现语义化）：把“异常状态”解释为“疾病样/功能异常状态”，将 Top-K 解释为“候选驱动基因/干预靶点”。
    
5. 定义主指标与次指标  
    主指标建议直接采用检索式指标：Top-1/Top-5、MRR、NDCG（报告里也建议 reverse ID 用 MRR，整体用 NDCG/mAP）。  
    次指标（生物意义体现）：错误是否“生物学上合理”（例如混淆的基因是否共享 GO/通路），这也是报告强调的“生物一致性”评估。
    
6. 规定必须做的反作弊协议  
    报告明确提醒 reverse ID 存在“直接看被敲基因自身表达”为零的作弊路径，并给出协议：在输入中**mask 被扰动基因的表达**，迫使模型学习下游效应。  
    再加上“按扰动 ID 进行 hold-out split”的泛化评估（训练见过 80 个基因，测试完全没见过的 20 个基因），避免记忆签名。
    

交付物：一页“评价卡模板”（指标、split、反作弊、基线清单），以及最终选用数据集的决定标准（规模、标签、可获取性）。

---

### 阶段 1：复现 scGPT 的“逆向检索”设定（MVP 复现，2–4 天）

任务目标：严格复现 scGPT 在 Norman 子集上的“Top-K retrieval”框架，作为你们后续所有故事的技术锚点。

scGPT 在论文中把 reverse perturbation prediction 明确定义为检索任务：用所有候选扰动条件的“参考库”与测试的“query（真实扰动后细胞）”做相似度检索，目标是检索到能产生该结果的扰动条件。  
其具体实验设定也给出了可操作细节：从 Norman 选 20 个基因构成 210 个组合，训练 39 个组合，其余为未见组合；并用多 control cell 生成参考库。

可用数据资源（首选）  
Norman et al. 2019（K562，CRISPRa，单基因+双基因，约 25k cells，标签清晰），报告的数据表也把它列为“最适合显式标签逆向任务”的数据之一。  
scGPT 也在文中说明 Norman 有大量单/双基因扰动并用于多种扰动评估。

你们要做的不是“重新造模型”，而是“按论文做下游应用复现”：

1. 基线 1：差异基因（naive baseline）/最近邻检索（raw 或 PCA）
2. 基线 2：线性分类器（logistic regression）
3. scGPT：冻结或轻量微调（只训检索头/LoRA）（deep research 报告也明确建议用 scGPT 作为 pretrained encoder + 轻量头）。

交付物：复现图表与表格（Top-K 命中、MRR、混淆矩阵），以及“mask perturbed gene”后的对照结果，证明不是靠作弊信号。  
备注：scGPT 原文还给出其在该任务上“Top-1 relevant、Top-8 correct”等结果与实验意义（减少随机试错次数，辅助规划实验），你们可以把这段作为报告动机。

---

### 阶段 2：从“复现”升级为“可推广的检索系统”（1–2 周）

任务目标：把“20 基因玩具子集”扩展到更现实的规模，并把“检索=靶点发现”说清楚。

1. 扩展数据规模的两条路径  
    路径 A（更稳）：Norman 全集（单基因为主，双基因作为加分），保持噪声可控与故事连贯。  
    路径 B（更像真实筛选）：Replogle 2022（K562/RPE-1，基因 KO/CRISPRi，报告表格给出百万级规模与显式扰动基因标签），但工程更重，适合你们 4×A40 的资源。
    
2. 伪 bulk 策略（强烈建议作为主版本）  
    你已确认可以先做 pseudo-bulk，这会显著降低单细胞噪声，并让检索更像真实药物筛选的“signature matching”。deep research 报告也明确 reverse ID 输入可以是单细胞或“平均的 many cells”。
    
3. 研究点放在“泛化与可靠性”，而不是结构创新  
    你们的主研究问题可以写成：  
    “在**hold-out 未见扰动基因**与**跨批次**场景下，scGPT embedding 是否比 raw 表达/PCA 更能保留‘扰动的因果指纹’？”这与报告强调的 realistic split 完全一致。
    

交付物：一个“检索系统卡”（数据、split、指标、基线、消融、错误分析），以及一个“泛化曲线”（seen→unseen 的性能落差与解释）。

---

### 阶段 3：把检索任务真正落到“靶点发现”应用闭环（2–3 周）

任务目标：不改变你们的核心技术（仍是检索/反推），但把 query 从“随机扰动后状态”替换为“疾病样/功能异常状态”，让输出 Top-K 真正对应“潜在治疗靶点”。

这里直接采用 scGPT 文中的假想应用：预测“能让细胞从疾病状态恢复的 CRISPR 靶基因”。  
实现上你们不需要真的有“疾病患者+对应 perturb-seq”的完美闭环，只需要一个有**已验证调控因子列表**的“疾病样刺激模型”，作为 ground truth。

可用数据资源（优先推荐）  
Yao et al. 2024（LPS 免疫刺激 + CRISPR KO 调控因子）：报告表格说明它提供“被验证的 LPS 反应调控因子列表（KO 会削弱炎症）”以及显式扰动基因标签，这正好把“Top-K 候选”变成“已验证的靶点集合命中率”问题。

如何把它写成闭环故事（CS 友好、可验证）

1. 定义异常状态：LPS 刺激下“过强炎症转录程序”的表达 profile。
    
2. 目标：找出“哪些基因 KO 会把该异常程序拉回去”（候选靶点）。
    
3. 方法：用你们在阶段 2 建好的检索器，把“异常状态”作为 query，在扰动 signature 库中检索最能匹配“恢复/抑制炎症”方向的扰动（具体可用负相关的 signature 相似度或“接近健康”距离）。
    
4. 验证：Top-K 是否显著富集在 Yao 提供的“validated regulators”列表中（这就是明确标签）。
    

交付物：一张闭环图（异常状态→候选靶点→验证命中→机制解释），一张 Top-K 富集/PR-AUC 图，以及 2–3 个高质量 case study（例如你们 top-10 中哪些基因在免疫通路上聚类）。

---

### 阶段 4：机制解释层（不改模型主干，用“解释+外部证据”增强生物意义，1–2 周）

任务目标：回应“CS 学生不懂生物故事”的最大痛点：让结果不仅是一个列表，而是一个“为什么可信”的链条。

你们 deep research 报告明确提出闭环评估应落在“检索到已知 disease genes / validated regulatory targets / true drug–target”而非低 MSE，并建议利用 ENCODE/ChIP 证据等外部 ground truth。  
最稳妥的机制增强方式是两步：

1. 通路/基因集层面的解释  
    对 Top-K 候选做 pathway enrichment（GSEA/hypergeometric），报告也把它列为关键的“生物一致性”指标之一。
    
2. 调控靶基因层面的验证（可选加分）  
    如果你们选到的候选里包含 TF/调控因子，可以进一步用 TF target gene set（ENCODE/ChEA/MSigDB TFT 等）验证“预测到的下游效应是否合理”，报告给了这种验证思路（检查 Gene B 是否在 A 的 ChIP target 集合中）。
    

交付物：一页“机制证据卡”（Top pathways、关键下游基因、外部证据来源与命中统计），把“输出=靶点发现”讲成可审阅的科学论证。

---

## 资源与算力如何匹配你们的约束（最多 4×A40，3 天训练）

你们的定位是“下游应用与小改”，因此训练预算主要花在阶段 1–2 的 scGPT 轻量适配上。deep research 报告已经把“用 scGPT 作为 pretrained encoder、加轻量头/LoRA，在 3 天单 GPU 预算内完成”作为可行策略；你们有 4×A40，只会更稳。

同时，报告也提醒必须纳入线性/最近邻等强基线，因为已有研究显示复杂模型在某些扰动预测设置下未必优于简单方法；你们的贡献点应落在“任务定义 + 评估协议 + 生物闭环”。

---

## 你们最终报告的结构建议（与老师“讲故事”对齐）

1. 问题与动机：为什么 VCC 式指标不等价于靶点发现；为什么逆向任务更贴近应用（引用闭环定义）。
    
2. 方法：reverse perturbation 作为 Top-K retrieval（引用 scGPT 的任务设定）。
    
3. 实验 1（MVP）：Norman 子集复现与反作弊验证。
    
4. 实验 2（泛化）：hold-out genes + pseudo-bulk + 多基线对照。
    
5. 实验 3（靶点发现闭环）：Yao LPS 场景，命中 validated regulators + pathway 解释。
    
6. 讨论：你们的系统如何用于“规划扰动实验、减少试错次数”（scGPT 也强调这一点）。
