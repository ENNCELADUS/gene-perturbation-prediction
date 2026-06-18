# Final Presentation Speech Draft

Date: 2026-06-18

Scope: speech draft only. This file follows the final-presentation plan and design
spec, but it does not modify the Marp deck, the plan, or the design spec.

Language: Chinese oral script, with model names, experiment IDs, and metrics kept
in English or numerals where clearer.

Update note: exp07 is no longer pending. The oral script includes the completed
Tier-1 no-coverage-flag result from
`docs/experiment/07_k562_sl_pair_perturbseq_augmented.md`. exp08 remains pending.

## Slide 1 - From SL Promise to an Honest Ranking Model

大家好，我今天汇报的是 K562 合成致死候选基因对排序项目。

一句话概括：我们想把三类信息接在一起，第一是 DepMap 的基因依赖性，第二是 Perturb-seq 观测到的扰动转录组响应，第三是冻结的扰动基础模型 STATE，然后用它们来排序 K562 里面可能的 synthetic-lethal gene pairs。

这里我会反复强调一个词：honest。这个项目目前做的是 benchmark 上的 SL-pair classification 和 ranking，不是已经经过实验验证的 SL target discovery。所以后面每个结果都会围绕一个问题展开：在不泄漏、不过度声称的前提下，我们到底能不能比简单 baseline 更好？

## Slide 2 - What is synthetic lethality and why it is hard

先从 synthetic lethality 的概念开始。

如果两个基因单独失活时细胞都还能活，但是两个同时失活时细胞死亡，这就是 synthetic lethality。对癌症来说，这个概念有吸引力，因为肿瘤里可能已经有一个基因因为突变或者缺失而失活；如果我们找到它的 synthetic-lethal partner，就可能得到一个更有选择性的药物靶点。

临床上最经典的例子是 BRCA1/2 突变肿瘤里的 PARP inhibitors。这个例子说明 SL 不是纯理论概念，它确实可以转化成治疗策略。

但困难在规模和 context dependence。人类基因对大约有两亿个，而且一个基因对是不是 SL 往往依赖细胞系、突变背景和组织来源。实验上不可能逐一筛，所以我们需要一个计算模型，先把候选 partner 排序出来。

## Slide 3 - The task we actually solve

这一页是整个报告的边界声明。

我们用的是 Feng et al. 2024 的 SynLethDB-derived benchmark，负样本是 Rand 1:1 balanced。过滤到 K562 和 DepMap 后，候选空间是 9,471 个基因。

它有三种 split。CV1 是 pair-level split，也就是说基因本身可能在训练集出现过，只是这个组合没见过，所以它最容易，也最容易被 gene degree 这类 shortcut 利用。CV2 是一个基因 unseen，难度中等。CV3 是两个基因都 unseen，是最难的 cold-start setting。

所以我们真正解决的是 SL-pair classification 和 ranking，不是 validated SL target discovery。这里的 Rand negatives 只是未确认的 non-SL，不等于真实实验验证的非 SL。后面所有模型 claim，我都会重点看 CV2 和 CV3，因为这两个才是更诚实的泛化表面。

## Slide 4 - Data sources and why each one

这个项目用了四类数据，每一类都有明确角色。

DepMap CRISPRGeneEffect 是 dependency label，也就是我们记作 C 的信号。它是 population-level fitness readout，K562 对应的 cell line ID 是 ACH-000551。

Replogle K562 gwps Perturb-seq 提供扰动后的转录组响应。它的重要性在于这是 CRISPRi，也就是 loss-of-function，所以和 DepMap 的 CRISPR knockout label 在方向上比较一致。这个数据大约有 199 万个细胞，覆盖 6,070 个候选基因。

ESM2-650M 提供基因身份的连续表示。它对 held-out genes 很关键，因为 one-hot 对 unseen gene 没有语义，而 protein embedding 至少能把没见过的基因放进一个连续空间。

最后，Feng 2024 和 SynLethDB 给我们 SL-pair label，也就是 D。

为什么从 K562 开始？因为 K562 是这个 proof-of-concept 里唯一同时有深度 Perturb-seq 和 DepMap dependency 的细胞系。

## Slide 5 - Stage 1: observed transcriptome predicts dependency

Stage 1 先回答一个基础问题：观测到的扰动转录组，能不能预测这个基因在 DepMap 里的 dependency score？

exp01 用 pseudobulk delta expression，接 PCA 和 Ridge。结果是 Replogle 五折交叉验证大约 0.49 Spearman，迁移到 Adamson 数据也有 0.50 Spearman，AUROC 是 0.886。这个结果说明 transcriptome-to-dependency 这条桥是通的。

然后 exp02 做了一个必要审计：这个信号会不会只是模型学到了一个通用的死亡或者 viability axis？结果不是。NAR viability score 单独预测只有 0.244，而最好的 pseudobulk baseline 是 0.494。把 NAR axis residualize 掉以后，转录组仍然有 0.503。

所以这里的结论是：dependency prediction 里确实有 transcriptomic signal，不只是“所有细胞都快死了”这种泛化死亡信号。

## Slide 6 - Stage 2: dependency-aware representation as the bridge

Stage 2 是从简单 pseudobulk 往 single-cell representation 过渡，也是整个故事里的 bridge。

exp03 做 single-cell set learning。最后最强的是 scVI128 delta，加 frozen-GMM distribution regression。它在 Adamson transfer 上达到 Spearman 0.666，AUROC 0.911，比 attention MIL 更稳；而且 HVG setting 反而伤害结果。

exp04 进一步做 leakage-free predicted-B loop，也就是 forward A to B to C，评估时不使用 test gene 自己的观测 response bag。

exp05 接上 frozen STATE forward model，形成一个 STATE-based 的 A to B to C pipeline。

这页的核心 takeaway 是：前面这些实验不仅是在做 dependency regression，它们给了我们一种 dependency-aware representation。也就是说，我们有办法把一个 perturbation 变成和 dependency 相关的 embedding。这个 embedding 思路会被带到后面的 SL-pair task 里。

## Slide 7 - Stage 3: dependency-only floor

进入 Stage 3 后，任务从单基因 dependency prediction 变成基因对 SL ranking。

exp06 是最简单但非常重要的 baseline。输入只有两个基因在 K562 里的 DepMap GeneEffect scalars，然后构造五个 swap-invariant features：min、max、sum、product 和 absolute difference，再预测这个 pair 是不是 SL。

最强的 XGBoost baseline 在 CV2 上 AUROC 是 0.704，NDCG@10 是 0.042；到 CV3，AUROC 掉到 0.596，NDCG@10 只有 0.002。这说明 dependency-only features 对 both-gene cold-start 基本不够。

同时 degree probe 在 CV1 的 NDCG@10 到了 0.197，超过 XGBoost 的 0.0505。这证明 CV1 很容易被 train-positive degree 这种 graph shortcut 利用。所以 exp06 给了我们一个很清楚的 bar：后面的模型必须在 CV2/CV3 上超过它，不能只看 CV1。

## Slide 8 - Does observed Perturb-seq add lift? exp07 update

exp07 的问题是：在 exp06 的 GeneEffect pair features 之上，加上 Replogle gwps 观测到的每个基因的 perturbation response embedding，会不会带来提升？

覆盖率是关键限制。Replogle gwps 单基因覆盖是 6,070 / 9,471，也就是 64.09%。如果按随机 pair 估算，两个基因都 covered 大约是 41%。在实际 CV1/CV2/CV3 Rand benchmark rows 里，both-covered fraction 是 51.17%，所以大约一半 pair 至少有一个基因要走 fallback embedding。

完成的 run 是 Tier-1: PCA-delta mean-pool，zero fallback，不加 coverage flag。这个设置很重要，因为它避免模型直接用 coverage flag 去学 well-studied gene 或 graph degree shortcut。

主要结果是 CV2 positive。对于最强的 XGBoost head，B_transcript 把 CV2 AUROC 从 0.704 提到 0.751，把 NDCG@10 从 0.042 提到 0.094，MAP@10 从 0.034 提到 0.079。也就是说，observed Perturb-seq features 在 one-gene-held-out ranking 上确实带来明显增益。

CV3 仍然困难。B_transcript 的 CV3 AUROC 从 0.596 提到 0.630，但是 NDCG@10 没有改善，0.002 变成 0.001。Logistic head 的 CV3 ranking 有一点提升，NDCG@10 从 0.004 到 0.008，但这还不是一个 clean solution。

所以 exp07 的诚实结论是：observed transcriptome helps under CV2，但是还没有解决 both-gene cold-start 的 top-k ranking。这个结果是正面的，但它的边界也很清楚。

## Slide 9 - e2e DL centerpiece pt.1: problem and architecture

exp08 是这次报告的方法中心。这里这张图展示的是它的完整网络架构：左边和中间的主干是 train 和 eval 共享的 forward path，右边红色虚线框是只在训练时使用的 supervision。

问题来自本地 STATE checkpoint 的 closed-vocab one-hot pert_encoder。它只有 2,024 个基因，其中和 SL universe 重合的是 1,542 个，也就是 9,471 个候选基因里的 16.3%。换句话说，83.7% 的 SL universe 对 STATE 来说是 out-of-vocab。

这对 SL benchmark 尤其麻烦，因为 CV2/CV3 里面有 held-out genes，而 one-hot perturbation ID 对这些基因没有连续语义。

所以我们不再直接输入 one-hot perturbation ID，而是从每个基因的 ESM2 embedding 开始。具体做法是：先从 SL benchmark pairs 里收集所有 unique gene symbols，对每个基因查询 UniProt 的 reviewed human canonical protein sequence，然后把氨基酸序列送进 ESM2，对 residue-level hidden states 做 mean-pooling，去掉 special tokens，得到每个基因一个 1280 维 embedding。这些 embedding 只生成一次，并缓存给所有 CV folds 复用；没有解析到 UniProt 序列的基因走 fallback strategy。

图中蓝色的 Perturbation Adapter 是 trainable 的，它把 1280 维 ESM2 vector 映射成 328 维 perturbation token，用来替代原始 STATE pert_encoder 的输出。后面的 STATE transformer 和 decoder 全部 frozen。

给定这个 perturbation token 和固定的 K562 control-cell template，frozen STATE 会预测这个基因的 post-perturbation response bag。然后我们用非参数的 MeanStd pooling，把 response bag 转成固定长度的 per-gene embedding，也就是图里的 e_g。到这里，每个候选基因都得到一个在同一坐标系里的 response-based representation。

## Slide 10 - e2e DL centerpiece pt.2: leakage-safe training and the bar

Slide 10 接着讲图的下半部分：怎么从 per-gene embedding 变成 gene-pair SL score，以及红色监督信号什么时候使用。

对一个 gene pair，我们取出 pooled embeddings e_a 和 e_b，再加上 GeneEffect features，送进 symmetric pair head。这个 head 被设计成 f(e_a, e_b) = f(e_b, e_a)，因为 synthetic lethality 是 unordered gene-pair relation。输出就是 P(SL)，也就是这个 pair 是 synthetic lethal 的预测概率。

训练时有三类 supervision，也就是图里红色虚线部分。第一，L_SL 是 train pair labels 上的 binary cross-entropy。第二，L_bag 把 predicted response bag 对齐真实 GWPS response bags，但只用于 covered train genes。第三，L_distill 把 adapter 输出锚定到 STATE 原始 one-hot pert_encoder token，但只用于 STATE 原始词表里存在的基因。

评估时这些红色监督 target 都不用。我们只是对所有 universe genes 生成 e_g，用训练好的 pair head 填满 SL score matrix，然后计算 ranking metrics。

关键的 leakage rule 是：CV2/CV3 held-out genes 只能通过 adapter(ESM2) 加 frozen STATE 进入模型，不能用它们自己的 observed response bag 做监督。这样评估才是真正的 unseen-gene generalization。

要打败的 bar 仍然是 exp06：CV2 AUROC 0.704，CV3 AUROC 0.596，同时要看 ranking，尤其是 CV2/CV3 的 NDCG@10 和 covered-pair slice。

这里也要和 exp07 区分：exp07 已经证明 observed gwps features 在 CV2 上有 lift；exp08 要进一步问的是，能不能通过 ESM2 adapter 加 frozen STATE，把这种 response representation 推到 held-out 或 OOV genes 上。当前状态是 code 和 unit tests complete，CV2/CV3 cluster gates 仍然 pending，所以不汇报编造的 exp08 metric。

## Slide 11 - Parallel route: cross-cell-line selectivity

exp09 是另一条平行路线：完全不使用 transcriptome，而是使用 DepMap across 1,208 cell lines。

核心问题是：如果 gene_a 在某些细胞系里 defective，那么 knockout gene_b 是否在这些细胞系里更 lethal？这里 defective 的定义是 composite OR，包括 mutation、copy-number loss 或 low expression。

结果显示，这条路线对 classification 有帮助。B_xcl 在 CV2 的 AUROC 是 0.742，比 exp06 B 的 0.704 高大约 0.039；NDCG@10 是 0.086，比 0.042 高大约 0.044。CV3 AUROC 也从 0.596 提到 0.645，大约提升 0.050。

但 ranking 没有完全解决。CV3 NDCG@10 基本是 flat，B_xcl 是 0.001，而 exp06 B 是 0.002。

所以 exp09 的结论是：cross-cell-line selectivity 能改善 cold-start classification，但不能单独修好 both-gene cold-start top-k ranking。它和 exp07 是互补路线：一个用 observed transcriptome，一个用 cross-cell-line dependency context。

## Slide 12 - Closing the loop

最后回到开头的 honest framing。

我们实际构建的是一个 leakage-controlled SL-pair ranking adapter，包含 classification 和 ranking，但不是 validated SL target discovery。这个边界很重要，因为 benchmark negatives 是 unconfirmed non-SL，CV1 也明显可以被 degree shortcut 利用。

这个项目的 recurring discipline 有四点。第一，先做简单 floor，比如 exp06 dependency-only。第二，用 CV2/CV3 作为真正的 bar。第三，负结果或者部分正结果也要保留，比如 exp07 明确支持 CV2，但没有解决 CV3 top-k ranking。第四，pending 就标 pending，比如 exp08 还没有 cluster metric。

往后走，最直接的路径是：先完成 exp08 的 CV2/CV3 cluster gates，看 frozen-STATE plus ESM2 adapter 是否真的能泛化到 held-out genes；再结合 exp09 的 cross-cell-line selectivity；最后才考虑 TCGA patient-context transfer，把 cell-line benchmark 往 tumor-context dependency mapping 推进。

所以这次报告的结论不是“我们发现了新的 SL target”，而是“我们建立并审计了一个更诚实的 SL-pair ranking pipeline，并且知道每条证据路线现在能解决什么、不能解决什么”。

## Appendix - Results scoreboard

如果有人问完整 scoreboard，我会用这一页作为 backup。

这里先澄清口径：DDGCN、GRSMF、SL2MF 这些 SOTA rows 不是泛泛引用别的数据集结果，而是放在同一个 K562 过滤后的 Rand 1:1 CV1/CV2/CV3 benchmark 上做参考，所以它们确实是这个 benchmark 里的强基线。

但我不会把这页讲成一个已经完全公平的模型排行榜。原因是这些 SOTA 方法的特征、训练策略和报告实现来自不同 pipeline，而我们的模型是当前 repo 里同一个 official metric harness 下的 ablation ladder。所以这页有两个读法：第一，SOTA rows 告诉我们这个 K562-filtered benchmark 上现有方法能到什么量级；第二，我们自己模型之间最可信的是 within-harness comparison。

exp06 的 B 是 dependency-only XGBoost baseline。exp09 的 B_xcl 在同一个 harness 下，把 CV2 NDCG@10 从 0.0421 提到 0.0864，这是 apples-to-apples 的 lift。

另外，按照最新 exp07 文档，observed Perturb-seq 的 B_transcript 在 CV2 上也有明确 lift：NDCG@10 从 0.042 到 0.094，MAP@10 从 0.034 到 0.079。但是如果 appendix slide 还没有更新，我会口头说明这是 2026-06-18 完成后的更新结果，不把它混进尚未更新的 slide table 里。

exp08 仍然 pending。这里的重点是保持口径一致：SOTA rows 是同一个 K562-filtered benchmark 上的参考强基线；我们的因果性结论和 lift claim 只来自同一个实现 harness 内的 exp06、exp07、exp09 ablation。
