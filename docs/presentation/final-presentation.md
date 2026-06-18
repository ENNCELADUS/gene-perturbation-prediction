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
