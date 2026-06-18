# exp08 — STATE-Adapter End-to-End DL Model for K562 SL-Pair Ranking

Status: Design approved 2026-06-17. Implementation not started.
Provenance chain: exp06 (dependency-only) → exp07 (real-bag features, locked MVP) → **exp08 (STATE-adapter DL, this doc)**.

## 1. Goal & Scope

Beat the exp06 dependency-only baseline on the **CV2/CV3 official per-anchor ranking
metric** by adding a transcriptomic-response signal that generalizes to held-out
genes. exp08 is a separate experiment from the locked exp07 (which explicitly
excludes any forward model / e2e learned set encoder); exp08 deliberately
introduces both, so it lives under its own experiment id rather than re-scoping
exp07.

Core idea (one line): **freeze STATE's transformer backbone; replace only its
perturbation input layer with a trainable adapter fed by ESM2 gene embeddings, so
all 9,471 genes — including held-out ones — get a STATE-predicted response in one
coordinate system.**

### Task definitions (unchanged from exp06/exp07)

- Classification sample: gene pair `(a, b)` → `sl_label ∈ {0,1}`.
- Ranking sample (primary metric): anchor gene `a` → rank all candidate partners
  `b` over the 9,471-gene K562-filtered universe, evaluated against `a`'s
  test-positive partners, with seen/train pairs masked and the diagonal zeroed.
  This is exp06's `official_ranking_metrics`, reused verbatim.

### In scope / out of scope

In scope: CV1/CV2/CV3 `Rand` 1:1 splits; gwps as the only Perturb-seq source;
frozen STATE backbone + trainable pert-adapter; ESM2 gene identity; the exp06
metric harness reused verbatim.

Out of scope: new cell lines; non-`Rand` negatives; biological SL claims ("SL
target" language); changing the `D` label, `C = GeneEffect(K562,g)`, the Feng-2024
benchmark, the metric protocol, or the split definitions.

## 2. Key Finding That Shapes the Design

The local STATE checkpoint
(`model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/`) is a **closed-vocabulary,
no-gwps** model:

- `config.yaml`: `pert_rep: onehot`, trained on `replogle_nogwps_v2`.
- `pert_onehot_map.pt`: exactly **2,024 perturbation genes**, each a 2024-dim
  one-hot.
- `pert_encoder.0.weight` is `(328, 2024)` — STATE's perturbation encoder is a
  single linear layer from the one-hot to the 328-dim hidden token.

Measured overlap against the 9,471-gene SL universe:

| Set | Count | % of SL universe |
| --- | ---: | ---: |
| SL genes with a real STATE one-hot token | 1,542 | 16.3% |
| SL genes with no STATE token | 7,929 | 83.7% |

Consequence: a naive "frozen STATE encodes all 9,471 genes" plan fails — 84% of
the universe is out-of-vocab, and `PerturbationVectorAdapter` (model.py:135) would
hand those genes a trainable mean-one-hot that receives **no gradient for CV2/CV3
held-out genes** (they never appear in train pairs). The bottleneck is the
perturbation *representation*, not the 64% bag coverage. The fix: replace STATE's
one-hot `pert_encoder` with a trainable adapter fed by a continuous,
gene-generalizable embedding (ESM2), keeping the rest of STATE frozen.

## 3. Architecture

```
                         ESM2(gene)  [1280-d, all 9,471 genes]
                              |
                    +---------v----------+
                    |  Pert-Adapter|  trainable: 1280 -> 328
                    |  replaces STATE     |
                    |  pert_encoder       |
                    +---------v----------+
                       328-d pert token
                              |
   K562 control      +--------v---------+
   template cells -->|  STATE backbone  |  FROZEN (transformer + decoder)
   [n_cells,input]   |  one-hot path     |
                     |  bypassed         |
                     +--------v---------+
                    predicted response bag
                       [n_cells, output_dim]
                              |
                    +---------v----------+
                    | Pooling head |  trainable: bag -> e_g
                    +---------v----------+
                     per-gene embedding e_g
                              |
         e_a, e_b --> symmetric pair head --> P(SL)
```

The 8-layer Llama backbone and decoder stay frozen, preserving STATE's learned
response geometry. A held-out gene reached only via `adapter(ESM2(gene))` lands in
the same space as covered genes — this is what makes the single-coordinate-system
constraint hold for all 9,471 genes, not just the 16% in STATE's vocab.

## 4. Components

New package `src/sl_dl_model/` (sibling to `sl_benchmark_baseline/`), CLI
`uv run python -m sl_dl_model`. Five core components (4.1–4.5) across the modules
listed in Section 10, plus verbatim reuse of the exp06 metric/data path.

### 4.1 `gene_embeddings.py` — ESM2 per-gene identity (precompute, cached)

- One-time: map each of the 9,471 SL-universe gene symbols → UniProt canonical
  protein sequence → ESM2 (`esm2_t33_650M`, 1280-d, mean-pooled over residues).
  Cache to `.npz` keyed by symbol.
- Genes with no resolvable sequence are recorded as `embedding_missing` and handled
  by the fallback path (never silently zeroed).
- Pure function of gene identity → split-independent, computed once for all folds.
- ESM2 size is a config knob (650M default; 150M as a faster ablation).

### 4.2 `encoder.py` — frozen STATE + trainable pert-adapter

- Wraps `load_state_model` (model.py:799) frozen; reuses `StateForwardAdapter`
  (model.py:72) for the `ctrl_cell_emb / pert_emb / pert_name / batch` plumbing.
- **`PertAdapter` (new):** MLP `D_esm → hidden → 328`, where `D_esm` is the ESM2
  embedding dim (1280 for 650M, 640 for 150M; resolved from config). Its output
  replaces what STATE's `pert_encoder` produces, fed in as `pert_emb`. Only this
  part of the STATE path trains.
- One shared K562 control template (fixed sample of `non-targeting` gwps cells),
  reused for every gene's forward, so the response delta is comparable across genes.

### 4.3 `embedding.py` — pooling head `bag → e_g`

- Trainable permutation-invariant pool over the predicted bag. Default: masked
  mean + std concat (reuse masked-mean from single_cell.py:810). GMM-occupancy
  pooling (`FixedGMMFeatureizer`, model.py:220) available as a config-swappable
  alternative.

### 4.4 `pair_head.py` — symmetric scorer `(e_a, e_b) → P(SL)`

- Swap-invariant combine mirroring exp06 symmetry:
  `[e_a + e_b, |e_a − e_b|, e_a ⊙ e_b]` concatenated with exp06's 5-feature
  GeneEffect block `[min, max, sum, prod, |diff|]`. MLP → logit → sigmoid.
- GeneEffect block retained so exp08 strictly supersets exp06's feature set; any
  lift is attributable to the transcriptomic block.

### 4.5 `train.py` / `evaluate.py` / `config.py` / `__main__.py`

- `train.py`: 3-loss training loop (Section 6). **Orchestrated by HuggingFace
  `Accelerator` (`from accelerate import Accelerator`), DDP by default** — a
  deliberate departure from exp06's single-process sklearn path; mirrors the exp05
  AIVC `accelerate launch` pattern (`scripts/state.sh`). **tqdm** per-epoch
  progress bars. Frozen-STATE handling reused from `AivcModel.train()`
  (model.py:346). A `scripts/sl_dl_model.sh` Slurm wrapper mirrors
  `scripts/state.sh`.
- `evaluate.py`: caches `e_g` once per fold for all 9,471 genes (genes drive STATE
  cost, not pairs), then **imports `sl_benchmark_baseline.metrics` verbatim** and
  reuses `_build_score_matrix` chunking to fill the 9,471² matrix.
- `config.py`: frozen `SLDLConfig` dataclass from YAML. Blocks: `state`,
  `gene_embedding`, `pert_adapter`, `pooling`, `pair_head`, `loss`, plus reused
  `split_types / folds / ranking_k / seed / input_csv / output_dir`.
- `__main__.py`: CLI entrypoint mirroring `sl_benchmark_baseline/__main__.py`.

### 4.6 Verbatim reuse (not reimplemented)

`sl_benchmark_baseline/metrics.py` (official ranking + classification),
`data.py` (`load_benchmark`, `fold_split`), and `evaluate.py`'s gene-universe /
seen-mask / diagonal-zero logic. This is what makes exp08-vs-exp06 a true ablation
rather than a cross-harness comparison.

## 5. Data Flow & Leakage Control

```
ESM2 cache (9,471 genes, split-independent) ----------------+
gwps h5ad (1.99M cells)                                      |
  +- control template: fixed sample of non-targeting cells  | (split-independent)
  +- per-gene real response bags (6,070 covered genes)       |
SL pair CSV (CV1/CV2/CV3, balanced)                          |
                                                             v
   PER FOLD (split_type, fold_id):
     train_genes   = genes appearing in TRAIN pairs of this fold
     covered_train = train_genes intersect gwps-covered    <-- ONLY these supervise bags
     ---------------------------------------------------------
     fit: PertAdapter + pooling head + pair head on train pairs, with:
            - token-distill anchor  -> in-vocab intersect train_genes
            - real-bag supervision  -> covered_train only
            - SL BCE                -> all train pairs
     ---------------------------------------------------------
     eval: e_g = pool(STATE(adapter(ESM2(g)), control)) for ALL 9,471 genes
           score 9,471^2 matrix -> seen-mask -> diag-zero -> official metrics
```

### Leakage rule (this is what makes CV2/CV3 valid)

- A gwps-covered gene appearing **only in test pairs** does **not** contribute its
  real bag to the bag-supervision loss, and its token is not used for distill
  fitting. It is reached at eval time purely through `adapter(ESM2(gene))` + frozen
  STATE. Held-out genes are genuinely unseen by every trainable component.
- ESM2 embeddings and the control template are functions of gene identity / control
  cells only — never of SL labels — so precomputing them once across folds is
  leakage-free (same logic exp06 uses for GeneEffect).
- Standardizer statistics (GeneEffect block + any embedding normalization) fit on
  **train-fold rows only**, applied to test.

### Consequence

For CV3 (both genes held out), *every* trainable component sees neither gene — the
model rides entirely on ESM2→STATE generalization. This is the hardest surface and
where a null result is most likely; it is reported honestly rather than engineered
around.

## 6. Training Objective

Three losses, summed with configurable weights; **STATE backbone frozen
throughout**. Only `PertAdapter`, pooling head, and pair head receive gradient.

```
L_total = lambda_sl * L_SL + lambda_distill * L_distill + lambda_bag * L_bag
```

### 6.1 L_SL — task loss (BCE only)

Binary cross-entropy on `P(SL)` over all train pairs. The only loss with an exp06
analogue, keeping exp08 a clean superset. Ranking (NDCG/MAP) emerges from calibrated
probabilities. RankNet (`aivc_model._pairwise_ranknet_loss`) stays wired but
`lambda_rank = 0` in V1 — a one-line flip if BCE underperforms on NDCG.

### 6.2 L_distill — adapter anchor (in-vocab ∩ train genes)

MSE between the adapter's 328-d output `adapter(ESM2(g))` and STATE's original
one-hot token `pert_encoder(onehot(g))`, for the ~1,542 in-vocab genes also in this
fold's train set. Pins the adapter into STATE's existing coordinate system so
out-of-vocab genes extrapolate into a meaningful region. **Decay default: stays on
at reduced weight after warm-up** (regularizer against drift), not decayed to zero.

### 6.3 L_bag — real-response supervision (covered train genes)

Aligns the predicted response bag against the real gwps bag for covered train genes.
Reuse the AIVC losses (`_mean_delta_loss` + `_energy_distance`, model.py) on the bag
in STATE output space, before pooling. Grounds the response space in real
transcriptomics so the pooled `e_g` is biologically meaningful.

### 6.4 Scheduling & freeze table

Warm-up: distill + bag for a few epochs (ground the response space), then anneal in
L_SL — mirrors exp05's `b_loss_anneal_epochs`. Warm-up length and all weights are
config knobs. Default weights: `lambda_sl=1.0, lambda_distill=0.5, lambda_bag=1.0`.

| Component | State |
| --- | --- |
| STATE transformer backbone + decoder | frozen |
| STATE original `pert_encoder` (one-hot) | frozen, distill target only |
| `PertAdapter` (ESM2→328) | trains |
| Pooling head, pair head | trains |

## 7. Evaluation

### Metric path — exp06's, imported verbatim

`evaluate.py` caches `e_g` once per fold for all 9,471 genes, then calls
`sl_benchmark_baseline.metrics.official_ranking_metrics` and
`official_classification_metrics` unchanged. Same seen-pair masking to −999999
(both directions), same diagonal zeroing, same per-anchor macro-average, same
`ranking_k = [10, 20, 50]`, seed 17. Score matrix filled with the existing
`_build_score_matrix` chunking.

### Primary success criterion — CV2 and CV3 official ranking

Beat the in-harness exp06 baseline on NDCG@k and MAP@k. Targets: CV2 NDCG@10 >
0.042, CV3 NDCG@10 > 0.002. CV1 is degree-gameable (the degree probe already hits
NDCG@10 0.197 there), so CV1 is reported but **does not count as a win**.

exp06 reference values (5-fold means, `official_metrics_summary.csv`): XGBoost
CV2 AUROC 0.704 / AUPR 0.732 / NDCG@10 0.042 / MAP@10 0.034; CV3 AUROC 0.596 /
NDCG@10 0.002.

### Baseline = exp06 re-run inside exp08's harness

Add an exp06-equivalent mode (transcript block zeroed, GeneEffect-only pair head)
that runs through the identical code path. Comparing to the stored
`official_metrics_summary.csv` would be a cross-harness comparison and invalid.

### Honesty checks (baked in)

1. **Covered-pair diagnostic slice** — metrics restricted to both-covered pairs
   (~41%). Separates "transcriptome useless" from "signal diluted by fallback
   pairs." A real win shows larger lift on this slice.
2. **Coverage-flag ablation** — run with and without any coverage indicator exposed
   to the pair head; report both. Coverage correlates with SL-graph degree.
3. **Effect size, not just point estimate** — report mean ± std across the 5 folds
   and flag any lift within fold noise (exp06 CV2 NDCG@10 std ≈ 0.008) as null.

### Artifacts (mirror exp06 layout)

`fold_metrics.csv`, `summary.csv`, `manifest.json` per split under
`results/experiments/08_*/cv{1,2,3}/`, plus a combined `official_metrics_summary.csv`.
Manifest records `candidate_gene_count=9471`, `seed=17`, `ranking_k`, ESM2 model id,
STATE checkpoint sha, `state_pert_vocab_overlap=1542`, `gwps_coverage_gene_count=6070`,
pooling type, loss weights, `coverage_flag_included`.

## 8. Phasing

Five phases, each independently verifiable, each with a gate.

- **Phase 0 — Harness parity (no STATE, no ESM2).** Stand up `sl_dl_model/` with
  config/CLI, reused data+metric path, GeneEffect-only pair head. **Gate:**
  reproduce exp06 numbers within fold noise. If not, the harness is wrong — stop.
- **Phase 1 — ESM2 + pert-adapter + frozen STATE plumbing.** Precompute ESM2 cache;
  wire `PertAdapter → frozen STATE → pooling → e_g`; train with distill anchor only.
  **Gate:** adapter reproduces STATE's original token on held-out in-vocab genes
  (distill MSE low); `e_g` finite and gene-varying.
- **Phase 2 — SL classifier (BCE).** Add pair head + L_SL. **Gate:** beat exp06 on
  CV2 classification (AUROC > 0.704 / AUPR > 0.732).
- **Phase 3 — Bag supervision.** Add L_bag on covered train genes. **Gate
  (primary):** beat exp06 on CV2/CV3 official ranking, lift concentrated on the
  covered-pair slice. This is the experiment's pass/fail.
- **Phase 4 — Robustness & ablations.** Coverage-flag on/off; pooling swap
  (mean/std vs GMM); optional RankNet (`lambda_rank>0`) if BCE underperformed on
  NDCG; ESM2 650M vs 150M; reporting polish.

## 9. Risks

1. **Single-gene → pair signal gap** (MLE Risk 3): frozen STATE bags for `a` and `b`
   are both single-perturbation responses; SL is combinatorial. Inherent to the
   premise — the pair head must do the combinatorial work; CV3 most likely null.
   Accepted, documented; negative result is publishable.
2. **Adapter extrapolation quality** (forced by this checkpoint): out-of-vocab genes
   (84%) rely entirely on ESM2→adapter generalizing into STATE's space. Distill
   anchor + bag supervision are the defenses; Phase 1's gate measures it directly.
3. **Coverage = degree proxy** (MLE Risk 4): handled by coverage-flag ablation +
   covered-slice diagnostic, but the implicit signal (fallback embeddings are
   distributionally distinct) cannot be fully ablated. Reported as a caveat.
4. **Weak/noisy CV2/CV3 effect size** (MLE Risk 6): exp06 fold std is large relative
   to means; report ± std and flag within-noise wins as null.
5. **Cost**: ESM2 precompute is one-time; per fold STATE runs 9,471 forwards (genes,
   not pairs). DDP + bag/embedding caching keeps it tractable — confirm on the first
   fold before scaling to all 15 (3 splits × 5 folds).

### Terminology guardrail

exp08 stays a benchmark-adapter extension. No "SL target" / biological SL claims;
`Rand` negatives are unconfirmed non-SL. Use "SL candidate prioritization" only with
context-specificity evidence (none claimed here).

## 10. File Map

New (`src/sl_dl_model/`): `gene_embeddings.py`, `encoder.py`, `embedding.py`,
`pair_head.py`, `train.py`, `evaluate.py`, `config.py`, `__main__.py`.
Config: `configs/experiments/08_k562_sl_pair_state_dl/`. Slurm wrapper:
`scripts/sl_dl_model.sh`. Results: `results/experiments/08_*/`.

Reused verbatim: `src/sl_benchmark_baseline/{metrics,data,evaluate}.py`.
Reused as patterns: `src/aivc_model/model.py` (`StateForwardAdapter` L72,
`PerturbationVectorAdapter` L135, `FixedGMMFeatureizer` L220, freeze pattern L346,
`_pairwise_ranknet_loss`, `_energy_distance`, `_mean_delta_loss`, `load_state_model`
L799), `src/aivc_model/train.py` (Accelerate/DDP loop), `src/aivc_model/prepare.py`
(`load_gene_bags` for control template + real bags).




