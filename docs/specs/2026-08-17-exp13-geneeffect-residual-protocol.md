# Experiment Protocol: Exp13 Cell-Line GeneEffect Residual Benchmark

**Status:** Stage 0 and the formal Stage 1 response run are complete. The strict two-phase Stage 2
implementation exists, but no Stage 2 result exists: its 226-line Tx1/q_sc/raw-UMI inputs,
target-universe ESM2, registered copy-prior, and Stage 1 compatibility/input manifest must pass preflight; that manifest records incomplete historical training lineage.
Supersedes T2 (`results/tx1-hvg-geneeffect-phase-f.md`, marked superseded).
**Authority:** [`01-blueprint.md`](../01-blueprint.md) is the contract; this document is its
executable form for the GeneEffect residual track and may not relax it. **Companion:**
[`03-experiment-protocol.md`](../03-experiment-protocol.md) is the SL-pair protocol; its §7
defers its dependency-residual metric to this document. **Benchmark:**
[`data/cell-line-geneeffect-226.md`](../data/cell-line-geneeffect-226.md).

## Scope statement (binding)

**Exp13 is a standalone cell-line GeneEffect residual benchmark. Its results are not
evidence for cross-context synthetic lethality.** The single frozen backbone pass this
document registers is valid only under this contract. Any later SL held-out-context reuse
would require every SL val/test context excluded from both dependency fitting and
feature-model fitting — an out-of-fold backbone per inner group, per
[`03-experiment-protocol.md`](../03-experiment-protocol.md) §5 — which this protocol does not
build. This benchmark (`cell_line_geneeffect_226_split`) and `context_screen_v2` never
substitute for each other; see [`01-blueprint.md`](../01-blueprint.md) §8.

## 1. Objective

Score a context-residual dependency prediction, $\hat y_{g,c}=\hat\mu_g+\hat\delta_{g,c}$
(`01` §3), on cell lines excluded from every fitting and selection step. No pair label, no
SL score, no interaction quantity is computed anywhere in this pipeline.

## 2. Benchmark

`cell_line_geneeffect_226_split` (`configs/benchmarks/cell_line_geneeffect_226_split.json`,
`.csv`, `_audit.json`) is the sole membership authority — 172 train / 27 validation / 27 test
cell lines, patient-grouped and source×lineage-balanced (MILP, `deterministic_tie_break_seed:
20260816`). DepMap 26Q1 GeneEffect covers 224/226 members; PC9 (`ACH-000779`) and HeLa
(`ACH-001086`) are unlabeled train members excluded from every supervised residual, context
model, and nearest-label donor fit (170 labeled train lines). Val and test are fully labeled
(27/27 each). Source: `docs/data/cell-line-geneeffect-226.md`.

## 3. Scored Gene Universe

The scored universe is **GeneEffect train/val/test coverage ∩ finite train-side K562 copy-prior
coverage ∩ ESM2-resolvable**, frozen before Stage 2 and shared by every model and baseline.
GeneEffect coverage alone (≥5 finite train, ≥3 finite val, ≥3 finite test observations) is
17,931 genes (`configs/benchmarks/cell_line_geneeffect_226_split_audit.json:
common_genes_train_ge5_val_ge3_test_ge3`), the **pre-prior/pre-ESM2 upper bound**, not the final
universe. The copy prior is pinned donor `ACH-000551`; its missing values are globally dropped,
never filled or method-specific. Symbol → UniProt/isoform mapping follows the existing
`--require-complete-coverage` gate and top-reviewed-hit convention
(`scripts/precompute_esm2_embeddings.py`), caching to `data/esm2/symbol_to_sequence.json`. A gene
unresolvable by ESM2 is dropped before freeze, never retroactively from an already-scored run.

## 4. Model

`01` §3's amended five-block composition (`Delta`, `s_{g,c}`, `q_sc_{g,c}`, `e_g`, `z_c` →
$h_\delta$) is the model under test. `mu_hat_g` is the empirical train-line mean, `mu_g^{tr}`
(`01` §4) — no learned $h_\mu$, because §3's scored universe already requires GeneEffect
train coverage for every scored gene by construction. Current instantiation: Tx1-3B basal
embedding (2560-d) as $F_\omega$ input, ST-checkpoint HVG panel (2000-d) as $B_c$/output
space, ESM2 adapter $p_g=A_\phi(E_{\text{ESM2}}(\text{protein}(g)))$ (1280-d → 2024-d) as
$F_\omega$'s perturbation input, with $e_g$ the raw 1280-d ESM2 embedding supplied to
$h_\delta$ directly. `Delta` is 4000-d (2 × 2000, mean+var); `z_c` is 5120-d (2 × 2560).
Rank zero writes `condition_features/stage1_frozen`; every launched rank then loads the supervised-train+validation features once into its own device cache.
Launcher/Accelerator-auto-detected 2- or 4-rank DDP covers frozen/eval-Stage-1 $h_\delta$ warmup and joint tuning after STATE/ESM adapters unfreeze for $L_{resp}+\lambda_{dep}L_{dep}$. `conditions_per_rank` is per rank in both phases where relevant. Masked Huber uses $\delta=1$, Pearson $\beta=1$, and $\lambda_{dep}$ is the clipped median gradient-norm ratio on eight train batches.

## 5. Sampling and Features

- **Cell sampling.** $M_c=128$ cells per line, `cell_set_len = 64` per ST window. A line with
  fewer than 128 basal cells is padded by sampling with replacement, after every distinct
  cell is used once; the true distinct count and padding fraction are recorded per line in
  `run_manifest.json`. The deterministic subsample order is `sha256(model_id + "|" +
  cell_barcode)`, ascending; the first $M_c$ (post-padding) are taken. Fixed once, reused
  identically across every run — never reseeded per run.
- **`s_{g,c}`.** Energy distance of $\hat B_{c,g}$ to the basal bag $B_c$; cross-cell response
  dispersion; fraction of cells beyond a declared shift threshold. Computed inside the
  forward pass — unrecoverable later.
- **`q_sc_{g,c}`.** Mean expression, fraction expressing, expression variance of gene $g$ in
  context $c$, from basal single cells only.
- **Projection.** `Delta` (4000-d) is projected to 256-d by one fixed, seeded sparse random
  projection with no fit step (so no train/test asymmetry). Seed `20260828` is pinned before
  Stage 2 and never redrawn. Unprojected interpretables kept alongside:
  $\lVert\Delta_{\text{mean}}\rVert$, cosine to the basal mean, and `Delta` at $g$'s own HVG
  index (own-gene shift) when $g$ is in the panel.
- **Collator seed.** `max_length: 2048` with `sampling: true` means `_sample` draws genes by
  unseeded `torch.randperm`, and most basal cells exceed 2048 detected genes. Seed before
  every forward pass and record it in `run_manifest.json`, as the cell order is pinned above;
  seeding makes the encode bit-identical (§6).
- **Coverage bits.** `q_sc` available, `g` in the HVG panel, own-gene shift available —
  explicit masks, nothing zero-filled.

## 6. Stage 0 — Tx1 Input Representation (closed, branch 1)

152 of the 179 new-atlas lines were published as Kinker `processed_cpm` (19/27 test, 23/27
val lines), so $x_c=E_{\text{Tx1}}(r_c)$ depended on whether the collator reads a CPM row like
the counts underneath it. **It does not:** per-cell cosine 0.92-0.95 against the raw encode,
and — unlike gene-subsampling noise, which pools away to 0.9997 — the shift survives pooling
to the per-line mean at 0.972-0.987 ([result](../results/exp13-stage0-tx1-input-representation.md)).
"Swap only $z_c$" was never a fallback: ST consumes the embedding as its Stage-1 *input* (§4),
so an untrustworthy one voids a line's whole forward pass, not one feature block.

**Branch 1 is taken.** SCP542 publishes `UMIcount_data.txt`, raw pre-QC UMI counts covering
all 40,670 selected cells and all 152 lines, with cell-line identity in the file.
`scripts/prepare_kinker_umi_h5ad.py` ingests it to per-line h5ads passing
`assert_tx1_input_contract` (0 non-integer values in 151,986,988 nonzeros; 40,670/40,670 line
labels reconciled against the CPM-derived selection). Split membership is unchanged — only the
numeric source moved — and the frozen line manifest, whose hash the split builder pins, is not
edited. The CPM artifacts stay the comparison arm and must not re-enter the Tx1 path.

## 7. Stage 1 Objective and Validation

Registered in `configs/experiments/13_geneeffect_226/stage1_response.yaml` and
snapshotted to `stage1_objective.json` before Stage 1 trains:

- Per-anchor response metrics (mean-delta MSE, energy distance) for each of the four
  Perturb-seq anchors (K562, HCT116, Jurkat, HepG2 — `03-experiment-protocol.md` §3.2).
- A held-out perturbation-gene set per anchor, excluded from Stage 1 training and reserved
  for the response-model's own generalization check.
- The four-line weighting used to combine anchor losses into one Stage 1 objective.

Basal-copy and null-shuffle losses are required reported baselines, not pass/fail gates.

## 8. Metrics and Baselines

**Primary metric:** macro per-gene Spearman across the 27 test lines, residual-only
($\hat\delta$ vs. $\delta=y-\mu^{tr}$); per-line values reported alongside. **Selection**
(checkpoint, hyperparameters) may use the identical metric on the 27 validation lines, but
validation labels never enter gradient fitting; test labels enter neither fitting nor selection.

Baselines and the model are scored identically by `residual_ladder.py` /
`residual_metrics.py`: `copy_prior`, `nearest_line`, `context_pca_ridge[z_c]`
(`src/aivc_model/residual_ladder.py`). `gene_mean` predicts $\hat\delta\equiv0$ by
construction (`residual_ladder.py:7`), so its per-gene Spearman across lines is undefined —
report it as **"not evaluable / constant prediction"** with its coverage count, never as a
score of 0.

This bring-up runs the full five-block model for one seed only: no ablation and no multi-seed
scientific claim. Any later virtual-cell ablation must jointly remove
`{Delta_proj, s_{g,c}, own-gene shift}`; dropping `Delta_proj` alone is not that test.

## 9. Leakage and Integrity

- Validation and test lines are absent from response/dependency fitting, `mu_g^{tr}` fitting, and projection/PCA fitting.
  Validation alone may select checkpoints/hyperparameters; test never enters selection.
- `mu_g^{tr}` is fit on the 170 labeled train lines only, never re-added inside a scored
  quantity, and never centered on a fold that includes the row being predicted (per
  `CLAUDE.md`'s residual-ladder rule).
- Coverage masks are explicit (§5); nothing is zero-filled.
- One gene universe (§3) across model and every baseline; genes unresolvable by ESM2 are
  dropped from the frozen manifest before any run, never mid-run.
- Every held-out-line result is qualified with Tx1 Tahoe-100M pretraining exposure.

## 10. Required Outputs

```text
cell_line_geneeffect_226_split.json    copy of the tracked split file, with its hash
esm2_gene_universe_manifest.json       symbol->UniProt/isoform mapping, coverage, drop list
stage1_objective.json                  §7, pinned before Stage 1 trains
condition_features/stage1_frozen/      one-time generated warmup features
warmup/, joint/, checkpoint_selection.json    selected checkpoints, metadata, logs, decisions
geneeffect_residual_predictions.csv
geneeffect_residual_metrics.json       primary + per-line + response before/after + baselines
run_manifest.json                      commit, seeds, M_c/cell_set_len, projection/gene universe, DDP settings
complete.json                          successful terminal run; no failure.json
```

Dot-status files and live log output are diagnostic progress only. Completion requires exited workers, `complete.json`, and no `failure.json`.
Planned metrics are not results; a claim enters `results/` only after the run completes and its checks pass.
