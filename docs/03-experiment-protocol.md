# Experiment Protocol: Context-Conditioned SL Benchmark

**Status:** §2 step 1 done — `context_screen_v2` built on the HPC 2026-08-15 with provenance,
filter audit and context statistics, reproducing v1's counts exactly. The split (§2.3) is
unassigned, no head is fitted, no model has run. Supersedes the withdrawn STATE-to-SLIdR
protocol; SLIdR was never implemented, so no code retires with it.
**Authority:** [`01-blueprint.md`](01-blueprint.md) is the contract; this document is its
executable form and may not relax it.

## 1. Objective

Evaluate the three-stage pipeline of `01` §1 on cell lines excluded from every fitting and
selection step:

```text
basal single cells + perturbation gene
  -> predicted post-perturbation cells   (Perturb-seq supervision)
  -> predicted DepMap GeneEffect         (mu + delta heads)
  -> pair score in a held-out context    (trained SL head)
```

Two questions, answered separately. Does the backbone predict held-out-context GeneEffect
residuals? And does a pair score built from predicted profiles rank experimental SL hits in
a context the model never saw? GeneEffect is a single-gene quantity; only the second stage
is scored against pair labels.

## 2. Prerequisite: Benchmark v2

### 2.1 Defects in v1 that force the rebuild

Each was computed directly from `derived/context_screen_v1/sl_context_pairs.csv`.

- **Duplicate screens.** K562/JURKAT share 9,219 rows (100% of JURKAT, Jaccard 0.772) and
  HELA/PC9 share 2,523 rows (100% of PC9, Jaccard 0.983), both at label agreement exactly
  1.0000. These rows carry `source_n_evidence == 2, source_row_count == 1`: one aggregated
  source row reported "tested in 2 lines, unanimous" and the builder copied its single label
  to both contexts. 97.9% of the 11,999 cross-context recurring pairs are this artifact.
- **Degenerate anchor in A549.** All 392 A549 positives contain TRA2A and no A549 row
  containing TRA2A is negative, so `1[TRA2A in {a,b}]` scores AUPR 1.0 on that context.
- **Missing positives in repeated patterns.** Nine contexts carry 933–941 negatives and zero
  positives (GI1, HS936T, HS944T, HSC5, IPC298, PATU8988S, PK1, MEL202, MELJUSO); five carry
  exactly 684 negatives and zero positives (A427, CAL27, CAL33, MCF10A, MCF7); THP1 carries
  1,332 positives and zero negatives. Identical counts across unrelated lineages indicate a
  filter or explosion artifact rather than biology.

### 2.2 Build

Re-run `scripts/build_sl_context_benchmark.py` against `sl_integrated_pairs.csv` with two
changes, writing to `derived/context_screen_v2/`. Do not overwrite v1.

1. **Emit `source_row_id`** for every exploded row. This links contexts exploded from one
   aggregate row. It **cannot** link separate rows produced by the same underlying screen —
   the source carries no study or evidence identifier — so it is used only to keep duplicate
   contexts on the same side of the split. No independence claim rests on it.
2. **Audit every filter** in the v1 preprocessing contract for rows and per-context positives
   removed: `sources == screen`, `evidence_types == experimental_screen`, `conflict == 0`,
   the all-evidence-unanimous rule, `n_evidence == n_cell_lines == n_context_tokens`, and the
   atomic-context token rule. Publish a per-filter, per-context drop table and account
   specifically for the zero-positive contexts above.

### 2.3 Published split

The split ships as a `split` column in the dataset, with the canonical copy at the **tracked**
path `configs/benchmarks/context_screen_v2_split.json` — `/data/` is gitignored in full, so a
manifest written only under `derived/` is neither distributed nor independently verifiable.
The dataset copy is a convenience mirror; the tracked file is the authority, and a mismatch
is a hard error.

Membership is decided once, by this rule:

- a context is **eligible** with at least 50 positives and 50 negatives;
- a context is **executable** if it has DepMap GeneEffect and basal single-cell input;
- **only executable contexts enter the benchmark at all** — train, validation and test alike.
  Arm A needs a predicted profile for every context it touches, so an eligible-but-
  non-executable context is unusable on any side, not just on test;
- the four response anchors (K562, HCT116, Jurkat, HepG2) are **pinned to train**. They
  supply the only Perturb-seq supervision, so placing one in test would contradict §8's
  requirement that test contexts be absent from response training;
- contexts sharing a `source_row_id` group stay on the same side.

Assignment is deterministic, not discretionary: sort the remaining executable context groups
by ModelID, then allocate to test, validation and train in that order under counts fixed in
the manifest before any context's difficulty is examined. Record the sort key, the counts,
and the resulting assignment in the tracked file.

HELA can never be executable: it has neither a 26Q1 GeneEffect target nor a compatible basal
single-cell input, and the §3.5 acquisitions supply basal cells, not GeneEffect. RPE1 and
HAP1 DepMap-CRISPR membership must be **verified, not assumed**. HAP1's v1 counts (56,994
positives against 20 negatives) fail the eligibility rule.

Publish per-context statistics beside the split: class counts, prior, distinct genes
appearing in positives, and the top gene's share of positives — the last is where the A549
degeneracy becomes visible to anyone using the benchmark.

### 2.4 Deliverables, frozen before any model run

```text
configs/benchmarks/context_screen_v2_split.json    TRACKED — the canonical split
derived/context_screen_v2/sl_context_pairs.csv     with source_row_id and split
derived/context_screen_v2/filter_audit.csv         per-filter, per-context drops
derived/context_screen_v2/context_statistics.csv
derived/context_screen_v2/manifest.json            source and output hashes
```

Everything under `derived/` is gitignored; only the split manifest is tracked. Write
`data/sl-context-screen-v2.md` as the card and mark the v1 card historical.

## 3. Data Contract

### 3.1 Basal input

Tahoe lines use their Tahoe-100M DMSO cells. K562, HCT116, Jurkat and HepG2 use
non-targeting Perturb-seq control cells. The source inventory is
`../results/phase_a_tx1_20260724/cell_line_manifest.csv` (42 lines); the v2 split supersedes
its `role` column but not its identity, basal-source, or coverage fields.

### 3.2 Perturb-seq supervision

The response module trains only on genetic-perturbation anchors:

| ModelID | Cell line | Basal control |
| --- | --- | --- |
| ACH-000551 | K562 | non-targeting Perturb-seq cells |
| ACH-000971 | HCT116 | non-targeting Perturb-seq cells |
| ACH-000995 | Jurkat | non-targeting Perturb-seq cells |
| ACH-000739 | HepG2 | non-targeting Perturb-seq cells |

No perturbation data from any test context may enter response training.

### 3.3 Dependency supervision

One frozen DepMap 26Q1 `CRISPRGeneEffect.csv` release, joined by ModelID. Targets are
$\mu^{tr}$ and $\delta$ as defined in `01` §4, with $\mu^{tr}$ fit on train-side contexts
only and genes needing at least three training observations.

### 3.4 SL labels

`derived/context_screen_v2/` is the sole pair-label source. Feng2024, Horlbeck, DepMap
co-dependency and every other label set stay out. The negative class is an experimental
screen non-hit in the named context, not a universal non-SL assertion, and context
assignment remains `silver_inferred`.

### 3.5 Required acquisitions

Basal single-cell expression is missing for most eligible contexts. Of the ten that clear
v1's permissive `>= 10/10` gate, only K562, Jurkat, A549 and HT29 appear in the 42-line
manifest. Acquire basal cells for the remainder before freezing the split, and record the
source and accession per context in the v2 card. RPE1 is the strongest candidate: Replogle
2022's essential-genome RPE1 arm is a genetic Perturb-seq dataset, so it could serve as both
a fifth response anchor and a large SL context.

## 4. Backbone Training

**Stage 1.** Fit the response module on the four anchors under $L_{\text{resp}}$ alone
(mean-delta MSE plus energy distance). Record its converged response metrics.

**Stage 2.** Unfreeze and optimize $L_{\text{resp}} + \lambda_{\text{dep}} L_{\text{dep}}$
over the anchors and the train-side dependency contexts. $L_{\text{resp}}$ remains active as
an anchor. **Report response metrics before and after Stage 2 in the same table.** If they
collapse, $\lambda_{\text{dep}}$ is misconfigured and the run is not finished.

**Selection.** Validation contexts from `split_manifest.json` select the checkpoint and all
hyperparameters ($\beta$, $\lambda_{\text{dep}}$, architecture widths). They are never
promoted into training. $\alpha$ is additionally frozen against a numeric $\hat y$-vs-$y$
calibration band on GeneEffect-only validation **before any SL label is read** — it may
never be tuned on SL performance, because inflating $\alpha$ degrades $\psi$ and flatters
the model against its own baseline.

Declare $G_{\text{var}}$, the delta-variance gene set, from train-side contexts before any
run: state the rule, the resulting gene count, and the exclusion of genes with fewer than
five non-missing training observations.

## 5. Out-of-Fold Feature Generation

Arm A requires that every backbone-derived block of an **SL training** row — the profile,
the target-context values, and $\hat\mu$ — come from a single model that excluded that row's
context group and refit $\mu^{tr}$ without it. Partition the train-side SL contexts into
inner groups and fit one model per group under the finally selected hyperparameters; models
from the hyperparameter search are not reused, since they need not share those settings.

The feature standardizer is fit only on complete out-of-fold vectors. Mixing an in-sample
profile block with an out-of-fold context block inside one vector is prohibited: it would
make training rows and test rows come from different distributions and then fit the scaler
on the mixture. There is no non-out-of-fold fallback for Arm A.

## 6. SL Head

Stage 3 minimizes BCE over train-side SL contexts, **context-balanced** so each contributes
equally, with positives and negatives reweighted inside each context. Without this, one
context dominates: RPE1 alone is 90,520 of 127,323 v1 eligible rows.

Feature construction follows `01` §3 exactly — 24 dimensions, all swap-invariant, with
$\psi_{\text{add}}$ excluded because it duplicates the `yhat_ac + yhat_bc` feature and would
make the C2 baseline non-independent of the context block. Reuse the summary-statistic
pattern at `src/sl_profile_baseline/features.py:103-139`, not the raw-profile form at
`:165-171`. **Do not reuse `sl_profile_baseline/data.py:90-92` or `features.py:84`** — both
zero-fill uncovered genes, the same silent-failure pattern as `selectivity.py:224`. Mask
explicitly instead. Arm A has no missingness; in Arm B a pair with a missing profile leaves
the common universe rather than being zero-filled.

Three arms on one identical pair universe: **Arm A** out-of-fold predicted, **Arm B**
measured DepMap over the same `C_ref` columns, and **Arm B-full** measured over all DepMap
columns as the honest ceiling.

## 7. Metrics and Baselines

Metrics carry the number of the stage in §4 that they score. **Stage 1** reports the
response metrics before and after Stage 2 training, per §4.

**Stage 2** uses two surfaces. A dependency split over `C_dep`, disjoint from the SL
contexts and holding out at least eight, carries the per-gene across-context Spearman on the
residual over $G_{\text{var}}$ — the axis the redesign is justified by, and the one R1's
ladder baselines. The SL test contexts carry per-context cross-gene Spearman only; it is
reportable and cannot support a context claim.

**Stage 3** reports AUPR per test context and the macro, both as $\text{AUPR}-\text{prior}$.
AUROC is secondary. Coverage and post-filter class counts precede every performance number.
Stratify each AUPR by whether both, one, or neither endpoint appeared in SL training.
Uncertainty uses a two-way dyadic bootstrap over both endpoints, 2,000 replicates, per
context; a one-way anchor-gene bootstrap is invalid because pairs have two endpoints.

Baselines, all reported as per-context lift, none permitted to remove a context after its
result is seen:

| ID | Baseline | Shortcut removed |
| --- | --- | --- |
| C1 | pair identity / train-context label frequency | pair memorization |
| C1b | anchor-gene frequency, `max(r_a, r_b)` | gene-level memorization C1 misses |
| C2 | `psi` alone, predicted **and** measured | ranking that is only the null |
| C3 | the `muhat` block alone | pan-essentiality |
| C5 | R1's best residual predictor replacing the backbone | a backbone beating no simple prior |
| C6 | matched context-ablated head | a context claim with no context information |
| C4 | Arm B, Arm B-full | reference only, never a bar |

C1 is a lift, not an absolute bar. On v1 it reached AUPR 1.0000 on JURKAT, HELA and PC9 and
0.506 macro against a 0.083 prior, so an absolute bar would void every run; §2.3 handles
those contexts through the split instead.

**C6** is an identically trained head on the 21 context-invariant dimensions with the three
context dimensions ablated, plus fixed deterministic context derangements and a permutation
null. Declare the minimum per-context incremental $\text{AUPR}-\text{prior}$ attributable to
the context block before reading any test label. Statistical distinguishability alone does
not clear it, and below it no context claim is licensed at any absolute AUPR.

Report $A-B$ both in full and restricted to the context block; only the restricted form
isolates out-of-sample GeneEffect cost, since 21 of 24 dimensions are context-invariant in
both arms.

## 8. Leakage Rules

- Test contexts are absent from response training, dependency training, $\mu^{tr}$, SL-head
  training, hyperparameter and checkpoint selection, standardizer fitting, calibration and
  thresholding.
- Validation contexts select only; they are never promoted into training.
- Join contexts by DepMap ModelID through a checked-in `context -> ModelID` map. Fail loudly
  on an unmapped context rather than dropping it. Never join on informal name — DepMap's
  `CellLineName` for K562 is `K-562`.
- Fit the standardizer, any calibrator, and any threshold inside the training side only.
  **Per-context z-scoring of $\hat y$ is forbidden**: it consumes the test context's own
  distribution and erases the quantity under test.
- One common, label-independent pair universe across arms. Missing scores stay missing,
  never imputed to zero and never counted as negatives.
- Report every test context. None may be dropped after its result is inspected.
- Qualify every result with the Tx1 Tahoe-100M pretraining exposure.

## 9. Compute Contract

The dominant cost is not backbone fitting but materializing $\hat\delta_{g,c}$ for every
benchmark gene across every `C_ref` context, once per inner model — order
`genes x contexts x cells-per-bag` Tx1-3B forward passes, with backbone-specific caches
(`tx1_predicted_response_cache.py`, `tx1_response_gene_bags_cache.py`) rebuilt each time.
Pin the bag size, the cell-sampling seed, and the cache layout, and record a measured
GPU-hour estimate for one inner model before launching the full set. If the total is
infeasible, that must surface before the first run, not after.

Report per run: the number of response contexts, total response cells, and Stage-1 and
Stage-2 $L_{\text{resp}}$.

## 10. Required Outputs

```text
split_manifest.json          copy of the tracked configs/benchmarks/ file, with its hash
checkpoint_selection.json
geneeffect_predictions.csv
geneeffect_metrics.json
sl_features.parquet
sl_scores.csv
sl_metrics.json
run_manifest.json
```

`run_manifest.json` records the git commit, input and checkpoint hashes, the DepMap release,
exact context lists, the gene universe, $G_{\text{var}}$, all hyperparameters with the
surface each was selected on, the $\alpha$ calibration band, and seeds. `sl_scores.csv`
records context ModelID, canonical pair, arm, score, label, endpoint-seen stratum, and every
exclusion reason. Planned metrics are not results; a claim enters `results/` only after the
frozen run completes and its integrity checks pass.
