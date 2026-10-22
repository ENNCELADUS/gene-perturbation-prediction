# PIE stage-one ladder after Task 2 (Tasks 3–9): Implementation Plan

**Spec:** [the PIE stage-one design](2026-10-16-pie-geneeffect-ladder-design.md), as revised 2026-10-17. **Status:** approved
2026-10-17; Wave 0 done 2026-10-17 ([record](../../results/ladder_inputs/README.md)); Wave 1A built 2026-10-17
on `feat/stage-one` (protocol §16), Task 3 run and reported 2026-10-22
([record](../../results/stage_one_architecture/README.md)); Wave 1B (Task 8) run and reported 2026-10-22
([record](../../results/state_telemetry_seed0/README.md)).

## Context

The PIE stage-one design (`docs/specs/2026-10-16-pie-geneeffect-ladder-design.md`) was revised and approved on
2026-10-17. §3.5, Task 8 (state telemetry) and Task 9 (cell states) were added after a cell-states literature survey and a
Codex review. Task 2 is built on the H20, checked 2026-10-17 17:56 UTC:
- **Done:** the frozen-PIE pass assembled (17,607 genes × 1,036 lines); gene sources, context text, cells, mean channel
  and summaries.
- **Still running:** the last Tx1 shard (1/16).
- **Branch:** `feat/ladder-inputs` is unmerged; the host is at `c219707`, the laptop 2 commits ahead, and the spec
  revision is uncommitted.

The user chose to build Task 3 (architecture stage) and Task 8 (state telemetry) in parallel after closing Task 2.

Rules from CLAUDE.md:
- strict configs; no gates in code (people read reports and decide);
- one seed; validation selects, test is scored once;
- Conventional Commits; merge to `main` once a branch's H20 path has run;
- GPU work through `hpc/run.sh`, ending local work with the exact launch command.

The plan is executed wave by wave.

## Waves

| Wave | Branch | Builds | People read, then decide |
| --- | --- | --- | --- |
| 0 | `feat/ladder-inputs` | Close Task 2: last Tx1 shard, doc fixes, two precompute guards, merge | `results/ladder_inputs/README.md` |
| 1A | `feat/stage-one` | Shared stage-one stack and Task 3's independent rows | Task 3 results: the hybrid's Δ, the leading family |
| 1B (parallel) | `feat/state-telemetry` | Task 8 | which vocabularies Task 9 precomputes |
| 2 | `feat/stage-one-prior-input` | The prior-input rows (protocol §16) first; Task 3's second half and Task 4 wait on their reading | whether ΔE adds to the prior; architecture; gene query |
| 3 | `feat/data-ladder`, `feat/cell-states` | Task 5, alongside Task 9's frozen-state pilot and planted state effects | ladder contrasts; states pilot, before fine-tuned states are built |
| 4 | same | Task 6, Task 7, Task 9 fine-tuned states | planted recovery, cell-count curve, drift; before end-to-end states are built |
| 5 | `feat/cell-states` | Task 9 end-to-end states | final per-state account |

Waves 2–5 are outlined here and get their file-level detail when the previous wave's reading is recorded.

## Cross-cutting decisions (my calls)

- **Environments.**
  - Add `scikit-learn` to `.venv-pie` as a `geneeffect` dependency group in `third_party/pie/pyproject.toml`, re-locked;
    `hpc/pie/setup.sh` syncs it. The scorer `src/eval/geneeffect.py` imports sklearn.
  - Lift `make_scheduler` (`src/training/trainer.py:97`, a module that imports accelerate) into `src/training/schedule.py`.
  - The prior reference and the sklearn mixtures run in the project env.
  - cNMF, starCAT and scvi-tools run in a new `.venv-states` (`hpc/states/setup.sh`).
  - Environments exchange files only.
- **Residency.**
  - Δ arrays are gene-major (17,607 × 1,036 × w), about 56 GB each for the 768-d ones. Open them with memmap and copy
    to the GPU in bf16 by gene chunk (they are already centred within each gene).
  - Cell banks are GPU-resident and ragged.
- **Gene axis.** Δ exists for the 17,607 panel genes. The 180 genes outside the panel predict the base; scoring keeps the
  protocol's 4,323 selective genes, with a secondary row on the 4,269 in the panel.
- **Δ input.** `[Δ centred within gene ; frame mean Δ̄_g]`, with no per-gene parameter.
- **ΔE states.** The code stores only Δ̂ coordinates on the basis.
  - before(c) = (x_c − x̄_frame)Cᵀ, from `mean/pie_input.parquet`;
  - Δ̂Cᵀ = dE + frame_mean_dE[g] + μ_ΔCᵀ;
  - after = before + Δ̂Cᵀ.
  - The share of x_c's variance kept is reported.
- **Run directories.** `outputs/stage_one/<task>/<config stem>[__fold<k>]/`. `_bind` and `_locked` move from
  `src/experiments/context_prior.py:520–545` to `src/experiments/run_dir.py`.
- **Out-of-fold telemetry (Task 3 rows).** `--fold k` drops fold k from training and refits `m_g` without it. μ_g, σ_g
  and the selective genes stay the protocol's; selection stays on validation.
- **State pseudo-contexts in `pie_precompute`.**
  - Name each context `ACH-x|<set>|<k>`, with a map from name to owning line; context text is looked up by owner.
  - Reuse the main `pie/dataset`; write outputs to `ladder/states/pie/<set>/` (under the state inputs, with a fingerprint of them; revised 2026-10-17 after review).
  - Run context-major batches of 64, each running its reference pass inside the batch and evicting its cache entries
    afterwards. Pad the last batch so `torch.compile` doesn't recompile.
  - Run the encoder once per owner (exact, because the latents read no expression).
  - Each set includes a `|whole` context per line as a bf16 noise floor.
  - Centre state deltas on the main pass's `frame_means`.
- **New tracked tables.**
  - `configs/benchmarks/single_cell_source_chemistry.csv`: the assay per deposit (per line or arm where a deposit mixes kits), for scVI's nuisance and Task 5's
    covariate.
  - `configs/context_prior/reference/kinker_programs.tsv`: Kinker's 12 programs, with provenance.
  - Task 5's ambient score doesn't exist; it is reported as unavailable.
- **V(c, g).** Every run writes the 32-d V for all lines × panel genes in fp16, with training lines marked as in-sample.
- **Recorded by people, never computed by a rule:** the hybrid's Δ, the winning family, the gene query, Task 8's
  vocabulary list, and the go-ahead for fine-tuned and end-to-end states. Recorded 2026-10-22, after Tasks 3 and 8:
  the hybrid's Δ is ΔE and the leading family FiLM; since no row reached the prior, FiLM reads the prior's prediction
  as an input (not as the base: the final model is one network) before the rest of Wave 2; Task 9 precomputes the
  cycle, recurrent programs, line-centred scVI K 6 and HVG K 8 vocabularies, with QC K 8 as its control.

## Wave 0: close Task 2

1. When Tx1 shard 1/16 exits, check that `manifest.json` lists every Tx1 shard and that
   `tx1_cache.missing_lines` is empty for both `real/` and `shuffled/`.
2. **Doc fixes**, in one `docs(ladder)` commit together with the spec revision:
   - spec §3.2: the before-state wording;
   - spec §6: the new files from this plan;
   - `docs/data/ladder-inputs.md:35`: no RRID column (it lives in `context_text_provenance.json`), plus the before-state
     note;
   - protocol §15: the same ΔE wording.
3. **Guards** in `src/experiments/pie_precompute.py`, against silent reuse:
   - `reference_pass` (PP:807) stores `lines` and refuses a `reference.npz` whose row count or lines differ;
   - `write_dataset` (PP:244) refuses an existing dataset whose `context_to_id` or genes differ.
   - Tests go in `tests/test_pie_precompute.py`.
4. **Records.** Write `results/ladder_inputs/README.md` (frame counts, coverage per source, line and gene, the pilot,
   seconds per gene from `pie/run.json`) and update CLAUDE.md's "Current state".
5. **Merge.** Run the suite to a file and `.venv/bin/ruff check .`; push; pull on the H20; merge into `main`; delete the
   branch.

## Wave 1A: shared stage-one stack and Task 3 (`feat/stage-one`)

**New files:**

- `src/data/ladder_reader.py` (numpy, pandas, torch; no `pie` import):
  - `read_ladder`, wrapping `ladder_inputs.read_genes`/`read_lines` (:346/:350);
  - `check_pie_index`, which fails closed unless `pie/index.json` matches `lines.parquet` and the panel;
  - `open_delta` (memmap), `frame_means`, `basis`, `profile_states`;
  - `training_targets(…, exclude_fold)`: GE − m_g from `read_depmap_matrix`, as `ladder_inputs.build_lines:288` does.
    It refuses lines that aren't `train_*` and runs `splits.assert_fit_eligible`;
  - `scoring_truth`, `gene_sources`, `context_text` (via `gene_sources.read_pie_dense`), `mean_channel`, `pie_input`.
- `src/model/stage_one.py`:
  - `FiLM`; `CilantroTail` (→64→32→1, zero-initialised output, returns (z, V32)), transcribed from the H20 copy of
    `third_party/cilantro_sl` FiLMNet128;
  - `GeneEncoder` (masked sum of per-source projections); `ContextEncoder`;
  - `DeltaFiLM`, one class for virtual cell + FiLM, multi-encoder, the no-Δ FiLM, and the lineage-, text-, `x_cg`- and
    component-only rows;
  - `DirectReadout`, `Hybrid`, `build_model`.
- `third_party/pie/src/pie/model/geneeffect.py`:
  - per-channel masked SDPA cross-attention;
  - `GeneEffectTrunk` (latents ← channels → blocks → gene queries), with token and own-evidence dropout;
  - strict `warm_start` with an explicit name map.
- `src/training/stage_one_trainer.py`, plain torch on one GPU:
  - AdamW with `schedule.make_scheduler`; EMA via `torch.optim.swa_utils.AveragedModel`;
  - masked MSE in σ units;
  - validation at epoch 0, where the zero-initialised tail equals the base;
  - reuses `checkpoint.record_validation`, `TrainState`, `capture_rng_state`/`restore_rng_state` and the strict
    `load_checkpoint`;
  - checkpoints record membership, using `_membership`/`_restored_split` lifted from `src/data/prepared.py:124,260`
    into `src/data/splits.py`; resume refuses a conflicting config.
- `src/experiments/stage_one.py`:
  - train, score validation and then test once, and export `row.json`, `predictions.parquet` and `V.npy`;
  - the per-gene ridge on line-centred ΔH (dual closed form on the GPU);
  - `report`, which writes `results.md` with every §4.1 metric, the calibration slope, contrasts with 1,000-resample
    `paired_line_bootstrap` intervals (`src/eval/metrics.py:605`) and the out-of-fold table.
- `src/experiments/stage_one_prior.py` (project env): refits the default prior on the 981 frame lines' `mean_channel`
  rows, via `context_prior.prior_inputs:300`, `stages:363`, `prior.fit_prior:266` and `crossfit:364` with the frame
  folds. It writes validation, test and out-of-fold predictions and the 128 component scores, which the shared
  nonlinear reader of 128 components needs.

**Edits:**
- `src/eval/`: move `long_frame`, `score` and `gain` from `src/experiments/context_prior.py:234–290` into
  `src/eval/geneeffect.py`, and re-import them from `context_prior`. Add `calibration_slope` to `src/eval/metrics.py`.
  The scorer receives `(m_g − μ_g)/σ_g + z`.
- `src/experiments/config.py`: `validate_stage_one_config` / `load_stage_one_config`, following the `_GROUPS` and
  `_require_keys` pattern, with groups data, model, train and eval.
- `hpc/run.sh`:
  - `stage-one CONFIG --run-id ID [--fold K]` (`.venv-pie`);
  - `stage-one-queue --gpus … [--folds main,0..4] CONFIG...`, one process per GPU, skipping runs whose `row.json`
    exists;
  - `stage-one-report`;
  - `stage-one-prior` (project env).

**Configs** (`configs/stage_one/architecture/`):
- `direct_readout`, `direct_readout_warm`;
- `film_{dH,dE,dL}`, `multi_{dH,dE,dL}`;
- `nodelta_film`, `permuted_{film,multi}_{dH,dE,dL}`;
- `components_reader`, `lineage_only`, `text_only`, `own_x_only`;
- `prior_reference`;
- `report_3a`, which also lists the untrained rows: the base, the prior and the ΔH ridge.

**Tests:**
- Cilantro FiLM and tail parity (skipped when the H20 copy is absent).
- The zero-initialised tail outputs 0; a gene absent from training decodes; an unlisted gene masks only its own evidence.
- Chunked attention equals unchunked; duplicating tokens leaves a channel unchanged; `warm_start` is strict.
- Reader: a mismatched index fails; the target is GE − m_g; validation and test lines are refused; `exclude_fold`
  refits the base; the ΔE before and after states round-trip.
- Calibration slope is 1 on scaled truth.
- Trainer: epoch 0 equals the base; resume refuses another config; a checkpoint refuses another membership.
- Config: unknown or missing keys raise.
- End to end on two synthetic lines: train, select, validation, then test.

Fixtures reuse `tests/test_ladder_inputs.py:build_world` (:41) and `tests/test_pie_precompute.py:tiny_pie` (:135).

**Docs:** protocol §16 "Stage-one model"; `hpc/README.md` recipes; afterwards
`results/stage_one_architecture_seed0/README.md`.

**Compute:** about 19 configs × (main + 5 folds). An FiLM epoch runs in seconds to minutes once resident; the direct
trunk takes about 1 min per epoch. Under a day on 4 × H20 (an estimate; the first epoch logs replace it).

**Launch (H20, after pull and the `.venv-pie` re-sync):**
```
hpc/run.sh stage-one-prior configs/stage_one/architecture/prior_reference.yaml --run-id prior_reference_seed0
nohup hpc/run.sh stage-one-queue --gpus 0,1,2,3 --folds main,0,1,2,3,4 configs/stage_one/architecture/*.yaml > outputs/launches/stage_one_3a.log 2>&1 &
hpc/run.sh stage-one-report configs/stage_one/architecture/report_3a.yaml
```

## Wave 1B: Task 8 state telemetry (`feat/state-telemetry`)

**New and edited code:**
- `src/data/cell_states.py`:
  - `fit_sample`: at most 500 cells per line, equal weight, `train_cells` only;
  - `cycle_vocabulary`: a 3-component GMM on the S and G2/M columns of `cells/*.npz` qc;
  - `mixture_vocabulary(latent, K, centred)`; a deterministic `project`;
  - `state_masses_and_inputs`: sparse `qᵀU / Σq`, `effective_count`, `merge_small(10)`, with mass and weighted
    expression kept;
  - `permute_memberships`, `qc_states`, `cycling_weights(p + 0.05)`;
  - `per_cell_log_cp10k`, which recomputes u_i from the raw h5ad via `tx1_cache.read_basal_source`,
    `ladder_cells.library_sizes`, `_scaled(PIE_TARGET_SUM)` and `align_columns`, filling unlisted genes as `build_mean`
    does (`ladder_inputs.py:593`) so that Σπx = x̄ exactly.
- `src/data/prepare/state_programs.py` (`.venv-states`):
  - consensus NMF per training line, clustering of programs across lines with cycle programs set aside, starCAT spectra;
  - scVI with library size modelled, chemistry as the nuisance, and S/G2M as covariates;
  - latents written for every cell; a manifest records the spectra normalisation, the background program and the
    usage-to-weight rule.
- `src/experiments/state_telemetry.py` (project env), steps `fit | project | inputs | report`:
  - **fit:** the cycle, programs, scVI × K{4,6,8} × {centred, absolute}, HVG-PCA and Tx1 × K{4,6,8} fits, and QC states,
    each at three fitting seeds.
  - **project:** `train_cells` and `val` only. Test is added after the choice is recorded.
  - **inputs:** one raw-h5ad pass per line writes `ladder/states/<set>/{inputs,masses}.parquet` for every set, its
    permuted seeds 0–2, the QC states, the cycling-weighted input and the whole-line input.
  - **report:**
    - identity against Kinker's 12, Gavish's 41, hallmarks and Tirosh, via `noise_inputs.program_scores`;
    - the share of mass explained by deposit and by QC measures;
    - recurrence across lines;
    - SVD of the line × state masses in centred log-ratio form;
    - agreement between a line's two deposits, through the second deposits in `noise_diagnosis.yaml`;
    - stability, with states matched by the Hungarian method;
    - PIE's gap for ΔH, ΔE and the own slots, against the uncentred `pie/genes/<g>.npz`.
- `src/experiments/pie_precompute.py`: a `states` step (`--set`, `--shard` over owners) with a `StateTables`, the
  pseudo-context design above, outputs per line that resume, and `assemble_states`. Task 8 runs it on
  `pilot_genes(tables, 200, 0)` (PP:229).
- `config.py`: `validate_states_config`. `hpc/run.sh`: `cell-states CONFIG --steps …` (with `STATES_PYTHON` for the
  `.venv-states` steps) and `pie-precompute --steps states|assemble_states --set NAME`.
- `hpc/states/setup.sh`, plus the chemistry and Kinker tables.

**Tests:**
- State masses times inputs reproduce the line's PIE input, after merging too.
- Permuted memberships keep each state's mass, effective count and the line input.
- A fit rejects validation and test lines; projecting twice gives identical memberships.
- Two states of one line give different PIE outputs.
- A `|whole` context equals the main pass.
- The cache is empty after each batch; padding changes nothing; the hoisted owner encoder equals the unhoisted one.

**Docs:** protocol §17 "Cell states"; the data card `docs/data/cell-states.md`; `results/state_telemetry_seed0/README.md`.

**Compute:**
- Fits run on CPU, except scVI (1 GPU, under 1 h).
- PIE: about 140k contexts × 200 genes ≈ 28M pairs ≈ 25–30 h on 4 × H20.
- The queue runs real vocabularies and permuted seed 0 first (about 13 h); people may stop before seeds 1–2.

**Launch:**
```
CFG=configs/stage_one/states/telemetry.yaml
CUDA_VISIBLE_DEVICES=0 STATES_PYTHON=.venv-states/bin/python hpc/run.sh cell-states $CFG --steps fit,project,inputs
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i nohup hpc/run.sh pie-precompute $CFG --steps states --set ALL --shard $i/4 > outputs/launches/states_$i.log 2>&1 & done
wait   # every shard has exited (from another shell: until no pie_precompute process is left)
hpc/run.sh pie-precompute $CFG --steps assemble_states --set ALL
hpc/run.sh cell-states $CFG --steps report
```
`hpc/README.md` ("The cell states") gives the same recipe.
Waves 1A and 1B share the 4 GPUs: Task 3's queue and the Task 8 PIE pass each take their own GPU ids, set at launch.

## Waves 2–5 (outline)

- **Wave 2.**
  - Task 3's second half: the hybrid with the recorded Δ; the winner's bulk-encoder, 377-line and protocol-mean rows; a
    separate bulk encoder over `mean/pseudobulk.parquet` and the bulk profiles, aligned on the 343 paired lines.
  - Task 4 (`configs/stage_one/gene_query/`): each source added to `gene_text`; the fold-masked GeneEffect profile (five
    masks; held-out lines average the five; the outer fold removed in out-of-fold runs, spec §8); then the best two.
- **Wave 3.**
  - **Task 5:** `CellBank` channels (HVG, shuffled HVG, QC, Tx1 real and shuffled), per-gene spread evidence, and the
    gene-to-cell block only if the direct readout or the hybrid wins; the nine contrasts with per-line QC covariates.
  - **Task 9, frozen-state pilot:** the state precompute on the chosen sets, over the selective genes in the panel and
    all lines with cells (test included now). `StateMixture`, with the fixed average and a log-sum-exp row that saves
    the aₖ and KL terms. `StateExpressionReader` (E(x_ck) plus x_ckg). The four primary contrasts. The direction table
    against Task 8's identities, with TP53 status from DepMap. Timing of a minimal differentiable PIE forward and
    backward. About 55M pairs, about 50 h.
  - **Task 9, planted state effects:** `src/data/planted_terms.py:planted_state_targets`, with recovery of each f_k
    from the saved scores.
- **Wave 4.**
  - **Task 6:** planted marginal, joint and rare terms, residualised on training lines.
  - **Task 7:** a `draws` step in `ladder_inputs` with the bridge refitted per draw, and a PIE draw pass through
    pseudo-contexts (`ACH-x|draw|n|d`, about 4.7M pairs).
  - **Task 9, fine-tuned states, after people read the pilot:** `src/model/pie_differentiable.py` (gradients enabled,
    dropout off, only membership-independent work cached, the zero flag from membership support) and a
    `DifferentiableMixture` initialised from the scVI GMM, with a KL term to the start and a weak usage prior, plus
    drift reports. Tests: it matches the precompute's outputs and agrees with finite differences.
- **Wave 5.** Task 9 end-to-end states: the same mixture randomly initialised on the frozen line-centred latent, against
  permuted memberships and the whole-line input.

## Verification

- **Each wave, locally:**
  - `uv run python -m pytest tests -q > pytest.txt 2>&1`, against a baseline taken before the wave;
  - `.venv/bin/ruff check .` and `ruff format` on touched files;
  - the synthetic end-to-end tests above.
- **On the H20:**
  - the `.venv-pie` import check: `python -c "import src.eval.geneeffect, src.eval.metrics, src.context_prior.targets"`;
  - one short run per new command before the queue;
  - confirm `row.json`, `results.md` and the run manifests exist and that epoch-0 validation equals the base row.
- **Silent-error checks with tests:**
  - stale control cache;
  - `reference.npz` or dataset reuse;
  - an out-of-fold base containing the fold's labels;
  - the profile row seeing its own or the outer fold;
  - the Δ index order against `lines.parquet` and the panel;
  - the `|whole` noise floor.
- **Merge** each branch into `main` once its suite and lint pass and its H20 path has run.
