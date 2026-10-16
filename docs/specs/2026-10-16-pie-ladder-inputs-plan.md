# PIE stage-one inputs (Task 2): Implementation Plan

**Spec:** [the PIE stage-one design](2026-10-16-pie-geneeffect-ladder-design.md) §5 Task 2, as revised in the Task 2 grill of 2026-10-16. **Status:** agreed 2026-10-16; implementation on `feat/ladder-inputs`.

## Context

Task 1 is done: the vendored PIE (`third_party/pie/`, `21a526a`, `.venv-pie`) reproduces its released checkpoints on
the H20. Task 2 (`docs/specs/2026-10-16-pie-geneeffect-ladder-design.md` §5) builds every fixed input that Tasks 3–7
read, under `data/geneeffect_joint/v2/ladder/` with its own manifest:
- the line set, base and folds;
- the mean channel and PIE's input;
- the cell channels;
- Tx1 embeddings of all cells;
- Cellosaurus text;
- the gene sources;
- the frozen PIE's deltas.

The whole of Task 2 is in this one plan, with no cut by consumer.

Exploration changed some of the spec's facts, and this grill (2026-10-16) settled the choices those changes opened.

## Decisions (grill 2026-10-16) and the spec edits they need

| Topic | Decision |
| --- | --- |
| Frame | 981 labelled lines: the 377 single-cell training lines (28 lymphoid/myeloid and ~6 without bulk RNA included) and the prior's 604 non-haematopoietic bulk-only labelled lines. The prior's 947 hold only 343 of the 377. `m_g` and the five patient-grouped folds (seed 0) cover the 981. |
| Mean channel | A line with cells always reads its cells. Bulk RNA is the input only for bulk-only lines. Each training line's cells go through a bridge fitted on the paired lines outside its fold; validation and test lines go through the bridge fitted on all paired lines. The 343 paired lines' bulk is stored only because the bridge (and Task 3's bulk-encoder alignment) needs it. |
| Pseudo-bulk | PIE's per-cell normalisation first: `log1p(mean_i CP10K_i)`, built new; the existing sum-then-CPM cache is not used. The bridge is refitted on it (Task 3's prior reference is refitted on the same input). PIE's own input stays the mean over cells of `log1p` CP10K, as PIE computes it (`pie.process.normalize`, `prep/labels.py:pseudobulk_h5ad`). |
| Text | Every arm reads the Cellosaurus text, as PIE does. No text-off PIE pass. The spec's "every arm with text has a text-off row" and the "ΔL without text" control are removed; the line-permuted-Δ control keeps gene identity covered, and the "Cellosaurus text only" reference row stays. |
| Where | H20 container 30838 only (4 × H20, 224 CPUs, 1.5 TB RAM, 981 TB free); GPU jobs run one after another. |
| Storage | No limit binds: deltas in fp32, plus the uncompressed `Δ̂` panel for the selective genes (~320 GB) for the full-panel check. |

Facts found during exploration, also written into the spec:
- About 1,036 lines enter the PIE pass (the 981 plus 55 held-out lines), so about 18.5M pairs (the spec says 7.8M), roughly 17 h by the spec's FLOP rate.
- The lines have 573,532 cells, about 1.15M with the shuffled copies (the spec says ~2M).
- The H20 cannot reach OpenAI, so the Cellosaurus embedding runs on the Mac.
- Cellosaurus's live release is 56.0, PIE's pin.
- ESM C 600M and ESM3-open are ungated on hf-mirror.
- The AlphaFold DB API host does not resolve from the H20, but `ftp.ebi.ac.uk/pub/databases/alphafold` does.

Routine choices (stated, not asked):
- **Bulk-only lines' PIE input.** DepMap's `log2(TPM+1)` is converted to `log1p(TPM/100)`, i.e. CP10K. A panel gene the bulk does not list takes the frame-line mean.
- **Doublet score.** `scanpy.pp.scrublet` per line, NaN for lines too small to simulate. The count is reported.
- **Per-gene spread.**
  - Detection excess is the detected fraction minus `mean_i(1 − exp(−μ_g L_i / 10⁴))`.
  - SD excess is the SD minus the SD of one Poisson replicate of the line's cells at `λ_ig = μ_g L_i / 10⁴` (seed 0).
  - Both use `log1p` CP10K.
- **Summaries.**
  - Program features come from all cells. The noise diagnosis's per-cell program scores are already over every cell.
  - Cluster shares come from the noise diagnosis's K-means. It is refitted the same way on training cells and then assigns every cell.
- **Pilot genes.** 200 random decode-panel genes (seed 0); the selective share is reported. The ΔE 512-component basis is fitted on the pilot's frame-line pairs.
- **ESM3-open.** Reads sequence plus the AFDB structure of the same UniProt accession, provided the sequences match. A gene without a matching structure has the source masked (§4.4's rule).

## Implementation (branch `feat/ladder-inputs`)

Step 0: save this plan as `docs/specs/2026-10-16-pie-ladder-inputs-plan.md`, apply the spec edits above, and baseline
the suite (`uv run python -m pytest tests -q > pytest.txt 2>&1`).

### 1. Config and manifest

- `configs/stage_one/ladder_inputs.yaml`, plus a strict family in `src/experiments/config.py` (`validate_ladder_config` and `load_ladder_config`, following the `_*_GROUPS` pattern at `:178-219`). It covers:
  - paths: split, gene effect, bulk, Model.csv, basal raw UMI, prepared root, ladder root, PIE checkpoint and sources, Tx1 model, Gene2vec, AFDB;
  - folds (5, seed 0);
  - the excluded lineages;
  - HVGs: the 2,000 of STATE's order, in the prepared space at the recorded `T`;
  - pilot genes (200);
  - ΔE components (512);
  - batch sizes.
- `ladder/manifest.json` records the expression spaces:
  - `log1p_cp10k` for PIE and pseudo-bulk;
  - `log1p_normalize_total` at `T` for the HVG cells;
  - raw UMI for Tx1;
  - the frame and its counts, the folds, the per-step provenance and SHA-256 sums.

  A root grows only under identical settings, as `prepare.py:260-292` does.

### 2. CPU preparation: `src/data/prepare/prepare_ladder_inputs.py` (project env, `.venv-tx1` on the H20)

Steps are resumable, write per line atomically, and skip any output that already exists.

- **`lines`**
  - Builds the frame from `load_geneeffect_split` and `load_bulk_lines` (`src/data/extra_lines.py:94`), `read_models` (`src/data/depmap.py:78`) and the lineage drop, with `assert_fit_eligible` on every fit line.
  - `m_g` is the mean GeneEffect over the 981.
  - Folds come from `patient_folds` (`src/context_prior/folds.py:10`) over the 981.
  - Lineage, patient, RRID and any flag for a sibling of a held-out line are written to `lines.parquet`.
- **`cells`** (per line with cells, all cells, read through `read_basal_source` at `src/data/tx1_cache.py:104`):
  - PIE input `x_c`: the mean of `log1p` CP10K on the decode panel. The panel is the GeneEffect genes that have a `gene_text` entry, aligned by symbol and PIE's aliases, with coverage reported. Reuse `compute_q_sc(target_sum=1e4)` (`src/data/q_sc.py:27`) where it matches.
  - Pseudo-bulk `log1p(mean CP10K)`.
  - Per-gene spread excess.
  - Per-cell QC and cycle values: log UMI, genes detected, mito and ribo fractions, S and G2/M scores from `program_scores` (`src/data/noise_inputs.py`), scrublet.
  - HVG cells and the shuffled HVG cells (each HVG permuted across the line's cells, seed 0).
  - The mean HVG profile.
  - Raw counts shuffled per gene, for Tx1.
- **`mean`**
  - Puts the new pseudo-bulk and the bulk into the prior's space with the existing helpers: `measured_genes`, `fill_unmeasured`, `quantile_reference`, `quantile_normalize` (`src/context_prior/space.py`).
  - Training lines with cells go through `oof_bridged` (`src/context_prior/bridging.py:70`) with the frame folds. Validation and test go through `fit_bridge` on all 343 paired lines.
  - Bulk-only lines read bulk.
  - Stores the mean channel, the raw quantile-normalised pseudo-bulk, the paired bulk, and PIE's `x_c` for bulk-only lines (TPM → CP10K).
- **`summaries`**: program fractions and means (`program_features`, `src/context_prior/noise.py:188`) and cluster shares at 8 and 16 (`cluster_proportions`, `noise.py:210`).
- **`tx1`** (GPU, `--shard i/4`, one process per GPU)
  - Embeds every cell and every count-shuffled cell with `_build_tx1_encoder` (`src/model/tx1.py:89`) through `encode_lines`. It runs in fp32, matching the existing cache, with no cell cap.
  - Writes to a new root `data/tx1_basal_embeddings/v2_all/{real,shuffled}/<ModelID>/`.
  - Timed on a sample first.
  - Stopping it loses at most the line in progress.

### 3. Cellosaurus text: `src/data/prepare/cellosaurus_context.py` (Mac)

- Runs with only `pie[sources]` importable:
  ```
  uv run --isolated --python 3.12 --with-editable 'third_party/pie[sources]' python -I -m src.data.prepare.cellosaurus_context
  ```
  The module imports only `pie` and the standard library.
- Uses the RRIDs from Model.csv for the ~1,036 lines.
- Fetches each record with PIE's `CellosaurusClient` (release 56.0) and renders it with `render_context`.
- Embeds with `text-embedding-3-large` through PIE's OpenAI embedder and writes a PIE dense source `context_text` keyed by ModelID, with provenance.
- The user runs it with `! OPENAI_API_KEY=… uv run …`, so the key never touches a file.
- The output is copied to the H20 with a SHA-256 check, then deleted on the Mac. Lines without an RRID are counted.

### 4. Gene sources

- **`src/data/prepare/embed_protein_sources.py`** (new `.venv-esm` built by `hpc/esm/setup.sh`, EvolutionaryScale `esm`; 1 GPU)
  - Embeds with ESM C 600M and ESM3-open, using the reviewed human UniProt sequences behind PIE's `esm2`, pooled as PIE pools `esm2`.
  - ESM3 adds AFDB human-proteome structures from `ftp.ebi.ac.uk`.
- **`src/data/gene_sources.py`**
  - Reads and aligns `gene_text` and `string_space` (PIE_sources), ESM C, ESM3 and Gene2vec onto the GeneEffect gene axis. Gene2vec comes from `data/cilantro/official/gene2vec_embs.pt`, mapped by Ensembl via Geneformer's dict as in `src/experiments/cilantro_coverage.py:45-59`.
  - Reports coverage per source.

### 5. Frozen PIE: `src/data/prepare/pie_precompute.py` (`.venv-pie`, imports `pie` lazily like `pie_reproduction.py`)

- **Loading.**
  - `best_auprc.ckpt` of `PIE_replogle_xdataset` loads through PIE's `load_checkpoint` and `PieModel` with `strict=True`, as in `predict.py:169-254`.
  - It runs in bf16 autocast under `inference_mode`.
  - The same-perturbation evidence is the checkpoint's training evidence, served by `serve_evidence` under a dataset name that is not in the artifact, so nothing is excluded and the context blocks are zero.
  - Every evidence key is present, with `have=0` where a block is absent.
- **DepMap masked.** `depmap_gene_effect` is popped from each batch's `source_tokens` and `source_masks` after collate. It cannot leave the source list, because the pinned `source_dims` require it. `_build_kv` then skips it, and the carrier covers rows left empty.
- **`pilot`**
  - Re-scores Replogle `test_seen` and `test_unseen` with the mask, through `pie.evaluate.run_eval` and a masking collate. These numbers sit beside Task 1's.
  - Then 200 genes × every line, timed.
  - Fits the 512-component ΔE basis on the frame-line pairs and reports the variance it keeps.
  - For each delta, reports the variance across lines within a gene against the variance between genes.
  - Writes `pilot.md`. **Stop: the user reads it before `full` launches.**
- **`full`**: gene-major, with the genes split across 4 processes.
  - Each gene g's sources and evidence are built once.
  - The encoder runs once per distinct context text (ΔL is stored per text).
  - The decoder runs over line batches.
  - One reference pass per line, with every perturbation source and every evidence block absent.
  - A forward hook on `_decode_chunk` captures the last decoder-layer features.
  - The pass writes:
    - ΔH (panel-pooled, 768-d);
    - ΔE (the softmax expectation of `Δ̂` over the Δp bin centres at PIE's temperature, projected onto the 512-component basis);
    - the uncompressed `Δ̂` for the selective genes;
    - ΔL;
    - the own-gene slot `h_g`, `Δ̂_g`, `x_cg`.
  - Each gene shard is written atomically, and the pass resumes by skipping finished shards.
- **`assemble`**: centres each delta within its gene on the frame-line mean (the mean is stored) and writes `pie/{dH,dE,dL,own}.npy` with an index.

### 6. Runner and docs

- `hpc/run.sh` gets three commands:
  - `ladder-inputs CONFIG --steps …` (project env; `tx1` with `--shard`);
  - `pie-precompute CONFIG --steps pilot|full|assemble [--shard i/4]` (`.venv-pie`, PIE env vars as `pie-reproduce` sets them);
  - `gene-sources CONFIG` (`.venv-esm`, then project env).
- `hpc/README.md` gets the launch recipes.
- New protocol section §15, "Stage-one inputs".
- Data card `docs/data/ladder-inputs.md`.
- Record `results/ladder_inputs/README.md`: frame counts, coverage per source, line and gene, the pilot, and timings.
- Update the current state in CLAUDE.md.

### 7. Tests (synthetic fixtures; PIE tests `importorskip("pie")` and run in `.venv-pie` on the H20)

- The HVG shuffle keeps every gene's values per line and removes the correlation between genes.
- The count shuffle keeps each gene's raw-count multiset.
- Pseudo-bulk equals `log1p` of the mean per-cell CP10K, and `x_c` equals PIE's `normalize_log1p` then mean.
- The spread excess is ≈ 0 on Poisson data.
- `m_g`, the folds and the bridge refuse validation and test lines.
- No training line's bridged row uses a bridge fitted on its own fold.
- TPM → CP10K conversion.
- The manifest refuses other settings.
- PIE batches carry no `depmap_gene_effect` token.
- On a tiny random `PieModel`, the hooked decoder features equal `_decode_chunk`'s.
- The ΔE expectation equals the bins' mean under the softmax.
- Strict checkpoint load.
- A two-line end-to-end run (lines → cells → mean → tiny PIE pass → assemble) writes a complete manifest.

## Verification and launch (30838)

1. On the Mac:
   - Run the suite and lint: `.venv/bin/ruff check .` and `ruff format` on the touched files.
   - Commit and push.
   - Pull on 30838 with `ssh -A`.
   - Run the PIE-dependent tests there in `.venv-pie`, and run the two-line end-to-end on real assets.
2. `OMP_NUM_THREADS=32 nohup hpc/run.sh ladder-inputs configs/stage_one/ladder_inputs.yaml --steps lines,cells,mean,summaries > outputs/launches/ladder_inputs.log 2>&1 &`
3. The Mac Cellosaurus step (the user runs it with their key). Copy the output to the H20.
4. `CUDA_VISIBLE_DEVICES=0..3` `hpc/run.sh pie-precompute … --steps pilot`. Then `gene-sources` (1 GPU). While the user reads `pilot.md`, `ladder-inputs --steps tx1` runs on the 4 GPUs (stoppable).
5. After the user's read: stop Tx1, run `pie-precompute --steps full --shard i/4` ×4, then `assemble`, then resume Tx1.
6. Confirm every manifest step and each coverage table from the outputs. Write the results record. Merge `feat/ladder-inputs` into `main`, push, and delete the branch.
