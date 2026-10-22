# Module map

One line per module, from its header. Commands are in [`hpc/README.md`](../hpc/README.md), rules in
[`CLAUDE.md`](../CLAUDE.md). `data` and `model` never import `training`, `eval` or `experiments`; each research
track's commands live in `experiments/<track>/`, retired ones in `experiments/historical/`.

### `data/`

- `basal`: Raw-count helpers and memory-bounded readers of Perturb-seq sources.
- `batches`: Batch records shared by the joint GeneEffect data, model and training code.
- `cell_states`: Cell-state vocabularies (PIE stage-one design §3.5).
- `context_pca`: Train-fit, eigen-scaled principal components of the Tx1 line context ``z_c`` (numpy only: preparation workers import this module without torch).
- `datasets`: GeneEffect rows and anchor response conditions over prepared inputs.
- `depmap`: Pinned DepMap matrices as ModelID x gene-symbol frames, and Model.csv.
- `embeddings`: Precomputed ESM2 gene embeddings keyed by upper-case symbol.
- `expression`: STATE's expression space, as arc-state's ``preprocess_train``.
- `extra_lines`: DepMap lines outside the 226 that join the training side.
- `gene_sources`: Gene sources of the stage-one gene encoder (ladder design §4.4), aligned to the GeneEffect gene axis.
- `io`: Atomic file writes every step shares, and plain-JSON conversion.
- `labels`: DepMap GeneEffect labels by ModelID and what the training lines fit from them.
- `ladder_cells`: Per-line cell inputs of the stage-one ladder (PIE's input, pseudo-bulk, per-gene spread, per-cell QC, HVG cells), each over every raw basal cell of the line.
- `ladder_reader`: Readers of the stage-one inputs (``data/geneeffect_joint/v2/ladder/``, protocol §15) and their GPU-resident form, each array checked against ``lines.parquet`` and the decode panel; numpy, pandas and torch only, so ``.venv-pie`` reads them too.
- `noise_inputs`: Per-line inputs of the noise diagnosis from a line's raw basal cells.
- `prepared`: Prepared joint-training inputs (``lines/<ModelID>.npz``, ``response/`` and the ``prepared_inputs.json`` manifest, written last), opened with train-only fitting or a checkpoint's restored state.
- `prior_offsets`: The linear context prior's export (``hpc/run.sh prior-export``) as the joint model reads it.
- `pseudobulk`: Per-line pseudo-bulk.
- `q_sc`: Per-gene basal expression summaries ``q_sc`` of one cell line.
- `response`: The response anchors' observed perturbation responses, from source to prepared targets.
- `splits`: The fixed train/validation/test split of cell lines and the guard that keeps held-out and unlabelled lines out of every fit.
- `tx1_cache`: Tx1-3B basal embedding cache, ``<ModelID>/{embeddings,hvg}.npy, obs.parquet``.

### `data/builders/`

- `bootstrap_exp13_uniprot_cache`: Build the Exp13 UniProt cache from local reviewed artifacts.
- `build_cell_line_geneeffect_226_split`: Build the 226-context GeneEffect split (the final split's single-cell core).
- `build_cell_line_geneeffect_split`: Build the final cell-line split (docs/data/cell-line-geneeffect-split.md), its JSON and ``.csv`` table, from the 226 split and the two extra-line files, which stay its provenance and enter no fit.
- `build_context_reference`: Fetch and pin the linear context prior's reference tables, committed under configs/context_prior/reference/ because the H20 cannot fetch them.
- `build_exp13_esm2_universe`: The Exp13 ESM2 gene universe.
- `build_extra_bulk_lines`: Build configs/benchmarks/extra_bulk_lines_26Q1.json from the 26Q1 files (docs/data/extra-bulk-lines-26q1.md).
- `build_extra_single_cell_lines`: The frozen single-cell identity map and the extra single-cell membership built from it (docs/data/single-cell-atlas-search.md).
- `build_extra_single_cell_sources`: Build the extra single-cell lines' raw-UMI h5ad from their public deposits, one wave of ``configs/data/extra_single_cell_sources_26Q1.json`` per run.
- `build_pc9_hela_basal`: Build Tx1-ready PC9 and HeLa basal-count artifacts on the HPC.
- `build_sl_screen_benchmark`: Build the SL screen benchmark (docs/data/sl-screen-benchmark.md, blueprint §5).
- `build_state_programs`: Fetch and pin the noise diagnosis's reference cell-state programs under configs/context_prior/reference/.
- `cellosaurus_context`: Cellosaurus text of the labelled lines as PIE's ``context_text``, keyed by ModelID.
- `demultiplex`: Naming each called cell's line in a deposited library.
- `embed_protein_sources`: Protein gene sources for the stage-one gene encoder (ladder design §4.4).
- `materialize_exp13_original47_raw_umi`: Materialize the 47 original Exp13 contexts as capped raw-UMI h5ads.
- `plan_second_sources`: Plan one second deposit per labelled single-cell training line for the noise diagnosis's deposit swap (protocol §14), from tracked files only; the plan is what ``build_extra_single_cell_sources --second-source`` builds.
- `precompute_esm2_embeddings`: Fetch UniProt sequences for a supplied gene universe and embed them with ESM2, writing the table with its UniProt mapping and provenance.
- `prepare_cell_line_atlas_raw_umi`: Convert the 27 Breast/CCLA raw-UMI lines to audited Tx1-contract h5ad.
- `prepare_kinker_umi_h5ad`: Build per-line raw-UMI Kinker h5ads that pass the Tx1 input contract from Broad SCP542's ``UMIcount_data.txt``, since published CPM cannot be turned back into counts.
- `recount_extra_single_cell_sources`: Re-count the extra single-cell libraries whose deposits hold no raw counts.
- `single_cell_readers`: Readers for the extra single-cell deposits' counts.

### `model/`

- `features`: Response features of one condition.
- `head`: The GeneEffect heads (protocol §4).
- `initialization`: Build the joint GeneEffect model fresh from released STATE, or restore it.
- `joint`: Joint GeneEffect model.
- `losses`: GeneEffect training objectives on residual-unit predictions.
- `normalization`: Train-only standardization of the head's feature blocks.
- `perturbation`: Fixed per-gene ESM2 vectors mapped to STATE perturbation vectors by an adapter.
- `response`: Response prediction over STATE sentences and the response losses.
- `response_mlp`: STATE-free response model.
- `stage_one`: Stage-one architecture arms (PIE stage-one design §3.3; protocol §16).
- `state`: STATE driven by ESM2 perturbation tokens on log-space basal HVG cells (2000 genes in STATE's own order, through its released 2000->328 basal encoder); Tx1 embeddings never reach STATE, only the GeneEffect head's context.
- `tx1`: Tx1-3B (Tahoe-x1) loaded from its released safetensors without flash-attn, and its basal-cell ``EncoderFn``.

### `training/`

- `checkpoint`: Epoch-boundary training state, selective-gene Spearman selection and checkpoints.
- `sampling`: Training epochs.
- `schedule`: The learning-rate schedule shared by the joint and stage-one trainers, apart from ``src.training.trainer`` (which imports accelerate) so ``.venv-pie`` can import it.
- `stage_one_trainer`: The stage-one training loop (PIE stage-one design §4.1; protocol §16), plain torch on one device for ``.venv-pie``.
- `trainer`: Joint training loop.

### `eval/`

- `accuracy`: GeneEffect scoring.
- `metrics`: Per-unit correlations (NaN, never 0, where undefined), the paired line bootstrap of the selective Spearman, and the calibration slope.

### `baselines/`

- `prior_controls`: Controls built on the linear context prior.
- `residual`: The residual control ladder (gene mean, copy prior, nearest line, context-PCA ridge), fitted once on a fixed split's supervised training lines and scored on its validation and test lines.
- `tx1_gmm`: Basal Tx1 population distributions with train-only per-gene ridge readouts.

### `context_prior/`

- `alignment`: Celligner's contrastive PCA on paired profiles.
- `bridge`: The per-gene affine bridge from quantile-normalised pseudo-bulk to bulk, and what every bridge remedy shares.
- `folds`: Patient-grouped, seeded folds of training-side lines.
- `gene_features`: Features of gene g in line c.
- `noise`: Statistics of the noise diagnosis.
- `prior`: The linear context prior.
- `reference`: The pinned reference tables of the linear context prior.
- `ridge`: Closed-form ridge stages of the linear context prior, each with a per-gene intercept.
- `single_cell`: Per-gene single-cell statistics of a line's basal cells (the prepared ``q_sc`` caches: log-space mean, detected fraction, log-space variance) stacked on the prior as a two-stage ridge (protocol §13).
- `space`: One expression space for bulk and pseudo-bulk.
- `targets`: The metric's fixed definitions, fitted on the labelled training lines only.
- `views`: Context views a query line can supply, computed identically from bulk and bridged pseudo-bulk.

### `context_prior/remedies/`

- `affine`: The reference bridge.
- `contrastive`: Contrastive-PCA remedy.
- `denoise`: Low-rank denoising.
- `gating`: Reliability gating.
- `noise_matched`: Noise-matched fitting.
- `rank_value`: Cilantro-SL's input reductions as a bridge remedy.

### `experiments/`

- `config`: What every track's strict config loader shares.
- `prepare`: Prepare fixed joint-training inputs once, in STATE's log expression space (``log1p(x * T / L_cell)``, Tx1 on raw UMI); the only reader of raw data.
- `run_dir`: What every experiment command does with its run directory.

### `experiments/joint/`

- `all`: One automatic run.
- `baselines`: Evaluate train-fitted residual baselines on the prepared common panel.
- `compare_runs`: Paired line bootstrap of selective Spearman between two runs of one split.
- `config`: The joint GeneEffect model's config (``all``, ``revision``, ``test``; the prior and the ladder read it too).
- `evaluate`: Score a saved joint checkpoint on one split with its saved preprocessing, never fitting (``hpc/run.sh test``).
- `pipeline`: What the joint model's runs (``all`` and ``revision``) share.
- `response_arms`: The six response-model arms the comparison scores in STATE's log expression space.
- `response_comparison`: Six-arm response-model comparison in STATE's log expression space.
- `revision`: One revision run.
- `train`: Train the joint GeneEffect model into a run directory (resumes from ``last.pt``), as one process or under ``accelerate launch``.
- `tx1_gmm_ridge`: Fit a basal-only Tx1 GMM64-ridge baseline and evaluate saved models.

### `experiments/joint/readout/` (`python -m` runs `run`)

- `cache`: Compact, immutable raw feature cache and train-only readout-head preprocessing.
- `compare`: Pair-aligned diagnostic differences and context-cluster uncertainty.
- `heads`: The readout head on the frozen joint backbone's cached features.
- `run`: The readout-head run (``all``'s last step).

### `experiments/prior/`

- `base`: The fitted linear context prior every prior command shares.
- `config`: The linear context prior's configs.
- `context_prior`: Linear context prior experiments.
- `data_selected_analysis`: What the data-selected stage of the reference prior carries.
- `diagnosis`: What the noise diagnosis reads.
- `noise_diagnosis`: What limits the default prior.
- `noise_report`: The noise diagnosis's ``results.md``.
- `noise_sections`: The noise diagnosis's sections (protocol §14).
- `prior_export`: Export of a prior config's reference row for the single-cell correction.
- `single_cell_stack`: Single-cell statistics stacked on a prior run's export (protocol §13).

### `experiments/stage_one/` (`python -m` runs `run`)

- `config`: The stage-one configs.
- `ladder_inputs`: The stage-one inputs under ``ladder_root`` (PIE stage-one design §5 Task 2; protocol §15).
- `pie`: The frozen PIE as the ladder runs it.
- `pie_decode`: PIE's decode as it runs it (``PieModel._decode_group``, step for step), with what does not change across genes hoisted, the gene pass over every line, and the check that it agrees with PIE's own data path.
- `pie_precompute`: The frozen PIE's deltas for every line x panel gene, and on the cell-state pseudo-contexts (PIE stage-one design §3.2, §3.5; protocol §15), in ``.venv-pie``, with PIE's ``depmap_gene_effect`` token never served.
- `pie_reproduction`: PIE reproduction (PIE stage-one design Task 1).
- `pie_states`: The frozen PIE on cell-state pseudo-contexts (design §3.5, Task 8).
- `prior_reference`: The default prior's reference row refitted on the stage-one frame's mean channel, the like-for-like linear reference (PIE stage-one design Task 3; protocol §16), with its out-of-fold predictions and the expression components the components reader reads.
- `report`: A stage-one report.
- `resident`: What a stage-one run keeps on the device.
- `ridge`: The per-gene ridge on the decoder-feature delta (a report row).
- `run`: Stage-one training runs, their queue over GPUs, and the report entry point (PIE stage-one design §3.3, §4, Task 3; protocol §16), in ``.venv-pie``.
- `setup`: A ladder config and everything it names.

### `experiments/states/`

- `config`: The cell-state telemetry's config (protocol §17).
- `inputs`: The ``inputs`` step.
- `programs`: Program (cNMF, starCAT-style reference and usages) and scVI fits of the cell-state vocabularies (protocol §17), run in ``.venv-states`` from the plan the ``fit`` step writes; it exchanges files only.
- `report`: The ``report`` step.
- `state_telemetry`: The cell-state telemetry CLI (protocol §17).
- `telemetry`: A cell-state telemetry run.
- `vocabularies`: The state vocabularies.

### `experiments/historical/`

- `audit_tx1_basal_batch`: Historical audit of basal-source confounding.
- `build_sl_context_benchmark`: Historical builder of the context_screen_v2 SL pair-context benchmark (``docs/data/sl-context-screen.md``) from the integrated raw table, its only pair-label input; the tracked split contract supplies context identities and registration evidence, never labels.
- `cilantro_coverage`: Historical pre-GPU audit of the context screen's coverage by Cilantro-SL.
- `prepare_kinker_cpm_h5ad`: Historical builder of per-line Kinker/scCCLE processed-CPM h5ads for a CPM sensitivity analysis, superseded by raw-UMI preparation; every artifact marks the matrix as non-raw and the standard Tx1 raw-count contract as blocked.
- `stage0_tx1_input_probe`: Historical probe of whether Tx1-3B encodes a CPM-rescaled cell as it encodes the raw counts underneath it, with the collator's unseeded gene subsampling (``_sample``, ``torch.randperm``) isolated by seeding each arm just before its forward pass.
