<div align="center">

  <h1 style="margin-top: 10px;">Generalizable Synthetic-Lethality Discovery by Virtual-Cell Composition</h1>

  <h2>Study context-conditioned synthetic-lethality ranking in held-out cancer cell lines.</h2>

  <div align="center">
    <a href="https://github.com/ENNCELADUS/gene-perturbation-prediction/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/ENNCELADUS/gene-perturbation-prediction"/></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg"/></a>
    <a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/managed%20with-uv-261230.svg"/></a>
    <a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/lint-ruff-orange.svg"/></a>
  </div>

  <p>
    <a href="#why-this-project">Why This Project?</a>
    ◆ <a href="#quick-start">Quick Start</a>
    ◆ <a href="#research-framing">Research Framing</a>
    ◆ <a href="#installation">Installation</a>
    ◆ <a href="#architecture">Architecture</a>
    ◆ <a href="#results">Results</a>
  </p>

</div>

> **Status (2026-10-02):** The seed-0 joint GeneEffect run is trained, tested and baselined: its Huber loss beats the context-blind gene mean by 0.08% and its residual correlations trail a Tx1 context-PCA ridge, a working path and not a result. On that frozen backbone, a readout head with an explicit gene-specific context slope lifts validation residual Pearson from 0.05 to 0.13 and ties the eight-component context-PCA ridge. No tested interface transfers a perturbation response to a held-out cell line. The pipeline now runs in STATE's log expression space, with STATE on its own released basal encoder and Tx1 feeding only the GeneEffect head, and one command runs preparation, a six-arm response-model comparison, joint training and validation evaluation; it has no result yet. Further model decisions use validation. No SL model has run. [`Protocol`](docs/03-geneeffect-protocol.md) · [`Diagnostics`](results/p1_response_pathway_diagnostics/README.md) · [`Joint result`](results/joint_geneeffect_seed0/README.md) · [`Research statement`](docs/01-blueprint.md).

The central question of the active direction:

> Can a dependency profile predicted by a perturbation-response-trained virtual cell rank synthetic-lethal pairs in a cancer cell line that was excluded from every fitting and selection step, beyond what a declared null and a context-ablated model already achieve?

The intuition is compositional: **a cell line's dependency profile is what makes a pair lethal there.** If a virtual cell can predict how a gene's fitness cost shifts with cellular context, then the shape of that shift across lines should carry pair-specific signal that a curated SL graph can only memorize. The bar is deliberately internal — the gene-mean block, the null baseline, and a context-ablated head must all be beaten before any context claim is licensed. Nothing here estimates a genetic interaction; see the rules in the [GeneEffect](docs/03-geneeffect-protocol.md#11-rules) and [SL](docs/04-sl-ranking-protocol.md#8-rules) protocols.

## *Latest News* 🔥

- **[2026/10]** **One expression space and one command.** STATE's released checkpoint was trained on whole-library-normalised `log1p` expression, but the joint pipeline fed it raw counts through a newly initialised Tx1 basal layer. Every expression quantity except Tx1's input is now `log1p(x·T/library size)` on STATE's 2,000 highly variable genes, STATE reads basal cells through its own released encoder, and Tx1 feeds only the GeneEffect head. `hpc/run.sh all` runs preparation, a six-arm leave-one-anchor-out response-model comparison (what the STATE transformer adds over a plain MLP, and what Tx1 adds as a representation), joint training and validation evaluation to one `summary.md`. The closed diagnostic harnesses were deleted; their evidence stays. [`Design`](docs/specs/2026-10-02-expression-space-and-all-pipeline-design.md) · [`Protocol §9`](docs/03-geneeffect-protocol.md#9-response-model-comparison-and-the-all-run).
- **[2026/09]** **Response pathway diagnosed and its numeric-space defect corrected.** Seed-0 diagnostics: response preparation had cached raw UMI counts for a STATE decoder trained in log space, leaving the backbone's response block without perturbation-identity use. After the correction every adapted interface learns its training lines, but none beats no-change on a held-out cell line (pooled leave-one-anchor-out ratios 1.09–26); the untrained released checkpoint beats no-change on HepG2 and Jurkat, not K562 or HCT116. A gene-specific context slope raises validation residual Pearson 0.05 → 0.13 at three head seeds and ties the eight-component context-PCA ridge. No context or SL claim. [`Result`](results/p1_response_pathway_diagnostics/README.md) · [`Protocol §8`](docs/03-geneeffect-protocol.md#8-where-the-model-stalls-response-pathway-diagnostics).
- **[2026/09]** **Seed-0 joint GeneEffect run trained, tested and baselined.** Test Huber 0.01612 against 0.01613 for the gene mean (0.08% lower); residual Pearson 0.054 against 0.122 for the Tx1 context-PCA ridge. A working training and evaluation path, not a context-modelling result. [`Result`](results/joint_geneeffect_seed0/README.md).
- **[2026/09]** **Staged GeneEffect protocol (Exp13) completed — negative point estimate.**
  The selected model reached held-out test macro per-gene Spearman 0.0225, below
  context-PCA ridge (0.0851) and nearest-line (0.0462); its macro per-line score was
  0.0217 versus 0.0993 and 0.0577. This one-seed GeneEffect result licenses no positive
  context or SL claim. [`Result`](results/exp13_stage2_full/README.md).
- **[2026/08]** **Tx1 does not read CPM like raw counts.** Measured per-cell cosine 0.92–0.95 against the raw encode, and unlike gene-subsampling noise the shift survives pooling to the per-line mean (0.972–0.987), so the 152 Kinker `processed_cpm` lines were rebuilt from SCP542 raw UMI counts. Also found: the collator subsamples genes with an unseeded `randperm` above 2048 detected genes, so runs pin a collator seed. [`Result`](results/exp13_stage0/README.md).
- **[2026/08]** **Nine-context SL split built.** K562/JURKAT/OVCAR8/HAP1/HT29 are train, A549 validation, and 22RV1/PC9/HELA test; PC9/HELA are SL-label-only, with cross-side source rows and pairs isolated. [`Data card`](docs/data/sl-context-screen.md) · [`Protocol`](docs/04-sl-ranking-protocol.md).
- **[2026/07]** **Tx1-conditioned STATE few-shot GeneEffect gate completed — negative.** On a 28 train / 5 validation / 9 test GeneEffect split and a 587-gene slice, the Tx1-3B-conditioned STATE model failed to beat copy-K562 + 10 labels (`Delta rho = -0.0048`, 95% CI `[-0.0941, 0.0769]`, registered `rho_min = 0.05`). The HVG-conditioned control was also negative (`Delta rho = 0.0326`, 95% CI `[-0.0602, 0.1181]`). Both few-shot curves deteriorated with larger k. [`Result`](results/tx1-hvg-geneeffect-phase-f.md).
- **[2026/07]** **K562 counterfactual co-dependency kill-test against Horlbeck completed — negative.** The frozen K562 forward-model backbone composed into a symmetrized counterfactual co-dependency score does not recover measured Horlbeck K562 genetic interactions over the 83,028 covered pairs (|Spearman| < 0.01; AUROC ≈ 0.52, below the single-gene floor; no dose-response), across both pooler reference conventions. The composition mechanism was not extended across cell lines. [`Result`](results/exp05-bridge-a-horlbeck-kill-test.md).
- **[2026/07]** **HCT116 frozen-backbone audit closed negative.** Direct K562 GeneEffect transfer remained strong (Spearman 0.554), but the response head collapsed and added no independent HCT116 signal. Single-gene backbone evidence, not cross-cell-line SL. [`Result`](results/exp05-hct116-frozen-backbone-transport.md).

## Why This Project?

This project tests whether basal cell state and predicted perturbation response can improve GeneEffect prediction and, in a separate protocol, SL-pair ranking in held-out cell lines. No SL graph enters the features. Single-gene predictions alone do not measure genetic interaction.

- **🔗 Composes, not memorizes** — Turns a single-gene fitness model into a pairwise score against a declared null baseline, instead of reading topology off an SL graph.
- **🧊 Gene features** — ESM2 identity and predicted perturbation response define the inputs; the current benchmark does not hold out genes or establish unseen-gene performance.
- **🌐 Context-resolved evaluation** — The generalization axis is the cell line, on a fixed held-out split. Pan-essentiality is a controlled variable, not an assumption, via the gene-mean / context-residual split.
- **🪜 Honest baseline ladder** — Gene mean → K562 copy prior → nearest line → context-PCA ridge on Tx1 and on log HVG features, so every gain is measured against a simpler control.
- **🚪 Train-only fitting** — Model updates and fitted preprocessing use training cell lines; validation selects checkpoints and test evaluation remains separate.
- **📏 Terminology guardrails** — Dependency prediction, essentiality ranking, and SL candidate prioritization are kept strictly distinct (see [Terminology](#terminology-guardrails)).

## Quick Start

```bash
# 1. Clone and sync the environment (uv-managed, project-local .venv)
git clone git@github.com:ENNCELADUS/gene-perturbation-prediction.git
cd gene-perturbation-prediction
uv sync

# 2. Verify the environment
uv run python -c "import anndata, scanpy, torch, scvi; print('environment ok')"

# 3. Run the test suite (uses synthetic fixtures, no external data needed)
uv run python -m pytest tests -q
```

> **Prerequisites**: Python 3.11–3.12 and [`uv`](https://docs.astral.sh/uv/). Running the pipeline additionally requires Perturb-seq `*.h5ad` files, DepMap labels and model weights, which are **not** committed to git (see [Data Sources](#data-sources-and-roles)).
>
> See [Installation](#installation) for optional dependencies and [the launcher guide](hpc/README.md) for the GPU environment.

## Research Framing

> **Status:** the SL contract and protocol are written, and SL ranking is scored on a benchmark this project proposes; its current build is the nine-context screen table, whose raw-filter audit is incomplete. The Feng 2024 SL benchmark is not used. No SL model has run. Research statement: [`docs/01-blueprint.md`](docs/01-blueprint.md); rules: [`docs/04-sl-ranking-protocol.md`](docs/04-sl-ranking-protocol.md#8-rules).

```text
Given a cancer cell line described only by its basal single-cell transcriptome —
no CRISPR screen, no SL screen — rank unordered gene pairs by the probability
that the pair is an experimental synthetic-lethal hit in that line.
```

The generalization axis is the **cell line**. Graph and knowledge-graph SL predictors need the query gene to already be a node, so they cannot score an unscreened gene at all and cannot condition on a cellular context; this program reaches both by reading from gene identity and predicted perturbation biology instead of graph topology. Genes are *not* held out here, so no unseen-gene claim is available from this benchmark.

- **Expression space.** Tx1 reads raw UMI counts. Every other expression quantity is `log1p(x·T/library size)` sliced to STATE's 2,000 highly variable genes, with library size over all genes and `T` the median library size of the Jurkat and HepG2 non-targeting cells, computed at preparation and recorded.
- **Joint GeneEffect training.** STATE, initialised from its released Replogle checkpoint, reads log-normalised basal cells through its own basal encoder and is driven by an ESM2 adapter's perturbation token; frozen Tx1 embeddings feed only the context of the five-block residual head. STATE, the adapter and the head train together with GeneEffect Huber regression over the 170 labeled training lines; every fourth update also fits response distributions on all conditions of four anchor lines.
- **Validation and testing.** Every epoch logs fixed-model `val_*` GeneEffect metrics over the 27 validation lines, the only validation split, and `train_eval_*` diagnostics over a fixed 27-line subset of the training lines (recorded in `run.json`). Early stopping and `best.pt` use only minimum `val_geneeffect_loss`. Use `src.evaluate --split train` for checkpoint diagnostics; test is evaluated only explicitly. Feature ablations use the five `model.head_blocks` flags; removing response from the readout requires disabling both `use_delta_proj` and `use_s`.
- **Response-model comparison.** Six arms, leave one response anchor out: no change, global mean effect, the released STATE checkpoint, STATE as in the joint model, an MLP on log HVG cells and an MLP on Tx1 cell embeddings.
- **Separate SL proposal.** A lightweight pair head would score unordered pairs from predicted residual profiles against a declared null. It requires its own out-of-fold fitting and evaluation and is not implemented.

The full contract — task definition, objective, split, controls, and claim boundaries — lives in the research vault, not here:

- [`docs/01-blueprint.md`](docs/01-blueprint.md) — the research statement: question, approach and where the work stands.
- [`docs/02-literature-review.md`](docs/02-literature-review.md) — related work and the novelty boundary.
- [`docs/03-geneeffect-protocol.md`](docs/03-geneeffect-protocol.md) — the executable protocol of the implemented GeneEffect track: benchmark, expression space, model, training, evaluation, results.
- [`docs/04-sl-ranking-protocol.md`](docs/04-sl-ranking-protocol.md) — the SL-pair protocol that builds on it, and its prerequisites.
- [`docs/specs/2026-10-02-expression-space-and-all-pipeline-design.md`](docs/specs/2026-10-02-expression-space-and-all-pipeline-design.md) — the design of the current expression space, STATE wiring and `all` run.
- [`docs/data/`](docs/data/) — one card per dataset. Read the card before using the file.

## Installation

For a quick setup, see [Quick Start](#quick-start) above. This section covers detailed setup and optional dependencies.

### Environment Setup

```bash
git clone git@github.com:ENNCELADUS/gene-perturbation-prediction.git
cd gene-perturbation-prediction

uv python install 3.11      # if 3.11 is not already available
uv sync                     # creates project-local .venv, installs all dependencies
uv run python -c "import anndata, scanpy, torch, scvi; print('environment ok')"
```

`uv sync` installs the core stack (anndata, scanpy, scvi-tools, torch, scikit-learn, accelerate, arc-state) plus the `dev` group (pytest, ruff, xgboost). Optional extras are declared in `pyproject.toml`:

- **`baseline`** — `xgboost`.
- **`research`** — `datasets`, `scib` for additional analysis.
- **`viz`** — `matplotlib`, `seaborn`, `networkx`, `tabulate` for plotting.

### Day-to-Day Commands

```bash
uv run ruff check .                     # lint
uv run ruff format <files you touched>  # format; never the whole tree
uv run python -m pytest tests -q        # full test suite (synthetic fixtures)
```

### Current Entrypoints

Run commands from the repository root. The [joint configuration](configs/geneeffect_joint.yaml)
names supplied datasets, model initialization and cache paths; unknown or missing keys are
errors. One command runs the whole route; rerunning it with the same run id resumes and
skips finished steps.

```bash
# Preparation, response-model comparison, joint training, validation evaluation,
# baselines, readout head and summary.md under outputs/geneeffect_joint/<run id>/
# Every GPU step uses every visible GPU, or only those --gpus lists
hpc/run.sh all configs/geneeffect_joint.yaml [--run-id <id>] [--gpus 0,1,2,3]

# Explicit testing of a selected checkpoint (never run by `all`)
hpc/run.sh test outputs/geneeffect_joint/<id>/train/best.pt

# Standalone evaluation of a checkpoint on validation (or train, for diagnostics)
uv run python -m src.evaluate --checkpoint outputs/geneeffect_joint/<id>/train/best.pt --split val
```

> Raw `*.h5ad`, `*.csv`, checkpoints, and large artifacts are gitignored. The pipeline requires Perturb-seq and DepMap data you supply locally.

## Architecture

- `src/data/`: fixed splits, batch records, the expression transform, basal/response caches and preparation tools.
- `src/model/`: STATE and ESM2 adapters, the STATE-free response MLP, live features, residual head, readout head and losses.
- `src/training/`: sampling, optimization and resumable checkpoints.
- `src/eval/`: common validation/test scoring and metrics.
- `src/baselines/`: residual controls fitted on training lines.
- `src/experiments/`: preparation, the response-model comparison, the readout head, baselines and the `all` run; `historical/` retains completed probes and the SL benchmark builder.
- `src/train.py`, `src/evaluate.py`: thin module entry points.
- `hpc/`: launcher and [operator guide](hpc/README.md); `scripts/` contains operational utilities.
- `configs/`: current joint config, fixed benchmark membership and small input provenance.
- `outputs/`: ignored generated runs; `results/`: tracked reports and small evidence.

## Results

Current GeneEffect-track results are in the [protocol §7](docs/03-geneeffect-protocol.md#7-results) and [`results/`](results/). The sections below are historical evidence from retired routes; their raw local outputs and implementations are not in the tree, and they are not results of the active context-conditioned SL protocol. Consolidated table: [`results/prior-internal-evidence.md`](results/prior-internal-evidence.md).

### HCT116 Frozen-K562-Backbone Transport (one-shot audit, 2026-07-21)

On the 1,652-gene primary cohort, direct K562 GeneEffect transfer retained
Spearman **0.554**, while the frozen response head reached **-0.001** with a
collapsed prediction standard deviation of 0.059 versus 0.409 for HCT116
GeneEffect. A follow-up analysis controlling for K562 GeneEffect gave
partial Spearman about -0.005. The failed path is HCT116 observed response through the frozen K562
fitness head; this is not a pairwise SL or cross-cell-line SL result. Full
protocol, metrics, and interpretation: [`results/exp05-hct116-frozen-backbone-transport.md`](results/exp05-hct116-frozen-backbone-transport.md).

### Single-Cell Bag → Dependency (Adamson K562 external transfer)

The best distribution/prototype regressor (K64-centered Ridge) reaches Adamson **Spearman ≈ 0.67**, **AUROC ≈ 0.91**, **AUPRC ≈ 0.80**, with held-out-gene Spearman ≈ 0.64 — clearing the original distribution-regression gate and beating the earlier scVI128 single-head gated-attention row. Full tables: [`results/prior-internal-evidence.md`](results/prior-internal-evidence.md).

## Data Sources and Roles

| Source | Role | Notes |
| --- | --- | --- |
| Perturb-seq / CROP-seq / CRISPRi-seq | Response supervision | Genetic-perturbation responses and non-targeting controls of the four anchor lines (K562, HepG2, Jurkat, HCT116). |
| Single-cell basal atlases | Context input | Basal raw-UMI cells of the 226 GeneEffect benchmark lines; see [`docs/data/cell-line-geneeffect-226.md`](docs/data/cell-line-geneeffect-226.md). |
| DepMap 26Q1 | Supervision and context | CRISPR GeneEffect scores, joined by ModelID; see [`docs/data/depmap-public-26q1.md`](docs/data/depmap-public-26q1.md). |
| Integrated SL screen pairs | SL pair labels | Experimental screen hits and screened non-hits in named cell lines, the source of the nine-context SL table; see [`docs/data/sl-context-screen.md`](docs/data/sl-context-screen.md). |
| TCGA / patient omics | Disease context | Future biomarker framing only; not evidence of cell-line or patient generalization under the current protocol. |

> **Data rules**: Prioritize CRISPRi or knockout Perturb-seq for DepMap alignment. Cross-cell-line claims require context-specific pairwise SL/GI labels; single-gene GeneEffect transfer is insufficient. Norman CRISPRa is auxiliary, and DepMap labels are population-level fitness readouts, not single-cell death.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) — instructions for AI coding agents.
- [`docs/01-blueprint.md`](docs/01-blueprint.md) — the research statement; start here. [`results/`](results/) holds registered evidence.
- [`docs/03-geneeffect-protocol.md`](docs/03-geneeffect-protocol.md) and [`docs/04-sl-ranking-protocol.md`](docs/04-sl-ranking-protocol.md) — the two executable protocols.
- [`docs/data/`](docs/data/) — dataset cards for downloaded data.
- [`hpc/README.md`](hpc/README.md) — running the pipeline on the GPU host.

## Contributing

This is a research repository. When contributing:

```bash
# Fork, then clone your fork
git clone git@github.com:YOUR_USERNAME/gene-perturbation-prediction.git
cd gene-perturbation-prediction
uv sync

# Create a feature branch
git checkout -b feature/your-feature-name

# Verify before committing
uv run ruff check .
uv run python -m pytest tests -q

git commit -m "feat: description"   # Conventional Commits
git push -u origin feature/your-feature-name
```

Follow the **Plan → Confirm → Code** workflow for non-trivial research or implementation changes, use Conventional Commits (`feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `ci`), and respect the terminology guardrails below.

## Terminology Guardrails

- Say **dependency / GeneEffect prediction** for the single-gene supervised task; **SL candidate prioritization** for the pairwise score — never "validated SL target" (benchmark negatives are unconfirmed).
- Do not claim SL from single-gene essentiality; a declared null baseline is required.
- **Never write "interaction" about a model result.** The pair head takes the null as one input among many and predicts no joint outcome, so the model-minus-null gap is incremental label ranking. An interaction claim needs a joint or measured genetic-interaction quantity.
- The generalization axis is the **cell line**. Genes are not held out, so no unseen-gene claim is available.
- Qualify every held-out-context result with foundation-model pretraining exposure: task-label holdout is not representation-pretraining holdout, Tx1 saw Tahoe-100M, and STATE's released checkpoint saw K562, HepG2 and Jurkat.
- A pan-essentiality lift is not an SL result — the gene-mean block must be ablated, not assumed away.
- No significance claim across contexts: one split with few test contexts admits no valid family-wise inference over the baselines, arms, and strata.
- Do not call DepMap GeneEffect a single-cell death label; it is a single-gene relative growth-rate effect.
- Norman CRISPRa is auxiliary only, never aligned to knockout labels without the modality caveat.
- The Feng 2024 SL benchmark is not used; [`docs/02-literature-review.md`](docs/02-literature-review.md) cites it as background on evaluation axes, not as a bar this program reproduces.

---

<div align="center">
  <p>
    <strong>Active direction: generalizable synthetic-lethality discovery by virtual-cell composition, evaluated on held-out cell lines.</strong><br>
    <sub>See <a href="docs/01-blueprint.md">docs/01-blueprint.md</a> for the research statement.</sub>
  </p>
</div>
