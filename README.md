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

> **Status (2026-09-06):** Current GeneEffect development uses one joint trainer with recurring four-line response supervision. Validate every epoch and select the checkpoint with minimum validation GeneEffect loss; training, collation and projection seeds are 0. This protocol has no scientific run yet. Historical Exp13 Stage 2 reached test macro per-gene Spearman 0.0225 versus context-PCA ridge 0.0851 and nearest-line 0.0462, a negative point estimate. `context_screen_v2` remains a separate, unrun SL track. [`Joint design`](docs/specs/2026-09-06-modular-joint-training-design.md) · [`Historical result`](docs/results/exp13_stage2_full/README.md) · [`Research contract`](docs/01-blueprint.md).

The central question of the active direction:

> Can a dependency profile predicted by a perturbation-response-trained virtual cell rank synthetic-lethal pairs in a cancer cell line that was excluded from every fitting and selection step, beyond what a declared null and a context-ablated model already achieve?

The intuition is compositional: **a cell line's dependency profile is what makes a pair lethal there.** If a virtual cell can predict how a gene's fitness cost shifts with cellular context, then the shape of that shift across lines should carry pair-specific signal that a curated SL graph can only memorize. The bar is deliberately internal — the gene-mean block, the null baseline, and a context-ablated head must all be beaten before any context claim is licensed. Nothing here estimates a genetic interaction; see [`docs/01-blueprint.md`](docs/01-blueprint.md) §8 for what that forbids.

## *Latest News* 🔥

- **[2026/08]** **Exp13 Stage 0 closed — Tx1 does not read CPM like raw counts.** Measured per-cell cosine 0.92–0.95 against the raw encode, and unlike gene-subsampling noise the shift survives pooling to the per-line mean (0.972–0.987), so the 152 Kinker `processed_cpm` lines were rebuilt from SCP542 raw UMI counts. Also found: the collator subsamples genes with an unseeded `randperm` above 2048 detected genes, so runs must pin a collator seed. [`Result`](docs/results/exp13_stage0/README.md) · [`Protocol §6`](docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md).
- **[2026/09]** **Joint GeneEffect runtime implemented.** Data, model, training and evaluation now have separate modules. One trainer revisits the four response anchors throughout GeneEffect regression; validation loss selects checkpoints. The new protocol has not run on research data. Registered negatives stay in [`docs/results/`](docs/results/); Git history holds the staged implementations.
- **[2026/09]** **Exp13 formal Stage 2 completed — negative point estimate.**
  The selected model reached held-out test macro per-gene Spearman 0.0225, below
  context-PCA ridge (0.0851) and nearest-line (0.0462); its macro per-line score was
  0.0217 versus 0.0993 and 0.0577. The 226-line run is terminally verified, but this
  one-seed GeneEffect result licenses no positive context or SL claim. [`Result`](docs/results/exp13_stage2_full/README.md) ·
  [`Exp13 protocol`](docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md).
- **[2026/08]** **Nine-context split built.** K562/JURKAT/OVCAR8/HAP1/HT29 are train, A549 validation, and 22RV1/PC9/HELA test; PC9/HELA are SL-label-only, with cross-side source rows and pairs isolated. [`Contract`](docs/01-blueprint.md) · [`Protocol`](docs/03-experiment-protocol.md).
- **[2026/07]** **T2 registered primary gate completed — negative.** On the frozen 28 train / 5 validation / 9 test GeneEffect split and 587-gene slice, Tx1-3B-ST failed to beat copy-K562 + 10 labels (`Delta rho = -0.0048`, 95% CI `[-0.0941, 0.0769]`, registered `rho_min = 0.05`). HVG-ST was also negative (`Delta rho = 0.0326`, 95% CI `[-0.0602, 0.1181]`). Both few-shot curves deteriorated with larger k. T2 is paused for redesign and the remaining baseline ladder is closeout work. [`Result`](docs/results/tx1-hvg-geneeffect-phase-f.md).
- **[2026/07]** **K562 Bridge-A-vs-Horlbeck mechanism kill-test completed — negative.** The frozen exp05 backbone composed into a symmetrized counterfactual co-dependency score does not recover measured Horlbeck K562 genetic interactions over the 83,028 exp05-covered pairs (|Spearman| < 0.01; AUROC(s_A → strong-SL) ≈ 0.52, below the single-gene floor; no dose-response), across both pooler reference conventions. Per the kill-test rule the composition mechanism is **paused for redesign and not extended across cell lines**. [`Result`](docs/results/exp05-bridge-a-horlbeck-kill-test.md).
- **[2026/07]** Horlbeck 2018 K562 fitness-GI map acquired and coverage-audited (448 genes, 100,128 pairs; 83,028 exp05-covered), and an execution plan set: two parallel development tracks — a **K562 Bridge-A-vs-Horlbeck mechanism kill-test** on the frozen exp05 backbone, and **DepMap cross-cell-line GeneEffect** transfer (single-gene backbone, not cross-cell-line SL). Neither opens held-out-cell-line SL labels.
- **[2026/07]** HCT116 frozen-backbone audit closed negative: direct K562 GeneEffect transfer remained strong (Spearman 0.554), but the response head collapsed and added no independent HCT116 signal. This is single-gene backbone evidence, not cross-cell-line SL. [`Closeout`](docs/results/exp05-hct116-frozen-backbone-transport.md).
- **[2026/07]** Research contract expanded from a K562-only formulation to a **general SL discovery model**. Feng2024 CV2/CV3 test genes withheld from SL-pair/graph training; held-out-cell-line splits separately test unseen contexts. K562 remains the current backbone and one mechanistic anchor.
- **[2026/07]** Two composition bridges — counterfactual co-dependency and virtual double knockout — were specified against SLMGAE/KR4SL and measured GI. Both were retired by the 2026/08 reformulation; neither is an active target.
- **[2026/06]** Stage 3 SL pair benchmark adapter and dependency-only baseline shipped (`src/sl_benchmark_baseline/`); official-metric CV1/CV2/CV3 rerun completed.
- **[2026/06]** Experiment 05 AIVC STATE A→B→C forward-model pipeline reviewed, including a frozen-STATE feature ablation track.
- **[2026/05]** Single-cell Deep Sets, attention-MIL, and distribution/prototype regressors landed with Adamson K562 external transfer.
- **[2026/05]** Pseudobulk delta baseline ladder established as the Stage 1 dependency-prediction floor.

## Why This Project?

This project tests whether basal cell state and predicted perturbation response can improve GeneEffect prediction and, in a separate protocol, SL-pair ranking in held-out cell lines. No SL graph enters the features. Single-gene predictions alone do not measure genetic interaction.

- **🔗 Composes, not memorizes** — Turns a single-gene fitness model into a pairwise score against a declared null baseline, instead of reading topology off an SL graph.
- **🧊 Gene features** — ESM2 identity and predicted perturbation response define the inputs; the current benchmark does not hold out genes or establish unseen-gene performance.
- **🌐 Context-resolved evaluation** — The generalization axis is the cell line, on a published held-out split. Pan-essentiality is a controlled variable, not an assumption, via the gene-mean / context-residual split.
- **🧪 Observed-first methodology** — Validates that *observed* response carries dependency signal before trusting any *predicted* transcriptome, so forward-model error never silently inflates results.
- **🪜 Honest baseline ladder** — Dummy → ridge → PCA → tabular nonlinear → MIL/foundation models, so every gain is measured against a simpler control.
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
uv run python -m pytest
```

> **Prerequisites**: Python 3.11–3.12 and [`uv`](https://docs.astral.sh/uv/). Running the full pipeline additionally requires Perturb-seq `*.h5ad` files and DepMap labels, which are **not** committed to git (see [Data Sources](#data-sources-and-roles)).
>
> See [Installation](#installation) for optional dependencies and [the launcher guide](hpc/README.md) for the GPU environment.

## Research Framing

> **Status:** contract, protocol and row-level split are built; the raw-filter audit remains incomplete. No model has run. Live contract: [`docs/01-blueprint.md`](docs/01-blueprint.md).

```text
Given a cancer cell line described only by its basal single-cell transcriptome —
no CRISPR screen, no SL screen — rank unordered gene pairs by the probability
that the pair is an experimental synthetic-lethal hit in that line.
```

The generalization axis is the **cell line**. Graph and knowledge-graph SL predictors need the query gene to already be a node, so they cannot score an unscreened gene at all and cannot condition on a cellular context; this program reaches both by reading from gene identity and predicted perturbation biology instead of graph topology. Genes are *not* held out here, so no unseen-gene claim is available from this benchmark.

- **Joint GeneEffect training.** Frozen Tx1 embeddings feed STATE and the five-block residual head. Train STATE, the ESM2 adapter and head together with Huber regression over 170 labeled training lines; every fourth update also reconstructs response distributions on four anchors.
- **Validation and testing.** Every epoch reports total and individual losses, Pearson, Spearman, RMSE, MAE and coverage. Early stopping and `best.pt` use only minimum `val_geneeffect_loss`. Evaluate the selected checkpoint explicitly on test after training.
- **Separate SL proposal.** A lightweight pair head would score unordered pairs from predicted residual profiles against a declared null. It requires its own out-of-fold fitting and evaluation; this refactor does not implement it.

The full contract — task definition, objective, split, controls, and claim boundaries — lives in the research vault, not here:

- [`docs/01-blueprint.md`](docs/01-blueprint.md) — the research contract: task, objective, evaluation, and claim boundaries.
- [`docs/02-literature-review.md`](docs/02-literature-review.md) — related work and the novelty boundary.
- [`docs/03-experiment-protocol.md`](docs/03-experiment-protocol.md) — the SL-pair executable protocol and its prerequisites.
- [`docs/specs/2026-09-06-modular-joint-training-design.md`](docs/specs/2026-09-06-modular-joint-training-design.md) — current GeneEffect training and evaluation.
- [`docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md`](docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md) — historical staged GeneEffect protocol.
- [`docs/data/`](docs/data/) — one card per dataset. Read the card before using the file.

### Prior Program (Retired as Roadmap, Kept as Evidence)

Through mid-2026 the project's roadmap was a **staged, context-ranking** dependency → synthetic-lethality program (distinct from the active *composition* direction above):

```text
cell line + perturbation gene
    → observed or predicted post-perturbation transcriptome
    → dependency / essentiality score
    → context-specific target ranking
```

That staged program is **retired as the roadmap**. Its exp08/exp08b implementation was removed, and the exp05 forward-model stack followed in 2026/08 after T1 and T2 closed negative; the remaining Tx1/Exp13 code supports the current substrate; dependency, SL-benchmark and DDGCN support files are archived. Historical results remain **prior evidence** for the new direction — see [Results](#results) and [`docs/results/prior-internal-evidence.md`](docs/results/prior-internal-evidence.md).

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

`uv sync` installs the core stack (anndata, scanpy, scvi-tools, torch, scikit-learn, arc-state) plus the `dev` group (pytest, ruff, xgboost). Optional extras are declared in `pyproject.toml`:

- **`baseline`** — `xgboost` for the tabular nonlinear model ladder.
- **`research`** — `datasets`, `scib` for additional analysis.
- **`viz`** — `matplotlib`, `seaborn`, `networkx`, `tabulate` for plotting.

### Day-to-Day Commands

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run python -m pytest      # full test suite (synthetic fixtures)
```

### Current Entrypoints

Run module commands from the repository root. The [joint configuration](configs/geneeffect_joint.yaml)
names supplied datasets, model initialization and cache paths. Prepare inputs once;
training opens those caches without rebuilding raw data on every worker.

```bash
# Single-process fixed-input preparation
hpc/run.sh prepare configs/geneeffect_joint.yaml

# Joint training on visible GPUs, or epoch-boundary resume
hpc/run.sh train configs/geneeffect_joint.yaml --run-id joint_seed0
hpc/run.sh train configs/geneeffect_joint.yaml --resume outputs/geneeffect_joint/joint_seed0/last.pt

# Explicit testing of the selected checkpoint
hpc/run.sh test outputs/geneeffect_joint/joint_seed0/best.pt

# Validation or the train-fitted baseline ladder
uv run python -m src.evaluate --checkpoint outputs/geneeffect_joint/joint_seed0/best.pt --split val
uv run python -m src.experiments.baselines --config configs/geneeffect_joint.yaml \
  --split test --out-dir outputs/geneeffect_joint/baselines_seed0
```

> Raw `*.h5ad`, `*.csv`, checkpoints, and large artifacts are gitignored. The pipeline requires Perturb-seq and DepMap data you supply locally.

## Architecture

- `src/data/`: fixed splits, batch records, basal/response caches and preparation tools.
- `src/model/`: STATE/ESM2 adapters, live features, residual head and losses.
- `src/training/`: sampling, optimization, distributed execution and resumable checkpoints.
- `src/eval/`: common validation/test scoring and metrics.
- `src/baselines/`: residual controls fitted on training lines.
- `src/experiments/`: preparation and experiment wiring; `historical/` retains selected probes.
- `src/train.py`, `src/evaluate.py`: thin module entry points.
- `hpc/`: launcher and [operator guide](hpc/README.md); `scripts/` contains operational utilities.
- `configs/`: current joint config, fixed benchmark membership and small input provenance.
- `outputs/`: ignored generated runs; `docs/results/`: tracked reports and small evidence.

The old staged implementation and commands remain in Git at `e6341d2`.

The context-conditioned SL model has not run. Historical dependency/Feng2024 implementations, matching tests and outputs were moved out of the active directories on 2026-09-05. See [archive inventory](docs/archive-inventory-2026-09-05.md).

## Results

Historical evidence from retired routes; their raw local outputs and implementations are archived. These numbers are not results of the active context-conditioned SL protocol. Consolidated table: [`docs/results/prior-internal-evidence.md`](docs/results/prior-internal-evidence.md).

### HCT116 Frozen-K562-Backbone Transport (one-shot audit, 2026-07-21)

On the 1,652-gene primary cohort, direct K562 GeneEffect transfer retained
Spearman **0.554**, while the frozen response head reached **-0.001** with a
collapsed prediction standard deviation of 0.059 versus 0.409 for HCT116
GeneEffect. A follow-up analysis controlling for K562 GeneEffect gave
partial Spearman about -0.005. The failed path is HCT116 observed response through the frozen K562
fitness head; this is not a pairwise SL or cross-cell-line SL result. Full
protocol, metrics, and interpretation: [`docs/results/exp05-hct116-frozen-backbone-transport.md`](docs/results/exp05-hct116-frozen-backbone-transport.md).

### Single-Cell Bag → Dependency (Track 2, Adamson K562 external transfer)

The best distribution/prototype regressor (K64-centered Ridge) reaches Adamson **Spearman ≈ 0.67**, **AUROC ≈ 0.91**, **AUPRC ≈ 0.80**, with held-out-gene Spearman ≈ 0.64 — clearing the original distribution-regression gate and beating the earlier scVI128 single-head gated-attention row. Full tables: [`docs/results/prior-internal-evidence.md`](docs/results/prior-internal-evidence.md).

### Dependency-Only SL Floor (official-metric CV, 2026-06-17)

Models: **A** = symmetric logistic regression (honest floor), **B** = XGBoost (nonlinear interactions), **C** = preferential-attachment degree probe (CV-gameability control).

| Split | Holdout | Model B AUPR | Model B AUROC |
| --- | --- | ---: | ---: |
| CV1 | pair-level (easiest) | 0.812 | 0.795 |
| CV2 | one gene unseen | 0.732 | 0.704 |
| CV3 | both genes unseen (hardest) | 0.609 | 0.596 |

The degree-probe control (Model C) scores highest on CV1 — a reminder that pair-level splits are gameable from graph degree alone, which is exactly why CV2/CV3 are the generalization surfaces. Combined summary: `results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_summary.csv`.

## Data Sources and Roles

| Source | Role | Notes |
| --- | --- | --- |
| Perturb-seq / CROP-seq / CRISPRi-seq | Mechanistic response input | Post-perturbation scRNA-seq, pseudobulk signatures, delta expression. |
| DepMap / Achilles / CCLE | Supervision and context | CRISPR gene-effect scores, dependency labels, omics, lineage, mutation context. |
| SL_benchmark 2024 (SynLethDB-derived) | General SL pair benchmark | Official 9,845-gene labels, CV1/CV2/CV3 splits, pair-ranking metrics, and SOTA zoo. Not cell-line-specific. |
| CancerSCEM / SCAR / CancerSEA | State annotation | Apoptosis, stress, cell-cycle, EMT, DNA-damage interpretation. |
| Cell-line-resolved combinatorial screens | Cross-context evaluation | Required for held-out-cell-line SL/GI evaluation; roles must be frozen before modeling. |
| TCGA / patient omics | Disease context | Future biomarker framing only; not evidence of cell-line or patient generalization under the current protocol. |
| LINCS L1000 / Tahoe-100M | Later extensions | Bulk or drug perturbation expansion once the gene-perturbation task is stable. |

> **Data rules**: Prioritize CRISPRi or knockout Perturb-seq for DepMap alignment. K562 is the current backbone, not the target scope. Cross-cell-line claims require context-specific pairwise SL/GI labels; single-gene GeneEffect transfer is insufficient. Norman CRISPRa is auxiliary, and DepMap labels are population-level fitness readouts, not single-cell death.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) — instructions for AI coding agents.
- [`docs/01-blueprint.md`](docs/01-blueprint.md) — the research contract; start here. [`docs/results/`](docs/results/) holds registered evidence.
- [`docs/data/`](docs/data/) — dataset cards for downloaded data.
- [`docs/discussion/`](docs/discussion/) — project discussion notes.

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
uv run python -m pytest

git commit -m "feat: description"   # Conventional Commits
git push -u origin feature/your-feature-name
```

Follow the **Plan → Confirm → Code** workflow for non-trivial research or implementation changes, use Conventional Commits (`feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `ci`), and respect the terminology guardrails below.

## Terminology Guardrails

- Say **dependency / GeneEffect prediction** for the single-gene supervised task; **SL candidate prioritization** for the pairwise score — never "validated SL target" (benchmark negatives are unconfirmed).
- Do not claim SL from single-gene essentiality; a declared null baseline is required.
- **Never write "interaction" about a model result.** The pair head takes the null as one input among many and predicts no joint outcome, so the model-minus-null gap is incremental label ranking. An interaction claim needs a joint or measured genetic-interaction quantity.
- The generalization axis is the **cell line**. Genes are not held out, so no unseen-gene claim is available; historical CV1/CV2/CV3 statements below describe the retired Feng2024 track only.
- Qualify every held-out-context result with foundation-model pretraining exposure: task-label holdout is not representation-pretraining holdout, and Tx1 saw Tahoe-100M.
- A pan-essentiality lift is not an SL result — the gene-mean block must be ablated, not assumed away.
- No significance claim across contexts: one split with few test contexts admits no valid family-wise inference over the baselines, arms, and strata.
- Do not call DepMap GeneEffect a single-cell death label; it is a single-gene relative growth-rate effect.
- Norman CRISPRa is auxiliary only, never aligned to knockout labels without the modality caveat.
- Feng2024 is retired as a target; its numbers in `docs/02-literature-review.md` are field background, not a bar this program reproduces.

---

<div align="center">
  <p>
    <strong>Active direction: generalizable synthetic-lethality discovery by virtual-cell composition, evaluated separately for unseen genes and unseen cell lines.</strong><br>
    <sub>See <a href="docs/01-blueprint.md">docs/01-blueprint.md</a> for the live research contract.</sub>
  </p>
</div>
