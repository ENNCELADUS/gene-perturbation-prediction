<div align="center">

  <h1 style="margin-top: 10px;">Generalizable Synthetic-Lethality Discovery by Virtual-Cell Composition</h1>

  <h2>Discover synthetic-lethal gene pairs that generalize to genes withheld from SL-pair training and to held-out cancer cell-line contexts.</h2>

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

> **Status (2026-08-15):** Active direction — **context-conditioned synthetic-lethality ranking**. The research contract has been reformulated as a formal ML task: basal single cells plus a perturbation gene produce a predicted response, a gene-mean / context-residual GeneEffect decomposition, and a trained pair head scored on cell lines held out of every fitting and selection step. The Feng2024 benchmark and the train-free SLIdR stage are **dropped**; the benchmark is now the context-conditioned screen table, rebuilt as `context_screen_v2` on 2026-08-15 with row provenance and a filter audit. Its published split is the remaining prerequisite. T1 (Bridge-A vs Horlbeck), T2 (few-shot cross-cell-line GeneEffect, `Delta rho = -0.0048`, 95% CI `[-0.0941, 0.0769]`), and the HCT116 transport audit are all closed negative and remain binding single-gene evidence, **not** SL results. No model has run under the new contract. Live contract: [`docs/01-blueprint.md`](docs/01-blueprint.md).

The central question of the active direction:

> Can a dependency profile predicted by a perturbation-response-trained virtual cell rank synthetic-lethal pairs in a cancer cell line that was excluded from every fitting and selection step, beyond what a declared null and a context-ablated model already achieve?

The intuition is compositional: **a cell line's dependency profile is what makes a pair lethal there.** If a virtual cell can predict how a gene's fitness cost shifts with cellular context, then the shape of that shift across lines should carry pair-specific signal that a curated SL graph can only memorize. The bar is deliberately internal — the gene-mean block, the null baseline, and a context-ablated head must all be beaten before any context claim is licensed. Nothing here estimates a genetic interaction; see [`docs/01-blueprint.md`](docs/01-blueprint.md) §8 for what that forbids.

## *Latest News* 🔥

- **[2026/08]** **Contract reformulated as a formal ML task.** The program drops Feng2024 and the train-free SLIdR stage, and adopts a single generalization axis — the cell line. The backbone now forms an explicit perturbation delta, conditions on ESM-2 gene identity, and splits GeneEffect into a context-blind gene mean and a context residual trained on the across-context axis; the response module is fine-tuned jointly rather than frozen. A trained pair head scores unordered pairs from the predicted residual profile against a declared null baseline. Benchmark rebuild and published split are prerequisites. [`Contract`](docs/01-blueprint.md) · [`Protocol`](docs/03-experiment-protocol.md).
- **[2026/07]** **T2 registered primary gate completed — negative.** On the frozen 28 train / 5 validation / 9 test GeneEffect split and 587-gene slice, Tx1-3B-ST failed to beat copy-K562 + 10 labels (`Delta rho = -0.0048`, 95% CI `[-0.0941, 0.0769]`, registered `rho_min = 0.05`). HVG-ST was also negative as a diagnostic (`Delta rho = 0.0326`, 95% CI `[-0.0602, 0.1181]`). Both few-shot curves deteriorated with larger k. This is binding single-gene backbone evidence, not cross-cell-line SL; T2 is paused for redesign and the remaining baseline ladder is closeout work. [`Result`](docs/results/tx1-hvg-geneeffect-phase-f.md).
- **[2026/07]** **K562 Bridge-A-vs-Horlbeck mechanism kill-test completed — negative.** The frozen exp05 backbone composed into a symmetrized counterfactual co-dependency score does not recover measured Horlbeck K562 genetic interactions over the 83,028 exp05-covered pairs (|Spearman| < 0.01; AUROC(s_A → strong-SL) ≈ 0.52, below the single-gene floor; no dose-response), across both pooler reference conventions. Per the kill-test rule the composition mechanism is **paused for redesign and not extended across cell lines**. Development diagnostic, not a formal MECHANISTIC verdict. [`Result`](docs/results/exp05-bridge-a-horlbeck-kill-test.md).
- **[2026/07]** Horlbeck 2018 K562 fitness-GI map acquired and coverage-audited (448 genes, 100,128 pairs; 83,028 exp05-covered), and an execution plan set: two parallel development tracks — a **K562 Bridge-A-vs-Horlbeck mechanism kill-test** on the frozen exp05 backbone, and **DepMap cross-cell-line GeneEffect** transfer (single-gene backbone, not cross-cell-line SL). Neither opens held-out-cell-line SL labels.
- **[2026/07]** Formal HCT116 frozen-backbone audit closed negative: direct K562 GeneEffect transfer remained strong (Spearman 0.554), but the response head collapsed and added no independent HCT116 signal. This is single-gene backbone evidence, not cross-cell-line SL. [`Closeout`](docs/results/exp05-hct116-frozen-backbone-transport.md).
- **[2026/07]** Research contract expanded from a K562-only formulation to a **general SL discovery model**. Feng2024 CV2/CV3 test genes withheld from SL-pair/graph training; held-out-cell-line splits separately test unseen contexts. K562 remains the current backbone and one mechanistic anchor.
- **[2026/07]** Two composition bridges — counterfactual co-dependency and virtual double knockout — were specified against SLMGAE/KR4SL and measured GI. Both were retired by the 2026/08 reformulation; neither is an active target.
- **[2026/06]** Stage 3 SL pair benchmark adapter and dependency-only baseline shipped (`src/sl_benchmark_baseline/`); official-metric CV1/CV2/CV3 rerun completed.
- **[2026/06]** Experiment 05 AIVC STATE A→B→C forward-model pipeline reviewed, including a frozen-STATE feature ablation track.
- **[2026/05]** Single-cell Deep Sets, attention-MIL, and distribution/prototype regressors landed with Adamson K562 external transfer.
- **[2026/05]** Pseudobulk delta baseline ladder established as the Stage 1 dependency-prediction floor.

## Why This Project?

Most SL predictors do link prediction over a curated SL graph — powerful on seen genes, weak on unseen ones, and usually not conditioned on an explicit cellular context. This project brings signal from *outside* the graph: a virtual cell that predicts cancer-cell fitness from perturbation response, composed into a pairwise interaction for both general pair ranking and held-out-cell-line transfer.

- **🔗 Composes, not memorizes** — Turns a single-gene fitness model into a pairwise score against a declared null baseline, instead of reading topology off an SL graph.
- **🧊 Inductive by construction** — The score reads from gene features (ESM2 identity + predicted perturbation biology), so it is defined for genes no screen has touched, where transductive SOTA has no node at all.
- **🌐 Context-resolved evaluation** — The generalization axis is the cell line, on a published held-out split. Pan-essentiality is a controlled variable, not an assumption, via the gene-mean / context-residual split.
- **🧪 Observed-first methodology** — Validates that *observed* response carries dependency signal before trusting any *predicted* transcriptome, so forward-model error never silently inflates results.
- **🪜 Honest baseline ladder** — Dummy → ridge → PCA → tabular nonlinear → MIL/foundation models, so every gain is measured against a simpler control.
- **🚪 Fold-local, no-leakage CV** — A→B models, featurizers, GMM prototypes, and C-heads are all fit on train genes only, inside each fold.
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
> **Need more options?** See [Installation](#installation) below for detailed setup, optional dependency groups, and the AIVC STATE exception.

## Research Framing

> **Status:** contract, literature boundary, and executable protocol established under the reformulated task. The benchmark rebuild and its published split are prerequisite work; basal single-cell input is still missing for most eligible contexts. T1, T2, and the HCT116 transport audit are closed negative. No model has run under this contract. Live contract: [`docs/01-blueprint.md`](docs/01-blueprint.md).

```text
Given a cancer cell line described only by its basal single-cell transcriptome —
no CRISPR screen, no SL screen — rank unordered gene pairs by the probability
that the pair is an experimental synthetic-lethal hit in that line.
```

The generalization axis is the **cell line**. Graph and knowledge-graph SL predictors need the query gene to already be a node, so they cannot score an unscreened gene at all and cannot condition on a cellular context; this program reaches both by reading from gene identity and predicted perturbation biology instead of graph topology. Genes are *not* held out here, so no unseen-gene claim is available from this benchmark.

- **Stage 1 — response.** Basal cells plus a perturbation gene produce predicted post-perturbation cells, supervised on four Perturb-seq lines.
- **Stage 2 — dependency.** The perturbation delta, gene identity, and context vector produce GeneEffect as a context-blind gene mean plus a context residual, supervised across many lines.
- **Stage 3 — pairs.** A lightweight head scores unordered pairs from the predicted residual profile against a declared non-interaction null baseline. The gap is incremental label ranking, **not** an interaction estimate.

The full contract — task definition, objective, split, controls, and claim boundaries — lives in the research vault, not here:

- [`docs/01-blueprint.md`](docs/01-blueprint.md) — the research contract: formal task, objective, evaluation, and claim boundaries.
- [`docs/02-literature-review.md`](docs/02-literature-review.md) — related work and the novelty boundary.
- [`docs/03-experiment-protocol.md`](docs/03-experiment-protocol.md) — the executable protocol and its prerequisites.
- [`docs/data/`](docs/data/) — one card per dataset. Read the card before using the file.

### Prior Program (Retired as Roadmap, Kept as Evidence)

Through mid-2026 the project's roadmap was a **staged, context-ranking** dependency → synthetic-lethality program (distinct from the active *composition* direction above):

```text
cell line + perturbation gene
    → observed or predicted post-perturbation transcriptome
    → dependency / essentiality score
    → context-specific target ranking
```

That staged program is **retired as the roadmap**. Its exp08/exp08b implementation has been removed; the remaining dependency, exp05 forward-model, SL-benchmark, and DDGCN code supports active work and retained baselines. Historical results remain **prior evidence** for the new direction — see [Results](#results) and [`docs/results/prior-internal-evidence.md`](docs/results/prior-internal-evidence.md).

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

### Running the Pipeline

The normal entrypoint is the `vcc-dep-baseline` CLI:

```bash
uv run vcc-dep-baseline --help
```

Subcommands: `build-features`, `build-cell-bags`, `build-external-cell-bags`, `build-external-features`, `run-cv`, `run-single-cell-cv`, `evaluate-single-cell-external`, `run-distribution-cv`, `evaluate-distribution-external`, `run-predicted-b-cv`, `fit-final`, `summarize`, `organize-artifacts`, `viability-axis-report`. Most runner subcommands accept `--resume` and repeatable selection flags (`--scope`, `--feature-set`, `--model`, `--fold`, `--weighting`).

### The AIVC STATE Exception

Experiment 05 (AIVC STATE forward model) is **not** part of the CLI. Run it as a direct module:

```bash
# Direct training
uv run python src/aivc_model/train.py \
  --config configs/experiments/05_aivc_a_to_b_to_c/state_hf_hvg_replogle_k562_ranknet_freeze_state.yaml

# Slurm wrapper (accelerate launch, 4 GPUs)
bash scripts/state.sh

# Frozen-STATE feature ablation
uv run python src/aivc_model/state_feature_ablation.py \
  --config configs/experiments/05_aivc_a_to_b_to_c/state_frozen_feature_ablation.yaml
```

> Raw `*.h5ad`, `*.csv`, checkpoints, and large artifacts are gitignored. The pipeline requires Perturb-seq and DepMap data you supply locally.

## Architecture

### Conceptual Framing

The prior dependency pipeline closes a triangle: two edges (data → response,
response → dependency) are provided by existing data, and the model focuses on
the **transcriptomic response → single-gene GeneEffect** edge.

### Pipeline Tracks

```
                       h5ad + DepMap labels
                                │
         ┌──────────────────────┼──────────────────────────┐
         ▼                      ▼                            ▼
┌──────────────────┐  ┌──────────────────────┐  ┌────────────────────────┐
│ TRACK 1          │  │ TRACK 2              │  │ TRACK 3                │
│ Pseudobulk Delta │  │ Single-Cell Deep Sets│  │ Distribution / Proto   │
│                  │  │                      │  │                        │
│ build-features   │  │ build-cell-bags      │  │ run-distribution-cv    │
│   → features.npz │  │   → bags.npz (PCA)   │  │   (FrozenGMM /         │
│ run-cv           │  │ run-single-cell-cv   │  │    CloudPred-style)    │
│   (Repeated      │  │   (DeepSetsRegressor)│  │ GMM occupancy features │
│    Stratified    │  │ evaluate-single-cell │  │   → Ridge / forest head│
│    KFold)        │  │   -external (Adamson)│  │ evaluate-distribution  │
│ fit-final        │  │                      │  │   -external (Adamson)  │
└────────┬─────────┘  └──────────┬───────────┘  └───────────┬────────────┘
         │                       │                          │
         └───────────────────────┴──────────────────────────┘
                                 │
                                 ▼
                  ArtifactStore (fold metrics, predictions,
                   model manifests, top-k candidates, resume state)

   ┌─────────────────────────────────────────────────────────────────┐
   │ FORWARD MODEL (Exp 05, src/aivc_model/ — direct module, not CLI) │
   │  basal state + perturbation → STATE A→B→C → B_hat → C predictor  │
   └─────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │ STAGE 3 SL ADAPTER (src/sl_benchmark_baseline/)                  │
   │  (gene_a, gene_b) → swap-invariant GeneEffect features → P(SL)   │
   └─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Module | Role |
| --- | --- |
| `src/dependency_baseline/` | Multi-track CV pipeline: `features.py`, `datasets.py`, `models.py`, `cell_bags.py`, `single_cell.py`, `distribution.py`, `predicted_b.py`, `evaluation.py`. |
| `src/aivc_model/` | AIVC STATE A→B→C forward model (`model.py`, `prepare.py`, `train.py`) plus frozen-STATE feature ablation. |
| `src/sl_benchmark_baseline/` | Dependency-only SL pair baseline with official-metric evaluator. |
| `config.py` | Frozen dataclasses loaded from YAML; `SelectionConfig` narrows scopes/features/models/folds at runtime. |
| `artifacts.py` | `ArtifactStore` — incremental parquet writes, checkpoints, run manifests, resume state. |
| `metrics.py` | Spearman, Pearson, RMSE, MAE, R² for regression; AUROC, AUPRC, top-k enrichment for ranking. |

### Design Decisions

- **Observed-before-predicted** — Stage 1/2 validate observed response signal before any forward-model dependence, so predicted-transcriptome error is isolated.
- **Fold-local everything** — In predicted-B and distribution tracks, A→B fitting, featurization, GMM prototypes, and the C head all train on train genes only.
- **Config-driven runs** — Model ladders, selection filters, predicted-B settings, and viability-axis residualization live in YAML under `configs/experiments/`, grouped by experiment number. The numbered write-ups that used to accompany each experiment are archived locally (gitignored, not tracked in git); see [Documentation](#documentation) for what's actually tracked.

## Results

Headline numbers from the implemented baselines. These are the **floor and baselines** for the active composition direction (see [Research Framing](#research-framing)) — the dependency-only SL floor and the observed-transcriptome result the composition must beat, plus the single-gene forward-model signal it builds on. Consolidated table: [`docs/results/prior-internal-evidence.md`](docs/results/prior-internal-evidence.md).

### HCT116 Frozen-K562-Backbone Transport (formal one-shot audit, 2026-07-21)

On the 1,652-gene primary cohort, direct K562 GeneEffect transfer retained
Spearman **0.554**, while the frozen response head reached **-0.001** with a
collapsed prediction standard deviation of 0.059 versus 0.409 for HCT116
GeneEffect. A post-unseal diagnostic controlling for K562 GeneEffect gave
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

> **Data rules**: Prioritize CRISPRi or knockout Perturb-seq for DepMap alignment. K562 is the current backbone, not the target scope. Cross-cell-line claims require untouched cell lines with pairwise SL/GI labels; single-gene GeneEffect transfer is insufficient. Norman CRISPRa is auxiliary, and DepMap labels are population-level fitness readouts, not single-cell death.

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
