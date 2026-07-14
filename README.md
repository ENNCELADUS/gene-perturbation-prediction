<div align="center">

  <h1 style="margin-top: 10px;">Cancer Dependency Prediction from Perturbation-Induced Transcriptomes</h1>

  <h2>Predict DepMap-style CRISPR gene-effect scores from observed or predicted post-perturbation transcriptomes — and prioritize synthetic-lethality candidates with context evidence.</h2>

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

> **Status (2026-07-14):** The dependency → synthetic-lethality program below (central question, Research Framing) shipped the pipeline and the numbers in [Results](#results); it is now **retired as the roadmap** and kept as prior evidence. The active research direction is **cell-fate outcome dynamics** — see [Research Framing](#research-framing) below and the vault at [`docs/README.md`](docs/README.md) for the live contract, gate verdicts, and next steps.

The central question that shaped the codebase below:

> Given a cancer cell line and a gene perturbation, can the post-perturbation transcriptomic response predict whether that perturbation creates a meaningful fitness, vulnerability, or dependency phenotype?

This framing is deliberately narrower than "predict synthetic lethality end-to-end." DepMap/Achilles CRISPR gene-effect scores are population-level dependency labels — not single-cell death labels and not strict SL labels. The project first learns the link from perturbation-induced transcriptomic response to cancer dependency, then layers context specificity on top to prioritize SL-like candidates.

## *Latest News* 🔥

- **[2026/06]** Stage 3 SL pair benchmark adapter and dependency-only baseline shipped (`src/sl_benchmark_baseline/`); official-metric CV1/CV2/CV3 rerun completed.
- **[2026/06]** Experiment 05 AIVC STATE A→B→C forward-model pipeline reviewed, including a frozen-STATE feature ablation track.
- **[2026/05]** Single-cell Deep Sets, attention-MIL, and distribution/prototype regressors landed with Adamson K562 external transfer.
- **[2026/05]** Pseudobulk delta baseline ladder established as the Stage 1 dependency-prediction floor.

## Why This Project?

Most virtual-cell work stops at predicting the transcriptome. This project asks the next question — does that response actually predict whether a gene is a dependency — and builds an honest, leakage-controlled evaluation chain to answer it.

- **🔗 Closes the loop** — Connects perturbation → transcriptomic response → DepMap dependency, instead of treating forward prediction as the end goal.
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

> **Status:** literature funnel complete; decision **`narrow-or-pivot`** for both candidates below. Next step: three public-data reanalyses, then a bounded feasibility study. Live roadmap and status board: [`docs/README.md`](docs/README.md).

```text
The same net fitness loss can arise from completely different cellular dynamics —
division suppression, cell loss, or loss followed by survivor regrowth — and it is
unknown whether early molecular state prospectively distinguishes those trajectories
in genetic loss-of-function.
```

A single endpoint net-fitness readout — DepMap Chronos GeneEffect is one instance — does not uniquely determine the underlying division / death / recovery dynamics: Chronos is structurally incapable of separating them, because that decomposition lies outside what it estimates. Two candidate research questions are carried in parallel, each with its own estimand, evidence hierarchy, and falsifier; no unit has been selected yet:

- **Candidate A — lineage/clone level.** Within a fixed context and time horizon, does an early post-perturbation molecular state predict the subsequent division, persistence/recovery, and extinction trajectory of its linked lineage, beyond an independently measured net fitness?
- **Candidate B — population level.** Under comparable, independently measured net fitness, does the early single-cell state distribution carry incremental information about independently measured future population dynamics?

The full contract — estimands, frozen significance thresholds, gate verdicts, and the decision record — lives in the research vault, not here:

- [`docs/README.md`](docs/README.md) — vault index and status board.
- [`docs/01-research-direction.md`](docs/01-research-direction.md) — the frozen research contract.
- [`docs/02-significance-criteria.md`](docs/02-significance-criteria.md) — frozen significance thresholds.
- [`docs/03-review-findings.md`](docs/03-review-findings.md) — gate-by-gate verdicts.
- [`docs/04-decision-and-roadmap.md`](docs/04-decision-and-roadmap.md) — the decision and what runs next.

### Prior Program (Retired as Roadmap, Kept as Evidence)

Through mid-2026 the project's roadmap was a staged dependency → synthetic-lethality program:

```text
cell line + perturbation gene
    → observed or predicted post-perturbation transcriptome
    → dependency / essentiality score
    → context-specific target ranking
```

That staged program is **retired as the roadmap**. It is not retired as code: `src/dependency_baseline/`, `src/aivc_model/`, and `src/sl_benchmark_baseline/` all still run (see [Architecture](#architecture)), and the three planned public-data reanalyses reuse this pipeline. Its results are **prior evidence** for the new direction, not a roadmap to keep executing — see [Results](#results) for the numbers and [`docs/results/prior-internal-evidence.md`](docs/results/prior-internal-evidence.md) for the consolidated table.

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

The project closes a triangle: two edges (data → response, response → dependency) are provided by existing data, and the model focuses on the **transcriptome → death / essentiality** edge.

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

Headline numbers from the retired dependency → SL program's implemented baselines. These are now **prior evidence** feeding the cell-fate research direction (see [Research Framing](#research-framing)), not production claims and not claims about cell-fate dynamics. Consolidated table: [`docs/results/prior-internal-evidence.md`](docs/results/prior-internal-evidence.md).

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
| SL_benchmark 2024 (SynLethDB-derived) | External SL pair benchmark | Gene-pair SL labels, CV1/CV2/CV3 splits, pair-ranking metrics. Used via a pair-level adapter; not a K562 GeneEffect label source. |
| CancerSCEM / SCAR / CancerSEA | State annotation | Apoptosis, stress, cell-cycle, EMT, DNA-damage interpretation. |
| TCGA / patient omics | Disease context | Biomarker framing and translational interpretation after cell-line proof-of-concept. |
| LINCS L1000 / Tahoe-100M | Later extensions | Bulk or drug perturbation expansion once the gene-perturbation task is stable. |

> **Data rules**: Prioritize CRISPRi or knockout Perturb-seq for DepMap alignment. K562 is the proof-of-concept line. Norman CRISPRa is auxiliary, not primary label alignment. DepMap labels are population-level fitness readouts, not single-cell death.

## Documentation

- [`CONTEXT.md`](CONTEXT.md) — glossary of A / B / B_hat / C / D evaluation semantics, plus the cell-fate direction's load-bearing terms (`F_net`, T1/T2/T3, Candidate A/B, evidence tiers).
- [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) — instructions for AI coding agents.
- [`docs/README.md`](docs/README.md) — research vault index: contract, significance criteria, gate verdicts, decision and roadmap, results.
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

- Say **dependency prediction** or **essentiality ranking** for the supervised DepMap task.
- Say **SL candidate prioritization** only after adding context-specificity evidence.
- Do not call DepMap gene-effect scores single-cell death labels.
- Do not equate gene essentiality with synthetic lethality.
- Do not align CRISPRa activation perturbations with CRISPR knockout dependency labels without an explicit modality caveat.

---

<div align="center">
  <p>
    <strong>A retained dependency-prediction pipeline; the active research direction is cell-fate outcome dynamics.</strong><br>
    <sub>See <a href="docs/README.md">docs/README.md</a> for the live contract, gate verdicts, and roadmap.</sub>
  </p>
</div>
