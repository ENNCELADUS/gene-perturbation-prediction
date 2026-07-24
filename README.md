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

> **Status (2026-07-23):** Active direction — **generalizable synthetic-lethality discovery by virtual-cell composition**. The first HCT116 frozen-K562-backbone single-gene transport audit is closed negative; it does not test pairwise or cross-cell-line SL. The Horlbeck 2018 K562 fitness-GI map is acquired and coverage-audited, and the **K562 Bridge-A-vs-Horlbeck mechanism kill-test completed negative** — the composed frozen backbone does not recover measured K562 epistasis (|Spearman| < 0.01; AUROC ≈ 0.52), so the composition mechanism is paused for redesign and not extended across contexts. With composition paused, the near-term focus is a **few-shot cross-cell-line GeneEffect backbone** (T2) — an accurate, context-conditioned single-gene predictor that both bridges depend on, pursued as the Phase 3 backbone-transfer exit and reported as backbone transfer, **not** an SL claim. The official Feng2024 SOTA comparison and the context-conditioned SL track remain pending. Live contract, acceptance criteria, roadmap, and results: [`docs/README.md`](docs/README.md).

The central question of the active direction:

> Can a perturbation-response-trained virtual cell be composed into an explicit pairwise interaction that beats strong SL-prediction SOTA for unseen genes and adds context-specific signal for cancer cell lines excluded from training?

The intuition is mechanistic: **synthetic lethality is a combination fitness outcome.** If a virtual cell predicts single-gene fitness from the perturbation "shockwave," composing it — simulating the loss of one gene and re-reading the other's dependency (Bridge A), or forwarding the joint double-knockout (Bridge B) — should expose the pair-specific interaction that curated SL graphs only memorize. Single-gene GeneEffect alone is a prior K562-filtered floor (CV2 AUROC 0.704; CV3 0.596), not the method or the formal general-benchmark bar.

## *Latest News* 🔥

- **[2026/07]** **Foundation-first refocus after the Bridge-A negative.** With the composition mechanism paused, the near-term work is a **few-shot cross-cell-line GeneEffect backbone**: an accurate, context-conditioned single-gene predictor `F(X_c, c, g)` that both bridges depend on, evaluated on a **fixed few held-out DepMap cancer lines seen only through basal single-cell state** (four CRISPRi Perturb-seq lines train; a lineage-stratified few of the Tahoe DMSO-basal∩DepMap lines are the unseen test) on the **differentially-essential slice** against copy-K562 / mean / nearest-line / lineage / CCLE-bulk / pseudobulk-basal baselines, with a **few-shot curve** (accuracy vs. k held-out-line labels). This is the roadmap Phase 3 backbone-transfer exit — reported as task-data-held-out backbone transfer, **not** an SL result. Task established; not yet run. See [`docs/04-roadmap.md`](docs/04-roadmap.md) §1.1.
- **[2026/07]** **K562 Bridge-A-vs-Horlbeck mechanism kill-test completed — negative.** The frozen exp05 backbone composed into a symmetrized counterfactual co-dependency score does not recover measured Horlbeck K562 genetic interactions over the 83,028 exp05-covered pairs (|Spearman| < 0.01; AUROC(s_A → strong-SL) ≈ 0.52, below the single-gene floor; no dose-response), across both pooler reference conventions. Per the kill-test rule the composition mechanism is **paused for redesign and not extended across cell lines**. Development diagnostic, not a formal MECHANISTIC verdict. [`Result`](docs/results/exp05-bridge-a-horlbeck-kill-test.md).
- **[2026/07]** Horlbeck 2018 K562 fitness-GI map acquired and coverage-audited (448 genes, 100,128 pairs; 83,028 exp05-covered), and an execution plan set: two parallel development tracks — a **K562 Bridge-A-vs-Horlbeck mechanism kill-test** on the frozen exp05 backbone, and **DepMap cross-cell-line GeneEffect** transfer (single-gene backbone, not cross-cell-line SL). Neither opens held-out-cell-line SL labels. See [`docs/04-roadmap.md`](docs/04-roadmap.md) §1.1.
- **[2026/07]** Formal HCT116 frozen-backbone audit closed negative: direct K562 GeneEffect transfer remained strong (Spearman 0.554), but the response head collapsed and added no independent HCT116 signal. This is single-gene backbone evidence, not cross-cell-line SL. [`Closeout`](docs/results/exp05-hct116-frozen-backbone-transport.md).
- **[2026/07]** Research contract expanded from a K562-only formulation to a **general SL discovery model**. Feng2024 CV2/CV3 test genes withheld from SL-pair/graph training; held-out-cell-line splits separately test unseen contexts. K562 remains the current backbone and one mechanistic anchor.
- **[2026/07]** Two composition bridges — counterfactual co-dependency and virtual double knockout — are specified to beat strong SOTA (SLMGAE, KR4SL), survive context/pan-essentiality controls, and match measured GI.
- **[2026/06]** Stage 3 SL pair benchmark adapter and dependency-only baseline shipped (`src/sl_benchmark_baseline/`); official-metric CV1/CV2/CV3 rerun completed.
- **[2026/06]** Experiment 05 AIVC STATE A→B→C forward-model pipeline reviewed, including a frozen-STATE feature ablation track.
- **[2026/05]** Single-cell Deep Sets, attention-MIL, and distribution/prototype regressors landed with Adamson K562 external transfer.
- **[2026/05]** Pseudobulk delta baseline ladder established as the Stage 1 dependency-prediction floor.

## Why This Project?

Most SL predictors do link prediction over a curated SL graph — powerful on seen genes, weak on unseen ones, and usually not conditioned on an explicit cellular context. This project brings signal from *outside* the graph: a virtual cell that predicts cancer-cell fitness from perturbation response, composed into a pairwise interaction for both general pair ranking and held-out-cell-line transfer.

- **🔗 Composes, not memorizes** — Turns a single-gene fitness model into a pairwise SL score via an explicit interaction null, instead of reading topology off an SL graph.
- **🧊 Inductive by construction** — The score reads from gene features (ESM2 identity + predicted perturbation biology), so it works on cold-start (CV2/CV3) genes where transductive SOTA breaks.
- **🌐 Context-resolved evaluation** — Unseen-gene performance on Feng2024 and unseen-cell-line performance are separate binding claims; neither is used as a proxy for the other.
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

> **Status:** general-model contract, claim-level acceptance criteria, literature review, and experiment roadmap established; the first HCT116 frozen-backbone audit closed negative, and the Horlbeck K562 GI map is acquired and coverage-audited. Phase 0 effect-size/eligibility registration, official SOTA reproduction, measured-GI model evaluation, multi-cell-line data audit, and contextual SL model remain pending. Live status board: [`docs/README.md`](docs/README.md).

```text
Discover synthetic-lethal pairs for genes withheld from the SL graph and for
cancer cell lines withheld from model development. The first axis is evaluated on
Feng2024 CV2/CV3; the second requires explicit held-out-cell-line data.
```

Graph/knowledge-graph SL predictors (SLMGAE, KR4SL, KG4SL, and the wider Feng2024 zoo) rely heavily on graph relations and degrade as genes are withheld (CV1 → CV2 → CV3). This program compares against them on the official 9,845-gene benchmark, then separately evaluates context-conditioned transfer to unseen cell lines. The main Feng2024 benchmark is not a K562 assay and does not test cell-line generalization.

- **Bridge A — counterfactual co-dependency.** Simulate loss of `a`, then predict `b`'s GeneEffect in the `a`-lost state; SL = the dependency spike. Uses only single-gene labels.
- **Bridge B — virtual double-knockout.** Forward the joint `a+b` perturbation → joint fitness; SL = the interaction residual vs. an explicit additive/min null. Validated against measured epistasis.

The full contract — estimands, acceptance criteria, gate verdicts, and the decision record — lives in the research vault, not here:

- [`docs/README.md`](docs/README.md) — vault index and status board.
- [`docs/01-blueprint.md`](docs/01-blueprint.md) — the frozen research contract.
- [`docs/02-acceptance-criteria.md`](docs/02-acceptance-criteria.md) — the bar a result must clear to count as an answer. Frozen before evidence.
- [`docs/03-literature-review.md`](docs/03-literature-review.md) — related work and novelty boundaries.
- [`docs/04-roadmap.md`](docs/04-roadmap.md) — the active experiment roadmap.

### Prior Program (Retired as Roadmap, Kept as Evidence)

Through mid-2026 the project's roadmap was a **staged, context-ranking** dependency → synthetic-lethality program (distinct from the active *composition* direction above):

```text
cell line + perturbation gene
    → observed or predicted post-perturbation transcriptome
    → dependency / essentiality score
    → context-specific target ranking
```

That staged program is **retired as the roadmap**. It is not retired as code: `src/dependency_baseline/`, `src/aivc_model/`, `src/sl_benchmark_baseline/`, `src/sl_dl_model/`, and `src/ddgcn/` all still run (see [Architecture](#architecture)), and the active composition direction reuses the exp05 forward model and the SL benchmark harness directly. Its results are **prior evidence and baselines** for the new direction — see [Results](#results) for the numbers and [`docs/results/prior-internal-evidence.md`](docs/results/prior-internal-evidence.md) for the consolidated table.

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

- [`CONTEXT.md`](CONTEXT.md) — glossary of A / B / B_hat / C / D evaluation semantics used by the SL benchmark and forward model, plus a clearly marked retired glossary for prior artifacts.
- [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) — instructions for AI coding agents.
- [`docs/README.md`](docs/README.md) — research vault index: contract, acceptance criteria, gate verdicts, decision and roadmap, results.
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
- Do not claim SL from single-gene essentiality; an explicit interaction null is required.
- CV2/CV3 support SL-pair/graph-gene-cold claims only; CV1 is the degree-gameable diagnostic. Cross-cell-line claims require separate untouched cell-line splits.
- Qualify “unseen gene”: Feng2024 establishes absence from SL-pair/graph training,
  not necessarily from auxiliary response, GeneEffect, or foundation-model
  pretraining data.
- A pan-essentiality lift is not an SL result — report the non-pan-essential slice.
- The virtual double-knockout is an extrapolation of a single-perturbation backbone; its benchmark rank is not mechanistic proof until it matches measured epistasis.
- Do not call DepMap GeneEffect a single-cell death label; it is a single-gene relative growth-rate effect.
- Norman CRISPRa is auxiliary only, never aligned to knockout labels without the modality caveat.
- A K562-mappable Feng2024 subset is not a K562-specific SL assay, and the main Feng2024 benchmark cannot establish cell-line generalization.

---

<div align="center">
  <p>
    <strong>Active direction: generalizable synthetic-lethality discovery by virtual-cell composition, evaluated separately for unseen genes and unseen cell lines.</strong><br>
    <sub>See <a href="docs/README.md">docs/README.md</a> for the live contract, acceptance criteria, and roadmap.</sub>
  </p>
</div>
