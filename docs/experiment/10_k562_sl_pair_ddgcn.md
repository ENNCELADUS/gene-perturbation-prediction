# exp10: DDGCN reproduction on the K562 SL-pair benchmark

DDGCN (Dual-Dropout GCN, Cai et al. 2020, *Bioinformatics*,
doi 10.1093/bioinformatics/btaa211; repo https://github.com/CXX1113/Dual-DropoutGCN)
reproduced under the exp06/07/08/09 CV1/CV2/CV3 protocol and official per-anchor
metrics. Featureless, transductive graph auto-encoder; official defaults
dropout=0.5, lr=0.01.

## Status

- Code: complete and tested (`src/ddgcn/`, 21 unit tests green).
- Run: executed on a CUDA box via
  `uv run python -m ddgcn run-cv --config configs/experiments/10_k562_sl_pair_ddgcn/ddgcn_cv.yaml`.
- Artifacts: `results/experiments/10_k562_sl_pair_ddgcn/run/` (gitignored).

## Results vs the exp06 floor

Fill the DDGCN rows from `results/experiments/10_k562_sl_pair_ddgcn/run/summary.csv`
(mean over 5 folds, `slice = full_universe`).

| model | split | AUROC | AUPR | NDCG@10 | MAP@10 |
|---|---|---|---|---|---|
| exp06 XGB (B) | CV2 | 0.704 | 0.732 | 0.042 | 0.034 |
| exp06 XGB (B) | CV3 | 0.596 | — | 0.002 | — |
| exp06 degree probe (C) | CV1 | — | — | 0.197 | — |
| ddgcn | CV1 | [fill] | [fill] | [fill] | [fill] |
| ddgcn | CV2 | [fill] | [fill] | [fill] | [fill] |
| ddgcn | CV3 | [fill] | [fill] | [fill] | [fill] |

Smoke reference (CV1 fold 0, 60 epochs, capped): AUROC 0.809, AUPR 0.844,
NDCG@10 0.083, MAP@10 0.055 — already in the degree-gameable CV1 range. Full-run
numbers will differ (5 folds, up to 2000 epochs).

## Interpretation

DDGCN is transductive and featureless — it learns from the SL adjacency itself,
so it behaves like a learned nonlinear degree probe. Read CV2/CV3 as the
generalization surfaces; a strong CV1 result mirrors the exp06 degree probe
(NDCG@10 0.197) and is not evidence of learned biology. CV3 cold-start has no
training edges touching test genes and is expected near-floor. A CV1-only win is
a null finding.

## Reproduction notes

- Model ported from `data/SL_benchmark/src/models/ddgcn.py` (matches the official
  PyTorch repo line-for-line); torch 2.12 sparse API (`torch.sparse_coo_tensor`).
- 9,471-gene K562 universe (NOT DDGCN's native 9,845); official metrics reused
  verbatim from `sl_benchmark_baseline.metrics`.
- Loss-plateau stopping, no validation split (zero test leakage). Final-epoch
  fused score matrix (geometric mean of the two reconstructions, rho=1.0).
- Device-aware: runs on CUDA when available, else CPU (BCE computed on CPU per
  the original).
