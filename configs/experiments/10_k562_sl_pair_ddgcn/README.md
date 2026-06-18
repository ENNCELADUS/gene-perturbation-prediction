# exp10: DDGCN reproduction on the K562 SL-pair benchmark

Reproduces DDGCN (Dual-Dropout GCN, Cai et al. 2020,
https://github.com/CXX1113/Dual-DropoutGCN) under the exp06/07/08/09 CV1/CV2/CV3
protocol and official per-anchor metrics.

Run all splits:

    uv run python -m ddgcn run-cv --config configs/experiments/10_k562_sl_pair_ddgcn/ddgcn_cv.yaml

Single split (partial rerun):

    uv run python -m ddgcn run-cv --config configs/experiments/10_k562_sl_pair_ddgcn/ddgcn_cv.yaml --split-type CV2

Hyperparameters are the official DDGCN defaults (dropout=0.5, lr=0.01, hidden
512->256, Kaiming, no bias, rho=1.0, Adam amsgrad, max 2000 epochs with
loss-plateau early stop). The model is transductive and featureless; CV1 is
topology-gameable (cf. exp06 degree probe) and CV3 cold-start is expected
near-floor. Compare on CV2/CV3.
