# Claim–Evidence Map — Transcriptome-Encoded SL Paper

Living table updated per writing task. Every major claim in the manuscript must
appear here with a supporting artifact and a status. Spec of record:
`docs/superpowers/specs/2026-06-22-paper-story-design.md`.

| Claim ID | Claim (as written) | Evidence (artifact / figure / table) | Status |
|---|---|---|---|
| CNEW | A virtual-cell framework (AIVC/STATE + ESM2 adapter) ranks SL pairs from *generated* responses and is architecturally able to score pairs where neither gene was screened. | `src/sl_dl_model/` (encoder, model, pooling, pair_head, losses); Fig. pipeline (`e2e_SL_DL.png`); Tab. method | supported (design); results preliminary (L1/L2) |
| L3 | STATE checkpoint is closed-vocab one-hot; only 16.3% (1,542/9,471) of the SL universe is in-vocab. | spec ledger; `docs/experiment/08`; deck slide 9 | supported |
| L2 | exp08 best-epoch selection reads the test fold (selection-matched), not a strict embedding-only ablation. | Method §3.5; spec L2 | stated as limitation |
| C1 | Perturbation transcriptomes predict a gene's own dependency; signal is real, not generic death. | Tab. foundation (exp01 CV 0.485, Adamson AUROC 0.886; exp02 0.244 vs 0.494 vs 0.503) | supported (same-line) |
| C2 | Single-cell distribution embeddings carry dependency signal across datasets (exploratory). | Tab. foundation (exp03 0.666/0.911), flagged Adamson-guided | supported as exploratory (L6/F3) |
| C3 | Observed transcriptome encodes SL partners when one partner is anchored (CV2). | Tab. transcriptome (exp07 CV2 NDCG@10 0.042->0.094, AUROC 0.704->0.751) | supported |
| C4 | Leakage-controlled dependency-only floor exists; CV1 is degree-gameable. | Tab. floor (exp06 CV2 B AUROC 0.704; degree probe CV1 NDCG@10 0.197, collapses to 0.001) | supported |
| C5 | Most genome-wide co-dependency signal is pan-essentiality; pair-specific SL is the hard residual. | Tab. decomposition (exp09 CV3 non-pan-ess. AUROC 0.583, AUPR 0.490) | supported |
| C6 | Cold-start (both genes unseen, CV3) is the open frontier and the method's target. | Fig. ladder; Tab. transcriptome CV3; Tab. method CV3 | supported as open problem |
| C7 | Under one shared K562 protocol, label-free functional methods are classification-competitive with published label-graph SL methods but trail them on ranking. | Tab. benchmark (DDGCN/GRSMF/SL2MF/SLGNN from teammate reproduction, `benchmark_published.csv`; vs exp06/exp07 functional rows); `personal_report_sl_gene_perturbation.md` §6 | supported (no win claimed) |

## Abstract claim alignment

Every claim sentence in the abstract maps to a supported row above:
- "more than doubles partner-ranking quality (NDCG@10 0.042->0.094)" → C3.
- "lift vanishes when both genes are unseen" → C6.
- "frozen STATE + ESM2 adapter ... 83.7% out-of-vocabulary" → CNEW / L3 (83.7% = 100% − 16.3% in-vocab).
- "preliminary ... does not yet clear a strong dependency-only floor" → CNEW (L1).
- "most genome-wide co-dependency signal is pan-essentiality" → C5.
- "benchmark-adapter label, not a validated SL assay" → L8 (scope guard).

No abstract or conclusion sentence introduces a claim absent from this table.
