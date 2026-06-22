# Claim–Evidence Map — Transcriptome-Encoded SL Paper

Living table updated per writing task. Every major claim in the manuscript must
appear here with a supporting artifact and a status. Spec of record:
`docs/superpowers/specs/2026-06-22-paper-story-design.md`.

| Claim ID | Claim (as written) | Evidence (artifact / figure / table) | Status |
|---|---|---|---|
| CNEW | A virtual-cell framework (AIVC/STATE + ESM2 adapter) ranks SL pairs from *generated* responses and is architecturally able to score pairs where neither gene was screened. | `src/sl_dl_model/` (encoder, model, pooling, pair_head, losses); Fig. pipeline (`e2e_SL_DL.png`); Tab. method | supported (design); results preliminary (L1/L2) |
| L3 | STATE checkpoint is closed-vocab one-hot; only 16.3% (1,542/9,471) of the SL universe is in-vocab. | spec ledger; `docs/experiment/08`; deck slide 9 | supported |
| L2 | exp08 best-epoch selection reads the test fold (selection-matched), not a strict embedding-only ablation. | Method §3.5; spec L2 | stated as limitation |
