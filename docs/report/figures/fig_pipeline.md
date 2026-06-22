# Pipeline figure brief (Figure 2) — exp08 architecture

Realized by the existing authored asset `docs/report/e2e_SL_DL.png`
(`\includegraphics`), which already renders the full forward path. No redraw
needed; the manuscript references it as the method overview.

Path shown (matches `src/sl_dl_model/`):
ESM2 gene embeddings (9,471 genes, 1280-d) → trainable Perturbation Adapter
(1280→pert_dim, replaces STATE's `pert_encoder`, breaks the one-hot closed
vocabulary) → frozen STATE backbone (transformer + decoder) → predicted response
bag (n_cells × gene_dim) → MeanStd Pooling (per-gene embedding) → Symmetric Pair
Head f(e_a,e_b)=f(e_b,e_a) → P(SL).

Side annotations: 3-part loss = SL BCE (L_SL) + token distillation to original
STATE one-hot path (L_distill) + real-gwps bag supervision (L_bag); only the
adapter and pair head are trainable; eval/inference notes (per-anchor ranking).
