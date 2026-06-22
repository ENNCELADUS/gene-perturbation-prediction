# Teaser figure brief (Figure 1) — close-loop arc

Realized as TikZ in `sections/` via `\input{figures/fig_teaser.tex}` so it
compiles with the manuscript (no external asset, no copyright issue).

Panels, left to right, with a return arrow closing the loop:

1. **Silence gene A → transcriptional shockwave.** A CRISPRi/knockout perturbation
   of gene A produces a genome-wide perturbation response (the "shockwave").
2. **Shockwave predicts A's own dependency.** The response maps to A's DepMap
   GeneEffect (foundation, exp01): transcriptome → own essentiality.
3. **Does the shockwave also name A's SL partners?** When one partner is anchored
   by a real profile (CV2) the observed transcriptome lifts partner ranking
   (exp07 ✓); when both genes are cold (CV3) it does not (✗).
4. **Generate the shockwave for unseen genes.** A virtual-cell foundation model
   (AIVC/STATE) driven by an ESM2 identity embedding *generates* the response for
   any gene, in principle reaching pairs neither gene was screened for (exp08,
   preliminary).

The closing arrow (4 → 3) is the thesis: a *predicted* shockwave should carry the
same partner signal the *observed* one does, extending it past screened genes.
Keep arrows = the loop; no copyrighted artwork.
