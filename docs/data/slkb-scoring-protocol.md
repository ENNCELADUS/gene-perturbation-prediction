# SLKB scoring protocol (raw sgRNA counts to SL / non-SL call)

This document describes how SLKB (Gu et al., *NAR* 2024) processes the data of combinatorial double-knockout (CDKO)
screens. It describes SLKB only. It does not recommend a label rule for this project.

Each claim has a label:

- **[V]**: verified from the source quoted.
- **[I]**: inferred from the code or from arithmetic. The source does not state it.

The SLKB pipeline code was read at commit `e7cd532`, 2023-11-03, of
[BirkanGokbag/SLKB-Analysis-Pipeline](https://github.com/BirkanGokbag/SLKB-Analysis-Pipeline/tree/e7cd532891d1a311a9b607ff8448d530bc2bcc69).
This repository is the PyPI package `slkb`. Its latest PyPI version is 1.0.11, and that version was not compared with the commit.
The sources use these short names:

- **Code**: `SLKB/__init__.py`. Line numbers are given as `L<n>`.
- **Web**: the R Shiny app in `SLKB/files/SLKB_webapp.zip`.
- **Paper**: the article, which has the main text and Tables 1–3.
- **Supp**: the supplementary file "Manuscript - Supplemental File.pdf".

Section 8 gives the URL of each source.

## 1. Scope

| Item | Rule | Source |
|---|---|---|
| Screen design | Only paired-guide CDKO screens. Each construct has two sgRNAs, and each sgRNA targets one gene or "no gene, i.e. the control sgRNA". | [V] Paper, Methods: "we focus on dual single-guided RNA (sgRNA) design … in which a sgRNA targets either one gene or no gene" |
| Studies and lines | There are 11 studies and 22 cell lines. Raw counts are available for 10 of the 11 studies. | [V] Paper, Abstract: "11 CDKO experiments in 22 cell lines"; Table 3: "11 (currently counts available for 10)" |
| Time points | The pipeline uses one initial and one final time point. Intermediate time points are not used. | [V] Paper: "The methods were evaluated across initial and final timepoint of experiments" |
| Methods that SLKB does not reproduce | Methods that the original publication does not describe in enough detail are not reproduced. | [V] Paper: "We did not attempt to regenerate data analysis methods that were inadequately detailed in the initial publication or shared with a computational software." |
| Designs that cannot be processed | SLKB cannot score these designs: no T0 sample, no single-gene (gene × control) constructs, more than two guides in one construct, drug-anchor or RNAi screens, and screens without guide sequences (GEMINI needs the sequences). | [I] The code needs `T0_counts`, `TEnd_counts`, `target_type` and `sgRNA_guide_seq` (L573–607, L1727) |

Table 1 of the Paper lists the studies. The study's own criteria are in brackets:

- Wong 2016 (LFC < −0.9, Q < 0.01)
- Han 2017 (SL score < −4)
- Shen 2017 (< −3)
- Horlbeck 2018 (< −3)
- Najm 2018 (Q < 0.01)
- Zhao 2018 (< −3)
- Ito 2021 (FDR < 0.05)
- Parrish 2021 (SL score < −0.5, FDR < 0.1)
- Diehl 2021 (LFC < −1)
- Thompson 2021 (FDR < 0.1)
- Tang 2022 (< −0.361)

The Paper does not say which study has no counts. See section 7.

## 2. Input for each study

[V] The schema and the pipeline docs (`docs/pipeline.md`) define these input tables:

- **Sequences**: `sgRNA_guide_name`, `sgRNA_guide_seq`, `sgRNA_target_name`. A control guide has the target `control`.
- **Counts**: one row per construct (ordered guide pair) and cell line. The columns are `guide_1`, `guide_2`, `gene_1`,
  `gene_2`, `study_origin`, `cell_line_origin`, `study_conditions` and `count_replicates`. The last two hold the replicate names and the counts, separated by ";".
  The caller supplies `study_conditions = [[T0 replicate names], [TEnd replicate names]]` and a list of control target names (L125–147, L275–350).
- **Original scores** (optional): `gene_1`, `gene_2`, `study_origin`, `cell_line_origin`, `SL_score`, `SL_score_cutoff`,
  `statistical_score` and `statistical_score_cutoff` (L168).

[V] Each construct gets a `target_type` (L249–266):

| `target_type` | Rule |
|---|---|
| `Dual` | Both genes are not controls, and `gene_1 ≠ gene_2`. |
| `Control` | Both guides are controls. This is a "dual-targeting control". |
| `Single` | All other constructs: one gene and one control, or two guides that target the same gene. |

The Supp defines a control as "both sgRNAs target either a non-human gene or intergenic human genome" [V].

A gene pair is the two gene names in alphabetical order. `gene_pair_orientation` records `A_B` or `B_A` (L354–359) [V].

## 3. Preprocessing

The five methods do not share one preprocessing step. Each method has its own preprocessing [V] (Paper: "Each approach has its own preprocessing steps").

| Step | Median-B/NB and sgRNA-B/NB | Horlbeck | MAGeCK | GEMINI |
|---|---|---|---|---|
| Replicates | Read per replicate. A replicate that is all NA is removed. Other NA values become 0 (L593–605). | Same | Same | Same |
| Filter | Remove a construct if any T0 replicate has a count < 35 (L609–620, L1011) [V]. | For each TEnd replicate, remove a guide if its median count as guide 1 or as guide 2 is < 35. Remove every construct that contains that guide (L717–723) [V]. | None in SLKB (L1523) [V] | None in SLKB (L1716) [V] |
| Pseudocount | +10 after the filter, before normalization (L1016) [V] | +10 (L725–728) [V] | 1, the MAGeCK default [I]. Paper Table 2: "a pseudo-count of 1 is added before normalization" [V] | 32, `CONSTANT = 32` (GEMINI.R) [V] |
| Normalization | Do this separately for each `target_type`. Scale each sample to the median library size of all T0 and TEnd samples in that target type (L1036–1043, `full_normalization=False`) [V]. | Divide by the ratio of the pseudocounted totals, T0 sum / TEnd sum (L725) [V]. | `mageck test --norm-method control --control-gene` on the `CONTROL\|CONTROL` constructs (L1574–1578) [V] | `gemini_calculate_lfc(normalize = TRUE)` [V] |
| LFC | log2 of median(TEnd) / median(T0) across the replicates (L1048–1052). sgRNA-derived uses one LFC per replicate. When the T0 and TEnd replicate counts differ, it first averages each time point to one replicate (L1290–1296, L1322) [V]. | One log2 LFC per replicate. Subtract the median LFC of the `Control` constructs, then divide by 6.3 (L728–742) [V]. | The `LFC` column of `out.sgrna_summary.txt` for each construct (L1603–1612) [V] | Inside GEMINI |
| B vs NB | B subtracts the median LFC of the `Control` constructs. NB does not. B is computed only if `Control` constructs exist (L1161, L1399) [V]. | — | — | — |

The source of 6.3 is the Supp: "we have set this number to 6.3 across different studies from our validation with the
original scores of Horlbeck et al. 2018" [V].

The Paper says when to use NB: "If a study had no dual targeting controls (16, 17), Median-NB and sgRNA-derived-NB scores
were used." References 16 and 17 are Zhao 2018 and Ito 2021 [V].

## 4. The five scores

For all scores except GEMINI, **a more negative value means more SL**. For GEMINI Strong, **a larger value means more SL**. The sort
order in Web (`compareSL_server.R`, `get_n_percent`) shows this: `decreasing = T` for `gemini_score_SL_score_Strong`, and
`decreasing = F` for all other scores [V].

### 4.1 Median-B / Median-NB (L971–1212)

1. Calculate the single-gene effect of each gene: the median LFC over its `Single` constructs, with both orientations pooled. A missing gene gets 0 [V].
2. Calculate the pair effect: the median LFC over the `Dual` constructs of the gene pair [V].
3. Calculate `SL = pair − single(g1) − single(g2)` [V].
4. Calculate `SE = 1.25·sqrt(var_pair/n_pair + var_g1/n_g1 + var_g2/n_g2)` and `Z = SL/SE` [V].
5. For Median-B, subtract the control median c from all three terms first. Thus `SL_B = SL_NB + c` [I, algebra].

The Supp gives `P = Pr(Z ≥ |z|)`. The code does not store P [V].

Output columns: `SL_score`, `standard_error`, `Z_SL_score`.

### 4.2 sgRNA-derived-B / NB (L1214–1497)

Do these steps for each replicate:

1. Calculate the LFC of each guide pair. The guide pair is sorted, so AB and BA are averaged (mean) [V].
2. Calculate `SL = LFC − EC(guide1) − EC(guide2)`. EC is the median LFC of that guide's `Single` constructs [V].
3. Calculate `SE = sqrt(se_pair² + se1² + se2²)`, where `se = 1.25·sqrt(var/n)` for the single-guide terms. Calculate `z = SL/SE`. If SE is NaN or 0, use SE = 1 [V].
4. Calculate the gene-pair score as the median z over the guide pairs. Divide it again by `1.25·sqrt(var/n)` (again with 0 or NaN → 1) [V].

Then take the **mean over the replicates** [V].

B subtracts the control median from the single and dual LFCs before step 1 (L1399–1468) [V].

Output column: `SL_score`, which is a z value.

### 4.3 Horlbeck GI score (L688–969)

This score is ported from Horlbeck's `GImap_tools` (L681) [V].

1. Take the phenotype of each construct: the mean of the per-replicate LFCs, then the mean of AB and BA. A construct that is NA in any replicate is removed (L749–762) [V].
2. Take the single phenotype of each guide in position A and in position B: the mean phenotype over its guide × `CONTROL` constructs. A missing guide gets 0. Same-gene `Single` constructs are reassigned to `Dual` (L693–694, L806–825) [V].
3. For each query guide, fit `y = a·x² + b·x + c`, with c fixed at the single phenotype of the query guide. Here x is the single phenotypes of the partner guides and y is the construct phenotypes (`quadFitForceIntercept`, L682) [V].
4. The GI is the residual. Divide it by the SD of the residuals of the control partners [V].
5. Average the A and B orientations, then make the matrix symmetric (L936–937) [V].
6. Calculate the gene-pair score as the **mean** GI over the guide pairs of the gene pair, and `SE = sem` (L950–951) [V].

The Supp says "median" for step 6 [V]. The code and the Supp do not agree.

### 4.4 MAGeCK score (L1502–1695)

1. Run MAGeCK `test` with the TEnd columns as treatment and `--paired` when the T0 and TEnd replicate counts are equal (L1541, L1574–1578) [V].
2. Take the per-construct `LFC` [V].
3. Calculate `SL = median LFC(dual constructs of the pair) − median LFC(singles of g1) − median LFC(singles of g2)` [V].
4. Calculate `SE = sqrt(sum of var/n)·√2` and `Z = SL/SE` [V].

MAGeCK gene-level RRA is not used [I].

The command string is defined only when `Control` constructs exist (L1574) [V]. See section 7.

Output columns: `SL_score`, `standard_error`, `Z_SL_score`.

### 4.5 GEMINI score (L1697–1804, `GEMINI.R`)

1. Run GEMINI ≥ 2.1.1 with `gemini_calculate_lfc(normalize = TRUE, CONSTANT = 32)`, `nc_gene = "CONTROL"` and
   `gemini_inference(n_iterations = 20, force_results = TRUE)`, then `gemini_score(Model)` [V].
2. Store three outputs: `strong`, `sensitive_lethality` and `sensitive_recovery` [V].
3. Use `Strong` in the vote [V].

The definition comes from Zamanighomi 2019: "|s g,h,l| − max(|y g,l|, |y h,l|) … Larger positive values indicate a
stronger lethality if s g,h,l is negative" [V].

## 5. The SL / non-SL call

| Item | Rule | Source |
|---|---|---|
| Unit | One study and one cell line. Every method is applied to each study and each cell line. | [V] Paper: "each study and cell line was equipped with five different scoring approaches". Web download: "Contains the Venn diagram contents of SLKB scoring methods, separated by study and cell line." |
| Top 10% | For each method, remove the NA scores. Sort by the SL direction and take the first `floor(n × 0.10)` gene pairs (Web `get_n_percent`) [V]. | Paper: "By default, the top 10% of predicted pairs for each algorithm were categorized as synthetic lethal." |
| Methods in the vote | Median-B, sgRNA-B, GEMINI Strong, Horlbeck and MAGeCK. Use the NB versions when the study has no dual-targeting controls. | [V] Paper, Results |
| Score column for each method in Web | Horlbeck `SL_score`, Median `Z_SL_score`, sgRNA `SL_score` (a z value), MAGeCK `Z_SL_score`, GEMINI `SL_score_Strong`. | [V] Web |
| Vote | SL if **at least 3 of the 5** methods have the pair in their top 10%. | [V] Paper: "identified a pair as SL only if at least three of the methods agreed … namely the majority vote" |
| Non-SL | All other scored records. | [I] 13,173 + 248,785 = 261,958, which equals the "261,958 gene pairs (127,688 unique)" with calculated scores (Paper, Table 3) |
| Ties | Ties are not handled. R `order()` keeps the row order, so ties at the 10% rank are broken by row order. | [I] Web |
| Pairs missing from a method | A pair that has no score from a method is removed before that method ranks. The 10% is of the pairs that the method scores, so a missing pair gets no vote from that method. In the joined view `calculated_sl_table`, `FROM gemini_score LEFT JOIN …`, a pair that has no GEMINI score is not present. | [I] Web and schema |
| Vote export in Web | `SLKB_SL_pairs_by_approach.csv` has a 0/1 column for each method and `total_count`. The ≥ 3 cut is not in the code. | [V] Web `downloadPairs` |
| Counts | 13,173 SL and 248,785 non-SL (study, line, pair) records. That is 5.0% SL. | [V] Paper, Table 3; [I] the percentage |
| Study-reported calls | Kept separately in `cdko_original_sl_results.SL_or_not`. The call is SL if `SL_score ≤ SL_score_cutoff`, and also `statistical_score ≤ statistical_score_cutoff` when both cutoffs are given. A study without a cutoff gets "Not SL" for all its pairs (L223–237). The Web has a toggle that shows them. The total is 16,059 SL and 264,424 non-SL. | [V] Code; Paper: "A toggle switch … filters the browsed data to significant SL pairs identified by each study" |

## 6. Output tables

| Download | File | Contents |
|---|---|---|
| rawSL | `SLKB_original_scores.csv` | One row for each study, cell line and gene pair: `gene_1`, `gene_2`, `gene_pair`, `study_origin`, `cell_line_origin`, `SL_or_not`, `SL_score`, `statistical_score`, `SL_score_cutoff`, `statistical_score_cutoff` [V] (Web `browse_data_server.R`). |
| calcSL | `SLKB_calculated_scores.csv` | One row for each study, cell line and gene pair. The columns follow the view `calculated_sl_table`: `median_nb/median_b_score_{SL_score,standard_error,Z_SL_score}`, `sgrna_derived_{b,nb}_score_SL_score`, `horlbeck_score_{SL_score,standard_error}`, `mageck_score_{SL_score,standard_error,Z_SL_score}`, `gemini_score_SL_score_{Strong,SensitiveLethality,SensitiveRecovery}` [V] (schema). [I] that the CSV equals this view: `server.R` reads a CSV whose path is blank in the shipped code. |
| predSL | `SLKB_predictions.xlsx` | The top-10% set of each method, separated by study and cell line [V]. The file is not in the zip. |
| SQL dump | figshare 22902839 | Counts, sequences, original scores and computed scores [V] (Web). Not opened. |

[V] `gene_pair_id` keys a gene pair within one study and cell line. "this is done in order to make sure that gene pairs are unique to each study and cell line" (`docs/pipeline.md`).

## 7. Open points

- The Paper does not say which of the 11 studies has no raw counts.
- Web uses Median-B or sgRNA-B whenever the NB column is not all NA (`if (sum(is.na(NB)) != nrow) → B`). NB is always computed, so the app always selects B. For Zhao 2018 and Ito 2021, B is all NA, and those methods then select no pairs. [I] This is possibly a bug, and it is different from the Paper's rule. It is not known how the published 13,173 were computed.
- The MAGeCK command is not defined when a study has no `Control` constructs (L1574–1580). It is not known how Zhao 2018 and Ito 2021 got MAGeCK scores.
- Horlbeck gene-level aggregation: the code uses the mean, and the Supp says the median.
- The Paper does not say how SLKB handled studies with more than two time points or more than one condition, which T0 it used (plasmid or day 0), or which TEnd it used.
- The vote is per (study, line). There is no rule for a pair that is SL in one line and non-SL in another. Both records are kept.
- It is not verified that the PyPI `slkb` 1.0.11 matches commit `e7cd532`, or that the published scores came from this code.
- The Horlbeck 2018 full text was not available from Europe PMC. Its rules are taken from SLKB's port and the Supp. The MAGeCK paper was not re-read: the normalization and pseudocount are SLKB's description.

## 8. Sources

- Gu et al., "SLKB: synthetic lethality knowledge base", *NAR* 2024, PMC10767912:
  https://europepmc.org/article/PMC/PMC10767912 (full text: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10767912/fullTextXML)
- SLKB supplement, which contains the Supplemental File and the Supplemental Tables:
  https://pmc-oa-opendata.s3.amazonaws.com/PMC10767912.1/gkad806_supplemental_files.zip
- SLKB pipeline code at commit `e7cd532`:
  https://github.com/BirkanGokbag/SLKB-Analysis-Pipeline/blob/e7cd532891d1a311a9b607ff8448d530bc2bcc69/SLKB/__init__.py
  - `SLKB/files/GEMINI.R`
  - `SLKB/files/SLKB_sqlite3_schema.sql`
  - `SLKB/files/SLKB_webapp.zip`, which contains `server_func/compareSL_server.R` and `server_func/download_data_server.R`
  - `docs/pipeline.md`
- SLKB docs: https://slkb.docs.osubmi.org/. Website: https://slkb.osubmi.org. Figshare (not opened): https://doi.org/10.6084/m9.figshare.22902839
- Zamanighomi et al., GEMINI, *Genome Biol* 2019: https://europepmc.org/article/PMC/PMC6624979
- Horlbeck et al., *Cell* 2018: https://doi.org/10.1016/j.cell.2018.06.010. Code: https://github.com/mhorlbeck/GImap_tools
- Li et al., MAGeCK, *Genome Biol* 2014: https://europepmc.org/article/PMC/PMC4290824
