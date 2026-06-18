Project 3: Synthetic lethality discovery with AI virtual cells
Background
Two genes are synthetic lethal (SL) when loss of either gene alone is tolerated, but simultaneous 
loss of both genes kills the cell (Figure 3). In cancer, one gene is already inactivated by a somatic 
mutation; its SL partner becomes a selective drug target. PARP inhibitors in BRCA1/2-mutant 
tumors are the canonical clinical proof. The challenge is that most SL gene pairs are 
context-dependent: the SL relationships hold in some cell lines, some tumor subtypes, or some 
mutational backgrounds, but not others [1].
Figure 3. The concept of synthetic lethality.
There are roughly 200 million possible human gene pairs. Only a tiny fraction can be tested 
experimentally, and even fewer across the many genetic and cellular contexts that matter in cancer. 
This is the practical bottleneck behind synthetic lethality (SL): the idea is powerful, but the search 
space is too large and too context-dependent to brute-force.
Combinatorial CRISPR screens can measure genetic interactions at scale, but they cover a small 
fraction of the gene-pair space and are expensive to run across many contexts. Dependency maps 
such as DepMap profile gene essentiality across hundreds of cancer cell lines using genome-wide 
CRISPR screens, providing a rich but indirect signal for SL [10]. Translating those cellline 
signals into tumors requires bridging transcriptional profiles, mutational context, and clinical data, 
a problem that the TCGA-linked dependency map work addresses directly [7].
Where AI virtual cells enter. The AI Virtual Cell (AIVC) agenda reframes perturbation 
prediction as a general modeling problem: To build a deep learning model that represents cellular 
states well enough to simulate how cells respond to interventions across conditions, scales, and 
modalities [3]. The Virtual Cell Challenge (VCC) operationalizes this as a community benchmark, 
turning generalization to unseen perturbations into a shared evaluation target [4].
12
These two threads fit together naturally. A useful SL pipeline does not need to solve all of cell 
biology. It needs to do three things:
1. Represent cellular context (tumor type, mutational background, expression state).
2. Predict or approximate the effect of perturbing a candidate partner gene.
3. Rank candidate SL pairs in a way that survives realistic held-out evaluation.
Goal: Develop a deep-learning system that uses perturbation data, dependency information, and 
synthetic lethality resources to rank candidate SL gene partners in a defined cancer context.
Strong projects do at least one of the following well:
 Learn a context-aware representation of cellular state.
 Use perturbationresponse modeling to estimate the effect of candidate interventions.
 Integrate SL knowledge bases or dependency maps to improve prioritization.
 Evaluate predictions under realistic held-out settings rather than random easy splits.
Learning Objectives
 Understand the biological motivation behind synthetic lethality and why cellular context 
determines whether an SL interaction holds.
 Learn how AIVC and perturbation foundation models frame cellular-response prediction, 
and what their current limits are.
 Work with DepMap, TCGA, SynLethDB, and related resources for dependency mapping, 
SL labels, and patient-context transfer.
 Design an evaluation protocol that avoids data leakage, weak negatives, and benchmark 
inflation.
 Build a baseline and a stronger model, then explain clearly what is improved and why.
Suggested Methods
Choose one of the following directions or combine several of them. Start simple and add 
complexity only when the baseline is understood.
1. Representation learning with simple baselines first
Begin with linear models, matrix factorization, or shallow MLP baselines over DepMap gene 
effect profiles. Test whether more complex architectures actually improve on a realistic split 
before investing in them. The literature strongly suggests this step is not optional [12].
13
2. Foundation-model (FM) adaptation
Start from a perturbation-aware encoder, then fine-tune or probe it for SL ranking or context 
representation. Be explicit about what the pretrained model contributes versus what is learned 
from SL labels. Current SOTA options:
 Tahoe-x1 [6]: 3B-parameter FM trained on Tahoe-100M, a giga-scale CRISPR perturbation 
atlas. Explicitly designed for contextdependent gene function prediction. It is probably the 
strongest pretrained option if you want a perturbation-native encoder.
 Stack [16]: Arc Institute in-context learning model that conditions on "example cells" as 
prompts and was used to generate the Perturb Sapiens simulated perturbation atlas. Strong 
for context generalization across unseen cell states.
 scGPT [5]: Widely used single-cell FM baseline. Less perturbation-specialized than 
Tahoex1 or Stack, but welldocumented and easy to finetune.
3. Graph or knowledge-based modeling
Build a graph neural network model combined with a knowledge graph over genes, pathways, 
perturbations, and known SL interactions. SynLethDB's knowledge graph component provides a 
natural starting point. Prior work in this family (KG4SL, SLGNN, GCATSL) provides useful 
baselines.
4. Hybrid score design
Combine predicted perturbation effect, dependency similarity (from DepMap), pathway proximity, 
and prior SL evidence (from SynLethDB or SLKB) into a calibrated ranking model. This is often 
more robust than any single signal alone.
5. Mechanistic interpretation
Add pathway-level or network-level explanations so the model produces biologically readable 
rationales alongside the scores. Paralog pairs are a good test case because the expected mechanism 
(redundancy buffering) is well understood [9].
Data Resources
Primary Data Sources
Resource What to Use Access
DepMap CRISPR gene effect scores 
(Chronos), copy number, mutation 
calls, expression profiles across 
https://depmap.org/portal/
14
1000+ cancer cell lines
TCGA (GDC) RNA-seq gene expression 
(FPKM/TPM), somatic mutation 
calls, clinical metadata across 33 
cancer types
https://portal.gdc.cancer.gov/
SynLethDB 2.0 Curated SL pairs, biomedical 
knowledge graph, multi-species
evidence with confidence scores
https://www.synlethdb.com/
Secondary Data Sources
Resource What to Use Access
SLKB CRISPR double-knockout (CDKO) 
derived SL scores; explicit 
context-dependence annotations
https://slkb.osubmi.org/
Tahoe-100M / VCC 
datasets
Large-scale single-cell perturbation 
atlas for training perturbation 
representations
https://arcinstitute.org/tools/virtu
alcellatlas and 
https://virtualcellchallenge.org/
scPerturb Harmonized collection of public 
singlecell perturbation datasets
https://projects.sanderlab.org/scp
erturb/
TCGA Translational 
Dependency Map
Cell-line-to-tumor dependency 
transfer with validated SL examples
See [7]
Evaluation Metrics
Evaluation quality matters as much as model quality. The SL benchmarking literature shows that 
random splits and weak negatives produce misleadingly optimistic results [2], and the perturbation 
modeling literature shows the same for forward prediction tasks [12, 13].
Metric Purpose
PRAUC / ROCAUC Overall SL classification or pair scoring 
performance
Recall@K / Hit@K (K = 10, 50) Candidate prioritization quality
Heldout gene split Test generalization to unseen genes (not just 
unseen pairs)
Heldout cellline or context split Test generalization to unseen cellular contexts
Baseline comparison Compare against a simple model (e.g., mean 
gene effect score, co-essentiality correlation)
15
Error analysis Where does the model fail? Which gene 
families, cancer types, or evidence types are 
hardest?
Stretch Goals (Optional)
 Double-perturbation modeling. Move from single-gene effects to explicit pairwise SL 
hypotheses using combinatorial perturbation data (CDKO screens or Tahoe-100M).
 Patient-context transfer. Connect cell-line or perturbation data to tumor expression or 
dependency context using TCGA. The translational dependency map [7] provides a validated 
benchmark for this.
 Mechanism-aware explanations. Identify pathways or subnetworks that explain predicted 
SL interactions. Paralogous SL pairs are a good test case because the expected mechanism is 
known [9].
 Transcriptomic SL signatures. Reproduce or extend the finding that SL interactions leave 
detectable buffering signatures in TCGA tumor expression data [8].
Useful Links
 DepMap Portal: https://depmap.org/portal/
 TCGA GDC Portal: https://portal.gdc.cancer.gov/
 SynLethDB 2.0: https://www.synlethdb.com/
 SLKB: https://slkb.osubmi.org/
 Virtual Cell Challenge: https://virtualcellchallenge.org/
 Arc Virtual Cell Atlas: https://arcinstitute.org/tools/virtualcellatlas
 scPerturb: https://projects.sanderlab.org/scperturb/
 PerturbArena: https://luyitian.github.io/PerturbArena/index.html
 Systema project page: https://brbiclab.epfl.ch/projects/systema/
References
[1] A. Huang, L. A. Garraway, A. Ashworth et al., "Synthetic lethality as an engine for cancer drug 
target discovery," Nature Reviews Drug Discovery, vol. 19, pp. 23-38, 2020, doi: 
10.1038/s41573-019-0046-z.
[2] Y. Feng, Y. Long, H. Wang et al., "Benchmarking machine learning methods for synthetic 
16
lethality prediction in cancer," Nature Communications, vol. 15, art. 9058, 2024, doi: 
10.1038/s41467-024-52900-7.
[3] C. Bunne et al., "How to build the virtual cell with artificial intelligence: Priorities and 
opportunities," Cell, vol. 187, no. 25, pp. 7045-7063, 2024, doi: 10.1016/j.cell.2024.11.015.
[4] Y. H. Roohani et al., "Virtual Cell Challenge: Toward a Turing test for the virtual cell," Cell, 
vol. 188, no. 13, pp. 33703374, 2025, doi: 10.1016/j.cell.2025.06.008.
[5] H. Cui et al., "scGPT: Toward building a foundation model for singlecell multiomics using 
generative AI," Nature Methods, vol. 21, no. 8, pp. 1470-1480, 2024, doi: 
10.1038/s41592-024-02201-0.
[6] S. Gandhi, F. Javadi, V. Svensson et al., "Tahoe-x1: Scaling Perturbation-Trained Single-Cell 
Foundation Models to 3 Billion Parameters," bioRxiv, 2025, doi: 10.1101/2025.10.23.683759.
[7] X. Shi, C. Gekas, D. Verduzco et al., "Building a translational cancer dependency map for The 
Cancer Genome Atlas," Nature Cancer, vol. 5, pp. 1176-1194, 2024, doi: 
10.1038/s43018-024-00789-y.
[8] S. Haider, R. Brough, S. Madera et al., "The transcriptomic architecture of common cancers 
reflects synthetic lethal interactions," Nature Genetics, vol. 57, pp. 522-529, 2025, doi: 
10.1038/s41588025021082.
[9] B. De Kegel, N. Quinn, N. A. Thompson et al., "Comprehensive prediction of robust synthetic 
lethality between paralog pairs in cancer cell lines," Cell Systems, vol. 12, no. 12, pp. 
1144-1159.e6, 2021, doi: 10.1016/j.cels.2021.08.006.
[10] C. Pacini, E. Duncan, E. Gonçalves et al., "A comprehensive clinically informed map of 
dependencies in cancer cells and framework for target prioritization," Cancer Cell, vol. 42, no. 2, 
pp. 301-316.e9, 2024, doi: 10.1016/j.ccell.2023.12.016.
[11] J. Wang, Q. Zhang, J. Han et al., "Computational methods, databases and tools for synthetic 
lethality prediction," Briefings in Bioinformatics, vol. 23, no. 3, article bbac106, 2022, doi: 
10.1093/bib/bbac106.
[12] C. AhlmannEltze, W. Huber, and S. Anders, "Deeplearningbased gene perturbation effect 
prediction does not yet outperform simple linear baselines," Nature Methods, vol. 22, pp. 
1657-1661, 2025, doi: 10.1038/s41592-025-02772-6.
[13] R. Viñas Torné, M. Wiatrak, Z. Piran et al., "Systema: a framework for evaluating genetic 
17
perturbation response prediction beyond systematic variation," Nature Biotechnology, 2025, doi: 
10.1038/s41587-025-02777-8.
[14] J. Wang, M. Wu, X. Huang et al., "SynLethDB 2.0: a web-based knowledge graph database 
on synthetic lethality for novel anticancer drug discovery," Database, 2022, article baac030, doi: 
10.1093/database/baac030.
[15] B. Gökbağ, S. Tang, K. Fan et al., "SLKB: synthetic lethality knowledge base," Nucleic Acids 
Research, vol. 52, no. D1, pp. D1418-D1428, 2024, doi: 10.1093/nar/gkad806.
[16] M. Dong, A. Adduri, D. Gautam et al., "Stack: In-Context Learning of Single-Cell Biology," 
bioRxiv, 2026, doi: 10.64898/2026.01.09.698608