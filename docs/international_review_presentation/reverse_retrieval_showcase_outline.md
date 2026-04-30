# AI Virtual Cell Models for Reverse Perturbation Retrieval

Working title: **AI Virtual Cell Models for Reverse Perturbation Retrieval from Transcriptomic Cell States**

Presentation goal: This is an education-outcome focused 5-minute showcase. The research result is supporting evidence, not the center. The audience should understand that the project trained me to connect a real biomedical question with AI modeling, rigorous evaluation, and critical reflection.

Design principle: each slide should contain only one short sentence on screen. The rest of the meaning should come from visual structure and oral explanation.

---

## Slide 1: Opening Question

**On-slide sentence**

> Can AI read a cell state and infer its hidden cause?

**Visual**

- Full-slide clean diagram, not a technical chart.
- Left: one abnormal/cancer-like cell icon.
- Middle: a compact transcriptomic fingerprint, shown as a barcode, heatmap strip, or short row of gene-expression bars.
- Right: a small ranked list of candidate genes, shown as `Gene A`, `Gene B`, `Gene C`.
- Use one strong arrow from left to right: `cell state -> molecular fingerprint -> candidate causes`.

**Image constraints**

- The cell image should look biological but simple; avoid dense microscopy images unless heavily cropped and visually clear.
- The transcriptome visual should look like a fingerprint, not a full heatmap matrix.
- Do not include formulas, metric names, model architecture, or citations on this slide.

**Oral purpose**

Introduce the central question in plain language: today we can measure what a cell looks like molecularly, but the harder question is what caused that state.

---

## Slide 2: Biological Motivation

**On-slide sentence**

> In cancer research, the useful target is often hidden among thousands of genes.

**Visual**

- Use a single cancer synthetic lethality-inspired scenario.
- Left: cancer cell with an existing defect or mutation.
- Center: many possible gene targets shown as a large cloud/grid of gene names or dots.
- Right: a small highlighted shortlist labeled only visually, not with long text.
- Main shape should be a funnel: `many possible genes -> prioritized shortlist -> experimental validation`.

**Image constraints**

- Keep the biology example intuitive. Do not show BRCA/PARP mechanism details unless later explicitly needed.
- Use no more than 5 visible gene labels; most candidates can be anonymous dots.
- Do not use the words `synthetic lethality` as large headline text. If used, it can appear only as a small speaker-note concept.

**Oral purpose**

Explain why reverse retrieval matters: experiments cannot test every possible gene combination cheaply, so an AI-generated shortlist can help researchers choose better next experiments.

---

## Slide 3: Project Framing

**On-slide sentence**

> I reversed the usual question: from cell state back to possible cause.

**Visual**

- Use two clearly separated horizontal lanes with opposite logic.
- Top lane, smaller and muted: a real biology experiment.
  - Left: a dish or cell icon labeled only as `cell`.
  - Middle: a small perturbation icon, such as a gene switch, CRISPR scissors, or drug droplet.
  - Right: a changed transcriptomic fingerprint or changed cell-state icon.
  - Arrow direction: `cell + known perturbation -> changed state`.
- Bottom lane, larger and brighter: my project framing.
  - Left: the observed changed transcriptomic fingerprint.
  - Middle: an AI / retrieval box.
  - Right: a ranked list of possible causal genes, such as `#1 Gene A`, `#2 Gene B`, `#3 Gene C`.
  - Arrow direction: `changed state -> AI retrieval -> possible causes`.
- Add a subtle curved arrow or rotation icon between the two lanes to show that the project flips the direction of the question.
- The bottom lane should occupy about 60-65% of the visual weight, because it is the project focus.
- Use icons and arrows; avoid equations.

**Image constraints**

- Do not make the top lane look like the project contribution. It is only the familiar scientific baseline: perturb first, then measure what changed.
- Do not show a full wet-lab workflow, sequencing machine, dataset split, or VCC challenge details.
- Do not include `cause and effect` as large text; explain this orally.
- Do not include `f(x,p)`, `g(x,y)`, probability scores, loss functions, or metric names.
- Keep visible labels minimal: `cell`, `known perturbation`, `changed state`, `AI retrieval`, `possible causes`.
- The ranked genes should look like a practical shortlist for follow-up experiments, not a final answer or a classification label.
- Visually mark uncertainty: use wording such as `possible causes` or a question mark near the ranked list, not `true cause`.

**Oral purpose**

Define the project task in one accessible contrast: in biology, we often perturb a known factor and observe how the cell changes; in my project, I started from the observed transcriptomic change and asked which perturbation gene could have produced it. This makes the project a reverse retrieval problem, where AI produces a prioritized hypothesis list for later validation.

---

## Slide 4: Model Intuition

**On-slide sentence**

> I used a foundation model to encode cell states and rank genes.

**Visual**

- Draw this as a simplified deep neural network architecture, not as a biological pathway.
- Overall flow should move left to right.
- Left side: two input streams.
  - Top input: `perturbed cell state`, shown as one transcriptomic fingerprint card with colored expression bars.
  - Bottom input: `matched control cells`, shown as 3-5 smaller faded fingerprint cards.
- Center-left: one large shared `scGPT encoder` block.
  - Shape: a deep-learning-style stack, such as 4-5 slightly offset trapezoids or slanted rounded rectangles.
  - Label inside: `scGPT encoder`.
  - Small subtitle inside or below: `pre-trained cell foundation model`.
  - Both the perturbed input and control input should feed into this same block, making it clear the encoder is reused.
- Center: two compact embedding outputs.
  - Top: `cell-state embedding`, shown as a small vector or dot in latent space.
  - Bottom: `control baseline`, shown as a faded vector or average icon.
  - Join them through a small diamond or subtraction node labeled `perturbation signal`.
- Center-right: `gene scoring head`.
  - Shape: a trapezoid or small neural head module, wider on the input side and narrower toward the output.
  - It receives the `perturbation signal`.
  - Next to it, include a small `gene embedding bank`, shown as a vertical stack of colored gene tokens. This suggests the model compares the perturbation signal against candidate genes.
- Right side: ranked candidate target genes.
  - Use a clean list: `#1 Gene A`, `#2 Gene B`, `#3 Gene C`.
  - Title the list `possible targets`, not `true targets`.
- Add a faint background motif behind the scGPT encoder: many tiny cell dots or faint transcriptome strips, suggesting large-scale pre-training.
- The visual should make the design intuition clear: scGPT reads cell states, the control branch removes background variation, and the scoring head turns the remaining signal into a ranked gene list.

**Image constraints**

- Do not draw a detailed transformer architecture.
- Do not show attention heads, tokenization, `[CLS]`, binning, loss functions, dot products, logits, or mathematical notation.
- Do not draw every neural network layer. The scGPT block can look deep through stacked shapes, but it should remain one module.
- Keep the control branch visually secondary. It supports the story by showing how the model isolates the perturbation signal; it should not dominate the slide.
- Avoid making the architecture look like a finished clinical diagnostic pipeline. This is an experimental research model that produces hypotheses.
- Use no more than six visible module labels: `perturbed cell state`, `matched controls`, `scGPT encoder`, `perturbation signal`, `gene scoring head`, `possible targets`.
- Use one accent color for the project path from `perturbed cell state` to `possible targets`; use gray or muted blue for the control branch.
- Leave open space for oral explanation. The architecture should be readable in 10 seconds.

**Oral purpose**

Explain the intuition: I did not train a model from scratch. I adapted scGPT, a single-cell foundation model, as an encoder for transcriptomic cell states. By comparing a perturbed cell with matched controls, the model focuses on the change that may be caused by a perturbation. A scoring head then converts that signal into a ranked list of candidate genes for follow-up experiments.

---

## Slide 5: Learning Outcome

**On-slide text**

Title:

> What I learned from this course project

Three bullets:

1. Translate a biomedical need into an AI task.
2. Adapt a foundation model for a real dataset.
3. Redesign test cases to better reflect real biological use.

**Visual**

- Pure text slide.
- No diagram, chart, pipeline, or result figure.
- Use three large, well-spaced lines as the main visual structure.
- Suggested layout:
  - Small top title: `What I learned from this course project`
  - Three centered or left-aligned takeaway lines below.
  - Each line may use a subtle number marker `01`, `02`, `03`, but avoid icons if the page starts to look crowded.

**Image constraints**

- This slide should feel like a student learning-outcome slide, not a result slide.
- Keep the design clean and quiet; no background photo is needed.
- Do not show any performance metrics, result tables, or model architecture.
- Keep every bullet to one short line.

**Oral purpose**

Connect the work to the showcase goal: the course project trained me to move from a real biomedical motivation to a concrete AI formulation, implement a foundation-model-based method, and reshape public-dataset evaluation so the test setting better matches real biological use.

---

## Overall Timing

- Slide 1: 35 seconds.
- Slide 2: 55 seconds.
- Slide 3: 60 seconds.
- Slide 4: 95 seconds.
- Slide 5: 55 seconds.
- Total: about 5 minutes with brief transitions.

## Visual Style

- Use a clean academic style with a restrained palette.
- Prefer white or very light background.
- Use one accent color for the reverse-retrieval path.
- Avoid dense biomedical pathway diagrams.
- Avoid screenshots of full paper figures unless redrawn and simplified.
- Keep all visible text large enough to read from the back of a room.

## Content To Avoid

- Do not explain Hit@K, Recall@K, MRR, DES, PDS, MAE in detail.
- Do not present the full Norman result table.
- Do not spend time defending scGPT performance against baselines.
- Do not describe Tahoe drug-target experiments unless asked in Q&A.
- Do not overclaim that the model discovers true drug targets. Say it prioritizes hypotheses for further validation.
