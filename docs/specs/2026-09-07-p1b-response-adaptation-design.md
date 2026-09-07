# P1-B: response functionality and interface adaptation

Status: approved design, implemented for seed-0 diagnostics. This document records
all decisions confirmed in the P1-B planning conversation. The earlier referenced
anchor-selection note is absent from this checkout; the decisions are included
here rather than reconstructing a separate note.

## Question and limits

Does the assembled Tx1 + ESM2 + ST backbone correctly use perturbation identity and
context to predict Perturb-seq responses? Does independent interface adaptation
improve it, and does subsequent ST adaptation add value? GeneEffect heads and
labels do not enter the response optimization or selection objective. No P1-A
retraining, multi-seed search, random-ST control, or new pretraining is included.

Jurkat is held out **from interface adaptation**. Original ST and Tx1 pretraining
non-exposure is not established. B-joint has already received all four anchors'
response supervision and is an exposed reference, not an independent transfer arm.
Improved response prediction is not evidence of improved GeneEffect or SL ranking.

## Membership and measured coordinates

| Role | ModelIDs | Conditions |
|---|---|---:|
| Train | K562 ACH-000551, HepG2 ACH-000739, HCT116 ACH-000971 | 27,361 |
| Internal validation | Existing condition holdouts in those three anchors | 3,047 |
| External evaluation | Jurkat ACH-000995, all conditions | 2,377 |

Existing response-cache keys and holdouts are authoritative. Preparation fails on
count disagreement. All fitting, checkpoint selection, early stopping and stage
choices use the three source anchors only. Jurkat response labels are never read
while fitting baselines, and do not affect the stage-two eligibility decision.

Keep fixed basal and observed bags, gene order, and ESM2 vectors. The ST output
remains 2,000-dimensional. Use exactly **1,957 common measured coordinates** for
all training losses and primary evaluation. Recover the mask by exact matching of
checkpoint symbols against each source's configured `target_gene_symbol_col`,
mirroring response-target alignment. Read H5AD `var` only through HDF5, or the
Orion gene-metadata column; no raw expression matrices or full observation tables
are needed. Do not infer missing coordinates from zero-valued target columns.
Store included/excluded symbols per source; fail if the intersection count differs.

Jurkat strata are all 2,377; training-seen 2,373; unseen four; native vocabulary
2,009; native-and-training-seen 2,006. The unseen genes are ATP6V0C, COX6C, DHX15,
HIST1H2BM. The main transfer analysis uses seen perturbations; the four unseen
conditions support descriptive results only. Persist original B-joint condition
holdout membership in exports.

## States and trainable parameters

| State | Definition |
|---|---|
| B-native | Original checkpoint, original inputs, original perturbation encoding; evaluation only |
| B-init | Original P0 config and seed-0 construction order, released checkpoint plus new interfaces |
| B-joint | Exact P0 backbone used by P1-A; evaluation only |
| B-interface | From B-init; train only new basal input weight and ESM2 adapter |
| B-continue | Conditional stage two: continue interface-only training from best B-interface |
| B-unfreeze | Same stage-two starting checkpoint, additionally train inherited ST parameters |

B-interface contains exactly 2,533,864 trainable parameters:

- `state_adapter.state_model.basal_encoder.0.weight`: 328 x 2560.
- `perturbations.adapter.net.0.weight` / bias: 512 x 1280 / 512.
- `perturbations.adapter.net.2.weight` / bias: 2024 x 512 / 2024.

All other parameters, including the inherited basal bias, remain frozen. Audit
actual source/destination checkpoint shapes and loaded names, then classify every
parameter; mismatched shapes, new unclassified parameters or missing inherited
parameters are errors. Do not infer provenance by treating a whole module as new.

ST and the assembled backbone stay in eval mode in **both stages**. Autograd stays
enabled through inherited operations so new interfaces receive gradients. Stage two
changes parameter training flags only, not dropout behavior. B-init is built through
the same public joint constructor as P0, but only the backbone is persisted and
restored thereafter. Head standardization and projection are irrelevant to response
prediction and are not refitted.

B-native is a functional reference, limited to native vocabulary support. Original
normalization/output scaling and batch mapping remain unverified; `int_counts:
false` does not establish them, and the referenced external original config is
unavailable. The current evaluator explicitly records B-native as unavailable and
never fabricates native inputs/predictions. Resolving these prerequisites is separate
work; it does not block assembled-backbone diagnosis.

## Objective, budget and selection

Per condition, compute FP32 mean-shift MSE plus energy distance, each weight one.
Both use the common measured coordinates. Average within anchor, then equally
across the three source anchors. The mean-shift control mean cancels algebraically;
this loss alone cannot demonstrate correct perturbation identity use.

| Parameter | Fixed value |
|---|---|
| Seeds | Training, sampling, inference: 0 |
| Global batch | 192 conditions, 64 per source anchor |
| Epoch | 257 updates; maximum source training pool 16,399 |
| Exposure/epoch | 16,448 per anchor; 49,344 total, including repeats |
| AdamW | beta=(0.9,0.999), epsilon=1e-8, weight decay=0.01 |
| Interface LR | 1e-4 in both stages |
| Inherited ST LR | 1e-6, B-unfreeze only |
| Gradient clipping | Combined trainable-parameter norm 1.0 |
| Precision | CUDA BF16 forward; FP32 loss/distance computation; CPU tests FP32 |
| Scheduler/warmup | None |
| Per-stage cap | 50 epochs / 12,850 updates |
| Early stopping | Internal equal-anchor response MSE+energy, patience 5 |

Source pools have independent deterministic permutations derived from seed 0 and
epoch; smaller pools cycle with fresh permutations. Every update is balanced.
Evaluate all unique source training and validation conditions each epoch. Report
individual/total losses, actual updates/exposures per anchor and parameter-group
update norms. Only validation loss selects checkpoints. Epoch zero is eligible;
strict reduction updates best, ties count toward patience. If no improvement occurs,
stop after five epochs and keep the starting checkpoint.

Run B-continue and B-unfreeze only when best B-interface internal validation loss
is at least 1% below B-init. This is a fixed execution trigger, not significance.
Both stage-two arms start from the exact same best checkpoint with fresh optimizers,
identical sampling order and their own patience-five/50-epoch budgets. Compare best
checkpoints and joined common-update points. Additional training alone is controlled
by B-continue; only its contrast with B-unfreeze identifies an unfreezing benefit.

Complete required training and record selected checkpoint hashes in ordinary
`external_evaluation.json` before scoring Jurkat. Once that record exists, further
training in that run directory is rejected. No external metric enters this decision.

## Response controls and exports

Three train-only bag baselines: no-change (basal bag); global train mean effect
(equal conditions within source anchor, equal anchors); perturbation mean effect
(equal available training-anchor donors). Add the fitted effect to the basal bag,
so all baselines support both MSE and energy distance. Missing gene donors remain
missing and coverage is reported; no fallback predictions.

Each held-out panel has ten fixed seed-0 derangements without self matches. Keep
basal controls and true target bags fixed, and replace only input perturbation
identity. Every state uses identical maps. Correct-identity advantage is wrong loss
minus correct loss. Also report output mean-effect change; output change alone is
not correctness. Recompute errors after excluding the **true target gene** coordinate
from correct and wrong outputs alike. If absent from the panel, the mask is unchanged
and the removal flag is false. Identical panels reuse wrong-identity inference.
Training exports use correct identities only; costly shuffles cover internal and
external holdouts. Singleton panels have undefined identity diagnostics.

Cross-context analysis compares predicted and observed effect differences for the
same gene. Source/source pairs require both conditions in internal holdouts;
Jurkat/source pairs require a source holdout and a training-seen Jurkat gene. Report
per-pair difference MSE, vector Pearson and undefined counts, including baseline
comparisons. Never treat anchor pairs as independent context replications.

Persist condition metrics and predicted/observed mean effects, not full cell
predictions. Report panel coverage, train/val/external distinction, identities,
original joint exposure, loss deltas, relative loss improvement, and 1,000 paired
gene-bootstrap intervals. Synchronize resampling by gene wherever it spans contexts.
Intervals condition on fixed anchors, seed and selected checkpoint; they do not
include initialization, new-context or selection variability. Do not label larger
identity/correlation diagnostics with a lower-is-better improvement percentage.

Interpretation: lower errors plus correct-identity advantage and better context
contrasts supports response functionality; error-only gains suggest improved average
reconstruction. Internal-only gains do not establish transfer. Responses improved
without downstream head testing remain response findings. Any future GeneEffect
check should reuse the explicit P1-A readout and A2 no-response reference.

## Execution and acceptance

`hpc/run.sh p1b` supplies prepare, evaluate, train-interface, train-stage2 and compare.
One process/GPU per arm; stage-two arms can use separate GPUs. Working code is
synced by Git only. Preparation is once per bundle, including a streaming hash of
cached targets; runtime opens existing caches without raw rebuilds or recursive
hashing. A real full batch checks allocation/finite gradients; OOM fails without
automatic batch changes. Training and evaluation status are separate. Checkpoints
contain weights, optimizer, RNG, selection state and history; resume is epoch-boundary.
Export retry uses fixed weights and does not optimize. Production joint behavior is
unchanged. This implementation does not launch jobs or push code.

Acceptance tests cover leakage-safe membership/baselines, masks, balanced exposure,
exact interface provenance, eval-mode gradients, epoch-zero early stopping, stage
gate, real STATE reconstruction/response fitting, exact resume, export retry,
derangements, true-own-coordinate exclusion, cross-context pairing, bootstrap,
matched stage-two budgets, CLI wiring and the external selection boundary.
