# P1-B: Diagnosing Response Functionality and Interface Adaptation in the Composite Backbone

## 1. Objectives and Fixed Data Boundaries

This experiment answers three questions:

1. Do B-init and B-joint correctly use perturbation identity to predict response?
2. Can independently training the new interface improve response prediction and cross-context transfer during the adaptation stage?
3. After interface adaptation, does unfreezing inherited ST parameters provide additional benefit?

**This round does not train the GeneEffect head, does not rerun P1-A, and does not perform multi-seed experiments or hyperparameter search.**

Use the already confirmed splits:

| Purpose | Data | Number of conditions |
|---|---|---:|
| Adaptation training | K562 + HepG2 + HCT116, excluding each context's condition holdout | 27,361 |
| Internal validation | Existing condition holdouts from the three anchor contexts above | 3,047 |
| Independent transfer evaluation | All cached Jurkat conditions | 2,377 |

- Jurkat is not used for mean fitting, training, early stopping, checkpoint selection, or unfreezing decisions.
- Preserve the existing fixed basal bags, observed bags, gene ordering, and training-side ESM2 mapping. Do not resample cells.
- The model continues to output the original 2,000 HVGs. HCT116 contains 43 padded coordinates, so **the primary training loss and primary evaluation metrics use the 1,957 coordinates that are actually observed in all four data sources**, avoiding treatment of zero padding as measured values. The panel is determined only by gene observability and does not use Jurkat response values.
- During preparation, verify the exact identity of these 1,957 coordinates. If the actual mapping does not match the audited count, stop preparation and report the discrepancy rather than silently changing the panel.
- Re-score B-init, B-joint, and all adapted states on this panel. Do not directly reuse the old 2,000-dimensional response loss for numerical comparison.

## 2. Experimental States and Trainable Parameters

| State / arm | Starting point and configuration | Diagnostic role |
|---|---|---|
| **B-native** | Native checkpoint, native HVG input, native one-hot perturbation encoding; evaluation only | Native functional reference |
| **B-init** | Composite architecture initialized using the P0 construction order and seed 0; evaluation only | Initial capability after interface replacement |
| **B-joint** | Trained backbone produced by P0; evaluation only | Change introduced by the original joint training |
| **B-interface** | Start from B-init; update only the new interface | Contribution of independent interface adaptation |
| **B-continue, conditional** | Continue from the best B-interface checkpoint; train interface only | Control for additional training budget |
| **B-unfreeze, conditional** | Start from the same checkpoint as B-continue; additionally unfreeze inherited ST parameters | Incremental gain from unfreezing |

**The trainable parameter scope for B-interface is fixed:**

- Newly initialized `basal_encoder.0.weight`, shape `328 × 2560`.
- ESM2 adapter: `1280 → 512 → 2024`, including the weights and biases of both Linear layers.
- Total: **2,533,864 trainable parameters**.
- Freeze the inherited basal bias, perturbation encoder, output layer, transformer, and batch embedding.
- At runtime, verify and export parameter ownership using the load report and `named_parameters()`. Do not classify an entire encoder as newly initialized based only on module prefixes. Raise an error if the actual ownership differs from the list above.

To isolate the effect of parameter updates, **keep ST in evaluation mode during both adaptation stages, with dropout disabled and running-statistic updates disabled, while preserving normal autograd**. Do not wrap the frozen ST forward pass in `no_grad()`, because the interface must still receive gradients propagated through ST. B-unfreeze changes only which parameters are trainable; it must not simultaneously change stochastic-layer behavior.

**Execution requirements for B-native:**

- Explicitly restore the native input normalization, output scale, gene ordering, and any required batch encoding.
- Use the native one-hot vocabulary. Do not substitute ESM2 embeddings or arbitrary zero vectors for native perturbation encoding.
- If the original numerical transformation or batch mapping cannot be recovered, mark B-native as **"comparable evaluation cannot be established"** and record the missing evidence. Continue diagnosing the other states, but do not generate a "native" result using approximate inputs.
- On Jurkat, the main common support set between B-native and the composite model contains **2,006 perturbations that were seen on the training side and are covered by the native vocabulary**.

## 3. Loss, Training Budget, and Early Stopping

Keep the existing two response-loss terms:

\[
L_{\mathrm{response}}
=\frac{1}{3}\sum_{c}
\operatorname{mean}_{g\in c}
\left[
L_{\mathrm{mean\ shift\ MSE}}(g,c)
+
L_{\mathrm{energy}}(g,c)
\right].
\]

Both loss terms have weight 1 and are not rescaled based on validation performance. First score each condition, then average within each anchor, and finally average the three anchors equally.

| Parameter | Recommended fixed value |
|---|---|
| Base seed for training, sampling, and evaluation | **0** |
| Global batch | **192 conditions: 64 from each training anchor** |
| Optimizer | AdamW, β=(0.9, 0.999), ε=1e−8 |
| New-interface learning rate | **1e−4** |
| Inherited ST learning rate in B-unfreeze | **1e−6** |
| Weight decay | **0.01**, applied only to trainable parameters |
| Gradient clipping | Global norm **1.0** across all trainable parameters |
| Precision | BF16 forward pass; loss, distance calculations, and metrics in FP32 |
| Scheduler / warmup | None |
| Maximum epochs per training stage | **50** |
| Early stopping | Internal-validation `response_loss`, patience=5 |
| Checkpoint rule | Update best only on strict decrease; ties or increases count toward patience |

**Definition of a balanced epoch:**

- Within each anchor, use an independent deterministic ordering derived from seed 0 and the epoch index. When a smaller pool is exhausted, reshuffle deterministically and cycle.
- One epoch is the number of updates required for the largest training pool, HCT116, to complete one pass:
  \[
  \lceil16399/64\rceil=257\ \text{updates}.
  \]
- Each epoch exposes 16,448 conditions per anchor, for **49,344 total condition exposures**. These are repeated exposures, not unique-condition counts.
- Maximum per stage: **12,850 updates**. Stop according to early stopping and do not backfill unused budget.
- At the end of each epoch, fully evaluate the unique training conditions and internal-validation conditions for all three anchors separately. Training-set evaluation does not affect checkpoint selection.

**Rule for entering stage 2:**

- Start B-continue and B-unfreeze only if the best B-interface checkpoint reduces internal-validation total response loss by **at least 1%** relative to B-init.
- The 1% threshold is a pre-specified execution trigger, not a statistical-significance criterion.
- Both arms start from the same best B-interface checkpoint and use newly initialized optimizers. Interface learning rate, data order, and stage budget must be identical.
- Early-stop each arm independently and compare both their best checkpoints and matched completed-update checkpoints. Evidence for an unfreezing benefit requires B-unfreeze to outperform B-continue.
- The stage starting point is itself eligible as the best checkpoint. If subsequent training does not improve it, retain the starting point and stop after five non-improving validations.

## 4. Evaluation, Response Controls, and Decision Rules

Complete all internal training and stage decisions first, freeze the checkpoints to be evaluated, and only then run Jurkat evaluation. **Do not use Jurkat results to retune parameters or decide whether to add an unfreezing stage.**

### Simple Response References

All baselines must output complete predicted bags so that both MSE and energy distance can be computed:

- **No-change:** directly predict the basal bag for that context.
- **Training mean effect:** first average condition effects within each training anchor, then average the three anchors equally, and add this fixed effect to the basal bag.
- **Perturbation mean:** for each gene, average its effect equally across training anchors that contain a training condition for that gene, then add that effect to the target basal bag. Use training conditions only. If no donor exists, leave the result missing and report coverage; do not backfill from another baseline.

### Correct Identity vs. Wrong Identity

- For each anchor and each reported subset, use seed 0 to generate **10 derangements with no self-matches**. Reuse the same mappings across all model states.
- Keep the basal bag, true target bag, and scoring coordinates fixed; replace only the input perturbation identity.
- Record:
  \[
  \text{identity advantage}
  =L_{\mathrm{wrong\ identity}}-L_{\mathrm{correct\ identity}}.
  \]
  Only a positive value supports correct use of perturbation identity. Also report the magnitude of output change, but do not use that as a substitute for predictive advantage.
- Report metrics on both the full scoring panel and a version with the **true target gene's own coordinate removed**. If that gene is not present in the scoring panel, the two are identical; record the count of such cases.
- Permutation runs are performed only during fixed-checkpoint diagnostics. They are not part of the training objective and do not increase per-epoch validation cost. The 10 permutations do not constitute multiple training seeds.

### Cross-Context Effect Differences

- Internal diagnostic: use perturbations shared across the condition holdouts of pairs of training-side anchors.
- External diagnostic: pair Jurkat with each donor anchor's condition holdout; the primary analysis is restricted to perturbations that were seen on the training side for Jurkat.
- Compare predicted and observed \(\Delta_{g,c}-\Delta_{g,c'}\). Report, for each anchor pair, effect-difference MSE, per-perturbation vector Pearson correlation, and the count of undefined correlations.
- Retain the same paired results for the simple baselines. Do not treat all anchor pairs as independent replicated contexts.

### Reporting Subsets and Interpretation Rules

For Jurkat, report separately: all 2,377 conditions, the primary set of 2,373 training-side-seen conditions, the 4 unseen conditions, and the native common-support set. For B-joint, additionally retain the original response-training / condition-holdout label so that already fitted conditions are not described as independent transfer.

For each state, report per-anchor MSE, energy, total response loss, relative improvement over baselines, identity advantage, cross-context metrics, and condition coverage. Use **1,000 paired perturbation bootstrap replicates with seed 0**; for cross-context comparisons, resample genes synchronously. The intervals are conditional on the current anchor, fixed checkpoint, and training seed, and do not estimate variation across anchors or initializations.

Decision rules:

- Lower response error together with improved correct-identity advantage and improved cross-context metrics supports a functional improvement in response modeling.
- Lower error without a clear identity advantage mainly supports better reconstruction and does not establish that the model learned perturbation-specific responses.
- Internal improvement without Jurkat improvement means the adaptation benefit has not transferred.
- Similar B-unfreeze and B-continue performance provides no evidence that additional unfreezing is needed.
- Do not infer GeneEffect improvement from response improvement in this round. A later experiment will compare the same explicit head on adapted R against the no-response P1-A arm A2.

## 5. Implementation, Validation, and Deliverables

Add an independent `hpc/run.sh p1b` entry point with `prepare`, `evaluate`, `train-interface`, `train-stage2`, and `compare`. Keep the production joint trainer unchanged. P1-B should reuse model construction, response loss, and cache reader code, while adding three-anchor sampling and an independent response selector rather than reusing the GeneEffect Huber selector.

- **prepare:** save the fixed splits, scoring panel, parameter-load ownership, native-compatibility audit, fitted baseline state, and permutation mappings. Do not compute Jurkat response scores.
- **execution:** each training arm uses one GPU and one process, with global batch fixed at 192. The two stage-2 arms may run in parallel on separate GPUs. Before production runs, test memory usage and finite gradients using the real batch. If OOM occurs, stop and report it rather than automatically shrinking the batch.
- **artifacts:** save source-code and input identities, configuration, per-epoch metrics, update norms for each parameter group, `best.pt`, `last.pt`, optimizer and RNG state, and separate training and evaluation state. Support resume at epoch boundaries. Evaluation export failures must be independently retryable.
- **comparison:** output per-condition metrics, mean-effect vectors, permutation summaries, cross-context pairing tables, learning curves, and a concise results report. Do not save the full predicted-cell matrix.

Use TDD to verify the following behaviors:

1. Jurkat labels and internal holdouts do not enter training or any baseline fitting. Changing Jurkat response values must not change training results or stage decisions.
2. B-interface updates only the specified new parameters, gradients pass through frozen ST, and B-unfreeze additionally updates inherited parameters.
3. Three-anchor equal-weight sampling, 257-update epochs, actual exposure counts, and resumed ordering are correct.
4. Early stopping uses only internal-validation total response loss and correctly handles ties, starting-point best checkpoints, and stage triggering.
5. Scoring coordinates are consistent across model states, padded coordinates are not treated as real targets, and own-coordinate removal uses the true gene.
6. Derangements contain no self-matches, all states use the same mappings, and missing metrics remain missing.
7. Checkpoint-resumed training matches uninterrupted training; when evidence for native input reconstruction is insufficient, B-native is explicitly marked unevaluable rather than producing a pseudo-native result.

The final conclusion must use the wording **"Jurkat was held out during the interface-adaptation stage."** The full pretraining exposure of ST / Tx1 has not yet been established, so do not claim that the entire model has never seen Jurkat.
