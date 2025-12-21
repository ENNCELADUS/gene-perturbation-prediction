# scGPT Retrieval Optimization (Urgent Fixes)

This roadmap turns the current diagnosis into an actionable, minimal set of fixes.
Focus is on wiring, correctness, and consistency before any hyperparameter tuning.

---

## 0) Current Symptoms (Observed)

- scGPT variants (frozen/head/LoRA) all sit at ~0.2% exact_hit@1.
- PCA baseline is ~17-18% exact_hit@1, so the task is learnable.
- Fine-tuning loss decreases, but validation loss stays near random.

---

## 1) Root Causes (Confirmed from code/logs)

1) Fine-tuned weights are not used in evaluation.
   - `src/evaluate/cell_eval.py` always uses `ScGPTEncoder` and ignores
     `model/scgpt_finetune/*.pt`.

2) LoRA never receives gradient.
   - `src/train/finetune.py` precomputes embeddings and only trains on those.
     LoRA modifies the backbone, but the backbone is not in the forward path.

3) Validation split is unseen conditions.
   - In `in_dist`, train/val are split by conditions, not cells.
     InfoNCE val is therefore near random by design.

4) Train/test mismatch on masking.
   - Fine-tuning uses unmasked expression.
   - Evaluation uses `mask_perturbed=True` by default.

5) Prototype averaging happens in expression space.
   - `build_prototype_library()` averages expression before scGPT encoding.
     For nonlinear encoders, this can distort retrieval geometry.

---

## 2) Urgent Fix Plan (Do First, Minimal Scope)

### A) Wire fine-tuned models into evaluation [DONE]

- Add a fine-tuned scGPT encoder that loads:
  - retrieval head weights
  - LoRA weights (if enabled)
- Use this encoder in `src/evaluate/cell_eval.py` when `encoder_type == "scgpt"`
  and a fine-tune checkpoint is provided.

Files:
- `src/model/scgpt.py`
- `src/evaluate/cell_eval.py`

Acceptance:
- Metrics change when swapping between frozen vs fine-tuned checkpoints.

---

### B) Make LoRA trainable (end-to-end or disable) [DONE]

- For `lora_head` mode, compute embeddings inside the training loop so that
  gradients flow to LoRA layers.
- If full end-to-end is too heavy, disable LoRA until wiring is correct.

Files:
- `src/train/finetune.py`

Acceptance:
- LoRA params have non-zero grad norms during training.

---

### C) Fix validation split for in_dist [DONE]

- For `track: in_dist`, use a cell-level split within the same conditions
  (not condition-level holdout).
- Reserve condition-level splits only for generalization tracks.

Files:
- `src/train/finetune.py`

Acceptance:
- Val loss decreases meaningfully under the corrected split.

---

### D) Align masking policy between train and eval [DONE]

- Apply `mask_perturbed_genes()` in fine-tuning before embedding extraction,
  matching evaluation settings.

Files:
- `src/train/finetune.py`
- `src/data/preprocessing.py`

Acceptance:
- Train/eval use identical masking flags.

---

### E) Build prototypes in embedding space for scGPT [DONE]

- Encode single cells first, then average embeddings per condition/prototype.
- Alternatively, use `library_type=raw_cell` for scGPT to validate the effect.

Files:
- `src/evaluate/cell_eval.py`
- `src/data/preprocessing.py`

Acceptance:
- scGPT retrieval improves vs expression-space averaging.

---

## 3) Post-Fix Checks (Quick Sanity)

- Compare frozen vs head-only vs LoRA on the same split.
- Verify that fine-tuned scGPT beats frozen scGPT and moves toward PCA.
- Ensure masking ON/OFF ablation behaves consistently.

---

## 4) Out of Scope (Not Urgent)

- Hyperparameter sweeps.
- New losses or heavy architecture changes.
- Full-model fine-tuning (risk of overfitting).
