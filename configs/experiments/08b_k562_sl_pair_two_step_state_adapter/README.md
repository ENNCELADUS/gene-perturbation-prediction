# exp08b Two-Step STATE-Adapter Configs

Run Step 1 first:

```bash
rtk uv run python -m sl_dl_model train-generator --config configs/experiments/08b_k562_sl_pair_two_step_state_adapter/default.yaml
```

Then run Step 2:

```bash
rtk uv run python -m sl_dl_model train-sl-head --config configs/experiments/08b_k562_sl_pair_two_step_state_adapter/default.yaml
```

Step 1 writes fold-local generator artifacts under:

```text
results/experiments/08b_k562_sl_pair_two_step_state_adapter/default_cv2_cv3/step1_generator/CV*_fold*/
```

Step 2 writes official metric artifacts under:

```text
results/experiments/08b_k562_sl_pair_two_step_state_adapter/default_cv2_cv3/step2_sl_head/
```

Use `direct_mlp.yaml` and `nn_copy.yaml` for the two §5.2 step-2 control rungs
(both flow through the identical `train-generator` → `train-sl-head` passes;
`nn_copy.yaml` skips generator training and caches the ESM2-nearest
train-covered real bag). Use `ablation_bag_only.yaml`,
`ablation_distill_only.yaml`, and `ablation_ema_scale.yaml` for attribution runs.
