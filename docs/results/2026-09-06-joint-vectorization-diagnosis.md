# Joint GeneEffect vectorization diagnosis

Measured on 2026-09-06 at `apg3op6hp3v99-0`, SSH port 30030,
`/2023533015/VCC_Project`. This continues the
[resident-input diagnosis](2026-09-06-joint-throughput-diagnosis.md).
The final implementation is `f73f4e0`. Engineering throughput measurements do not
establish completed training or improved scientific quality.

Production throughput increased from **2,081 to 2,230 rows/s (7.1%)** after
checkpoint-boundary deployment. Mean GPU utilization remained approximately 76%
and memory remained about 17.7 GiB/card. This iteration improved throughput and
reduced CPU/CUDA overhead; it did **not** saturate GPU compute or memory.

## Controlled probes

Real prepared inputs and STATE, seed 0, BF16, eight warmup updates, and response
batch 64/rank with replay every four updates. Single-GPU probes timed 64 updates;
DDP probes timed 128 updates and used the slower rank's time for each update.
Workers of the prior production job were temporarily suspended with automatic
resume protection. Later probes explicitly checked both GPUs at 0% utilization
before starting. Probe memory below excludes the suspended workers' allocations.
Full measurements and source paths are in the
[probe record](2026-09-06-joint-vectorization-probes.json).

| Implementation | GPUs | Dependency batch/rank | Rows/s | Peak allocated GiB/rank |
| --- | ---: | ---: | ---: | ---: |
| Prior resident-input baseline | 1 | 256 | 1,179 | 14.99 |
| Batch FP32 conversion, idle-GPU repeat | 1 | 256 | 1,214 | 14.99 |
| Plus resident normalization statistics | 1 | 256 | 1,201 | 14.99 |
| Plus batched perturbation expansion | 1 | 256 | 1,239 | 14.99 |
| Plus response distance batching | 1 | 256 | 1,260 | 15.01 |
| Final direct perturbation tensor path | 1 | 256 | 1,255 | 15.01 |
| Prior resident-input baseline | 2 | 256 | 2,043 | 15.15 |
| Final implementation | 2 | 256 | **2,227** | **15.16** |
| Larger batch, before response batching | 1 | 512 | 1,369 | 26.59 |
| Larger batch, before response batching | 1 | 1024 | 1,490 | 51.90 |

Final matched-probe gains are 6.5% single GPU and 9.0% DDP. These are short
observations, without a statistical confidence interval. Individual small
differences should not be ranked confidently. Removing the intermediate
perturbation unbind/stack did not establish an additional single-GPU gain.

The first batch-cast trial returned 549 rows/s, with roughly doubled CUDA matrix
multiply time despite unchanged matrix operations. The 1,214 rows/s repeat on
GPU 1, after verified idle telemetry, did not reproduce that regression. The
anomalous trial is retained in the evidence; its exact environmental cause was
not established and it is excluded from the implementation comparison.

## Retained changes and numerical boundary

- Convert the low-precision STATE output to FP32 once before splitting views.
- Retain immutable fitted normalization tensors by block, device and dtype;
  checkpoint statistics remain unchanged and reconstruct the cache on restore.
- Pass perturbations as one matrix and expand all equal STATE windows together.
- Group response bags by shape for batched distance kernels, restoring condition
  order before the existing loss reduction. Unequal bag shapes remain supported.

The original four-update profile recorded 18,128 kernel launches and 207 stream
synchronizations. Statistics caching reduced synchronizations to 167; batching
response distances and the direct tensor path reduced kernel launches to 11,069
(38.9% fewer). Synchronizations decreased by 19.3%. Remaining explicit
finite-value checks and rank error propagation are preserved.

The batch-cast/statistics/expansion probes retained the exact baseline final
GeneEffect loss `0.020633725449442863`. Response distance batching changes floating
point operation order: final single-GPU loss became `0.020615633577108383`, an
absolute difference of approximately 0.0000181 after 72 updates. Independent
condition loss/gradient tests pass on CPU and CUDA, but the new trajectory is
**not bitwise identical** to the prior implementation. No accuracy claim follows
from this short-run difference.

Production continuation keeps dependency batch 256, response batch 64, the same
learning rates, seeds, sampling and validation/checkpoint rules. Larger batches
increase memory use and row throughput but change updates per epoch and replay
exposure per dependency row. They are diagnostic probes, not silently substituted
checkpoint configurations.

## Verification

The final H20 source passed all **60** STATE/features/response tests, including
the CUDA shape-group loss/gradient comparison and restored statistics reuse.
Local affected STATE/response/end-to-end training checks passed **47**, with two
CUDA skips; final resume/distributed/integration checks passed **15**. Startup
objective checks passed **12** after updating an old dataset test double to
accept the worker device. The original performance regressions were observed
before their fixes. Ruff and whitespace checks passed.

## Deployment

The old `joint_seed0_20260906T170355Z_resident` finished epoch 1 and saved a
checkpoint with `next_epoch=2`, `global_step=11778`, `best_epoch=1`, and
`best_loss=0.01632782630622387`. The validation value belongs to the old
implementation and identifies the recovery point. Authorized replacement stopped
launcher 68058 and workers 68190/68191 only after this complete checkpoint existed.
Their directories and logs remain preserved; the deliberate stop is recorded in
the old launch directory's `vectorization_stop.json`.

The new run is **`joint_seed0_20260906T174004Z_vector`**, launched through
`hpc/run.sh train` at `f73f4e0`, with the exact same saved configuration and world
size. It copied `last.pt`, `best.pt`, config and completed-epoch metrics into a
new directory. The checkpoint SHA-256 is
`4d77cf8153c2851fcae1a38835770862dfd28d0bca911d7cfc082ff8a6ff960e`.
The old run had logged step 11779 when stopped, so only **one** uncheckpointed
update was replayed. Supervisor 70477, launcher 70478 and workers 70610/70611
were verified running and advancing in epoch 2.

Remote paths:

- Run: `outputs/geneeffect_joint/joint_seed0_20260906T174004Z_vector/`
- Launch/log: `outputs/launches/joint_seed0_20260906T174004Z_vector/`

The epoch-1 checkpoint is inherited. Successful resume and measured throughput
do not establish a new optimized validation result or terminal training completion.

## Production measurement

After at least 128 resumed warmup updates, three consecutive 15-second windows
covered epoch-2 steps 11992--12188 with actual optimizer state and per-step
logging. The windows completed 65, 65 and 66 updates, respectively: 2,218,
2,218 and 2,252 dependency rows/s. Aggregate throughput was **2,229.57 rows/s**,
7.1% above the prior 2,081.05 rows/s production observation.

| Production observation | Prior resident-input run | Vectorized continuation |
| --- | ---: | ---: |
| Dependency rows/s | 2,081 | **2,230** |
| GPU 0 / 1 mean utilization | 76.27% / 76.33% | 75.51% / 77.33% |
| GPU 0 / 1 memory MiB | 18,218 / 18,212 | 18,166 / 18,164--18,166 |

Each device has 45 telemetry samples. Utilization did not materially increase;
short-window variation is larger than the change in the two-device mean. These
are steady-training samples, not full-epoch measurements. The bounded telemetry
sampler exited; no recurring monitor was created. Exact times, launch provenance
and telemetry summaries are in the
[production record](2026-09-06-joint-vectorization-production.json). Raw
`performance.json` and `gpu_performance.csv` remain in the new launch directory.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=1 .venv-tx1/bin/python -m src.experiments.profile_joint \
  configs/geneeffect_joint.yaml --steps 64 --profile \
  --output outputs/profiles/joint-vector-single.json
.venv-tx1/bin/python -m accelerate.commands.launch \
  --num_processes 2 --num_machines 1 --mixed_precision bf16 --multi_gpu \
  --main_process_port 0 --module src.experiments.profile_joint \
  configs/geneeffect_joint.yaml --steps 128 \
  --output outputs/profiles/joint-vector-ddp.json
```

`--dependency-batch-size 512` or `1024` overrides only the debug probe's in-memory
configuration. It writes no formal training checkpoint. Use idle GPUs and report
throughput together with peak allocation; GPU occupancy alone is not a speedup.
