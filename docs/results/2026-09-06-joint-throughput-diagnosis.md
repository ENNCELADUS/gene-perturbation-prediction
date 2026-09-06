# Joint GeneEffect throughput diagnosis

Measured 2026-09-06 on `apg3op6hp3v99-0`, SSH port 30030,
`/2023533015/VCC_Project`. Optimization and bounded H20 measurements are complete
and synchronized through `main`. The original long-running process still uses
`1c5d906`; replacing it awaits explicit stop/resume approval. These are engineering
measurements, not new scientific results.

## Measurements

All probes used prepared inputs, real STATE, BF16, seed 0, dependency batch
256/rank, response batch 64/rank and replay every four updates. Eight warmup
updates preceded timing; CPU/operator profiling ran afterwards. Original workers
were temporarily suspended and automatically resumed, with artifacts untouched.
Full outputs/profiles remain under remote `outputs/profiles/`. The compact
[measurement record](2026-09-06-joint-throughput-probes.json) includes revisions,
timings, last-loss snapshots and source paths.

| Implementation | GPUs | Timed updates | Dependency rows/s | Median ordinary / replay | Peak allocated GiB/rank |
| --- | ---: | ---: | ---: | --- | ---: |
| Baseline `7a79352` (probe added to `1c5d906`) | 1 | 64 | 802 | 281 / 416 ms | 15.39 |
| Window views `abe57f0` | 1 | 64 | 849 | 266 / 397 ms | 15.31 |
| CPU-packed transfer `628c56f`, removed | 1 | 64 | 414 | 559 / 780 ms | 15.31 |
| Views + resident basal inputs `bbe4670` | 1 | 64 | **1,179** | **190 / 306 ms** | **14.99** |
| Same optimized model, DDP `f3d6118` | 2 | 128 | **2,043** | **215 / 349 ms** | **15.15** |

The matched single-GPU throughput gain is **46.9%**. All four single-GPU probes
produced the identical final GeneEffect loss `0.020633725449442863` after the
same seed/update sequence. This snapshot complements the gradient/resume checks;
it does not establish full trajectory equality or improved scientific quality.

Original two-GPU production throughput was about **1,484 rows/s**: steps
4923/4952/4981 at Unix times 1788712328.495/1788712338.498/1788712348.504.
The optimized DDP probe is 37.7% faster than that observation, but this is not a
matched production comparison: the probe starts fresh and omits per-step file
logging. A replacement production measurement remains pending.

Original GPU utilization averaged 57.97% / 59.60% over 96 samples/device. A partial
optimized DDP timed-window sample averaged 73.67% / 68.08% over 12 samples/device,
selected using result mtime with boundary margins. This is not a full-epoch
utilization claim. GPU-smi memory during probes includes suspended old workers;
the table measures only probe tensor allocation.

## Diagnosis and retained changes

Ranked hypotheses were per-condition transfer/chunk overhead, sequential replay
loss operations, cross-rank error/log collectives, and insufficient batch size.
The dominant removable overhead was repeated transfer of fixed inputs: baseline
profiling recorded 5,832 memcpy calls and 4,559 stream synchronizations over four
updates; dependency transfer alone cost about 75 ms/update. External py-spy
attachment failed on container process-memory permissions, so an in-process probe
provided the feedback loop.

- Complete STATE windows use views instead of newly transferred index tensors.
  Partial-window padding retains the original seeded indices.
- Datasets retain one sampled basal tensor per eligible ModelID on the worker
  device, reused across batches. Startup normalization and evaluation use this
  path too. These are fixed inputs; STATE/adapter features remain differentiable.
- CPU concatenation before transfer regressed because large CPU copies dominated.
  That implementation and its implementation-specific test were removed.

Batch sizes, learning rates, sampling streams, losses, replay interval, benchmark
membership, evaluation frequency and checkpoint selection are unchanged.

## Verification and remaining action

The no-copy and fixed-input reuse regressions failed before their respective fixes.
Exact indexed-reference values/input gradients were checked for full, partial and
short windows; retained tests cover padding and RNG isolation. H20 passed **37**
sampling/response tests, including CUDA residency. Local affected checks had
**47 passed, 1 CUDA skip**; two socket-blocked distributed-evaluation tests then
passed outside the sandbox. Distributed-update and real tiny-STATE integration
checks had **12 passed**. Ruff and whitespace checks passed. Single-GPU and
128-update two-rank H20 probes exited 0; distinct CUDA devices were checked.

The original run is `joint_seed0_20260906T160227Z_perf`. Its verified checkpoint
has `next_epoch=1`, `global_step=5889`, `best_epoch=0`, and validation GeneEffect
loss `0.016417402774095535`. This identifies the recovery point, not a new result.
Automatic approval rejected replacing the live job and trimming its metrics;
that command did not execute. The revised proposal preserves the entire old run
and resumes in a new directory after explicit stop/resume authorization. Its
uncheckpointed epoch prefix must be replayed. Old workers 66332/66333 resumed;
no replacement long-running job or monitor was launched.

## Bounded reproduction

```bash
CUDA_VISIBLE_DEVICES=0 .venv-tx1/bin/python -m src.experiments.profile_joint \
  configs/geneeffect_joint.yaml --steps 64 --profile \
  --output outputs/profiles/joint-single.json
.venv-tx1/bin/python -m accelerate.commands.launch \
  --num_processes 2 --num_machines 1 --mixed_precision bf16 --multi_gpu \
  --main_process_port 0 --module src.experiments.profile_joint \
  configs/geneeffect_joint.yaml --steps 128 \
  --output outputs/profiles/joint-ddp.json
```

Use otherwise idle GPUs for interpretable timing. Probes write no formal training
checkpoints. Future throughput checks should include CPU assembly/full-update
timing; memory occupancy alone cannot locate the limiting operation.
