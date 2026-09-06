"""Bounded H20 timing/profiling probe; never writes training checkpoints."""

import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import time

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
import torch

from src.data.prepared import load_inputs
from src.experiments.config import load_config
from src.experiments.geneeffect import _revision
from src.model.initialization import build_joint_model
from src.model.normalization import fit_startup_standardizer
from src.training.sampling import make_training_loaders
from src.training.distributed import require_distinct_devices
from src.training.trainer import make_optimizer, train_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.steps < 4 or args.steps % 4:
        parser.error("--steps must be a positive multiple of four")
    config = load_config(args.config)
    accelerator = Accelerator(
        mixed_precision=config["precision"],
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.device.type != "cuda":
        raise ValueError("probe requires CUDA workers")
    if args.profile and accelerator.num_processes != 1:
        raise ValueError("operator profiling requires a single worker")
    require_distinct_devices(accelerator)
    torch.set_num_threads(1)
    set_seed(0)
    inputs = load_inputs(config)
    model = build_joint_model(config, inputs).to(accelerator.device)
    fit_startup_standardizer(
        model, inputs, batch_size=config["train"]["dependency_batch_size"],
        accelerator=accelerator,
    )
    optimizer = make_optimizer(model, config)
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()
    loader, replay = make_training_loaders(inputs, config, 0, accelerator)
    batches = iter(loader)
    step = 0

    def update():
        nonlocal step
        batch = next(batches)
        response = next(replay) if step % 4 == 0 else None
        values = train_update(model, optimizer, batch, response, config, accelerator)
        step += 1
        return values

    for _ in range(8):
        update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = {"ordinary": [], "replay": []}
    for _ in range(args.steps):
        kind = "replay" if step % 4 == 0 else "ordinary"
        start = time.perf_counter()
        last_losses = update()
        torch.cuda.synchronize()
        times[kind].append(time.perf_counter() - start)
    peak = torch.tensor(torch.cuda.max_memory_allocated(), device=accelerator.device)
    if accelerator.num_processes > 1:
        for kind, durations in times.items():
            values = torch.tensor(durations, device=accelerator.device)
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
            times[kind] = values.cpu().tolist()
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
    # train_update already returns globally averaged loss scalars.
    result = {
        "revision": _revision(), "config": config, "steps": args.steps,
        "world_size": accelerator.num_processes,
        "seconds": sum(sum(t) for t in times.values()), "step_seconds": times,
        "peak_allocated_gib": peak.item() / 2**30,
        "last_losses": last_losses,
    }
    result["dependency_rows_per_second"] = (
        args.steps * config["train"]["dependency_batch_size"]
        * accelerator.num_processes / result["seconds"]
    )
    if accelerator.is_main_process:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if args.profile:
        cpu = cProfile.Profile()
        cpu.enable()
        for _ in range(8):
            update()
        cpu.disable()
        report = io.StringIO()
        pstats.Stats(cpu, stream=report).sort_stats("cumulative").print_stats(50)
        args.output.with_suffix(".cpu.txt").write_text(report.getvalue())
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as profile:
            for _ in range(4):
                update()
        args.output.with_suffix(".ops.txt").write_text(
            profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
