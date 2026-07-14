"""Small DDP invariants shared by exp05 orchestration and training."""

from collections.abc import Callable
import traceback

from accelerate import Accelerator
import torch


def require_exact_world_size(
    accelerator: Accelerator,
    expected: int = 4,
) -> None:
    """Reject authoritative exp05 training outside the locked DDP topology."""
    if accelerator.num_processes != expected:
        raise RuntimeError(
            f"authoritative exp05 requires exactly {expected} DDP ranks; "
            f"got {accelerator.num_processes}"
        )


def run_rank_zero_or_raise(
    accelerator: Accelerator,
    label: str,
    action: Callable[[], object],
) -> None:
    """Run one rank-zero action and raise its failure on every rank."""
    error_text: str | None = None
    error_summary: str | None = None
    if accelerator.is_main_process:
        try:
            action()
        except Exception as error:
            error_summary = f"{type(error).__name__}: {error}"
            error_text = traceback.format_exc()
    values = [error_summary, error_text]
    if accelerator.num_processes > 1:
        backend = str(torch.distributed.get_backend()).lower()
        device = accelerator.device if "nccl" in backend else torch.device("cpu")
        torch.distributed.broadcast_object_list(values, src=0, device=device)
    if values[1] is not None:
        raise RuntimeError(f"{label} failed on rank zero: {values[0]}\n{values[1]}")


def assert_all_ranks_stepped(
    accelerator: Accelerator,
    local_steps: int,
) -> tuple[int, ...]:
    """Require a positive optimizer-step count from every DDP rank."""
    local = torch.tensor(
        [local_steps],
        device=accelerator.device,
        dtype=torch.int64,
    )
    gathered = accelerator.gather(local).detach().cpu().reshape(-1)
    counts = tuple(int(value) for value in gathered.tolist())
    if len(counts) != accelerator.num_processes or min(counts, default=0) <= 0:
        raise RuntimeError(f"rank optimizer-step counts must all be positive: {counts}")
    return counts
