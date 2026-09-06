"""Device assignment and coordinated ordinary rank failures for joint training."""

from collections.abc import Callable
import traceback

from accelerate import Accelerator
import torch


def require_distinct_devices(accelerator: Accelerator) -> None:
    """Allow CPU/Gloo or require distinct local CUDA assignments."""
    if accelerator.num_processes == 1:
        return
    local_assignment = (accelerator.device.type, accelerator.device.index)
    assignments: list[object | None] = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(assignments, local_assignment)
    if all(
        isinstance(assignment, tuple) and assignment[0] == "cpu"
        for assignment in assignments
    ):
        return
    if any(
        not isinstance(assignment, tuple)
        or len(assignment) != 2
        or assignment[0] != "cuda"
        or not isinstance(assignment[1], int)
        for assignment in assignments
    ):
        raise RuntimeError(
            "multi-rank DDP requires CPU on every rank or distinct CUDA devices; "
            f"got assignments {tuple(assignments)}"
        )
    cuda_indices = tuple(int(assignment[1]) for assignment in assignments)  # type: ignore[index]
    if len(set(cuda_indices)) != accelerator.num_processes:
        raise RuntimeError(
            f"multi-rank DDP requires {accelerator.num_processes} distinct CUDA "
            f"device assignments; got {cuda_indices}"
        )


def raise_rank_errors(accelerator: Accelerator, label: str, error: str | None) -> None:
    """Exchange caught errors before a dependent collective or optimizer update."""
    errors = [error]
    if accelerator.num_processes > 1:
        errors = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(errors, error)
    if any(value is not None for value in errors):
        raise RuntimeError(f"{label} failed on a rank: {errors}")


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
