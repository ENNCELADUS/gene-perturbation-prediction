"""DDP invariants for the Exp13 Stage 1 response-model trainer."""

from collections.abc import Callable
import traceback

from accelerate import Accelerator
import torch

_CUDA_TOPOLOGY_MARKER = "_aivc_cuda_topology"


def require_distinct_devices(accelerator: Accelerator) -> None:
    """Require every DDP rank to sit on a distinct CUDA device.

    A single-process run is legal on any device (CPU, MPS, or a single CUDA
    device) and returns immediately without any collective. With more than
    one rank, every rank must report a CUDA device, and no two ranks may
    report the same CUDA device index -- two ranks sharing one GPU is the
    real failure this catches.
    """
    if accelerator.num_processes == 1:
        return
    verified = getattr(accelerator, _CUDA_TOPOLOGY_MARKER, None)
    if (
        isinstance(verified, tuple)
        and len(verified) == 2
        and verified[0] == accelerator.num_processes
    ):
        return
    local_assignment = (accelerator.device.type, accelerator.device.index)
    assignments: list[object | None] = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(assignments, local_assignment)
    if any(
        not isinstance(assignment, tuple)
        or len(assignment) != 2
        or assignment[0] != "cuda"
        or not isinstance(assignment[1], int)
        for assignment in assignments
    ):
        raise RuntimeError(
            "multi-rank DDP requires CUDA on every rank; "
            f"got assignments {tuple(assignments)}"
        )
    cuda_indices = tuple(int(assignment[1]) for assignment in assignments)  # type: ignore[index]
    if len(set(cuda_indices)) != accelerator.num_processes:
        raise RuntimeError(
            f"multi-rank DDP requires {accelerator.num_processes} distinct CUDA "
            f"device assignments; got {cuda_indices}"
        )
    setattr(
        accelerator,
        _CUDA_TOPOLOGY_MARKER,
        (accelerator.num_processes, tuple(assignments)),
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
