"""Small DDP invariants shared by exp05 orchestration and training."""

from collections.abc import Callable
import traceback

from accelerate import Accelerator
import torch

_CUDA_TOPOLOGY_MARKER = "_aivc_exp05_cuda_topology"


def require_exact_world_size(
    accelerator: Accelerator,
    expected: int = 4,
) -> None:
    """Require one authoritative rank on each of four distinct CUDA devices."""
    verified = getattr(accelerator, _CUDA_TOPOLOGY_MARKER, None)
    if isinstance(verified, tuple) and len(verified) == 2 and verified[0] == expected:
        return
    if accelerator.num_processes != expected:
        raise RuntimeError(
            f"authoritative exp05 requires exactly {expected} DDP ranks; "
            f"got {accelerator.num_processes}"
        )
    local_assignment = (accelerator.device.type, accelerator.device.index)
    assignments: list[object | None] = [None] * expected
    torch.distributed.all_gather_object(assignments, local_assignment)
    if any(
        not isinstance(assignment, tuple)
        or len(assignment) != 2
        or assignment[0] != "cuda"
        or not isinstance(assignment[1], int)
        for assignment in assignments
    ):
        raise RuntimeError(
            "authoritative exp05 requires CUDA on every rank; "
            f"got assignments {tuple(assignments)}"
        )
    cuda_indices = tuple(int(assignment[1]) for assignment in assignments)  # type: ignore[index]
    if len(set(cuda_indices)) != expected:
        raise RuntimeError(
            f"authoritative exp05 requires {expected} distinct CUDA device "
            f"assignments; got {cuda_indices}"
        )
    setattr(
        accelerator,
        _CUDA_TOPOLOGY_MARKER,
        (expected, tuple(assignments)),
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
