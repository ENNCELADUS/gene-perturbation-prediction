"""
Distributed Data Parallel (DDP) utility functions for multi-GPU training.
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def setup_ddp(timeout_minutes: int = 60):
    """
    Initialize distributed training if running under torchrun.

    Args:
        timeout_minutes: NCCL timeout in minutes (default: 60 for long validation).

    Returns:
        Tuple of (rank, local_rank, world_size, is_distributed)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # Increase timeout to handle long validation with DE analysis
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=timeout_minutes),
        )
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_ddp(is_distributed: bool):
    """Clean up distributed training."""
    if is_distributed:
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0
