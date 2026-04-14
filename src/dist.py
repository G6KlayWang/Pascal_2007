from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    if "LOCAL_RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    return True, rank, world_size, local_rank, device


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(world_size())
    return tensor
