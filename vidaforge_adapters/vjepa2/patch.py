from __future__ import annotations

import os
from typing import Any


def apply_vjepa_dataset_patch() -> None:
    """Patch V-JEPA's VideoDataset factory to use the TorchCodec adapter."""
    import src.datasets.video_dataset as official_video_dataset

    from .dataset import TorchCodecVideoDataset, make_videodataset

    official_video_dataset.VideoDataset = TorchCodecVideoDataset
    official_video_dataset.make_videodataset = make_videodataset


def apply_torchrun_distributed_patch() -> None:
    """Patch V-JEPA's Slurm-oriented distributed init for torchrun/k8s."""
    import src.utils.distributed as official_distributed

    official_distributed.init_distributed = init_torchrun_distributed


def apply_vjepa_runtime_patches() -> None:
    """Apply all runtime patches needed before importing V-JEPA train modules."""
    apply_vjepa_dataset_patch()
    apply_torchrun_distributed_patch()


def init_torchrun_distributed(
    port: int = 37129,
    rank_and_world_size: tuple[int | None, int | None] = (None, None),
) -> tuple[int, int]:
    """Torchrun-compatible replacement for V-JEPA's Slurm-only init."""
    import torch
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    env = os.environ
    rank, world_size = rank_and_world_size
    rank = _env_int("RANK", rank)
    world_size = _env_int("WORLD_SIZE", world_size)
    local_rank = _env_int("LOCAL_RANK", _env_int("SLURM_LOCALID", 0))

    if rank is None or world_size is None:
        # Last-resort Slurm compatibility for users launching official jobs.
        rank = _env_int("SLURM_PROCID", 0)
        world_size = _env_int("SLURM_NTASKS", 1)

    assert rank is not None
    assert world_size is not None

    env.setdefault("MASTER_ADDR", "localhost")
    env.setdefault("MASTER_PORT", str(port))

    if torch.cuda.is_available():
        torch.cuda.set_device(_cuda_device_index_for_process(local_rank))

    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
        world_size=int(world_size),
        rank=int(rank),
    )
    return int(world_size), int(rank)


def _env_int(name: str, fallback: Any = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return fallback if fallback is None else int(fallback)
    return int(value)


def _cuda_device_index_for_process(local_rank: int | None) -> int:
    visible_devices = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if len(visible_devices) == 1:
        return 0
    return int(local_rank or 0)
