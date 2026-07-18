from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

from .collate import collate_automodel_video
from .dataset import AutoModelMetaDataset
from .sampler import AutoModelBucketBatchSampler

if TYPE_CHECKING:
    from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild

try:
    from torchdata.stateful_dataloader import StatefulDataLoader
except ImportError:  # pragma: no cover - depends on the training environment
    StatefulDataLoader = None  # type: ignore[assignment]


@dataclass
class VidaForgeVideoDataloaderConfig:
    """Typed AutoModel config for loading VidaForge Stage 5 tensor caches."""

    cache_dir: str | Path
    model_type: str = "wan"
    device: str = "cpu"
    base_resolution: tuple[int, int] = (512, 512)
    drop_last: bool = True
    shuffle: bool = True
    dynamic_batch_size: bool = False
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    map_location: str = "cpu"
    limit: int | None = None

    def build(
        self,
        *,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
    ) -> DiffusionDataloaderBuild:
        """Build the per-rank dataloader using AutoModel's typed result contract."""
        from nemo_automodel.components.datasets.diffusion.loader import (
            DiffusionDataloaderBuild,
        )

        dataloader, sampler = build_video_multiresolution_dataloader(
            cache_dir=self.cache_dir,
            model_type=self.model_type,
            device=self.device,
            batch_size=batch_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            base_resolution=self.base_resolution,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            dynamic_batch_size=self.dynamic_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            map_location=self.map_location,
            limit=self.limit,
        )
        return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler)


def build_video_multiresolution_dataloader(
    *,
    cache_dir: str | Path,
    model_type: str = "wan",
    device: str = "cpu",
    batch_size: int = 1,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    base_resolution: Sequence[int] = (512, 512),
    drop_last: bool = True,
    shuffle: bool = True,
    dynamic_batch_size: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    map_location: str = "cpu",
    limit: int | None = None,
) -> tuple[DataLoader, AutoModelBucketBatchSampler]:
    """Build an AutoModel-compatible video dataloader for Stage 5 `.meta` caches."""
    if dynamic_batch_size:
        raise NotImplementedError(
            "dynamic_batch_size is not supported for temporal-spatial AutoModel "
            "buckets; use a fixed per-rank batch_size."
        )
    if len(tuple(base_resolution)) != 2:
        raise ValueError("base_resolution must contain [width, height]")

    dataset = AutoModelMetaDataset(
        cache_dir=cache_dir,
        map_location=map_location,
        device=device,
        limit=limit,
    )
    sampler = AutoModelBucketBatchSampler(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        num_replicas=dp_world_size,
        rank=dp_rank,
    )

    collate_fn = functools.partial(collate_automodel_video, model_type=model_type)
    dataloader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["persistent_workers"] = True

    dataloader_cls = StatefulDataLoader if StatefulDataLoader is not None else DataLoader
    dataloader = dataloader_cls(**dataloader_kwargs)
    return dataloader, sampler


def build_video_multilength_multiresolution_dataloader(
    **kwargs: object,
) -> tuple[DataLoader, AutoModelBucketBatchSampler]:
    """Explicit alias for configs that want temporal length in the target name."""
    return build_video_multiresolution_dataloader(**kwargs)
