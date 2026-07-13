from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from .dataset import AutoModelMetaDataset


class AutoModelBucketBatchSampler(Sampler[list[int]]):
    """Batch sampler that keeps each AutoModel batch within one T/H/W bucket."""

    def __init__(
        self,
        dataset: AutoModelMetaDataset,
        *,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        if num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self.global_batch_size = self.batch_size * self.num_replicas
        self._total_batches = self._calculate_total_batches()

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        bucket_keys = list(self.dataset.sorted_bucket_keys)
        if self.shuffle:
            permutation = torch.randperm(len(bucket_keys), generator=generator).tolist()
            bucket_keys = [bucket_keys[index] for index in permutation]

        for key in bucket_keys:
            group = self.dataset.bucket_groups[key]
            indices = list(group["indices"])
            if self.shuffle:
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[index] for index in permutation]

            indices = self._trim_or_pad_to_global_batch(indices)
            for global_start in range(0, len(indices), self.global_batch_size):
                rank_start = global_start + self.rank * self.batch_size
                batch = indices[rank_start : rank_start + self.batch_size]
                if len(batch) != self.batch_size:
                    raise RuntimeError(
                        "internal sampler error: rank batch is not full after "
                        "global batch alignment"
                    )
                yield batch

    def __len__(self) -> int:
        return self._total_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def bucket_usable_counts(self) -> dict[object, int]:
        return {
            key: self._usable_count(len(group["indices"]))
            for key, group in self.dataset.bucket_groups.items()
        }

    def bucket_dropped_counts(self) -> dict[object, int]:
        return {
            key: len(group["indices"]) - self._usable_count(len(group["indices"]))
            for key, group in self.dataset.bucket_groups.items()
        }

    def _calculate_total_batches(self) -> int:
        count = 0
        for group in self.dataset.bucket_groups.values():
            usable_count = self._usable_count(len(group["indices"]))
            count += usable_count // self.global_batch_size
        return count

    def _usable_count(self, count: int) -> int:
        if count <= 0:
            return 0
        if self.drop_last:
            return (count // self.global_batch_size) * self.global_batch_size
        return math.ceil(count / self.global_batch_size) * self.global_batch_size

    def _trim_or_pad_to_global_batch(self, indices: list[int]) -> list[int]:
        usable_count = self._usable_count(len(indices))
        if usable_count <= 0:
            return []
        if usable_count <= len(indices):
            return indices[:usable_count]

        padding_count = usable_count - len(indices)
        padding = [indices[index % len(indices)] for index in range(padding_count)]
        return indices + padding
